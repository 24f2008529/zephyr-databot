import io
import json
import os
import re
import sys
import time
import threading
import traceback
from contextlib import asynccontextmanager, redirect_stdout, redirect_stderr
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from openai import OpenAI, RateLimitError

# 1. Load environment variables from .env file immediately
load_dotenv()

# --- Configuration & Paths ---
LOG_FILE = Path("run.jsonl")
LOG_FILE.touch(exist_ok=True)  # Prevents 404 on GET /run.jsonl

BOT_TOKEN = os.getenv("BOT_TOKEN")
BASE_URL = os.getenv("BASE_URL", "http://localhost:8000").rstrip("/")

# --- Gemini API Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/").rstrip("/")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-3.6-flash")
APP_VERSION = "final-answer-guard-2026-07-29"

log_lock = threading.Lock()

def log_event(event: dict) -> None:
    """Thread-safe append of a JSON object to run.jsonl."""
    event_with_ts = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **event
    }
    with log_lock:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(event_with_ts) + "\n")

# Initialize OpenAI client pointing to Gemini's endpoint
openai_client = OpenAI(
    api_key=GEMINI_API_KEY,
    base_url=GEMINI_BASE_URL
)

# --- Python Code Execution Tool ---
MAX_OUTPUT_CHARS = 8000

def run_python(code: str, namespace: dict | None = None) -> str:
    """Executes Python code, capturing output and optionally preserving its state."""
    buffer = io.StringIO()
    # A single namespace is important: imports and variables from one tool call
    # must be available to later calls in the same user turn.
    exec_namespace = namespace if namespace is not None else {}
    exec_namespace.setdefault("__builtins__", __builtins__)

    try:
        with redirect_stdout(buffer), redirect_stderr(buffer):
            exec(code, exec_namespace, exec_namespace)
    except Exception:
        buffer.write("\n" + traceback.format_exc())

    output = buffer.getvalue()
    if not output.strip():
        output = "[Code executed successfully with no output to stdout/stderr]"

    if len(output) > MAX_OUTPUT_CHARS:
        output = (
            f"... [Output truncated, showing last {MAX_OUTPUT_CHARS} characters] ...\n"
            + output[-MAX_OUTPUT_CHARS:]
        )
    return output

# --- System Prompt & Tools ---
SYSTEM_PROMPT = """
You are an expert data analyst AI handling multi-turn user queries.

STRICT OPERATIONAL RULES:
1. Answer the user's latest query directly. Earlier messages in the chat session provide critical context for multi-turn requests.
2. Use the `run_python` tool to fetch datasets, parse files (CSV, XLSX, HTML tables, MOSPI data), and calculate values. NEVER guess numerical or statistical values that can be computed.
3. If dataset fetching/downloading fails after attempts, fall back gracefully to your internal knowledge base.
4. Output MUST be ONLY a single valid JSON object formatted exactly as requested by the question.
5. NEVER include markdown formatting (no ```json code fences), explanations, or conversational filler outside the JSON.
6. Insert the exact string `"PLACEHOLDER_LOG_URL"` inside the `"log_url"` key of your returned JSON object if a log_url is required.
7. If a message is a multi-turn setup (e.g., "I will send data next"), reply with a small valid JSON acknowledgement (e.g., {"status": "ready"}).
8. Keep API usage within the Gemini free-tier request limit: use at most 3 `run_python` calls per user turn, reuse variables/imports from earlier tool calls, and combine related web/data work into one tool call whenever possible.
9. The `answer` value must contain ONLY the value in the shape requested by the user. Do not include evidence, reasoning, Markdown, tables, headings, or citations inside `answer`. For a direct fact question with no explicit answer shape, return the shortest answer possible (for example, `"Assam"`, not an explanation of Assam).
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": "Executes Python code on the server to download, process, parse, and analyze data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Executable Python code snippet."
                    }
                },
                "required": ["code"]
            }
        }
    }
]

# --- Chat History Memory ---
chat_histories: dict[int, list[dict]] = {}
history_lock = threading.Lock()
MAX_HISTORY_TURNS = 20

def retry_delay_seconds(error: Exception, attempt: int) -> float:
    """Use the provider's suggested 429 delay when available."""
    match = re.search(r"retry in\s+([0-9.]+)\s*(?:s|seconds?)", str(error), re.IGNORECASE)
    if match:
        return min(float(match.group(1)) + 1, 75.0)
    return min(2 ** attempt, 30.0)

def create_completion_with_retry(*, chat_id: int, **kwargs):
    """Retry transient Gemini quota errors instead of failing the Telegram turn."""
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            return openai_client.chat.completions.create(**kwargs)
        except RateLimitError as error:
            if attempt == max_attempts:
                raise
            delay = retry_delay_seconds(error, attempt)
            log_event({
                "chat_id": chat_id,
                "event": "rate_limit_retry",
                "attempt": attempt,
                "delay_seconds": delay,
                "error": str(error),
            })
            time.sleep(delay)

def get_chat_history(chat_id: int) -> list[dict]:
    with history_lock:
        return list(chat_histories.get(chat_id, []))

def append_chat_history(chat_id: int, new_messages: list[dict]) -> None:
    with history_lock:
        history = chat_histories.get(chat_id, [])
        history.extend(new_messages)
        if len(history) > MAX_HISTORY_TURNS:
            history = history[-MAX_HISTORY_TURNS:]
        chat_histories[chat_id] = history

# --- Agent Loop ---
# --- Agent Loop ---
def run_agent_loop(chat_id: int, user_message: str) -> str:
    start_time = time.time()
    TIME_BUDGET = 210.0  # 210 seconds safety limit
    # Reserve the final model request for an answer after at most three tool
    # calls. This stays below the free-tier limit and never returns an empty
    # fallback just because the model kept asking for tools.
    # Three tool calls, one forced-final response, and one recovery response
    # if a provider emits a stray tool call despite tool_choice="none".
    MAX_STEPS = 5
    MAX_TOOL_CALLS = 3
    log_event({
        "chat_id": chat_id,
        "event": "agent_started",
        "app_version": APP_VERSION,
        "process_id": os.getpid(),
        "max_tool_calls": MAX_TOOL_CALLS,
    })

    history = get_chat_history(chat_id)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history + [{"role": "user", "content": user_message}]
    turn_added_messages = [{"role": "user", "content": user_message}]
    python_namespace: dict = {"__builtins__": __builtins__}
    tool_calls_used = 0

    step = 0
    while step < MAX_STEPS:
        step += 1
        elapsed = time.time() - start_time

        if elapsed >= TIME_BUDGET:
            log_event({"chat_id": chat_id, "event": "timeout_warning", "elapsed": elapsed})
            timeout_response = create_completion_with_retry(
                chat_id=chat_id,
                model=MODEL_NAME,
                messages=messages + [{
                    "role": "system",
                    "content": "WALL-CLOCK TIME EXHAUSTED. Output your best final JSON answer immediately. Do NOT request any tool calls."
                }],
                temperature=0
            )
            final_content = timeout_response.choices[0].message.content or "{}"
            turn_added_messages.append({"role": "assistant", "content": final_content})
            append_chat_history(chat_id, turn_added_messages)
            return final_content

        tool_budget_available = tool_calls_used < MAX_TOOL_CALLS
        completion_messages = messages
        if not tool_budget_available:
            completion_messages = messages + [{
                "role": "system",
                "content": (
                    "Your tool budget is exhausted. Return the best final JSON answer now; do not call tools. "
                    "Never return {} or an empty answer. Use the evidence already collected and, if needed, "
                    "your internal knowledge. Include the required top-level answer key. The answer must be only "
                    "the requested value and shape: no explanation, Markdown, evidence, table, or citations."
                )
            }]

        completion_kwargs = {
            "model": MODEL_NAME,
            "messages": completion_messages,
            "temperature": 0,
        }
        if tool_budget_available:
            completion_kwargs.update({"tools": TOOLS, "tool_choice": "auto"})
        else:
            # Gemini may otherwise reuse the previous turn's tool definition
            # and issue another call, leaving the response content empty.
            completion_kwargs.update({"tools": TOOLS, "tool_choice": "none"})

        response = create_completion_with_retry(chat_id=chat_id, **completion_kwargs)

        response_msg = response.choices[0].message

        # FIX: Exclude None values to prevent sending null fields to Gemini
        assistant_dict = response_msg.model_dump(exclude_none=True)
        messages.append(assistant_dict)

        if response_msg.tool_calls:
            for tool_call in response_msg.tool_calls:
                if tool_call.function.name == "run_python":
                    if not tool_budget_available:
                        # Do not execute a fourth tool call. Complete the tool
                        # protocol, then use the reserved recovery response to
                        # obtain the final JSON answer.
                        code_to_run = ""
                        tool_output = "Tool budget exhausted. Return the final JSON answer without tools."
                        log_event({
                            "chat_id": chat_id,
                            "event": "unexpected_tool_call_blocked",
                            "tool_calls_used": tool_calls_used,
                        })
                    else:
                        tool_calls_used += 1
                        try:
                            args = json.loads(tool_call.function.arguments)
                            code_to_run = args.get("code", "")
                        except Exception as parse_err:
                            tool_output = f"Error parsing arguments: {parse_err}"
                            code_to_run = ""
                        else:
                            tool_output = run_python(code_to_run, python_namespace)

                    log_event({
                        "chat_id": chat_id,
                        "event": "tool_call",
                        "tool_calls_used": tool_calls_used,
                        "code": code_to_run,
                        "output": tool_output
                    })

                    tool_message = {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_output
                    }
                    messages.append(tool_message)
        else:
            final_content = response_msg.content or "{}"
            turn_added_messages.append({"role": "assistant", "content": final_content})
            append_chat_history(chat_id, turn_added_messages)
            return final_content

    fallback_content = response_msg.content or "{}"
    turn_added_messages.append({"role": "assistant", "content": fallback_content})
    append_chat_history(chat_id, turn_added_messages)
    return fallback_content

# --- Defensive Layer: JSON Cleanup & log_url Injection ---
def sanitize_and_format_response(raw_response: str) -> str:
    """
    Extracts valid JSON, ensures log_url is populated with our real public server URL,
    and strips Markdown formatting.
    """
    cleaned = raw_response.strip()

    # Strip markdown code blocks if the model accidentally included them
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    # Extract first balanced JSON object from string
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(0)

    try:
        parsed = json.loads(cleaned)
    except Exception:
        parsed = {"answer": raw_response}

    if not isinstance(parsed, dict):
        parsed = {"answer": parsed}
    elif "answer" not in parsed:
        # The grading protocol always requires a top-level answer key, even if
        # a model returned only the requested inner object.
        parsed = {"answer": parsed}

    # Substitute the real public log URL
    real_log_url = f"{BASE_URL}/run.jsonl"
    parsed["log_url"] = real_log_url

    return json.dumps(parsed)

# --- Telegram Long-Polling Loop ---
def send_telegram_message(chat_id: int, text: str) -> None:
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    try:
        httpx.post(url, json=payload, timeout=10.0)
    except Exception as e:
        log_event({"chat_id": chat_id, "event": "send_message_error", "error": str(e)})

def telegram_poller():
    """Background thread running Telegram long-polling loop."""
    if not BOT_TOKEN:
        print("[WARNING] BOT_TOKEN is missing. Poller disabled.")
        return

    offset = 0
    poll_url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates"
    print("[INFO] Telegram long-poller active and waiting for updates...")

    while True:
        try:
            res = httpx.get(poll_url, params={"offset": offset, "timeout": 30}, timeout=40.0)
            if res.status_code == 200:
                data = res.json()
                for update in data.get("result", []):
                    offset = update["update_id"] + 1
                    message = update.get("message", {})
                    chat_id = message.get("chat", {}).get("id")
                    user_text = message.get("text")

                    if chat_id and user_text:
                        log_event({"chat_id": chat_id, "event": "user_message", "text": user_text})

                        try:
                            raw_reply = run_agent_loop(chat_id, user_text)
                            final_json_reply = sanitize_and_format_response(raw_reply)
                        except Exception as agent_err:
                            log_event({"chat_id": chat_id, "event": "fatal_agent_error", "error": str(agent_err), "trace": traceback.format_exc()})
                            final_json_reply = json.dumps({
                                "answer": "internal_error",
                                "log_url": f"{BASE_URL}/run.jsonl"
                            })

                        send_telegram_message(chat_id, final_json_reply)
                        log_event({"chat_id": chat_id, "event": "bot_reply", "reply": final_json_reply})
        except Exception as poll_err:
            log_event({"event": "poll_error", "error": str(poll_err)})
            time.sleep(5)

# --- Keep-Alive Self-Pinger ---
def keep_alive_pinger():
    """Pings local health endpoint every 10 minutes to keep server active."""
    time.sleep(10)
    while True:
        try:
            httpx.get(f"{BASE_URL}/health", timeout=10.0)
        except Exception:
            pass
        time.sleep(600)

# --- FastAPI App Lifecycle ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=telegram_poller, daemon=True).start()
    threading.Thread(target=keep_alive_pinger, daemon=True).start()
    yield

app = FastAPI(title="Data-Analyst Telegram Bot", lifespan=lifespan)

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "version": APP_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

@app.get("/run.jsonl")
def get_logs():
    return FileResponse(path=LOG_FILE, media_type="text/plain", filename="run.jsonl")
