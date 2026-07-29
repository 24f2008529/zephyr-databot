import os
from google import genai

# Make sure GEMINI_API_KEY is set in your environment
client = genai.Client()

response = client.models.generate_content(
    model="gemini-1.5-flash",
    contents="Give me a 1-sentence motivational quote.",
)

print(response.text)
