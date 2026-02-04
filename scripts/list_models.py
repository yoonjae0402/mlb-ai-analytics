
import os
from google import genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("No GEMINI_API_KEY found.")
    exit(1)

client = genai.Client(api_key=api_key)

print("Listing models...")
try:
    models = client.models.list()
    for m in models:
        print(f"Model: {m.name}")
except Exception as e:
    print(f"Error listing models: {e}")
