import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = "models/embedding-001"
result = genai.embed_content(model=model, content="hello")
print(f"Model: {model}")
print(f"Dimension: {len(result['embedding'])}")
