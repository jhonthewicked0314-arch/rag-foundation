import requests

API_KEY = "AIzaSyAjjX6eb4n2sOrOZkKkaE5S7AlfUbI7RdQ" # Paste your AIzaSy key here
url = f"https://generativelanguage.googleapis.com/v1beta/models?key={API_KEY}"

response = requests.get(url)
models = response.json().get('models', [])

print("🟢 FREE EMBEDDING MODELS YOU CAN USE:")
for m in models:
    if 'embedContent' in m.get('supportedGenerationMethods', []):
        print(f" - {m['name']}")

print("\n🟢 FREE CHAT MODELS YOU CAN USE:")
for m in models:
    if 'generateContent' in m.get('supportedGenerationMethods', []):
        if "gemini" in m['name'].lower():
            print(f" - {m['name']}")