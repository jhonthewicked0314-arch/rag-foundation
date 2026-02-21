import google.generativeai as genai

genai.configure(api_key="AIzaSyAkF77yIhxpKGm2fiadzs3WnQA7rpo27MA")

for model in genai.list_models():
    if "generateContent" in model.supported_generation_methods:
        print(model.name)