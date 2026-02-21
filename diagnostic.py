import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone

load_dotenv()

def diagnostic():
    api_key = os.getenv("GOOGLE_API_KEY")
    pc_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")

    print(f"Index Name: {index_name}")

    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=api_key
        )
        test_vec = embeddings.embed_query("test")
        print(f"Embedding Dimensions: {len(test_vec)}")
    except Exception as e:
        print(f"Embedding Error: {e}")

    try:
        pc = Pinecone(api_key=pc_key)
        index = pc.Index(index_name)
        stats = index.describe_index_stats()
        print(f"Pinecone Stats: {stats}")
    except Exception as e:
        print(f"Pinecone Error: {e}")

if __name__ == "__main__":
    diagnostic()
