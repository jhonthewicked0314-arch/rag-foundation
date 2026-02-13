import os
from flask import Flask, request, jsonify, render_template_string
from dotenv import load_dotenv

# LangChain / Pinecone / Google Imports
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

app = Flask(__name__)

# 1. Setup Models
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

index_name = os.getenv("PINECONE_INDEX_NAME")

# 2. Simple UI for testing
HTML_UI = """
<!DOCTYPE html>
<html>
<head><title>Gemini RAG Bot</title></head>
<body style="font-family: sans-serif; max-width: 600px; margin: 50px auto; padding: 20px; border: 1px solid #ccc; border-radius: 10px;">
    <h2>1. Ingest PDF</h2>
    <p>Place your PDF in the project folder and type the filename here:</p>
    <input type="text" id="filename" placeholder="example.pdf">
    <button onclick="ingest()">Process PDF</button>
    <p id="status"></p>
    <hr>
    <h2>2. Chat</h2>
    <input type="text" id="query" style="width: 80%;" placeholder="Ask a question...">
    <button onclick="ask()">Ask</button>
    <div id="response" style="margin-top: 20px; white-space: pre-wrap; background: #f4f4f4; padding: 10px;"></div>

    <script>
        async function ingest() {
            const fname = document.getElementById('filename').value;
            document.getElementById('status').innerText = "Processing...";
            const res = await fetch('/ingest', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({filename: fname})
            });
            const data = await res.json();
            document.getElementById('status').innerText = data.message || data.error;
        }

        async function ask() {
            const q = document.getElementById('query').value;
            document.getElementById('response').innerText = "Thinking...";
            const res = await fetch('/chat', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({question: q})
            });
            const data = await res.json();
            document.getElementById('response').innerText = data.answer || data.error;
        }
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_UI)

# 3. Route to process a PDF and save to Pinecone
@app.route('/ingest', methods=['POST'])
def ingest():
    try:
        data = request.json
        filename = data.get('filename')
        
        # Load PDF
        loader = PyPDFLoader(filename)
        docs = loader.load()
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        splits = text_splitter.split_documents(docs)
        
        # Upload to Pinecone
        PineconeVectorStore.from_documents(splits, embeddings, index_name=index_name)
        
        return jsonify({"message": f"Successfully indexed {len(splits)} chunks from {filename}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 4. Route to ask questions
@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        question = data.get('question')
        
        # Initialize Vector Store
        vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
        
        # Create RAG Chain
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, just say that you don't know. "
            "\n\n"
            "{context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(vectorstore.as_retriever(), question_answer_chain)
        
        response = rag_chain.invoke({"input": question})
        return jsonify({"answer": response["answer"]})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)