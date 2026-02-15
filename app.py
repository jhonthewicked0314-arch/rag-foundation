import os
from flask import Flask, request, jsonify, render_template_string
from werkzeug.utils import secure_filename
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

# 1. Get environment variables
google_api_key = os.getenv("GOOGLE_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
index_name = os.getenv("PINECONE_INDEX_NAME")

# 2. Setup Models (Using EXACT names from your list)
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001", 
    google_api_key=google_api_key
)

llm = ChatGoogleGenerativeAI(
    model="models/gemini-flash-latest", # <--- Added 'models/' prefix
    temperature=0.3, 
    google_api_key=google_api_key
)

# Configure Uploads
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

HTML_UI = """
<!DOCTYPE html>
<html>
<head>
    <title>Gemini RAG Bot</title>
    <style>
        body { font-family: 'Segoe UI', sans-serif; max-width: 800px; margin: 30px auto; background-color: #f9f9f9; }
        .container { background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }
        #drop-zone { border: 2px dashed #4285f4; border-radius: 10px; padding: 40px; text-align: center; color: #4285f4; cursor: pointer; margin-bottom: 20px; }
        #chat-box { height: 400px; overflow-y: auto; border: 1px solid #eee; padding: 15px; margin-bottom: 20px; background: #fafafa; border-radius: 8px; }
        .msg { margin-bottom: 10px; }
        .user-msg { color: #1a73e8; font-weight: bold; }
        .bot-msg { color: #333; background: #eef; padding: 8px; border-radius: 5px; }
        .input-area { display: flex; gap: 10px; }
        input[type="text"] { flex: 1; padding: 12px; border: 1px solid #ddd; border-radius: 6px; }
        button { padding: 10px 20px; background: #4285f4; color: white; border: none; border-radius: 6px; cursor: pointer; }
        #status { font-size: 0.9em; margin-top: 10px; color: #666; }
    </style>
</head>
<body>
    <div class="container">
        <h2>📚 Gemini PDF Chatbot</h2>
        <div id="drop-zone" onclick="document.getElementById('file-input').click()">
            Drag & Drop PDF here or Click to Upload
            <input type="file" id="file-input" accept=".pdf" onchange="handleFile(this.files[0])">
        </div>
        <div id="status"></div>
        <hr>
        <div id="chat-box"></div>
        <div class="input-area">
            <input type="text" id="query" placeholder="Ask a question about your PDF...">
            <button onclick="ask()">Send</button>
        </div>
    </div>
    <script>
        async function handleFile(file) {
            if (!file) return;
            const formData = new FormData();
            formData.append('file', file);
            document.getElementById('status').innerText = "⏳ Indexing PDF... please wait.";
            const res = await fetch('/upload', { method: 'POST', body: formData });
            const data = await res.json();
            document.getElementById('status').innerText = data.message || data.error;
        }

        async function ask() {
            const qInput = document.getElementById('query');
            const q = qInput.value;
            if (!q) return;

            const chatBox = document.getElementById('chat-box');
            chatBox.innerHTML += `<div class="msg user-msg">You: ${q}</div>`;
            qInput.value = "";
            
            // Show loading message
            const loadingId = "loading-" + Date.now();
            chatBox.innerHTML += `<div class="msg bot-msg" id="${loadingId}">Bot is thinking...</div>`;
            chatBox.scrollTop = chatBox.scrollHeight;

            try {
                const res = await fetch('/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({question: q})
                });
                const data = await res.json();
                document.getElementById(loadingId).innerText = "Bot: " + (data.answer || data.error);
            } catch (err) {
                document.getElementById(loadingId).innerText = "Bot: Error connecting to server.";
            }
            chatBox.scrollTop = chatBox.scrollHeight;
        }
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_UI)

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files.get('file')
    if not file or not file.filename.endswith('.pdf'):
        return jsonify({"error": "Please upload a valid PDF"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        loader = PyPDFLoader(filepath)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        splits = text_splitter.split_documents(docs)

        # Pass API Key explicitly
        PineconeVectorStore.from_documents(
            splits, 
            embeddings, 
            index_name=index_name, 
            pinecone_api_key=pinecone_api_key
        )
        os.remove(filepath)
        return jsonify({"message": f"Successfully indexed {filename}!"})
    except Exception as e:
        return jsonify({"error": f"Indexing error: {str(e)}"}), 500

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        question = data.get('question')
        
        # Pass API Key explicitly to prevent hanging
        vectorstore = PineconeVectorStore(
            index_name=index_name, 
            embedding=embeddings, 
            pinecone_api_key=pinecone_api_key
        )
        
        system_prompt = (
            "You are a helpful assistant. Use the following context to answer.\n\n{context}"
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
        return jsonify({"error": f"Chat error: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)