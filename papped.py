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

# 1. Load the environment
load_dotenv()

app = Flask(__name__)

# 2. Get environment variables
google_api_key = os.getenv("GOOGLE_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
index_name = os.getenv("PINECONE_INDEX_NAME")

# 3. Setup Models (Using names confirmed by your list_models.py script)
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001", # This IS in your list (31st line)
    google_api_key=google_api_key
)

llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest", # This is the free standard model from your list
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
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <style>
        body { font-family: 'Segoe UI', sans-serif; max-width: 850px; margin: 30px auto; background-color: #f4f7f6; color: #333; }
        .container { background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.08); }
        
        /* Drag & Drop Zone Styling */
        #drop-zone {
            border: 3px dashed #4285f4;
            border-radius: 15px;
            padding: 60px 20px;
            text-align: center;
            color: #4285f4;
            font-weight: bold;
            background: #f8faff;
            transition: all 0.2s ease;
            cursor: pointer;
            margin-bottom: 20px;
            position: relative;
        }
        #drop-zone.hover { 
            background: #e8f0fe; 
            border-color: #1a73e8; 
            box-shadow: inset 0 0 10px rgba(66,133,244,0.2);
        }
        #drop-zone p { margin: 0; pointer-events: none; } /* Prevents text from blocking the drop */
        
        input[type="file"] { display: none; }
        
        #chat-box { height: 450px; overflow-y: auto; border: 1px solid #eee; padding: 20px; margin-bottom: 20px; background: #fafafa; border-radius: 10px; display: flex; flex-direction: column; }
        .msg { margin-bottom: 15px; padding: 12px 16px; border-radius: 10px; line-height: 1.6; max-width: 85%; }
        .user-msg { background: #e3f2fd; color: #0d47a1; align-self: flex-end; border-bottom-right-radius: 2px; }
        .bot-msg { background: white; border: 1px solid #eee; align-self: flex-start; border-bottom-left-radius: 2px; }

        .input-area { display: flex; gap: 10px; }
        input[id="query"] { flex: 1; padding: 15px; border: 1px solid #ddd; border-radius: 8px; font-size: 16px; }
        button { padding: 10px 25px; background: #4285f4; color: white; border: none; border-radius: 8px; cursor: pointer; font-weight: bold; }
        #status { font-size: 0.9em; margin-top: 10px; color: #666; font-style: italic; }
    </style>
</head>
<body>
    <div class="container">
        <h2>📚 Gemini Knowledge Bot</h2>
        
        <!-- Drag & Drop Zone -->
        <div id="drop-zone">
            <p id="drop-text">📁 Drag & Drop PDF here or Click to Browse</p>
            <input type="file" id="file-input" accept=".pdf">
        </div>
        <div id="status">Ready.</div>

        <hr style="border:0; border-top:1px solid #eee; margin:25px 0;">

        <div id="chat-box"></div>
        <div class="input-area">
            <input type="text" id="query" placeholder="Ask about your PDF...">
            <button onclick="ask()">Send</button>
        </div>
    </div>

    <script>
        const dropZone = document.getElementById('drop-zone');
        const fileInput = document.getElementById('file-input');
        const status = document.getElementById('status');

        // --- BULLETPROOF DRAG & DROP LOGIC ---
        
        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }

        // 1. Prevent browser from opening the file when dropped
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, preventDefaults, false);
            document.body.addEventListener(eventName, preventDefaults, false);
        });

        // 2. Add visual 'hover' state
        ['dragenter', 'dragover'].forEach(eventName => {
            dropZone.addEventListener(eventName, () => dropZone.classList.add('hover'), false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, () => dropZone.classList.remove('hover'), false);
        });

        // 3. Handle actual drop
        dropZone.addEventListener('drop', (e) => {
            const dt = e.dataTransfer;
            const file = dt.files[0];
            handleFile(file);
        }, false);

        // 4. Handle click upload
        dropZone.onclick = () => fileInput.click();
        fileInput.onchange = (e) => handleFile(e.target.files[0]);

        async function handleFile(file) {
            if (!file || file.type !== "application/pdf") {
                alert("Please drop a valid PDF file.");
                return;
            }
            const formData = new FormData();
            formData.append('file', file);
            status.innerText = "⏳ Uploading & Indexing " + file.name + "...";
            
            try {
                const res = await fetch('/upload', { method: 'POST', body: formData });
                const data = await res.json();
                status.innerText = data.message || data.error;
            } catch (err) {
                status.innerText = "❌ Connection Error.";
            }
        }

        async function ask() {
            const qInput = document.getElementById('query');
            const q = qInput.value.trim();
            if (!q) return;

            const chatBox = document.getElementById('chat-box');
            chatBox.innerHTML += `<div class="msg user-msg"><b>You:</b> ${q}</div>`;
            qInput.value = "";

            const loadingId = "loading-" + Date.now();
            chatBox.innerHTML += `<div class="msg bot-msg" id="${loadingId}">⏳ Thinking...</div>`;
            chatBox.scrollTop = chatBox.scrollHeight;

            try {
                const res = await fetch('/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({question: q})
                });
                const data = await res.json();
                
                if (!res.ok) {
                    document.getElementById(loadingId).innerText = "Bot: " + (data.error || "Error");
                    return;
                }

                // Parse markdown stars into bold/lists
                const formattedAnswer = marked.parse(data.answer);
                document.getElementById(loadingId).innerHTML = `<b>Bot:</b><br>${formattedAnswer}`;
            } catch (err) {
                document.getElementById(loadingId).innerText = "Bot: Connection timeout. Try again.";
            }
            chatBox.scrollTop = chatBox.scrollHeight;
        }
        
        // Allow 'Enter' key to send message
        document.getElementById("query").addEventListener("keypress", function(event) {
          if (event.key === "Enter") {
            event.preventDefault();
            ask();
          }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_UI)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file"}), 400

    if file and file.filename.endswith('.pdf'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        try:
            loader = PyPDFLoader(filepath)
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            splits = text_splitter.split_documents(docs)
            
            # Use explicit keys
            PineconeVectorStore.from_documents(
                splits, 
                embeddings, 
                index_name=index_name, 
                pinecone_api_key=pinecone_api_key
            )
            os.remove(filepath)
            return jsonify({"message": f"Successfully indexed {filename}!"})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify({"error": "Invalid file type"}), 400

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        question = data.get('question')
        
        vectorstore = PineconeVectorStore(
            index_name=index_name, 
            embedding=embeddings, 
            pinecone_api_key=pinecone_api_key
        )
        
      system_prompt = (
            "You are a helpful assistant. Use the following pieces of context to answer the question. "
            "Answer concisely in PLAIN TEXT. "
            "IMPORTANT: Do not use Markdown, bolding (no stars like **), or special formatting. "
            "Just give a clean, simple text response."
            "\n\n"
            "{context}"
        )
        
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(vectorstore.as_retriever(), question_answer_chain)
        
        response = rag_chain.invoke({"input": question})
        return jsonify({"answer": response["answer"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)