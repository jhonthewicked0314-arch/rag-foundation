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

# 1. Get environment variables (Render automatically provides these)
google_api_key = os.getenv("GOOGLE_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
index_name = os.getenv("PINECONE_INDEX_NAME")

# 2. Setup AI Models
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001", 
    google_api_key=google_api_key
)

llm = ChatGoogleGenerativeAI(
    model="models/gemini-flash-latest", 
    temperature=0.3, 
    google_api_key=google_api_key
)

# 3. Global Vectorstore Reference (Optimized for Live Performance)
vectorstore = PineconeVectorStore(
    index_name=index_name, 
    embedding=embeddings, 
    pinecone_api_key=pinecone_api_key
)

# Configure Uploads
UPLOAD_FOLDER = '/tmp'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

HTML_UI = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🎓 College Admission Assistant</title>
    <!-- Marked.js for Rendering AI Formatting (Stars/Symbols) -->
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <style>
        :root { --primary: #1a237e; --bg: #f0f2f5; --text: #333; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; max-width: 800px; margin: 30px auto; background-color: var(--bg); color: var(--text); line-height: 1.6; }
        .container { background: white; padding: 40px; border-radius: 16px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); border-top: 6px solid var(--primary); }
        
        h2 { color: var(--primary); text-align: center; margin-bottom: 30px; }
        
        /* Drag & Drop Styling */
        #drop-zone { 
            border: 2px dashed var(--primary); 
            border-radius: 12px; 
            padding: 50px 20px; 
            text-align: center; 
            color: var(--primary); 
            cursor: pointer; 
            margin-bottom: 25px; 
            transition: all 0.3s ease;
            background: #f8f9ff;
        }
        #drop-zone.dragover { background: #e8eaf6; transform: scale(1.02); }
        
        #status { font-size: 0.95em; text-align: center; margin-top: -10px; margin-bottom: 20px; color: var(--primary); font-weight: 600; min-height: 1.5em; }
        
        /* Chat UI Styling */
        #chat-box { height: 450px; overflow-y: auto; border: 1px solid #ddd; padding: 20px; margin-bottom: 20px; background: #ffffff; border-radius: 10px; display: flex; flex-direction: column; gap: 15px; }
        .msg { max-width: 85%; padding: 12px 18px; border-radius: 12px; font-size: 0.95em; position: relative; }
        .user-msg { align-self: flex-end; background: var(--primary); color: white; border-bottom-right-radius: 2px; }
        .bot-msg { align-self: flex-start; background: #f1f1f1; color: #333; border-bottom-left-radius: 2px; border-left: 4px solid var(--primary); }
        
        /* Markdown Rendering Styles */
        .bot-msg p { margin: 0 0 10px 0; }
        .bot-msg ul, .bot-msg ol { padding-left: 20px; margin-bottom: 10px; }
        .bot-msg li { margin-bottom: 5px; }
        
        .input-area { display: flex; gap: 12px; }
        input[type="text"] { flex: 1; padding: 14px; border: 2px solid #eee; border-radius: 8px; outline: none; transition: border-color 0.3s; }
        input[type="text"]:focus { border-color: var(--primary); }
        
        button { padding: 12px 24px; background: var(--primary); color: white; border: none; border-radius: 8px; cursor: pointer; font-weight: bold; transition: opacity 0.2s; }
        button:hover { opacity: 0.9; }
        
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-thumb { background: #ccc; border-radius: 10px; }
    </style>
</head>
<body>
    <div class="container">
        <h2>🎓 College Admission Assistant</h2>
        <div id="drop-zone">
            Drop your College Prospectus (PDF) here or Click to Select
        </div>
        <input type="file" id="file-input" accept=".pdf" style="display:none">
        <div id="status">Ready to assist.</div>
        
        <div id="chat-box"></div>
        
        <div class="input-area">
            <input type="text" id="query" placeholder="Ask about admissions, courses, fees...">
            <button onclick="ask()">Ask Assistant</button>
        </div>
    </div>

    <script>
        const dropZone = document.getElementById('drop-zone');
        const fileInput = document.getElementById('file-input');
        const status = document.getElementById('status');
        const chatBox = document.getElementById('chat-box');

        // Prevent Loop & Repetitive Dialogs
        dropZone.onclick = () => fileInput.click();

        fileInput.onchange = (e) => {
            if (e.target.files.length) handleFile(e.target.files[0]);
        };

        // Proper Drag & Drop Handling
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(event => {
            dropZone.addEventListener(event, e => {
                e.preventDefault();
                e.stopPropagation();
            });
        });

        dropZone.addEventListener('dragover', () => dropZone.classList.add('dragover'));
        dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
        dropZone.addEventListener('drop', (e) => {
            dropZone.classList.remove('dragover');
            if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
        });

        async function handleFile(file) {
            if (!file || !file.name.endsWith('.pdf')) return;
            
            status.innerText = "⏳ Reading and Indexing " + file.name + "...";
            const formData = new FormData();
            formData.append('file', file);
            
            try {
                const res = await fetch('/upload', { method: 'POST', body: formData });
                const data = await res.json();
                
                if (res.ok) {
                    status.innerText = data.message || "Successfully indexed!";
                } else {
                    status.innerText = "❌ Indexing failed: " + (data.error || res.statusText);
                }
                fileInput.value = ""; 
            } catch (err) {
                status.innerText = "❌ Connection failed. Your PDF might be too large for a 30s timeout.";
            }
        }

        async function ask() {
            const input = document.getElementById('query');
            const q = input.value.trim();
            if (!q) return;

            chatBox.innerHTML += `<div class="msg user-msg">${q}</div>`;
            input.value = "";
            
            const loadingId = "loading-" + Date.now();
            chatBox.innerHTML += `<div class="msg bot-msg" id="${loadingId}">Assistant is typing...</div>`;
            chatBox.scrollTop = chatBox.scrollHeight;

            try {
                const res = await fetch('/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({question: q})
                });
                const data = await res.json();
                const botDiv = document.getElementById(loadingId);
                
                if (data.answer) {
                    botDiv.innerHTML = marked.parse(data.answer);
                } else {
                    botDiv.innerText = data.error || "I'm sorry, I encountered an issue.";
                }
            } catch (err) {
                document.getElementById(loadingId).innerText = "❌ Error connecting to server.";
            }
            chatBox.scrollTop = chatBox.scrollHeight;
        }

        // Support Enter Key
        document.getElementById('query').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') ask();
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
    file = request.files.get('file')
    if not file or not file.filename.endswith('.pdf'):
        return jsonify({"error": "Please provide a valid PDF prospectus."}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        loader = PyPDFLoader(filepath)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
        splits = text_splitter.split_documents(docs)

        # Update Pinecone Index with the provided key
        PineconeVectorStore.from_documents(
            splits, 
            embeddings, 
            index_name=index_name, 
            pinecone_api_key=pinecone_api_key
        )
        os.remove(filepath)
        return jsonify({"message": f"Successfully indexed {filename}! You can now ask questions."})
    except Exception as e:
        return jsonify({"error": f"Upload process failed: {str(e)}"}), 500

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        question = data.get('question')
        
        # Customized Persona for College Admission
        system_prompt = (
            "You are an official College Admission Assistant. Your role is to provide polite, "
            "professional, and accurate information about courses, fees, scholarships, and campus life "
            "based ONLY on the provided document context.\n"
            "If the information is NOT in the document, reply: 'I'm sorry, I couldn't find that specific information "
            "in our prospectus. Please contact the administrative office at the college for detail.'\n\n"
            "Context: {context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        # Optimize chain for live speed (k=4)
        doc_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(vectorstore.as_retriever(search_kwargs={"k": 4}), doc_chain)
        
        response = rag_chain.invoke({"input": question})
        return jsonify({"answer": response["answer"]})
    except Exception as e:
        return jsonify({"error": f"Assistant is busy: {str(e)}"}), 500

if __name__ == "__main__":
    # Handle Render's dynamic port assignment
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
