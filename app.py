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

# Configure Uploads
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Setup Models
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3, google_api_key=google_api_key)

HTML_UI = """
<!DOCTYPE html>
<html>
<head>
    <title>Gemini RAG Bot</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; max-width: 800px; margin: 30px auto; background-color: #f9f9f9; color: #333; }
        .container { background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }
        
        /* Drag & Drop Zone */
        #drop-zone {
            border: 2px dashed #4285f4;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            color: #4285f4;
            transition: 0.3s;
            cursor: pointer;
            margin-bottom: 20px;
        }
        #drop-zone.hover { background: #e8f0fe; border-color: #1a73e8; }
        
        input[type="file"] { display: none; }
        
        #chat-box { height: 300px; overflow-y: auto; border: 1px solid #eee; padding: 15px; margin-bottom: 20px; background: #fafafa; border-radius: 8px; }
        .msg { margin-bottom: 10px; line-height: 1.4; }
        .user-msg { color: #1a73e8; font-weight: bold; }
        .bot-msg { color: #333; }
        
        .input-area { display: flex; gap: 10px; }
        input[type="text"] { flex: 1; padding: 12px; border: 1px solid #ddd; border-radius: 6px; }
        button { padding: 10px 20px; background: #4285f4; color: white; border: none; border-radius: 6px; cursor: pointer; }
        button:hover { background: #357abd; }
        #status { font-size: 0.9em; margin-top: 10px; color: #666; }
    </style>
</head>
<body>
    <div class="container">
        <h2>📚 Gemini PDF Chatbot</h2>
        
        <!-- Drag and Drop Section -->
        <div id="drop-zone" onclick="document.getElementById('file-input').click()">
            Drag & Drop a PDF here or Click to Upload
            <input type="file" id="file-input" accept=".pdf" onchange="handleFile(this.files[0])">
        </div>
        <div id="status"></div>

        <hr>

        <!-- Chat Section -->
        <div id="chat-box"></div>
        <div class="input-area">
            <input type="text" id="query" placeholder="Ask a question about your PDF...">
            <button onclick="ask()">Send</button>
        </div>
    </div>

    <script>
        // Drag & Drop Logic
        const dropZone = document.getElementById('drop-zone');
        
        ['dragenter', 'dragover'].forEach(name => {
            dropZone.addEventListener(name, (e) => {
                e.preventDefault();
                dropZone.classList.add('hover');
            }, false);
        });

        ['dragleave', 'drop'].forEach(name => {
            dropZone.addEventListener(name, (e) => {
                e.preventDefault();
                dropZone.classList.remove('hover');
            }, false);
        });

        dropZone.addEventListener('drop', (e) => {
            const file = e.dataTransfer.files[0];
            handleFile(file);
        }, false);

        async function handleFile(file) {
            if (!file || file.type !== "application/pdf") {
                alert("Please upload a valid PDF file.");
                return;
            }

            const formData = new FormData();
            formData.append('file', file);

            document.getElementById('status').innerText = "⏳ Processing PDF... this may take 30 seconds.";
            
            try {
                const res = await fetch('/upload', { method: 'POST', body: formData });
                const data = await res.json();
                document.getElementById('status').innerText = data.message || data.error;
            } catch (err) {
                document.getElementById('status').innerText = "Error uploading file.";
            }
        }

        async function ask() {
            const qInput = document.getElementById('query');
            const q = qInput.value;
            if (!q) return;

            const chatBox = document.getElementById('chat-box');
            chatBox.innerHTML += `<div class="msg user-msg">You: ${q}</div>`;
            qInput.value = "";

            const res = await fetch('/chat', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({question: q})
            });
            const data = await res.json();
            chatBox.innerHTML += `<div class="msg bot-msg">Bot: ${data.answer || data.error}</div>`;
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
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and file.filename.endswith('.pdf'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # 1. Load and Split
            loader = PyPDFLoader(filepath)
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            splits = text_splitter.split_documents(docs)

            # 2. Embed and Store in Pinecone
            PineconeVectorStore.from_documents(splits, embeddings, index_name=index_name)

            # 3. Clean up (delete local file)
            os.remove(filepath)

            return jsonify({"message": f"Successfully indexed {filename}!"})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    return jsonify({"error": "Invalid file type. Please upload a PDF."}), 400

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        question = data.get('question')
        
        vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
        
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