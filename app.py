from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import os

from file_parser import FileParser
from embeddings import EmbeddingModel
from qa_engine import QAEngine
from vector_store import VectorStore

# =========================
# APP INIT
# =========================
app = FastAPI()

# =========================
# CORS FIX
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# INIT COMPONENTS
# =========================
parser = FileParser()
embedder = EmbeddingModel()

vector_store = VectorStore(dimension=384)
qa_engine = QAEngine(vector_store)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# =========================
# HOME
# =========================
@app.get("/")
def home():
    return {"message": "RAG system running"}


# =========================
# UPLOAD
# =========================
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    # save file
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # parse text
    text = parser.parse(file_path)

    # chunk text
    chunks = parser.chunk_text(text)

    # safety check (IMPORTANT FIX)
    if not chunks:
        return {"message": "No text found in file"}

    # embeddings (FORCE FLOAT32)
    embeddings = embedder.get_embeddings(chunks)

    # safety check (IMPORTANT FIX)
    if embeddings is None or len(embeddings) == 0:
        return {"message": "Embedding failed"}

    # store in FAISS
    vector_store.add(chunks, embeddings)

    return {"message": "File uploaded and processed successfully"}


# =========================
# ASK
# =========================
@app.post("/ask")
def ask(question: str):

    answer = qa_engine.ask(question)

    return {"answer": answer}