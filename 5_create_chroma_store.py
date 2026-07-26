# -*- coding: utf-8 -*-
import os
import pickle
import shutil
import chromadb
import langchain_chroma
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ==========================================
# Project Configuration & Paths
# ==========================================
PROJECT_PATH = os.getcwd()
DATA_PATH = os.path.join(PROJECT_PATH, "data")

CHUNKS_PATH = os.path.join(DATA_PATH, "chunks.pkl")
DB_PATH = os.path.join(DATA_PATH, "chroma_db")

# ==========================================
# Load Chunks
# ==========================================
if not os.path.exists(CHUNKS_PATH):
    print(f"Error: Could not find '{CHUNKS_PATH}'. Please run '03_chunking.py' first.")
    exit()

print("Loading chunks...")
with open(CHUNKS_PATH, "rb") as f:
    chunks = pickle.load(f)

print(f"Number of chunks loaded: {len(chunks)}")

# ==========================================
# Load Embedding Model
# ==========================================
print("\nLoading embedding model...")
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
print("Embedding model loaded successfully!")

# ==========================================
# Delete Old Chroma Database (If Exists)
# ==========================================
if os.path.exists(DB_PATH):
    shutil.rmtree(DB_PATH)
    print("Old Chroma database deleted successfully.")

os.makedirs(DB_PATH, exist_ok=True)

# ==========================================
# Create Chroma Vector Store
# ==========================================
print("\nCreating Chroma vector store (this may take a few moments)...")
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory=DB_PATH
)

print("Chroma database created successfully!")

# ==========================================
# Verify Chroma Database
# ==========================================
print("\n--- Environment Verification ---")
print("chromadb version:", chromadb.__version__)
print("langchain_chroma version:", langchain_chroma.__version__)

print("\n--- Chroma DB Storage Check ---")
print("Files generated in DB path:", os.listdir(DB_PATH))
print(f"Saved to: {DB_PATH}")