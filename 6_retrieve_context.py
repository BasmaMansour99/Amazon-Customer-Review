# -*- coding: utf-8 -*-
import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ==========================================
# Project Configuration & DB Path
# ==========================================
PROJECT_PATH = os.getcwd()

# مسار مجلد قاعدة البيانات المحلية (يمكنك تعديله إذا كان مجلد chroma_db في مكان آخر)
DB_PATH = os.path.join(PROJECT_PATH, "data", "chroma_db")

# ==========================================
# Load Embedding Model
# ==========================================
print("Loading embedding model...")
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
print("Embedding model loaded!")

# ==========================================
# Load Vector Store
# ==========================================
if not os.path.exists(DB_PATH):
    print(f"\nError: Could not find Chroma DB at path: '{DB_PATH}'")
    print("Please make sure you generated and saved the database in step 05.")
    exit()

print("Loading Chroma vector store from local storage...")
vector_store = Chroma(
    persist_directory=DB_PATH,
    embedding_function=embedding_model
)

# ==========================================
# Create Retriever
# ==========================================
retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 5
    }
)

print("Retriever created successfully!")

# ==========================================
# Test Retrieval
# ==========================================
query = "What is Amazon Fire Tablet?"

print(f"\nSearching for: '{query}'...")
results = retriever.invoke(query)

print(f"\nNumber of retrieved documents: {len(results)}\n")

# ==========================================
# Display Retrieved Context
# ==========================================
for i, doc in enumerate(results):
    print(f"======== Result {i+1} ========")
    print(doc.page_content[:500])
    print("-----------------------------------")