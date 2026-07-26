# -*- coding: utf-8 -*-

# ==========================================
# Import Libraries
# ==========================================
from langchain_huggingface import HuggingFaceEmbeddings

# ==========================================
# Load Embedding Model
# ==========================================
print("Loading embedding model (this might take a moment to download the model on the first run)...")

# تحميل نموذج تحويل النصوص إلى متجهات
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded successfully!")

# ==========================================
# Test Embedding Model
# ==========================================
sample_text = "Amazon Fire Tablet is an excellent device."

print(f"\nTesting embedding model with text: '{sample_text}'")

# تحويل النص التجريبي إلى متجه (Vector)
vector = embedding_model.embed_query(sample_text)

print(f"Vector Length: {len(vector)}")
print("First 10 elements of the vector:")
print(vector[:10])