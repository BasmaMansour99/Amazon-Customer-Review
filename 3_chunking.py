# -*- coding: utf-8 -*-
import os
import pickle
import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ==========================================
# Project Configuration
# ==========================================
PROJECT_PATH = os.getcwd()
DATA_PATH = os.path.join(PROJECT_PATH, "data")

# ==========================================
# Load Clean Dataset
# ==========================================
INPUT_PATH = os.path.join(DATA_PATH, "cleaned_data.csv")

print("Loading cleaned dataset...")
try:
    df = pd.read_csv(INPUT_PATH)
    print(f"Dataset Shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: Could not find '{INPUT_PATH}'. Please ensure the preprocessing script ran successfully.")
    exit()

# ==========================================
# Convert Rows to Documents
# ==========================================
print("\nConverting rows to documents...")
documents = []

for _, row in df.iterrows():
    review = f"""
Product Name: {row['name']}

Brand: {row['brand']}

Category: {row['categories']}

Review Title: {row['reviews.title']}

Review Text:
{row['reviews.text']}
"""

    metadata = {
        "id": row["id"],
        "product": row["name"],
        "brand": row["brand"],
        "rating": row["reviews.rating"],
        "username": row["reviews.username"],
    }

    documents.append(
        Document(
            page_content=review,
            metadata=metadata
        )
    )

print(f"Documents Created: {len(documents)}")

# ==========================================
# Split Documents into Chunks
# ==========================================
print("\nSplitting documents into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)
print(f"Total Chunks: {len(chunks)}")

# ==========================================
# Display First Chunk
# ==========================================
if chunks:
    print("\n=== First Chunk ===")
    print(chunks[0].page_content)
    print("\nMetadata:")
    print(chunks[0].metadata)
    print("===================\n")

# ==========================================
# Save Chunks
# ==========================================
OUTPUT_PATH = os.path.join(DATA_PATH, "chunks.pkl")

print("Saving chunks...")
with open(OUTPUT_PATH, "wb") as f:
    pickle.dump(chunks, f)

print("\n==========================================")
print("Chunks saved successfully!")
print(f"Total Documents: {len(documents)}")
print(f"Total Chunks: {len(chunks)}")
print(f"Saved to: {OUTPUT_PATH}")
print("==========================================")