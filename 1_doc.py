# -*- coding: utf-8 -*-
import os
import pandas as pd
from langchain_core.documents import Document

# ==========================================
# Project Configuration & Paths
# ==========================================
PROJECT_PATH = os.getcwd()
DATA_PATH = os.path.join(PROJECT_PATH, "data")

# يمكنك استخدام ملف cleaned_data.csv الذي تم تجهيزه سابقاً أو ملف CSV محلي آخر
INPUT_FILE = os.path.join(DATA_PATH, "cleaned_data.csv")

REQUIRED_COLUMNS = [
    "id",
    "name",
    "asins",
    "brand",
    "categories",
    "reviews.didPurchase",
    "reviews.doRecommend",
    "reviews.rating",
    "reviews.title",
    "reviews.text",
    "reviews.username",
]

# ==========================================
# Load Local Dataset
# ==========================================
print("Loading dataset...")

# البحث عن الملف المنظف، وإن لم يوجد يبحث عن الملف الأصلي في المجلد الحالي
if not os.path.exists(INPUT_FILE):
    INPUT_FILE = os.path.join(PROJECT_PATH, "1429_1.csv")

try:
    raw_df = pd.read_csv(INPUT_FILE, low_memory=False)
    print(f"Dataset loaded successfully from: {INPUT_FILE}")
    print(f"Dataset Shape: {raw_df.shape}")
except FileNotFoundError:
    print(f"Error: Could not find any input CSV file at '{INPUT_FILE}'.")
    exit()

# التأكد من وجود كافة الأعمدة المطلوبة وملء أي قيم فارغة لتفادي الأخطاء
for col in REQUIRED_COLUMNS:
    if col not in raw_df.columns:
        raw_df[col] = "Unknown"

raw_df = raw_df.fillna("Unknown")

# ==========================================
# Convert Rows to LangChain Documents
# ==========================================
print("\nConverting rows to LangChain Documents...")
documents = []

for _, row in raw_df.iterrows():

    review_text = f"""
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
        "didPurchase": row["reviews.didPurchase"],
        "recommended": row["reviews.doRecommend"]
    }

    documents.append(
        Document(
            page_content=review_text,
            metadata=metadata
        )
    )

# ==========================================
# Check the Created Documents
# ==========================================
print("\n==========================================")
print(f"Total Documents Created: {len(documents)}")

if documents:
    print("\n--- First Document Sample ---")
    print(documents[0].page_content)

    print("\n--- First Document Metadata ---")
    print(documents[0].metadata)
print("==========================================")