# -*- coding: utf-8 -*-
import os
import pandas as pd

# ==========================================
# Load Datasets Locally
# ==========================================
# يرجى التأكد من أن هذه الملفات موجودة في نفس المجلد، أو قم بتغيير المسار ليطابق مكانها في جهازك.

file1_path = r"C:\Users\User1\Desktop\RAG\code ipynb\data\1429_1.csv"
file2_path = r"C:\Users\User1\Desktop\RAG\code ipynb\data\Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv"
file3_path = r"C:\Users\User1\Desktop\RAG\code ipynb\data\Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv"

print("Loading datasets...")
df1 = pd.read_csv(file1_path, low_memory=False)
df2 = pd.read_csv(file2_path, low_memory=False)
df3 = pd.read_csv(file3_path, low_memory=False)

# ==========================================
# Merge Datasets
# ==========================================
raw_df = pd.concat([df1, df2, df3], ignore_index=True)
print(f"\nMerged dataset shape: {raw_df.shape}")

# ==========================================
# Data Preprocessing
# ==========================================
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

df = raw_df.copy()

# 1) Exact duplicate rows
before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print(f"Dropped {before - len(df)} exact duplicate rows.")

# 2) Duplicate reviews (same product + reviewer + text)
before = len(df)
df = df.drop_duplicates(
    subset=["id", "reviews.username", "reviews.text"]
).reset_index(drop=True)
print(f"Dropped {before - len(df)} duplicate reviews.")

# 3) Empty / missing review text
df["reviews.text"] = df["reviews.text"].astype(str).str.strip()
before = len(df)
df = df[~df["reviews.text"].isin(["", "nan", "None", "none", "NaN"])].reset_index(drop=True)
print(f"Dropped {before - len(df)} rows with empty/missing review text.")

# 4) Require a valid product id and ASIN
before = len(df)
df = df.dropna(subset=["id", "asins"]).reset_index(drop=True)
print(f"Dropped {before - len(df)} rows missing product id/asins.")

# 5) Fill missing categorical metadata
categorical_cols = ["id", "asins", "name", "brand", "categories", "reviews.title", "reviews.username"]
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].fillna("Unknown").astype(str).str.strip()
        df.loc[df[col] == "", col] = "Unknown"
        df.loc[df[col].str.lower() == "nan", col] = "Unknown"

# 6) Coerce rating to numeric, fill missing with median (with fallback to 3.0)
if "reviews.rating" in df.columns:
    df["reviews.rating"] = pd.to_numeric(df["reviews.rating"], errors="coerce")
    rating_median = df["reviews.rating"].median()
    if pd.isna(rating_median):  # في حال كان العمود بأكمله فارغاً
        rating_median = 3.0
    df["reviews.rating"] = df["reviews.rating"].fillna(rating_median)

# 7) Coerce purchase/recommend flags to boolean
for col in ["reviews.didPurchase", "reviews.doRecommend"]:
    if col in df.columns:
        df[col] = df[col].fillna(False).astype(bool)

# 8) قفل أمان أخير: التأكد من عدم وجود أي قيمة مفقودة في الأعمدة المطلوبة
print("\nChecking for any remaining missing values in required columns...")
for col in REQUIRED_COLUMNS:
    if col not in df.columns:
        print(f"  -> Warning: '{col}' missing entirely. Creating it.")
        df[col] = "Unknown"
    
    if df[col].isna().any():
        missing_count = df[col].isna().sum()
        print(f"  -> Fixing {missing_count} remaining NaNs in '{col}'...")
        if df[col].dtype == 'bool':
            df[col] = df[col].fillna(False)
        elif pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(0)
        else:
            df[col] = df[col].fillna("Unknown")

print("\nFinal cleaned shape:", df.shape)

# Sanity check
assert df[REQUIRED_COLUMNS].isna().sum().sum() == 0, "Unexpected missing values remain."
assert (df["reviews.text"].str.len() > 0).all(), "Empty review text slipped through cleaning."
print("Cleaning checks passed successfully!")

# ==========================================
# Project Configuration & Saving
# ==========================================
PROJECT_PATH = os.getcwd()
DATA_PATH = os.path.join(PROJECT_PATH, "data")
os.makedirs(DATA_PATH, exist_ok=True)
OUTPUT_PATH = os.path.join(DATA_PATH, "cleaned_data.csv")

# ==========================================
# Save Clean Dataset
# ==========================================
df.to_csv(OUTPUT_PATH, index=False)
print("\n==========================================")
print("Clean dataset saved successfully!")
print(f"Final Shape: {df.shape}")
print(f"Saved to: {OUTPUT_PATH}")
print("==========================================")