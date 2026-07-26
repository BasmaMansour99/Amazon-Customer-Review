# -*- coding: utf-8 -*-
"""
Amazon Intelligence Assistant — RAG Chatbot
--------------------------------------------
Reads Amazon product review data (a pre-built Chroma vector DB, or a raw/cleaned
CSV that it will index on first run), retrieves the most relevant review
excerpts for a question, and asks an LLM (Groq) to answer strictly from those
excerpts.

Designed to be pushed to GitHub and deployed on Streamlit Community Cloud.
"""

import os
import sys
import zipfile

# ==========================================
# SQLite fix (REQUIRED on Streamlit Cloud)
# --------------------------------------------
# Streamlit Cloud ships an old system sqlite3 that chromadb rejects.
# pysqlite3-binary provides a modern build; we swap it in before chromadb
# is imported anywhere. This must be the very first thing that runs.
# ==========================================
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    # Running locally on a machine with a modern sqlite3 is fine too.
    pass

import pandas as pd
import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_groq import ChatGroq

# ==========================================
# Paths & Constants
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "chroma_db")
ZIP_PATH = os.path.join(BASE_DIR, "data.zip")
LOGO_PATH = os.path.join(BASE_DIR, "assets", "logo.png")

TOP_K = 5
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL_NAME = "llama-3.3-70b-versatile"   # change here if you prefer another Groq model

REQUIRED_COLUMNS = [
    "id", "name", "asins", "brand", "categories",
    "reviews.didPurchase", "reviews.doRecommend", "reviews.rating",
    "reviews.title", "reviews.text", "reviews.username",
]

# ==========================================
# Page Configuration & Theme
# ==========================================
st.set_page_config(
    page_title="Amazon Intelligence Assistant",
    page_icon=LOGO_PATH if os.path.exists(LOGO_PATH) else "🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main { background-color: #0f172a; color: #f8fafc; }
    .main-title {
        font-size: 2.3rem; font-weight: 800;
        background: linear-gradient(90deg, #ff9900, #ffb84d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title { color: #94a3b8; font-size: 1rem; margin-bottom: 1.6rem; }
    section[data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #1f2937;
    }
    .stChatMessage { border-radius: 14px; }
    .stExpander { border: 1px solid #1f2937 !important; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# API Key (Groq)
# ==========================================
api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))

# ==========================================
# Helper: build LangChain Documents from a dataframe row
# (kept consistent with the offline pipeline scripts 1_doc.py / 3_chunking.py)
# ==========================================
def row_to_document(row) -> Document:
    def g(col, default="Unknown"):
        val = row[col] if col in row and pd.notna(row[col]) else default
        return val

    text = f"""Product Name: {g('name')}
Brand: {g('brand')}
Category: {g('categories')}
Review Title: {g('reviews.title')}
Review Text:
{g('reviews.text', '')}"""

    metadata = {
        "id": str(g("id")),
        "product": str(g("name")),
        "brand": str(g("brand")),
        "rating": g("reviews.rating", 0),
        "username": str(g("reviews.username")),
    }
    return Document(page_content=text, metadata=metadata)


def build_documents_from_csv(csv_path: str):
    df = pd.read_csv(csv_path, low_memory=False)

    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = "Unknown"
    df = df.fillna("Unknown")
    df["reviews.text"] = df["reviews.text"].astype(str).str.strip()
    df = df[~df["reviews.text"].isin(["", "nan", "None", "Unknown"])]

    return [row_to_document(row) for _, row in df.iterrows()]


# ==========================================
# Prepare / load the Chroma vector database
# ==========================================
def prepare_chroma_db(embedding_model):
    """
    Resolution order:
      1) Use an already-built chroma_db folder if present.
      2) Unzip data.zip (expected to contain data/chroma_db) if present.
      3) Otherwise, build the DB from any CSV found in /data
         (prefers cleaned_data.csv).
    Returns the persist directory path, or None if nothing could be prepared.
    """
    if os.path.exists(DB_PATH) and len(os.listdir(DB_PATH)) > 0:
        return DB_PATH

    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(ZIP_PATH):
        try:
            with zipfile.ZipFile(ZIP_PATH, "r") as zf:
                zf.extractall(BASE_DIR)
            if os.path.exists(DB_PATH) and len(os.listdir(DB_PATH)) > 0:
                return DB_PATH
        except Exception as e:
            st.warning(f"Could not extract data.zip: {e}")

    csv_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".csv")] \
        if os.path.exists(DATA_DIR) else []
    if not csv_files:
        return None

    preferred = "cleaned_data.csv"
    csv_name = preferred if preferred in csv_files else csv_files[0]
    csv_path = os.path.join(DATA_DIR, csv_name)

    st.info(f"⚡ First-time setup: building the vector database from **{csv_name}**. "
            f"This runs once and may take a few minutes...")

    documents = build_documents_from_csv(csv_path)
    if not documents:
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    os.makedirs(DB_PATH, exist_ok=True)
    Chroma.from_documents(documents=chunks, embedding=embedding_model, persist_directory=DB_PATH)
    return DB_PATH


# ==========================================
# Cached resources
# ==========================================
@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


@st.cache_resource(show_spinner=False)
def load_retriever():
    embedding_model = load_embedding_model()
    db_path = prepare_chroma_db(embedding_model)
    if not db_path:
        return None
    vector_store = Chroma(persist_directory=db_path, embedding_function=embedding_model)
    return vector_store.as_retriever(search_kwargs={"k": TOP_K})


@st.cache_resource(show_spinner=False)
def load_llm(_api_key):
    return ChatGroq(model=GROQ_MODEL_NAME, groq_api_key=_api_key, temperature=0.2)


# ==========================================
# RAG logic
# ==========================================
def build_grounded_prompt(query, docs):
    if not docs:
        return None
    context_text = "\n\n".join(doc.page_content for doc in docs)
    return f"""You are AutoAnalyst AI, an assistant that answers questions about
Amazon products using ONLY the customer review excerpts provided below.

Rules:
- Answer strictly using the information contained in the reviews below.
- Do not use outside knowledge or make assumptions beyond what is written.
- If the reviews do not contain enough information to answer the question,
  respond with exactly: Insufficient information.
- Keep the answer brief, highly accurate, and directly grounded in the review text.
- Where useful, mention which product(s) the answer is based on.

Question:
{query}

Customer review excerpts:
{context_text}

Answer:"""


# ==========================================
# Sidebar
# ==========================================
with st.sidebar:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, use_container_width=True)
    else:
        st.image("https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg", width=140)

    st.markdown("### 📊 System Status")

    if api_key:
        st.success("Groq API Connected", icon="✅")
    else:
        st.error("Groq API Key Missing", icon="🚨")

    retriever = load_retriever()

    if retriever is not None:
        st.success("Vector DB Ready", icon="🗄️")
    else:
        st.error("Vector DB / Data Missing", icon="⚠️")

    st.markdown("---")
    st.markdown("### 💡 Quick Tips")
    st.info(
        "• Ask about **battery life**, **screen quality**, or **build performance**.\n\n"
        "• The system relies **strictly** on verified Amazon customer reviews."
    )

# ==========================================
# Main Header
# ==========================================
st.markdown('<div class="main-title">🛒 Amazon Product RAG Intelligence</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">AI-Powered Insights Grounded Strictly in Real Customer Reviews</div>',
    unsafe_allow_html=True,
)

# ==========================================
# Validation & Chat
# ==========================================
if not api_key:
    st.warning("⚠️ **System not ready:** add `GROQ_API_KEY` in Streamlit Cloud → Settings → Secrets.")
elif retriever is None:
    st.error(
        "⚠️ **Database error:** no `data/chroma_db`, `data.zip`, or CSV file was found. "
        "Make sure one of these is included in the repo's `data/` folder."
    )
else:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("🔍 View retrieved review excerpts"):
                    for idx, src in enumerate(message["sources"], 1):
                        st.markdown(f"**Excerpt {idx}:**\n> {src.page_content}")

    if user_query := st.chat_input("Ask a question about Amazon products (e.g., How is the battery life?)..."):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing customer reviews..."):
                try:
                    docs = retriever.invoke(user_query)
                    prompt = build_grounded_prompt(user_query, docs)

                    if prompt is None:
                        response_text = "Insufficient information."
                        sources = []
                    else:
                        llm = load_llm(api_key)
                        response = llm.invoke(prompt)
                        response_text = response.content
                        sources = docs

                    st.markdown(response_text)

                    if sources:
                        with st.expander("🔍 View retrieved review excerpts"):
                            for idx, src in enumerate(sources, 1):
                                st.markdown(f"**Excerpt {idx}:**\n> {src.page_content}")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response_text,
                        "sources": sources,
                    })

                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")
