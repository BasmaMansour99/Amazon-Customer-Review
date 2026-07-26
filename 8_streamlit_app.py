# -*- coding: utf-8 -*-
"""
Amazon Intelligence Assistant — RAG Chatbot
--------------------------------------------
Reads Amazon product review data (a pre-built Chroma vector DB, or a raw/cleaned
CSV that it will index on first run), retrieves the most relevant review
excerpts for a question, and asks an LLM (Groq) to answer strictly from those
excerpts. English is the app's native language; any answer can be translated
to Arabic on demand with one click.

Designed to be pushed to GitHub and deployed on Streamlit Community Cloud.
"""

import os
import sys
import zipfile

# ==========================================
# SQLite fix (REQUIRED on Streamlit Cloud)
# --------------------------------------------
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
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

# Available logo paths
LOGO_PATH = os.path.join(BASE_DIR, "assets", "logo.png")
LOGO_ROOT_PATH = os.path.join(BASE_DIR, "logo.png")

# Increased TOP_K to retrieve more comprehensive context from the dataset
TOP_K = 10
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL_NAME = "llama-3.3-70b-versatile"

REQUIRED_COLUMNS = [
    "id", "name", "asins", "brand", "categories",
    "reviews.didPurchase", "reviews.doRecommend", "reviews.rating",
    "reviews.title", "reviews.text", "reviews.username",
]

# Suggested reference questions grounded in dataset contents
SUGGESTED_QUESTIONS_GENERAL = [
    "What is the battery life like for Kindle Paperwhite?",
    "How do users describe the screen and reading experience on Kindle Paperwhite?",
    "What are the main complaints regarding battery life or SD cards on Fire Tablets?",
    "How is Alexa used on Amazon Echo and Amazon Tap devices?",
]

def get_logo_path():
    if os.path.exists(LOGO_PATH):
        return LOGO_PATH
    elif os.path.exists(LOGO_ROOT_PATH):
        return LOGO_ROOT_PATH
    return None

active_logo = get_logo_path()

# ==========================================
# Page Configuration & Theme
# ==========================================
st.set_page_config(
    page_title="Amazon Review Analyzer",
    page_icon=active_logo if active_logo else "🛒",
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
    .sub-title { color: #94a3b8; font-size: 1rem; margin-bottom: 1.4rem; }
    section[data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #1f2937;
    }
    .stChatMessage { border-radius: 14px; }
    .stExpander { border: 1px solid #1f2937 !important; border-radius: 10px; }
    .try-asking-label {
        color: #94a3b8; font-size: 0.8rem; font-weight: 700;
        letter-spacing: 0.05em; margin-bottom: 0.4rem;
    }
    div[data-testid="stButton"] > button {
        border: 1px solid #334155;
        background-color: #1e293b;
        color: #f8fafc;
        border-radius: 10px;
        text-align: left;
    }
    div[data-testid="stButton"] > button:hover {
        border-color: #ff9900;
        color: #ff9900;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# API Key (Groq)
# ==========================================
api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))

# ==========================================
# Helper Functions
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


def prepare_chroma_db(embedding_model):
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
# Cached Resources
# ==========================================
@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


@st.cache_resource(show_spinner=False)
def load_vector_store():
    embedding_model = load_embedding_model()
    db_path = prepare_chroma_db(embedding_model)
    if not db_path:
        return None
    return Chroma(persist_directory=db_path, embedding_function=embedding_model)


@st.cache_resource(show_spinner=False)
def load_llm(_api_key):
    return ChatGroq(model=GROQ_MODEL_NAME, groq_api_key=_api_key, temperature=0.2)


def retrieve(vector_store, query, k=TOP_K):
    return vector_store.similarity_search(query, k=k)


def build_grounded_prompt(query, docs):
    if not docs:
        return None
    context_text = "\n\n".join(doc.page_content for doc in docs)
    return f"""You are AutoAnalyst AI, an assistant that answers questions about
Amazon products using ONLY the customer review excerpts provided below.

Strict Rules:
- Answer strictly using the information contained in the reviews below.
- Do not use outside knowledge or make assumptions beyond what is written.
- If the question asks about a non-Amazon product (e.g., Apple iPad Pro, Samsung Galaxy) or features not directly evaluated in the reviews, answer ONLY with: Insufficient information.
- If the reviews do not contain enough specific details to directly answer the question, respond with exactly: Insufficient information.
- Keep the answer brief, highly accurate, and directly grounded in the review text.

Question:
{query}

Customer review excerpts:
{context_text}

Answer:"""


def translate_to_arabic(text, _api_key):
    llm = load_llm(_api_key)
    prompt = (
        "Translate the following text into natural, fluent Arabic. "
        "Return ONLY the Arabic translation, nothing else.\n\n"
        f"Text:\n{text}"
    )
    response = llm.invoke(prompt)
    return response.content


# ==========================================
# Sidebar
# ==========================================
with st.sidebar:
    if active_logo:
        st.image(active_logo, use_container_width=True)
    else:
        st.image("https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg", width=140)

    # Button to start a new chat
    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_prompt = None
        st.rerun()

    st.markdown("---")
    st.markdown("### 📊 System Status")

    if api_key:
        st.success("Groq API Connected", icon="✅")
    else:
        st.error("Groq API Key Missing", icon="🚨")

    vector_store = load_vector_store()

    if vector_store is not None:
        st.success("Vector DB Ready", icon="🗄️")
    else:
        st.error("Vector DB / Data Missing", icon="⚠️")

    st.markdown("---")
    st.markdown("### 💡 Quick Tips")
    st.info(
        "• Ask about **battery life**, **screen quality**, or **build performance**.\n\n"
        "• The system relies **strictly** on verified Amazon customer reviews.\n\n"
        "• Every answer can be translated to Arabic with one click."
    )

# ==========================================
# Main Header
# ==========================================
st.markdown('<div class="main-title">🛒 Amazon Review Analyzer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Analyze. Understand. Improve. Grounded Strictly in Real Customer Reviews</div>',
    unsafe_allow_html=True,
)

# ==========================================
# Validation & Chat Loop
# ==========================================
if not api_key:
    st.warning("⚠️ **System not ready:** add `GROQ_API_KEY` in Streamlit Cloud → Settings → Secrets.")
elif vector_store is None:
    st.error(
        "⚠️ **Database error:** no `data/chroma_db`, `data.zip`, or CSV file was found. "
        "Make sure one of these is included in the repo's `data/` folder."
    )
else:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None

    # Welcome screen before starting chat
    if not st.session_state.messages:
        if active_logo:
            col_a, col_b, col_c = st.columns([1, 2, 1])
            with col_b:
                st.image(active_logo, width=280)

        st.markdown('<div class="try-asking-label">TRY ASKING</div>', unsafe_allow_html=True)

        suggestions = SUGGESTED_QUESTIONS_GENERAL

        cols = st.columns(2)
        for i, q in enumerate(suggestions):
            with cols[i % 2]:
                if st.button(q, key=f"suggestion_{i}", use_container_width=True):
                    st.session_state.pending_prompt = q
                    st.rerun()

    # Display chat history
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message.get("sources"):
                with st.expander("🔍 View retrieved review excerpts"):
                    for idx, src in enumerate(message["sources"], 1):
                        st.markdown(f"**Excerpt {idx}:**\n> {src.page_content}")

            if message["role"] == "assistant" and message["content"] != "Insufficient information.":
                if message.get("ar_translation"):
                    st.markdown("---")
                    st.markdown(f"🌐 **Arabic:**\n\n{message['ar_translation']}")
                else:
                    if st.button("🌐 Translate to Arabic", key=f"translate_{i}"):
                        with st.spinner("Translating..."):
                            message["ar_translation"] = translate_to_arabic(message["content"], api_key)
                        st.rerun()

    # Chat user input
    typed_query = st.chat_input("Ask a question about Amazon products (e.g., How is the battery life?)...")
    user_query = typed_query or st.session_state.pending_prompt
    st.session_state.pending_prompt = None

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing customer reviews..."):
                try:
                    docs = retrieve(vector_store, user_query)
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
                        "ar_translation": None,
                    })
                    st.rerun()

                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")
