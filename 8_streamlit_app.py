# -*- coding: utf-8 -*-
"""
Review Spark — Amazon Intelligence Assistant (RAG Chatbot)
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
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
DB_PATH = os.path.join(DATA_DIR, "chroma_db")
ZIP_PATH = os.path.join(BASE_DIR, "data.zip")

# Available image paths
LOGO_PATH = os.path.join(ASSETS_DIR, "logo.png")
BANNER_PATH = os.path.join(ASSETS_DIR, "banner.png")
LOGO_ROOT_PATH = os.path.join(BASE_DIR, "logo.png")

TOP_K = 10
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL_NAME = "llama-3.3-70b-versatile"

REQUIRED_COLUMNS = [
    "id", "name", "asins", "brand", "categories",
    "reviews.didPurchase", "reviews.doRecommend", "reviews.rating",
    "reviews.title", "reviews.text", "reviews.username",
]

SUGGESTED_QUESTIONS_GENERAL = [
    "What is the battery life like for Kindle Paperwhite?",
    "How do users describe the screen and reading experience on Kindle Paperwhite?",
    "What are the main complaints regarding battery life or SD cards on Fire Tablets?",
    "How is Alexa used on Amazon Echo and Amazon Tap devices?",
]

TEAM_MEMBERS = [
    {
        "name": "Basma Mansour",
        "title": "Assistant Lecturer | Data Analyst | AI & Machine Learning | Deep Learning | Python, SQL, Excel | Power BI, Tableau | Azure | Financial Management",
        "image_filename": "basma.png",
    },
    {
        "name": "Radwa El-Mahdy",
        "title": "Applied AI & Data Analytics Scholar | Senior Financial Professional | Digilians • MCIT | Transforming Financial Data into Strategic Intelligence | Python | SQL",
        "image_filename": "radwa.png",
    },
    {
        "name": "Marwa Baraka",
        "title": "Data Analytics & Applied AI | Junior Data Analyst | Python | Power BI | SQL | Passionate About Turning Data Into Insights",
        "image_filename": "marwa.png",
    },
    {
        "name": "Shorouk Khaled",
        "title": "Data Analyst skilled in Python | SQL | Power BI",
        "image_filename": "shorouk.png",
    },
]

def get_image_path(path_options):
    for path in path_options:
        if os.path.exists(path):
            return path
    return None

active_logo = get_image_path([LOGO_PATH, LOGO_ROOT_PATH])
active_banner = get_image_path([BANNER_PATH])

def get_member_image_path(filename):
    path = os.path.join(ASSETS_DIR, filename)
    if os.path.exists(path):
        return path
    jpg_path = os.path.join(ASSETS_DIR, filename.replace('.png', '.jpg'))
    if os.path.exists(jpg_path):
        return jpg_path
    return None

# ==========================================
# Page Configuration & Theme
# ==========================================
st.set_page_config(
    page_title="Review Spark",
    page_icon=active_logo if active_logo else "✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS Styling (Updated for Gold/Dark Theme)
st.markdown("""
<style>
    .main { background-color: #0b0f19; color: #f8fafc; }
    .main-title {
        font-size: 2.4rem; font-weight: 800;
        background: linear-gradient(90deg, #d4af37, #facc15);
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
    .stExpander { border: 1px solid #1f2937 !important; border-radius: 10px; background-color: #1e293b;}
    .try-asking-label {
        color: #d4af37; font-size: 0.9rem; font-weight: 700;
        letter-spacing: 0.05em; margin-top: 2rem; margin-bottom: 0.8rem;
    }
    div[data-testid="stButton"] > button {
        border: 1px solid #334155;
        background-color: #1e293b;
        color: #f8fafc;
        border-radius: 8px;
        text-align: left;
        padding: 0.5rem 1rem;
    }
    div[data-testid="stButton"] > button:hover {
        border-color: #d4af37;
        color: #d4af37;
    }
    /* Metric Cards Styling */
    div[data-testid="metric-container"] {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    div[data-testid="metric-container"] label {
        color: #94a3b8 !important;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #d4af37 !important;
        font-weight: 700;
    }
    .team-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .team-name {
        font-size: 1.2rem;
        font-weight: 700;
        color: #d4af37;
        margin-top: 0.5rem;
        margin-bottom: 0.3rem;
    }
    .team-title {
        font-size: 0.88rem;
        color: #cbd5e1;
        line-height: 1.4;
    }
/* Banner image styling */
.banner-container img {
    border-radius: 16px;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.5);
    margin: 0 auto 2rem auto;
    max-width: 60%; 
    display: block;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# API Key (Groq)
# ==========================================
api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))

# ==========================================
# Helper Functions (Keep exact same logic)
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
    st.info(f"⚡ First-time setup: building the vector database from **{csv_name}**...")
    documents = build_documents_from_csv(csv_path)
    if not documents:
        return None
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    os.makedirs(DB_PATH, exist_ok=True)
    Chroma.from_documents(documents=chunks, embedding=embedding_model, persist_directory=DB_PATH)
    return DB_PATH

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
# Sidebar Navigation
# ==========================================
with st.sidebar:
    if active_logo:
        st.image(active_logo, use_container_width=True)
    else:
        st.markdown("## Review Spark")

    st.markdown("### 📌 Navigation")
    app_mode = st.radio(
        "Choose Mode",
        options=["💬 Review Analyzer Chat", "👥 Meet the Team"],
        index=0,
        label_visibility="collapsed"
    )

    if app_mode == "💬 Review Analyzer Chat":
        st.markdown("---")
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

# ==========================================
# Main Content Area
# ==========================================

if app_mode == "👥 Meet the Team":
    st.markdown('<div class="main-title">✨ Meet the Review Spark Team</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">The brilliant minds behind Review Spark — Transforming Amazon Reviews into Actionable Intelligence</div>', unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns(2, gap="large")

    for idx, member in enumerate(TEAM_MEMBERS):
        target_col = col1 if idx % 2 == 0 else col2
        with target_col:
            st.markdown('<div class="team-card">', unsafe_allow_html=True)
            img_path = get_member_image_path(member["image_filename"])
            if img_path:
                st.image(img_path, width=220)
            else:
                st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=120)
            st.markdown(f'<div class="team-name">{member["name"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="team-title">{member["title"]}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

else:
    # Chat Mode Validation
    if not api_key:
        st.warning("⚠️ **System not ready:** add `GROQ_API_KEY` in Streamlit Cloud → Settings → Secrets.")
    elif vector_store is None:
        st.error("⚠️ **Database error:** no `data/chroma_db`, `data.zip`, or CSV file was found.")
    else:
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "pending_prompt" not in st.session_state:
            st.session_state.pending_prompt = None

        # ==========================================
        # Welcome Screen (Hero Banner & Metrics)
        # ==========================================
        if not st.session_state.messages:
            # Display Hero Banner if it exists
            if active_banner:
                st.markdown('<div class="banner-container">', unsafe_allow_html=True)
                st.image(active_banner, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="main-title">✨ Review Spark RAG Engine</div>', unsafe_allow_html=True)
                st.markdown('<div class="sub-title">Insights that ignite better decisions. Ask questions grounded strictly in real customer reviews.</div>', unsafe_allow_html=True)

            # Display Stats (Metrics similar to the reference image)
m1, m2, m3 = st.columns(3)
m1.metric(label="Evidence sources", value="34,000+")
m2.metric(label="Searchable chunks", value="Vector DB")
m3.metric(label="Sentiment Overview", value="85% Pos")
            
            st.markdown('<div class="try-asking-label">ASK THE EVIDENCE</div>', unsafe_allow_html=True)

            cols = st.columns(2)
            for i, q in enumerate(SUGGESTED_QUESTIONS_GENERAL):
                with cols[i % 2]:
                    if st.button(q, key=f"suggestion_{i}", use_container_width=True):
                        st.session_state.pending_prompt = q
                        st.rerun()

        # ==========================================
        # Chat Interface Loop
        # ==========================================
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
