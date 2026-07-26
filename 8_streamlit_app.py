# -*- coding: utf-8 -*-
import os
import zipfile
import pandas as pd
import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document

# ==========================================
# Page Configuration & Custom UI Theme
# ==========================================
st.set_page_config(
    page_title="Amazon Intelligence Assistant",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { background-color: #0f172a; color: #f8fafc; }
    .main-title {
        font-size: 2.2rem; font-weight: 700;
        background: linear-gradient(90deg, #ff9900, #ffb84d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title { color: #94a3b8; font-size: 1rem; margin-bottom: 2rem; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# Configuration & Local Paths
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "chroma_db")
ZIP_PATH = os.path.join(BASE_DIR, "data.zip")
TOP_K = 5

api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))

# Helper Function: Auto Unzip and Build Chroma DB
def prepare_chroma_db():
    # 1. Check if DB exists directly
    if os.path.exists(DB_PATH) and len(os.listdir(DB_PATH)) > 0:
        return DB_PATH

    # Create data directory if not created
    os.makedirs(DATA_DIR, exist_ok=True)

    # 2. Extract data.zip if present
    if os.path.exists(ZIP_PATH):
        try:
            with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
                zip_ref.extractall(BASE_DIR)
            if os.path.exists(DB_PATH) and len(os.listdir(DB_PATH)) > 0:
                return DB_PATH
        except Exception:
            pass

    # 3. Build Vector DB from CSV files in DATA_DIR
    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')] if os.path.exists(DATA_DIR) else []
    if csv_files:
        st.info("⚡ First time setup: Building Vector Database from CSVs... Please wait.")
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        documents = []
        
        for file in csv_files:
            file_path = os.path.join(DATA_DIR, file)
            df = pd.read_csv(file_path, low_memory=False)
            
            # Common text columns in Amazon Datasets
            possible_cols = ['reviews.text', 'review_body', 'text', 'review', 'content', 'reviews.title']
            text_column = next((col for col in possible_cols if col in df.columns), df.columns[0])
            
            for _, row in df.iterrows():
                text = str(row[text_column]) if pd.notna(row[text_column]) else ""
                if len(text.strip()) > 0:
                    documents.append(Document(page_content=text))
        
        if documents:
            vector_store = Chroma.from_documents(
                documents=documents,
                embedding=embedding_model,
                persist_directory=DB_PATH
            )
            return DB_PATH
            
    return None

# ==========================================
# Load Models & Retriever (Cached)
# ==========================================
@st.cache_resource
def load_retriever():
    actual_db_path = prepare_chroma_db()
    if not actual_db_path or not os.path.exists(actual_db_path):
        return None
        
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = Chroma(
        persist_directory=actual_db_path,
        embedding_function=embedding_model
    )
    return vector_store.as_retriever(search_kwargs={"k": TOP_K})

@st.cache_resource
def load_llm(api_key):
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key
    )

retriever = load_retriever()

# ==========================================
# Sidebar Dashboard
# ==========================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg", width=140)
    st.markdown("### 📊 System Status")
    
    if api_key:
        st.success("🟢 Gemini API Connected", icon="✅")
    else:
        st.error("🔴 API Key Missing", icon="🚨")
        
    if retriever is not None:
        st.success("🟢 Vector DB Ready", icon="🗄️")
    else:
        st.error("🔴 Vector DB Missing", icon="⚠️")

    st.markdown("---")
    st.markdown("### 💡 Quick Tips")
    st.info("""
    • Ask about **battery life**, **screen quality**, or **build performance**.  
    • The system relies **strictly** on verified Amazon customer reviews.
    """)

# ==========================================
# Core RAG Logic
# ==========================================
def retrieve_docs(retriever, query):
    return retriever.invoke(query)

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

Question:
{query}

Customer review excerpts:
{context_text}

Answer:"""

# ==========================================
# Main Header UI
# ==========================================
st.markdown('<div class="main-title">🛒 Amazon Product RAG Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI-Powered Insights Grounded Strictly in Real Customer Reviews</div>', unsafe_allow_html=True)

# Validation Checks
if not api_key:
    st.warning("⚠️ **System Not Ready:** Please configure `GEMINI_API_KEY` in Streamlit Cloud Secrets to proceed.")
elif retriever is None:
    st.error("⚠️ **Database Error:** Could not initialize Chroma DB. Please ensure CSV file is in data folder.")
else:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("🔍 View Retrieved Review Excerpts"):
                    for idx, src in enumerate(message["sources"], 1):
                        st.markdown(f"**Excerpt {idx}:**\n> {src.page_content}")

    if user_query := st.chat_input("Ask a question about Amazon products (e.g., How is the battery life?)..."):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing customer reviews..."):
                try:
                    docs = retrieve_docs(retriever, user_query)
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
                        with st.expander("🔍 View Retrieved Review Excerpts"):
                            for idx, src in enumerate(sources, 1):
                                st.markdown(f"**Excerpt {idx}:**\n> {src.page_content}")

                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response_text,
                        "sources": sources
                    })

                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")
