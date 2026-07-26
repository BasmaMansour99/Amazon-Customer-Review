# -*- coding: utf-8 -*-
import os
import zipfile
import pandas as pd
import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.documents import Document

# ==========================================
# 1. Page Configuration & Custom CSS UI
# ==========================================
st.set_page_config(
    page_title="Amazon Review Analyzer",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Manage Language State
if "lang" not in st.session_state:
    st.session_state.lang = "en"

# Translations Dictionary
t = {
    "en": {
        "title": "Amazon Review Analyzer",
        "subtitle": "Analyze. Understand. Improve. — Grounded in 34K+ Verified Amazon Reviews",
        "try_asking": "TRY ASKING / أسئلة استرشادية",
        "new_chat": "➕ New Chat",
        "chat_tab": "💬 Chat History",
        "status_title": "📊 System Status",
        "groq_ok": "🟢 Groq API Connected",
        "groq_err": "🔴 API Key Missing",
        "db_ok": "🟢 Vector DB Ready",
        "db_err": "⚠️ Vector DB Missing",
        "lang_select": "🌐 LANGUAGE / اللغة",
        "input_placeholder": "Ask a question in English or Arabic...",
        "analyzing": "Analyzing customer reviews...",
        "insufficient": "Insufficient information.",
        "card1": "What is the percentage of 5-star reviews for the Kindle?",
        "card2": "Compare sentiment of Fire Tablet vs Echo Dot",
        "card3": "ما هي أبرز الشكاوى حول بطاريات Amazon Basics؟",
        "card4": "ما هو المنتج الذي يملك أعلى تقييمات إيجابية؟"
    },
    "ar": {
        "title": "محلل مراجعات أوجست / أمزون",
        "subtitle": "حلّل. افهم. حسّن. — مستند إلى أكثر من 34 ألف مراجعة حقيقية",
        "try_asking": "أسئلة استرشادية / TRY ASKING",
        "new_chat": "➕ محادثة جديدة",
        "chat_tab": "💬 سجل المحادثات",
        "status_title": "📊 حالة النظام",
        "groq_ok": "🟢 متصل بـ Groq API",
        "groq_err": "🔴 مفتاح API مفقود",
        "db_ok": "🟢 قاعدة البيانات جاهزة",
        "db_err": "⚠️ قاعدة البيانات مفقودة",
        "lang_select": "🌐 اللغة / LANGUAGE",
        "input_placeholder": "اكتب سؤالك بالعربية أو الإنجليزية...",
        "analyzing": "جاري تحليل مراجعات العملاء...",
        "insufficient": "المعلومات غير كافية.",
        "card1": "ما هي نسبة التقييمات 5 نجوم لـ Kindle؟",
        "card2": "قارن بين مشاعر المستخدمين لـ Fire Tablet و Echo Dot",
        "card3": "Top complaints about Amazon Basics batteries",
        "card4": "Which product has the most positive reviews?"
    }
}[st.session_state.lang]

# RTL / LTR Direction Handling Based on Selected Language
direction = "rtl" if st.session_state.lang == "ar" else "ltr"

st.markdown(f"""
<style>
    /* Global Styles */
    .stApp {{
        background-color: #0b0f19 !important;
        color: #e2e8f0 !important;
        direction: {direction};
    }}
    
    header, footer {{visibility: hidden;}}
    
    section[data-testid="stSidebar"] {{
        background-color: #0d1322 !important;
        border-right: 1px solid #1e293b;
    }}
    
    /* Logo Header Styles */
    .brand-container {{
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        margin-top: 10px;
        margin-bottom: 20px;
        text-align: center;
    }}
    
    .main-title-text {{
        font-size: 2rem;
        font-weight: 800;
        color: #f59e0b;
        letter-spacing: 1px;
        margin-top: 10px;
    }}
    
    .main-subtitle-text {{
        font-size: 0.9rem;
        color: #94a3b8;
        margin-bottom: 25px;
    }}

    /* Buttons & Suggestion Cards Styling */
    .stButton>button {{
        background-color: #111827 !important;
        color: #f1f5f9 !important;
        border: 1px solid #1e293b !important;
        border-radius: 12px !important;
        padding: 14px !important;
        height: 80px !important;
        width: 100% !important;
        text-align: {("right" if st.session_state.lang == "ar" else "left")} !important;
        font-size: 14px !important;
        transition: all 0.2s ease-in-out;
    }}
    
    .stButton>button:hover {{
        border-color: #f59e0b !important;
        background-color: #1e293b !important;
        color: #ffffff !important;
    }}

    .stChatInputContainer {{
        border-color: #f59e0b !important;
        background-color: #111827 !important;
        border-radius: 12px !important;
    }}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Configuration & Secrets
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "chroma_db")
ZIP_PATH = os.path.join(BASE_DIR, "data.zip")
LOGO_PATH = os.path.join(BASE_DIR, "logo.png")
TOP_K = 5

groq_api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))

# ==========================================
# 3. Vector DB Auto-Preparation Logic
# ==========================================
def prepare_chroma_db():
    if os.path.exists(DB_PATH) and len(os.listdir(DB_PATH)) > 0:
        return DB_PATH

    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(ZIP_PATH):
        try:
            with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
                zip_ref.extractall(BASE_DIR)
            if os.path.exists(DB_PATH) and len(os.listdir(DB_PATH)) > 0:
                return DB_PATH
        except Exception:
            pass

    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')] if os.path.exists(DATA_DIR) else []
    if csv_files:
        st.info("⚡ First time setup: Building Vector Database... Please wait.")
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        documents = []
        
        for file in csv_files:
            file_path = os.path.join(DATA_DIR, file)
            df = pd.read_csv(file_path, low_memory=False)
            
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
# 4. Load Models & Retriever (Cached)
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
    return ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=api_key,
        temperature=0.0
    )

retriever = load_retriever()

# ==========================================
# 5. Sidebar Navigation & Controls
# ==========================================
with st.sidebar:
    # Render Logo Image in Sidebar if available
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=180)
    else:
        st.markdown("<h2 style='color:#f59e0b;'>AMAZON</h2>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # New Chat Action
    if st.button(t["new_chat"], use_container_width=True):
        st.session_state.messages = []
        st.rerun()
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Active Chat Indicator
    st.markdown(f"""
    <div style="background-color: #1e293b; padding: 10px; border-radius: 8px; color: #f59e0b; font-weight: bold;">
        {t['chat_tab']}
    </div>
    """, unsafe_allow_html=True)

    # Status Check
    st.markdown("---")
    st.markdown(f"**{t['status_title']}**")
    if groq_api_key:
        st.caption(t["groq_ok"])
    else:
        st.caption(t["groq_err"])
        
    if retriever is not None:
        st.caption(t["db_ok"])
    else:
        st.caption(t["db_err"])

    # Language Switcher
    st.markdown("---")
    st.markdown(f"**{t['lang_select']}**")
    l_col1, l_col2 = st.columns(2)
    with l_col1:
        if st.button("English", key="btn_en", use_container_width=True):
            st.session_state.lang = "en"
            st.rerun()
    with l_col2:
        if st.button("العربية", key="btn_ar", use_container_width=True):
            st.session_state.lang = "ar"
            st.rerun()

# ==========================================
# 6. Prompt Logic Supporting English & Arabic
# ==========================================
def retrieve_docs(retriever, query):
    return retriever.invoke(query)

def build_grounded_prompt(query, docs):
    if not docs:
        return None
    context_text = "\n\n".join(doc.page_content for doc in docs)

    return f"""You are Amazon Review Analyzer AI assistant. Answer the user's question using ONLY the review excerpts provided below.

Rules:
- Respond in the SAME language as the user's question (if asked in Arabic, answer in Arabic; if in English, answer in English).
- Answer strictly using the information contained in the reviews below.
- Do not use outside knowledge or make assumptions.
- If the reviews do not contain enough information, respond with exactly: "Insufficient information." (or "المعلومات غير كافية." if asked in Arabic).
- Keep the answer clear, structured, and accurate.

Question:
{query}

Customer review excerpts:
{context_text}

Answer:"""

# ==========================================
# 7. Main Interface & Question Cards
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# Landing Page with Image and Cards if Chat is Empty
if len(st.session_state.messages) == 0:
    st.markdown('<div class="brand-container">', unsafe_allow_html=True)
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=220)
    st.markdown(f'<div class="main-title-text">{t["title"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="main-subtitle-text">{t["subtitle"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="color: #64748b; font-size: 13px; font-weight: 700; margin-bottom: 15px;">{t["try_asking"]}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # 4 Cards Grid (Suggested Questions in Both Languages)
    col1, col2 = st.columns(2)
    prompt_to_send = None
    
    with col1:
        if st.button(t["card1"], key="c1"):
            prompt_to_send = t["card1"]
        if st.button(t["card3"], key="c3"):
            prompt_to_send = t["card3"]
            
    with col2:
        if st.button(t["card2"], key="c2"):
            prompt_to_send = t["card2"]
        if st.button(t["card4"], key="c4"):
            prompt_to_send = t["card4"]

    if prompt_to_send:
        st.session_state.messages.append({"role": "user", "content": prompt_to_send})
        st.rerun()

# Display Conversation History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Chat Input
user_input = st.chat_input(t["input_placeholder"])

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.rerun()

# Process Response via Groq
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    current_prompt = st.session_state.messages[-1]["content"]
    
    with st.chat_message("assistant"):
        if not groq_api_key:
            st.error("Please set GROQ_API_KEY in Streamlit Secrets.")
        elif retriever is None:
            st.error("Vector DB Error: Could not retrieve database.")
        else:
            with st.spinner(t["analyzing"]):
                try:
                    docs = retrieve_docs(retriever, current_prompt)
                    prompt = build_grounded_prompt(current_prompt, docs)
                    
                    if prompt is None:
                        response_text = t["insufficient"]
                    else:
                        llm = load_llm(groq_api_key)
                        response = llm.invoke(prompt)
                        response_text = response.content

                    st.markdown(response_text)
                    st.session_state.messages.append({"role": "assistant", "content": response_text})

                except Exception as e:
                    st.error(f"Error: {e}")
