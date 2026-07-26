# -*- coding: utf-8 -*-
import os
import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

# ==========================================
# Page Configuration & Custom UI Theme
# ==========================================
st.set_page_config(
    page_title="Amazon Intelligence Assistant",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional Styling
st.markdown("""
<style>
    /* Clean background & typography */
    .main {
        background-color: #0f172a;
        color: #f8fafc;
    }
    
    /* Custom Header Styling */
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #ff9900, #ffb84d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .sub-title {
        color: #94a3b8;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    
    /* Card Styles */
    .metric-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
    }
    
    /* Source Context Box */
    .source-box {
        background-color: #1e293b;
        border-left: 4px solid #ff9900;
        padding: 10px 15px;
        border-radius: 4px;
        font-size: 0.88rem;
        color: #cbd5e1;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# Configuration & Local Paths
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_PATH = os.path.abspath(os.path.join(BASE_DIR, ".."))
DB_PATH = os.path.join(PROJECT_PATH, "data", "chroma_db")
TOP_K = 5

api_key = st.secrets.get("OPENROUTER_API_KEY", os.getenv("OPENROUTER_API_KEY"))

# ==========================================
# Load Models & Retriever (Cached)
# ==========================================
@st.cache_resource
def load_retriever():
    if not os.path.exists(DB_PATH):
        return None
        
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embedding_model
    )
    return vector_store.as_retriever(search_kwargs={"k": TOP_K})

@st.cache_resource
def load_llm(api_key):
    return ChatOpenAI(
        model="openai/gpt-4o-mini",
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )

# ==========================================
# Sidebar Dashboard
# ==========================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg", width=140)
    st.markdown("### 📊 System Status")
    
    # API Status Check
    if api_key:
        st.success("🟢 API Key Connected", icon="✅")
    else:
        st.error("🔴 API Key Missing", icon="🚨")
        
    # Database Status Check
    if os.path.exists(DB_PATH):
        st.success("🟢 Vector DB Active", icon="🗄️")
    else:
        st.error("🔴 Vector DB Not Found", icon="⚠️")

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

retriever = load_retriever()

# Validation Checks
if not api_key:
    st.warning("⚠️ **System Not Ready:** Please configure `OPENROUTER_API_KEY` in `.streamlit/secrets.toml` to proceed.")
elif retriever is None:
    st.error(f"⚠️ **Database Error:** Could not locate Chroma DB at `{DB_PATH}`. Please create the embeddings first.")
else:
    # Initialize Chat History
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("🔍 View Retrieved Review Excerpts"):
                    for idx, src in enumerate(message["sources"], 1):
                        st.markdown(f"**Excerpt {idx}:**\n> {src.page_content}")

    # Chat Input
    if user_query := st.chat_input("Ask a question about Amazon products (e.g., How is the battery life?)..."):
        # Add User Message to History
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Generate Response
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
                    
                    # Display Sources in Expander
                    if sources:
                        with st.expander("🔍 View Retrieved Review Excerpts"):
                            for idx, src in enumerate(sources, 1):
                                st.markdown(f"**Excerpt {idx}:**\n> {src.page_content}")

                    # Save Assistant Response to History
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response_text,
                        "sources": sources
                    })

                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")