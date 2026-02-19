import os
import asyncio
import nest_asyncio
import streamlit as st
import sqlite3

from llm import LLMManager
from uploads import FileUpload
from vectorstore import VectorStore

from core.query_expander import QueryExpander
from core.retriever import Retriever
from core.context_builder import ContextBuilder
from core.agent_service import AgentService
from core.rag_pipeline import RAGPipeline

from services.rebuilder import rebuild_vectorstore
from services.upload_handler import process_uploaded_files

# ---------------- Config ----------------
UPLOADS_DIR = r"D:\PythonProjects\modular_1 - Copy\uploads"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
TOP_K_CHUNKS = 5

os.makedirs(UPLOADS_DIR, exist_ok=True)

# ---------------- Streamlit Setup ----------------
nest_asyncio.apply()
st.set_page_config(page_title="📄 Talk To My Docs", layout="wide")
st.title("📄 Talk to Your Documents")

# ---------------- Initialize Agent ----------------
async def init_manager():
    manager = LLMManager()
    await manager.initialize()
    return manager

if "agent_manager" not in st.session_state:
    st.session_state.agent_manager = asyncio.run(init_manager())

agent = st.session_state.agent_manager.get_agent()

# ---------------- Initialize Vectorstore ----------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = VectorStore()

vectorstore = st.session_state.vectorstore

# ---------------- Upload Service ----------------
if "upload_service" not in st.session_state:
    st.session_state.upload_service = FileUpload(
        upload_dir=UPLOADS_DIR,
        db_path=os.path.join(UPLOADS_DIR, "database.db")
    )

upload_service = st.session_state.upload_service

if "uploaded_docs_processed" not in st.session_state:
    st.session_state.uploaded_docs_processed = set()

# ---------------- Rebuild Vectorstore ----------------
rebuild_vectorstore(
    upload_service,
    vectorstore,
    st.session_state.uploaded_docs_processed,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)

# ---------------- Sidebar: Document List ----------------
st.sidebar.header("📂 Your Documents")

conn = sqlite3.connect(upload_service.db_path)
cur = conn.cursor()
cur.execute("SELECT filename, upload_date FROM documents ORDER BY upload_date DESC")
rows = cur.fetchall()
conn.close()

if rows:
    for filename, upload_date in rows:
        st.sidebar.markdown(f"- **{filename}** | {upload_date}")
else:
    st.sidebar.info("Upload any file first!")

# ---------------- Sidebar: Upload ----------------
st.sidebar.header("📤 Upload New Documents")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF / TXT / CSV",
    type=["pdf", "txt", "csv"],
    accept_multiple_files=True
)

if uploaded_files:
    with st.spinner("Processing documents..."):
        process_uploaded_files(
            upload_service,
            vectorstore,
            uploaded_files,
            st.session_state.uploaded_docs_processed,
            CHUNK_SIZE,
            CHUNK_OVERLAP
        )
        st.success("✅ Document chunks added to vectorstore!")

# ---------------- Build RAG Pipeline ----------------
expander = QueryExpander(agent)
retriever = Retriever(vectorstore, TOP_K_CHUNKS)
context_builder = ContextBuilder()
agent_service = AgentService(agent)

pipeline = RAGPipeline(
    expander,
    retriever,
    context_builder,
    agent_service
)

# ---------------- Chat ----------------
st.header("💬 Ask questions from your documents")
query = st.text_input("Enter your question here:")

if query and query.strip():
    if not rows and not uploaded_files:
        st.warning("Please upload documents first.")
    else:
        with st.spinner("Thinking..."):
            answer = asyncio.run(pipeline.run(query))
        st.subheader("Answer")
        st.markdown(answer)
else:
    st.info("Enter a question to chat with your documents.")
