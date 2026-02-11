import os
import asyncio
from datetime import datetime
import base64
from langchain_groq import ChatGroq
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_core.tools import Tool
from mcp_client import MCPClient
from mcp_client import upload_file_via_mcp, save_metadata_via_mcp
from llm import LLMManager
from vectorstore import VectorStore
from textprocessing import TextProcessor
from langchain.agents import create_agent

load_dotenv(override=True)



# ---------------- Helpers ----------------

def extract_text_from_pdf(file) -> str:
    reader = PdfReader(file)
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)

async def process_upload(file, write_tool: Tool, sql_tool: Tool):
    """Uploads a single file and saves its metadata."""
    if file.type == "application/pdf":
        content = extract_text_from_pdf(file).encode("utf-8")
    else:
        content = file.getvalue()

    await upload_file_via_mcp(write_tool, file.name, content)

    
    await save_metadata_via_mcp(sql_tool, file.name, "User uploaded")

    return {"filename": file.name, "content": content.decode("utf-8", errors="ignore")}

# ---------------- Streamlit Setup ----------------

st.set_page_config(page_title="📄 Talk To My Docs", layout="wide")
st.title("📄 Talk to Your Documents")

# ---------------- MCP Setup ----------------
mcp_client = MCPClient()
mcp_tools = asyncio.run(mcp_client.get_tools())


vectorstore = VectorStore()
llm_manager = LLMManager(tools=mcp_tools)





write_file_tool = next((t for t in mcp_tools if t.name == "write_file"), None)
read_file_tool = next((t for t in mcp_tools if t.name == "read_file"), None)
write_query_tool = next((t for t in mcp_tools if t.name == "write_query"), None)

if not all([write_file_tool, read_file_tool, write_query_tool]):
    st.error("MCP tools not loaded correctly.")
    st.stop()

# ---------------- Session State ----------------

if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = []

# ---------------- Sidebar Upload ----------------
st.sidebar.header("📤 Upload Documents")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF / TXT / CSV",
    type=["pdf", "txt", "csv"],
    accept_multiple_files=True
)


UPLOAD_DIR = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

import nest_asyncio
nest_asyncio.apply()

if uploaded_files:
    st.session_state.uploaded_docs.clear()

    with st.spinner("Processing documents..."):

        uploaded_docs = []

        for file in uploaded_files:
            # Save file to uploads folder
            file_path = os.path.join(UPLOAD_DIR, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getvalue())

            # Extract text from file
            if file.type == "application/pdf":
                content = extract_text_from_pdf(file_path)
            else:
                content = file.getvalue().decode("utf-8", errors="ignore")

            # Upload via MCP (async)
            async def upload_file_task(file_name, file_content):
                await upload_file_via_mcp(write_file_tool, file_name, file_content.encode("utf-8"))
                await save_metadata_via_mcp(write_query_tool, file_name, "User uploaded")

            asyncio.get_event_loop().run_until_complete(upload_file_task(file.name, content))

            # Add to session state
            uploaded_docs.append({
                "filename": file.name,
                "content": content,
                "path": file_path
            })

        st.session_state.uploaded_docs.extend(uploaded_docs)

        # Chunk documents and add to vectorstore
        processor = TextProcessor(chunk_size=500, chunk_overlap=50)
        docs = []
        for doc in st.session_state.uploaded_docs:
            doc_chunks = processor.process(doc["content"], doc["filename"])
            docs.extend(doc_chunks)
        vectorstore.add_documents(docs)

        st.sidebar.success("✅ Documents uploaded, saved & indexed")


import asyncio
import nest_asyncio
nest_asyncio.apply() 
# ---------------- Chat Section ----------------
st.header("💬 Ask questions from your documents")

query = st.text_input("Ask a question")

if query:
    if not st.session_state.uploaded_docs:
        st.warning("Please upload documents first.")
    else:
        with st.spinner("Thinking..."):
           
            matched_file = next(
                (d["filename"] for d in st.session_state.uploaded_docs if d["filename"] in query),
                None
            )

            if matched_file:
               
                doc_content = next(
                    (d["content"] for d in st.session_state.uploaded_docs if d["filename"] == matched_file),
                    ""
                )
                processor = TextProcessor(chunk_size=500, chunk_overlap=50)
                doc_chunks = processor.process(doc_content, matched_file)
                top_chunks = [c for c in doc_chunks if c.page_content.strip()][:3]
            else:
                
                top_chunks = vectorstore.similarity_search(query, k=3)

            
            if not top_chunks:
                answer = "I don't know"
            else:
                
                answer = asyncio.get_event_loop().run_until_complete(
                    llm_manager.generate_answer(query, top_chunks)
                )

        
        st.subheader("Answer")
        st.markdown(answer)

        st.subheader("Sources")
        for doc in top_chunks:
            st.markdown(f"- **{doc.metadata.get('filename', 'unknown')}**")
