import os
import asyncio
from datetime import datetime
import base64
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_core.tools import Tool
from mcp_client import load_mcp_tools, upload_file_via_mcp, save_metadata_via_mcp
from llm import LLMManager
from vectorstore import VectorStore
from textprocessing import TextProcessor

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

vectorstore = VectorStore()
llm_manager = LLMManager()

# ---------------- MCP Setup ----------------

mcp_client, mcp_tools = load_mcp_tools()

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

import nest_asyncio
nest_asyncio.apply() 


if uploaded_files:
    st.session_state.uploaded_docs.clear()

    with st.spinner("LLM is processing documents..."):

        async def upload_all_files():
            tasks = [process_upload(file, write_file_tool, write_query_tool) for file in uploaded_files]
            return await asyncio.gather(*tasks)

        uploaded_docs = asyncio.get_event_loop().run_until_complete(upload_all_files())


        st.session_state.uploaded_docs.extend(uploaded_docs)


        processor = TextProcessor(chunk_size=500, chunk_overlap=50)
        docs = []
        for doc in st.session_state.uploaded_docs:
            doc_chunks = processor.process(doc["content"], doc["filename"])
            docs.extend(doc_chunks)
        vectorstore.add_documents(docs)

        st.sidebar.success("✅ Documents uploaded & indexed")


# ---------------- Chat Section ----------------

st.header("💬 Ask questions from your documents")

query = st.text_input("Ask a question")

if query:
    if not st.session_state.uploaded_docs:
        st.warning("Please upload documents first.")
    else:
        with st.spinner("Thinking..."):
            # Check if user query mentions a specific uploaded file
            matched_file = None
            for doc in st.session_state.uploaded_docs:
                if doc["filename"] in query:
                    matched_file = doc["filename"]
                    break

            if matched_file:
                
                doc_content = next(
                    (d["content"] for d in st.session_state.uploaded_docs if d["filename"] == matched_file),
                    ""
                )
                processor = TextProcessor(chunk_size=500, chunk_overlap=50)
                doc_chunks = processor.process(doc_content, matched_file)

                top_chunks = doc_chunks[:3]

                # Generate answer using the LLMManager
                answer = llm_manager.generate_answer(
                    query,  
                    top_chunks  
                )
                top_docs = doc_chunks  
            else:
                # General similarity search across all documents
                top_docs = vectorstore.similarity_search(query, k=3)
                answer = llm_manager.generate_answer(
                    query,
                    top_docs
                )

        st.subheader("Answer")
        st.markdown(answer)

        st.subheader("Sources")
        for doc in top_docs:
            st.markdown(f"- **{doc.metadata.get('filename', 'unknown')}**")