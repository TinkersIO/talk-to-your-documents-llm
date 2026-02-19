import os
import asyncio
import nest_asyncio
import streamlit as st
from datetime import datetime
import sqlite3
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.schema import Document
from llm import LLMManager
from uploads import FileUpload
from vectorstore import VectorStore

# ---------------- Config ----------------
UPLOADS_DIR = r"D:\PythonProjects\modular_1\uploads"
os.makedirs(UPLOADS_DIR, exist_ok=True)
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
TOP_K_CHUNKS = 5  # number of chunks to retrieve per query

# ---------------- Streamlit Setup ----------------
nest_asyncio.apply()
st.set_page_config(page_title="📄 Talk To My Docs", layout="wide")
st.title("📄 Talk to Your Documents")

# ---------------- LLM Agent ----------------
async def init_manager():
    manager = LLMManager()
    await manager.initialize()
    return manager

if "agent_manager" not in st.session_state:
    st.session_state.agent_manager = asyncio.run(init_manager())

agent = st.session_state.agent_manager.get_agent()

# ---------------- Vectorstore ----------------
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

# ---------------- Rebuild Vectorstore from DB ----------------
def rebuild_vectorstore():
    conn = sqlite3.connect(upload_service.db_path)
    cur = conn.cursor()
    cur.execute("SELECT filename, filepath FROM documents")
    docs = cur.fetchall()
    conn.close()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks_to_add = []

    for filename, filepath in docs:
        if filename in st.session_state.uploaded_docs_processed:
            continue

        content = upload_service._extract_text(filepath)
        doc_chunks = splitter.split_text(content)

        for i, chunk in enumerate(doc_chunks):
            chunks_to_add.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "filename": filename,
                        "chunk_index": i,
                        "upload_date": datetime.now().isoformat()
                    }
                )
            )

        st.session_state.uploaded_docs_processed.add(filename)

    if chunks_to_add:
        vectorstore.add_documents(chunks_to_add)

rebuild_vectorstore()

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

# ---------------- Sidebar: File Upload ----------------
st.sidebar.header("📤 Upload New Documents")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF / TXT / CSV",
    type=["pdf", "txt", "csv"],
    accept_multiple_files=True
)

if uploaded_files:
    with st.spinner("Processing documents..."):
        new_files = [f for f in uploaded_files if f.name not in st.session_state.uploaded_docs_processed]

        if new_files:
            uploaded_docs = upload_service.upload_files(new_files)
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP
            )

            chunks_to_add = []

            for doc in uploaded_docs:
                doc_chunks = splitter.split_text(doc["content"])
                for i, chunk in enumerate(doc_chunks):
                    chunks_to_add.append(
                        Document(
                            page_content=chunk,
                            metadata={
                                "filename": doc["filename"],
                                "chunk_index": i,
                                "upload_date": datetime.now().isoformat()
                            }
                        )
                    )

                st.session_state.uploaded_docs_processed.add(doc["filename"])

            vectorstore.add_documents(chunks_to_add)
            st.success("✅ Document chunks added to vectorstore!")

# ---------------- Chat Section ----------------
st.header("💬 Ask questions from your documents")
query = st.text_input("Enter your question here:")

if query and query.strip():
    if not rows and not uploaded_files:
        st.warning("Please upload documents first.")
    else:
        with st.spinner("Thinking..."):

            # ---------------- Step 1: Query Expansion ----------------
            expansion_prompt = f"""
You are expanding a user query to improve document retrieval.
    Generate EXACTLY 3 alternative search queries.
    Do NOT number them.
    Do NOT add bullet points.
    Each query must be on a new line.
                                                        
    Add:
    - synonyms
    - related technical terms
    - alternative phrasings

    Return ONLY the queries.
    Do NOT explain

Original Question:
{query}

"""

            try:
                expanded = asyncio.run(agent.ainvoke({
                    "input": [
                        {"role": "system", "content": "You are a search query expansion assistant."},
                        {"role": "user", "content": expansion_prompt}
                    ]
                }))

                if isinstance(expanded, dict):
                    expanded_queries = expanded.get("output", "").split("\n")
                else:
                    expanded_queries = str(expanded).split("\n")

                expanded_queries = [q.strip() for q in expanded_queries if q.strip()]
            except:
                expanded_queries = []

            all_queries = [query] + expanded_queries

            # ---------------- Step 2: Retrieve relevant chunks ----------------
            retrieved_docs = []

            for q in all_queries:
                docs = vectorstore.similarity_search(q, k=TOP_K_CHUNKS)
                retrieved_docs.extend(docs)

            # Remove duplicate chunks
            unique_docs = {doc.page_content: doc for doc in retrieved_docs}
            relevant_docs = list(unique_docs.values())

            # ---------------- Step 3: Build context ----------------
            context_text = ""
            for doc in relevant_docs:
                meta = doc.metadata
                context_text += f"Filename: {meta.get('filename', 'unknown')}\n"
                context_text += f"Content:\n{doc.page_content}\n\n"

            # ---------------- Step 4: Agent tool prompt ----------------
            tool_prompt = f"""
You are a knowledge agent with access to MCP tools:
1. SQLite MCP tool - query document metadata.
2. Filesystem MCP tool - read document content.

Workflow Rules:
- Decide which MCP tool to call based on the user question.
- Retrieve information only via tools, do not hallucinate.
- Use the following document context from vectorstore if relevant:
{context_text}
- Rerank top chunks and summarize only after retrieving content.
- If information is not found → respond exactly: "I don't know".
- Always include filename and source path in your answer.
- Return only the final answer, do not show tool calls.
"""

            messages = [
                {"role": "system", "content": tool_prompt},
                {"role": "user", "content": query}
            ]

            # ---------------- Step 5: Call agent ----------------
            try:
                result = asyncio.run(agent.ainvoke({"input": messages}))
                if isinstance(result, dict):
                    answer = result.get("output", "I don't know")
                else:
                    answer = str(result)
            except Exception as e:
                answer = f"Error processing query: {e}"

        st.subheader("Answer")
        st.markdown(answer)
else:
    st.info("Enter a question to chat with your documents.")
