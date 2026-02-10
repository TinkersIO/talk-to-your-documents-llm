import os
import asyncio
import atexit
import PyPDF2
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv

load_dotenv(override=True) 
# ---------------- MCP Client ----------------
from langchain_mcp_adapters.client import MultiServerMCPClient

# ---------------- Local Modules ----------------
from loaders import FileLoader
from textprocessing import TextProcessor
from vectorstore import VectorStore
from llm import LLMManager
from langchain_core.documents import Document

# ---------------- MCP Servers ----------------
MCP_SERVERS = {
    "filesystem": {
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "./uploads"],
    },
    "sqlite": {
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@executeautomation/database-server", "./database.db"],
    },
}


class MCPClient:
    def __init__(self):
        self.mcp = MultiServerMCPClient(MCP_SERVERS)
        self.tools = {}

    async def init_tools(self):
        if not self.tools:
            tools = await self.mcp.get_tools()
            self.tools = {t.name: t for t in tools}

    async def get_tool(self, name):
        await self.init_tools()
        return self.tools[name]

    # ---------- Filesystem MCP ----------
    async def upload_file(self, filename, content: bytes):
        write_file = await self.get_tool("write_file")
        import base64
        encoded = base64.b64encode(content).decode("utf-8")
        path = f"uploads/{os.path.basename(filename)}"
        await write_file.arun({
            "path": path,
            "content": encoded,
            "encoding": "base64"
        })
        return path

    # ---------- SQLite MCP (INSERT ONLY) ----------
    async def save_metadata(self, metadata: dict):
        write_query = await self.get_tool("write_query")
        sql = f"""
        INSERT INTO documents (filename, upload_date, metadata)
        VALUES (
            '{metadata["filename"]}',
            '{metadata["upload_date"]}',
            '{metadata["metadata"]}'
        );
        """
        await write_query.arun({"query": sql})

    async def close(self):
        await self.mcp.close()


def run_async(coro):
    return asyncio.run(coro)


# ---------------- Streamlit App ----------------
st.set_page_config(page_title="📄 Talk To My Docs", layout="wide")
st.title("📄 Talk to Your Documents")

# ---------------- Init Components ----------------
mcp_client = MCPClient()
file_loader = FileLoader()
text_processor = TextProcessor()
vectorstore = VectorStore()  # <-- Local Qdrant + embeddings
llm = LLMManager()

# ---------------- Session State ----------------
if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = []

atexit.register(lambda: asyncio.get_event_loop().run_until_complete(mcp_client.close()))


# ---------------- Helpers ----------------
def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def process_documents_and_add_to_vectorstore(documents):
    all_docs = []
    for doc in documents:
        chunks = chunk_text(doc["content"])
        for chunk in chunks:
            processed_chunk = Document(
                page_content=chunk,
                metadata={"filename": doc["filename"], "mcp_path": doc["mcp_path"]}
            )
            all_docs.append(processed_chunk)

    # Add all chunks to vectorstore
    if all_docs:
        vectorstore.add_documents(all_docs)
    return all_docs


# ---------------- Sidebar Upload ----------------
st.sidebar.header("📤 Upload Documents")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF / DOCX / CSV / XLSX",
    type=["pdf", "docx", "csv", "xlsx"],
    accept_multiple_files=True
)

if uploaded_files:
    st.session_state.uploaded_docs.clear()

    with st.spinner("Uploading, processing and indexing documents..."):
        for file in uploaded_files:
            content = ""
            if file.type == "application/pdf":
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    content += page.extract_text() or ""
            else:
                content = file.getvalue().decode("utf-8", errors="ignore")

            mcp_path = run_async(
                mcp_client.upload_file(file.name, file.getvalue())
            )

            run_async(
                mcp_client.save_metadata({
                    "filename": file.name,
                    "upload_date": datetime.now().isoformat(),
                    "metadata": mcp_path
                })
            )

            st.session_state.uploaded_docs.append({
                "filename": file.name,
                "content": content,
                "mcp_path": mcp_path
            })

        process_documents_and_add_to_vectorstore(st.session_state.uploaded_docs)
        st.sidebar.success("✅ Documents uploaded & indexed successfully!")


# ---------------- Chat Section ----------------
st.header("💬 Ask questions from your documents")
query = st.text_input("Ask a question:")

if query and query.strip():
    if not st.session_state.uploaded_docs:
        st.warning("Please upload documents first.")
    else:
        with st.spinner("Searching documents..."):
            # Retrieve top 3 most relevant chunks from vectorstore
            top_docs = vectorstore.similarity_search(query, k=3)

            # Assemble context for LLM
            context = "\n\n".join(
                f"[Source: {r.metadata.get('filename', 'Unknown')}]\n{r.page_content}"
                for r in top_docs
            )

            # Generate answer
            answer = llm.generate_answer(query, context)

        st.subheader("Answer")
        st.markdown(f"💡 **{answer}**")

        st.subheader("Sources")
        for doc in top_docs:
            st.markdown(f"- **{doc.metadata.get('filename', 'unknown')}**")
else:
    st.info("📤 Upload documents and ask a question to get started.")
