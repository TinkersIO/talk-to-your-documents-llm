import os
import asyncio
import atexit
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv
import base64

# ---------------- MCP ----------------
from langchain_mcp_adapters.client import MultiServerMCPClient

# ---------------- Your RAG components ----------------
from loaders import FileLoader
from textprocessing import TextProcessor
from vectorstore import VectorStore
from llm import LLMManager

# ---------------- Load ENV ----------------
load_dotenv(override=True)

# ---------------- MCP Servers ----------------
MCP_SERVERS = {
    "filesystem": {
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "./uploads"],
    }
}

# ---------------- MCP Client ----------------
class MCPClient:
    def __init__(self):
        self.mcp = MultiServerMCPClient(MCP_SERVERS)
        self.tools = {}

    async def init_tools(self):
        if not self.tools:
            tools = await self.mcp.get_tools()
            self.tools = {t.name: t for t in tools}

    async def upload_file(self, filename: str, content: bytes) -> str:
        await self.init_tools()
        write_file = self.tools["write_file"]

        os.makedirs("uploads", exist_ok=True)
        safe_name = os.path.basename(filename)
        mcp_path = f"uploads/{safe_name}"

        encoded = base64.b64encode(content).decode("utf-8")
        await write_file.arun({
            "path": mcp_path,
            "content": encoded,
            "encoding": "base64",
        })
        return mcp_path

    async def close(self):
        await self.mcp.close()

# ---------------- Async helper ----------------
def run_async(coro):
    return asyncio.run(coro)

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="📄 Talk To My Docs", layout="wide")
st.title("📄 Talk to Your Documents")

# ---------------- Initialize components ----------------
file_loader = FileLoader()
text_processor = TextProcessor()
vectorstore = VectorStore()
llm_manager = LLMManager()
mcp_client = MCPClient()

MAX_TOP_DOCS = 3

atexit.register(lambda: asyncio.get_event_loop().run_until_complete(mcp_client.close()))

# ---------------- Sidebar: Upload ----------------
st.sidebar.header("📤 Upload Documents")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF / DOCX / CSV / XLSX",
    type=["pdf", "docx", "csv", "xlsx"],
    accept_multiple_files=True,
)

if uploaded_files:
    with st.spinner("Processing and indexing documents..."):
        for uploaded_file in uploaded_files:

            # 1️⃣ Upload file via MCP (filesystem sandbox)
            content = uploaded_file.getvalue()
            run_async(mcp_client.upload_file(uploaded_file.name, content))

            # 2️⃣ Load text locally
            text = file_loader.load(uploaded_file)
            if not text.strip():
                continue

            # 3️⃣ Chunk text
            docs = text_processor.process(text, uploaded_file.name)
            upload_time = datetime.now().isoformat()

            # 4️⃣ Add metadata
            for doc in docs:
                doc.metadata["filename"] = uploaded_file.name
                doc.metadata["upload_date"] = upload_time

            # 5️⃣ Store in vector DB
            vectorstore.add_documents(docs)

    st.sidebar.success("✅ Documents indexed successfully!")

# ---------------- Chat ----------------
st.header("💬 Ask questions from your documents")
query = st.text_input("Ask a question:")

if query and query.strip():
    with st.spinner("Searching documents..."):

        # 1️⃣ Vector similarity search
        top_docs = vectorstore.similarity_search(
            query=query,
            k=MAX_TOP_DOCS,
        )

        if not top_docs:
            st.warning("❌ Answer not found in uploaded documents.")
            st.stop()

        # 2️⃣ LLM answer grounded in docs
        answer = llm_manager.generate_answer(query, top_docs)

    # ---------------- Display ----------------
    st.subheader("Answer")
    st.markdown(f"💡 **{answer}**")

    st.subheader("Sources")
    for doc in top_docs:
        st.markdown(
            f"- **{doc.metadata.get('filename')}** "
            f"({doc.metadata.get('upload_date', '')})"
        )
else:
    st.info("📤 Upload documents and ask a question to get started.")
