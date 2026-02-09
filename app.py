import os
import asyncio
import cohere
import atexit
import PyPDF2
import time
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv
import base64
from cohere import Client

# ---------------- MCP Client ----------------
from langchain_mcp_adapters.client import MultiServerMCPClient

# ---------------- Local Modules ----------------
from loaders import FileLoader
from textprocessing import TextProcessor
from vectorstore import VectorStore
from llm import LLMManager
from langchain_core.documents import Document

# ---------------- Load ENV ----------------
load_dotenv(override=True)
client = Client(os.getenv("COHERE_API_KEY"))

# ---------------- Rate Limiting ----------------
CALLS_PER_MINUTE = 100
TIME_WINDOW = 60


class APIClient:
    def __init__(self):
        self.api_calls = 0
        self.start_time = time.time()

    def reset_calls(self):
        if time.time() - self.start_time >= TIME_WINDOW:
            self.api_calls = 0
            self.start_time = time.time()

    def rate_limit(self):
        self.reset_calls()
        if self.api_calls >= CALLS_PER_MINUTE:
            time.sleep(TIME_WINDOW)
            self.api_calls = 0

    def call_api(self, texts):
        self.rate_limit()

        try:
            self.api_calls += 1

            response = client.embed(
            texts=texts,
            model="embed-english-v3.0"
            )

            return response.embeddings

        except cohere.errors.TooManyRequestsError:
            print("Rate limit exceeded. Retrying...")
            time.sleep(60)
            return self.call_api(texts)


api_client = APIClient()

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

# ---------------- Session State ----------------
if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = []

# ---------------- Init Components ----------------
mcp_client = MCPClient()
file_loader = FileLoader()
text_processor = TextProcessor()
vectorstore = VectorStore()
llm_manager = LLMManager()

atexit.register(lambda: asyncio.get_event_loop().run_until_complete(mcp_client.close()))

# ---------------- Embed Helper ----------------
def process_documents_and_embed(docs):
    texts = [d["content"] for d in docs]
    embeddings = api_client.call_api(texts)
    for emb, doc in zip(embeddings, docs):
        doc["embedding"] = emb
    return docs


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

        process_documents_and_embed(st.session_state.uploaded_docs)
        st.sidebar.success("✅ Documents uploaded & processed successfully!")


# ---------------- Chat Section ----------------
st.header("💬 Ask questions from your documents")
query = st.text_input("Ask a question:")

if query and query.strip():
    if not st.session_state.uploaded_docs:
        st.warning("Please upload documents first.")
    else:
        with st.spinner("Searching documents..."):
            top_docs = st.session_state.uploaded_docs[:3]
            answer = "This is a simulated answer based on the documents provided."

        st.subheader("Answer")
        st.markdown(f"💡 **{answer}**")

        st.subheader("Sources")
        for doc in top_docs:
            st.markdown(f"- **{doc['filename']}**")

else:
    st.info("📤 Upload documents and ask a question to get started.")
