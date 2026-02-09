import os
import asyncio
import atexit
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv
import base64

# ---------------- MCP Client ----------------
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_groq import ChatGroq

# ---------------- Load ENV ----------------
load_dotenv(override=True)

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

# ---------------- MCP Client Wrapper ----------------
class MCPClient:
    """Simple MCP client with async filesystem + SQLite operations."""
    def __init__(self):
        self.mcp = MultiServerMCPClient(MCP_SERVERS)
        self.tools = {}
        self.llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model="llama-3.1-8b-instant",
            temperature=0,
        )

    async def init_tools(self):
        if not self.tools:
            tools_list = await self.mcp.get_tools()
            self.tools = {t.name: t for t in tools_list}

    async def get_tool(self, name):
        await self.init_tools()
        tool = self.tools.get(name)
        if not tool:
            raise ValueError(f"Tool '{name}' not found. Available: {list(self.tools.keys())}")
        return tool

    # ---------------- Filesystem ----------------
    async def upload_file(self, filename, content: bytes):
        write_file = await self.get_tool("write_file")

        # Ensure uploads folder exists
        os.makedirs("uploads", exist_ok=True)

        # Only filename, MCP sandbox requires relative path inside ./uploads
        safe_filename = os.path.basename(filename)
        mcp_path = f"uploads/{safe_filename}"

        encoded = base64.b64encode(content).decode("utf-8")
        await write_file.arun({
            "path": mcp_path,
            "content": encoded,
            "encoding": "base64"
        })
        return mcp_path

    async def read_file(self, filename):
        read_file = await self.get_tool("read_file")
        result = await read_file.arun({"path": filename})
        return base64.b64decode(result["content"])

    # ---------------- LLM ----------------
    async def generate_answer(self, query: str, docs: list):
        context = "\n\n".join(doc["page_content"] for doc in docs)
        prompt = f"""
Answer the question using the context below.
- Be concise (3-4 sentences max)
- Do NOT repeat the question

Context:
{context}

Question:
{query}
"""
        response = self.llm.invoke(prompt)
        return response.content.strip()

    # ---------------- Cleanup ----------------
    async def close(self):
        await self.mcp.close()


# ---------------- Async helper ----------------
def run_async(coro):
    return asyncio.run(coro)


# ---------------- Streamlit App ----------------
st.set_page_config(page_title="📄 Talk To My Docs", layout="wide")
st.title("📄 Talk to Your Docs")

mcp_client = MCPClient()

# Ensure MCP servers close on exit
atexit.register(lambda: asyncio.get_event_loop().run_until_complete(mcp_client.close()))

# ---------------- Sidebar: Upload ----------------
st.sidebar.header("📤 Upload Documents")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF / DOCX / CSV / XLSX",
    type=["pdf", "docx", "csv", "xlsx"],
    accept_multiple_files=True
)

# Keep track of uploaded docs for LLM context
uploaded_docs = []

if uploaded_files:
    with st.spinner("Uploading and indexing files..."):
        for file in uploaded_files:
            content = file.getvalue()
            # Upload file inside ./uploads via MCP
            file_id = run_async(mcp_client.upload_file(file.name, content))
            # Add to context for demo purposes
            uploaded_docs.append({
                "page_content": f"File: {file.name}",
                "file_id": file_id
            })
    st.sidebar.success("✅ Files uploaded successfully!")

# ---------------- Chat Section ----------------
st.header("💬 Chat with your documents")
query = st.text_input("Ask a question:")

if query.strip():
    if not uploaded_docs:
        st.warning("Please upload documents first.")
    else:
        with st.spinner("Generating answer..."):
            answer = run_async(mcp_client.generate_answer(query, uploaded_docs))
        st.subheader("Answer")
        st.markdown(f"💡 {answer}")


