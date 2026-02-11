import asyncio
import atexit
import base64
from datetime import datetime
from typing import List, Tuple
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools import BaseTool, Tool

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

# ---------------- Minimal MCP Client ----------------
class MCPClient:
    """Starts MCP servers and exposes tools; LLM decides usage."""

    def __init__(self):
        self._client = MultiServerMCPClient(MCP_SERVERS)
        self._tools: List[BaseTool] | None = None

    async def get_tools(self) -> List[BaseTool]:
        if self._tools is None:
            self._tools = await self._client.get_tools()
        return self._tools

    async def close(self):
        await self._client.close()

# ---------------- Async Cleanup on Exit ----------------
def _register_exit_cleanup(client: MCPClient):
    """Registers client.close() to run when Python exits."""
    def _cleanup():
        try:
            loop = asyncio.get_running_loop()
            # If an event loop is running, schedule close as a task
            loop.create_task(client.close())
        except RuntimeError:
            # No running loop, safe to use asyncio.run
            asyncio.run(client.close())
    atexit.register(_cleanup)

# ---------------- Sync Loader for Streamlit ----------------
def load_mcp_tools() -> Tuple[MCPClient, List[BaseTool]]:
    """Sync wrapper for Streamlit: returns (client, tools)"""
    return asyncio.run(_load_mcp_tools())

async def _load_mcp_tools() -> Tuple[MCPClient, List[BaseTool]]:
    client = MCPClient()
    tools = await client.get_tools()
    # Register cleanup immediately after client creation
    _register_exit_cleanup(client)
    return client, tools

# ---------------- MCP Helper Functions ----------------
async def upload_file_via_mcp(write_tool: Tool, filename: str, content: bytes) -> str:
    """Uploads file to MCP filesystem safely."""
    encoded = base64.b64encode(content).decode("utf-8")
    path = f"uploads/{filename}"
    await write_tool.arun({
        "path": path,
        "content": encoded,
        "encoding": "base64"
    })
    return path

async def save_metadata_via_mcp(sql_tool: Tool, filename: str, metadata: str):
    """Saves metadata to MCP SQLite safely."""
    upload_date = datetime.utcnow().isoformat()
    sql = f"""
    INSERT INTO documents (filename, upload_date, metadata)
    VALUES ('{filename}', '{upload_date}', '{metadata}');
    """
    await sql_tool.arun({"query": sql})
