import asyncio
import atexit
import os
from typing import List
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from uploads import UPLOADS_DIR

UPLOADS_DIR = r"D:\PythonProjects\modular_1\uploads"

class MCPClient:
    """
    Manages 2 MCP servers:
    - Filesystem MCP (read files)
    - SQLite MCP (query metadata)
    """

    MCP_SERVERS = {
        "filesystem": {
            "transport": "stdio",
            "command": "npx",
            "args": [
                "-y",
                "@modelcontextprotocol/server-filesystem",
                 str(UPLOADS_DIR),
            ],
        },
        "sqlite": {
            "transport": "stdio",
            "command": "npx",
            "args": [
                "-y",
                "@executeautomation/database-server",
                str(os.path.join(UPLOADS_DIR, "database.db")),
            ],
        },
    }

    def __init__(self):
        self._client = MultiServerMCPClient(self.MCP_SERVERS)
        self.tools: List[BaseTool] | None = None

        self._register_cleanup()

    async def initialize(self):
        """Load tools asynchronously (SAFE for Streamlit)."""
        self.tools = await self._client.get_tools()

        if not self.tools:
            raise RuntimeError("No MCP tools were loaded!")

        print("Loaded MCP tools:")
        for t in self.tools:
            print(" -", t.name)

    def get_tools(self) -> List[BaseTool]:
        if self.tools is None:
            raise RuntimeError("MCPClient not initialized. Call initialize() first.")
        return self.tools

    def _register_cleanup(self):
        async def cleanup():
            await self._client.close()

        def sync_cleanup():
            try:
                asyncio.run(cleanup())
            except RuntimeError:
                pass

        atexit.register(sync_cleanup)

