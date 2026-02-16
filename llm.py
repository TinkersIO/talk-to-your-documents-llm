import os
import asyncio
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

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

#  Plain LLM (NO TOOLS)
def get_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY"),
    )

#  MCP Agent (WITH TOOLS)
async def get_agent():
    llm = get_llm()
    client = MultiServerMCPClient(MCP_SERVERS)
    tools = await client.get_tools()
    agent = create_agent(model=llm, tools=tools)
    return agent