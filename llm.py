import os
from langchain_groq import ChatGroq
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from mcp_client import MCPClient


class LLMManager:
    """
    Handles LLM and MCP tool-calling agent.
    Async-initialized but Streamlit-safe.
    """

    def __init__(self):
        self.llm = None
        self.agent_executor = None
        self.mcp_client = MCPClient()

    # ---------------------------
    # ASYNC INITIALIZATION
    # ---------------------------
    async def initialize(self):
        """Initialize MCP tools and agent safely."""
        await self.mcp_client.initialize()
        self._create_agent()

    # ---------------------------
    # LLM SETUP
    # ---------------------------
    def _get_llm(self):
        if not self.llm:
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise RuntimeError("GROQ_API_KEY not set!")

            self.llm = ChatGroq(
                model="meta-llama/llama-4-maverick-17b-128e-instruct",
                temperature=0,
                groq_api_key=api_key,
            )

        return self.llm

    # ---------------------------
    # AGENT CREATION
    # ---------------------------
    def _create_agent(self):
        llm = self._get_llm()
        tools = self.mcp_client.get_tools()

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "Only take files from the uploads folder."
                "You are an intelligent assistant. "
                "Use tools whenever required to answer questions. "
                "Never guess. If information is missing, say 'I don't know'.",
            ),
            ("user", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])

        agent = create_tool_calling_agent(
            llm=llm,
            tools=tools,
            prompt=prompt,
        )

        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
        )

    # ---------------------------
    # PUBLIC ACCESS
    # ---------------------------
    def get_agent(self):
        if not self.agent_executor:
            raise RuntimeError(
                "LLMManager not initialized. Call await initialize() first."
            )
        return self.agent_executor
