class AgentService:
    def __init__(self, agent):
        self.agent = agent

    async def answer(self, query, context_text):
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

        result = await self.agent.ainvoke({"input": messages})

        if isinstance(result, dict):
            return result.get("output", "I don't know")

        return str(result)
