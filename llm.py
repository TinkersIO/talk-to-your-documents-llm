from langchain_groq import ChatGroq
from langchain_classic.agents import create_tool_calling_agent
from langchain_classic.agents import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import Tool


class LLMManager:
    def __init__(self, tools: list[Tool]):
        # ---------------- LLM ----------------
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0,
            max_tokens=700,
            timeout=30
        )

        # ---------------- System Prompt ----------------
        system_prompt = """
You are an intelligent document assistant.

You have access to tools that allow you to:
- Search document chunks
- Read full documents
- Query document metadata

IMPORTANT TOKEN SAFETY RULES:

1. Never load or process an entire large document at once.
2. If a document is large:
   - Break it into smaller logical sections.
   - Summarize each section individually.
   - Then combine the partial summaries into a final summary.
3. Prefer retrieving only relevant chunks instead of full documents.
4. Always minimize the amount of text sent to the model.
5. If content is too large to process safely, summarize progressively.

TASK RULES:

• If the user asks for a summary:
  → Retrieve the document in manageable parts.
  → Perform step-by-step summarization.
  → Return a concise final summary.

• If the user asks a specific question:
  → Retrieve only relevant sections.
  → Answer strictly from retrieved data.

• If answer cannot be found:
  → Say "I don't know".

Never hallucinate.
Always respect token limits.
"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])

        # ---------------- Agent ----------------
        self.agent = create_tool_calling_agent(
            self.llm,
            tools,
            prompt
        )

        self.executor = AgentExecutor(
            agent=self.agent,
            tools=tools,
            verbose=True  
        )

    # ---------------- Main Answer Function ----------------
    async def generate_answer(self, query: str, documents: list):
        if not documents:
            return "I don't know"
        MAX_CHARS = 2000
        context_chunks = []

        for doc in documents:
            text = doc.page_content
            for i in range(0, len(text), MAX_CHARS):
                context_chunks.append(text[i:i+MAX_CHARS])

    
        context = "\n\n".join(context_chunks[:5])
        prompt = f"""
    You are a helpful assistant.
    Answer the question using ONLY the provided context.
    If the context does not contain the answer, say "I don't know".

    Context:
    {context}

    Question: {query}
    """

        response = await self.llm.ainvoke(prompt)
        return response.content.strip()
