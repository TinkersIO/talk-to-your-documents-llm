import os
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain_core.tools import Tool
from langchain_classic.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents.stuff import create_stuff_documents_chain


class LLMManager:
    def __init__(self, tools=None):
        self.llm = ChatGroq (
            model="llama-3.1-8b-instant",
            temperature=0,
            max_tokens=1000,
            timeout=30
            )
        
        self.agent=None
        if tools:
            self.agent = create_agent(self.llm, tools=tools)

          


    # ---------------- Query Expansion ----------------
    def expand_query(self, query: str) -> list[str]:
        expansion_prompt = ChatPromptTemplate.from_template("""
        You are expanding a user query to improve document retrieval.
        Generate EXACTLY 3 alternative search queries.
        Do NOT number them.
        Do NOT add bullet points.
        Each query must be on a new line.

        Add:
        - synonyms
        - related technical terms
        - alternative phrasings

        Return ONLY the queries.

        User query: {query}
        """)

        prompt_text = expansion_prompt.format(query=query)
        response = self.llm.invoke(prompt_text)

        expanded_queries = [
            q.strip().lstrip("0123456789.- ")
            for q in response.content.split("\n")
            if q.strip()
        ][:3]

        return [query] + expanded_queries

    # ---------------- Answer Generation ----------------
    def generate_answer(self, query: str, documents=None):
        """
        Generate an answer for the given query.

    - If an agent is initialized, use the agent.
    - Otherwise, use the raw LLM on the provided documents.
    """

    # If agent exists, delegate to agent (documents not needed)
        if self.agent:
            response = self.agent.run(query)
            return response.strip()  # agent returns string

    # If no documents, return fallback
        if not documents:
            return "I don't know"

    # Convert documents to text
        docs_text = "\n\n".join(
            doc.page_content if hasattr(doc, "page_content") else str(doc)
            for doc in documents
        )

    # Create prompt
        answer_prompt = ChatPromptTemplate.from_template("""
    You are a helpful assistant.
    Answer the question using ONLY the provided context.
    If the context partially answers the question, summarize it.
    Keep your answer concise, in 5-6 lines.
    If the context does not contain the answer, say "I don't know".

    Context:
    {context}

    Question: {input}
    """)

        prompt_text = answer_prompt.format(
        context=docs_text,
        input=query
    )

    # Use raw LLM
        response = self.llm.invoke(prompt_text)
        return response.content.strip()
