from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents.stuff import create_stuff_documents_chain


class LLMManager:
    def __init__(
        self,
        model_name: str = "llama-3.1-8b-instant",
        temperature: float = 0
    ):
        self.llm = ChatGroq(
            model=model_name,
            temperature=temperature
        )

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

        chain = expansion_prompt | self.llm
        response = chain.invoke({"query": query})

        expanded_queries = [
            q.strip().lstrip("0123456789.- ")
            for q in response.content.split("\n")
            if q.strip()
        ][:3]

        return [query] + expanded_queries

    # ---------------- Answer Generation ----------------
    def generate_answer(self, query: str, documents):
        answer_prompt = ChatPromptTemplate.from_template("""
        You are a helpful assistant.
        Answer the question using ONLY the provided context.
        If the context partially answers the question, summarize it.
        If the context does not contain the answer, say "I don't know".

        <context>
        {context}
        </context>

        Question: {input}
        """)

        document_chain = create_stuff_documents_chain(
            llm=self.llm,
            prompt=answer_prompt
        )

        result = document_chain.invoke({
            "input": query,
            "context": documents
        })

        if isinstance(result, dict):
            return result.get("answer", "")
        return result

