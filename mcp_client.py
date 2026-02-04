import os
import uuid
from typing import List
from langchain_groq import ChatGroq


class MCPClient:
    def __init__(self, upload_dir: str = "uploads"):
        """
        Acts as a logical MCP client.
        Filesystem MCP responsibilities are simulated via local FS.
        SQLite MCP responsibilities are handled by the database module.
        """
        self.upload_dir = upload_dir
        os.makedirs(self.upload_dir, exist_ok=True)

        self.llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model="llama-3.1-8b-instant",
            temperature=0
        )

    # ---------------- Filesystem MCP ----------------
    def upload_file(self, uploaded_file) -> str:
        """
        Simulates uploading a file to Filesystem MCP server.

        Returns:
            file_id (str): unique identifier for the stored file
        """
        try:
            file_id = f"{uuid.uuid4()}_{uploaded_file.name}"
            file_path = os.path.join(self.upload_dir, file_id)

            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            return file_id
        except Exception as e:
            print(f"[MCP Filesystem Error] {e}")
            return ""

    def get_file_path(self, file_id: str) -> str:
        """
        Resolve file_id to actual filesystem path
        """
        return os.path.join(self.upload_dir, file_id)

    # ---------------- Query Expansion MCP ----------------
    def expand_query(self, query: str) -> List[str]:
        """
        Simulates MCP Query Expansion capability.

        In real MCP:
        - This could be handled by an LLM MCP server.
        """
        return [
            query,
            f"Explain {query}",
            f"Details about {query}",
        ]

    # ---------------- Answer Generation MCP ----------------
    def generate_answer(self, query, docs):
        context = "\n\n".join(doc.page_content for doc in docs)

        prompt = f"""
Answer the question using the context below.
- Answer in **3-4 concise sentences only**
- Be concise and clear
- Do NOT repeat the question
- Do NOT add headings
- Summarize in your own words

Context:
{context}

Question:
{query}
"""

        response = self.llm.invoke(prompt)
        return response.content.strip()
