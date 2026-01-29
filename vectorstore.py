from langchain_chroma import Chroma
from langchain_cohere import CohereRerank
from langchain_community.embeddings import CohereEmbeddings
import os


class VectorStore:
    def __init__(
        self,
        persist_directory: str = "chroma_db",
        collection_name: str = "talk_to_docs"
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        self.embedding = CohereEmbeddings(
        cohere_api_key=os.getenv("COHERE_API_KEY"),
        user_agent="talk-to-my-docs-app"
    )


        self.vectordb = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding,
            persist_directory=self.persist_directory
        )

    # -------- Add documents --------
   
        def add_documents(self, docs):
            self.vectordb.add_documents(docs)

    # -------- Similarity search --------
    def similarity_search(self, query: str, k: int = 6):
        return self.vectordb.similarity_search(query, k=k)

    # -------- Reranking --------
    def rerank(self, documents, query: str, top_n: int = 5):
        reranker = CohereRerank(
            model="rerank-english-v3.0",
            top_n=top_n
        )
        return reranker.compress_documents(
            documents=documents,
            query=query
        )
