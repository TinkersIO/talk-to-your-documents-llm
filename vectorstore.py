import uuid
from typing import List
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

# ------------------------------
# ✅ GLOBAL SINGLETON CLIENT
# ------------------------------
_qdrant_client = None

class VectorStore:
    def __init__(self, collection_name: str = "docs"):
        global _qdrant_client

        self.collection_name = collection_name

        # -------- Local embeddings (offline) --------
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        # -------- Local Qdrant (SINGLETON) --------
        if _qdrant_client is None:
            _qdrant_client = QdrantClient(path="qdrant_db")
        self.client = _qdrant_client

        # -------- Create collection if missing --------
        self._recreate_collection_once()

    # ------------------------------------------------
    # Create collection ONLY if not exists
    # ------------------------------------------------
    def _recreate_collection_once(self):
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE
                )
            )

    # ------------------------------------------------
    # Add documents (BATCH EMBEDDING ✅)
    # ------------------------------------------------
    def add_documents(self, documents: List[Document]) -> None:
        if not documents:
            return

        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        # -------- Local embeddings --------
        vectors = self.model.encode(texts, convert_to_numpy=True)

        points = []
        for vector, text, metadata in zip(vectors, texts, metadatas):
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector.tolist(),
                    payload={
                        "content": text,
                        **{k: str(v) for k, v in metadata.items()}
                    }
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    # ------------------------------------------------
    # Similarity search
    # ------------------------------------------------
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        query_vector = self.model.encode([query], convert_to_numpy=True)[0]

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=k
        )

        documents = []
        for hit in results.points:
            payload = hit.payload or {}
            documents.append(
                Document(
                    page_content=payload.get("content", ""),
                    metadata={key: value for key, value in payload.items() if key != "content"}
                )
            )

        return documents

    # ------------------------------------------------
    # Optional: Clear collection manually
    # ------------------------------------------------
    def clear_collection(self):
        self.client.delete_collection(self.collection_name)

