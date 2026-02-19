class Retriever:
    def __init__(self, vectorstore, top_k: int = 5):
        self.vectorstore = vectorstore
        self.top_k = top_k

    def retrieve(self, queries):
        retrieved_docs = []

        for q in queries:
            docs = self.vectorstore.similarity_search(q, k=self.top_k)
            retrieved_docs.extend(docs)

        unique_docs = {doc.page_content: doc for doc in retrieved_docs}
        return list(unique_docs.values())
