class ContextBuilder:
    def build(self, relevant_docs):
        context_text = ""

        for doc in relevant_docs:
            meta = doc.metadata
            context_text += f"Filename: {meta.get('filename', 'unknown')}\n"
            context_text += f"Content:\n{doc.page_content}\n\n"

        return context_text