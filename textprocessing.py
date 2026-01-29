from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class TextProcessor:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def process(self, text: str, filename: str):
        """
        Convert raw text into chunked LangChain documents
        """
        chunks = self.splitter.split_text(text)

        documents = []
        for chunk in chunks:
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={"filename": filename}
                )
            )

        return documents
