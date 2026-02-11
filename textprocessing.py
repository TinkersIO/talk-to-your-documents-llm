from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class TextProcessor:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initializes the text processor with chunking parameters.
        """
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def process(self, text: str, filename: str) -> list[Document]:
        """
        Split raw text into chunks and wrap them as LangChain Documents.

        Args:
            text: The full text to split.
            filename: The source filename (used in document metadata).

        Returns:
            List of chunked Document objects.
        """
        return [
            Document(page_content=chunk, metadata={"filename": filename})
            for chunk in self.splitter.split_text(text)
        ]
