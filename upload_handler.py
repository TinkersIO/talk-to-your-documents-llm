from datetime import datetime
from langchain_classic.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def process_uploaded_files(upload_service, vectorstore, uploaded_files, processed_set, chunk_size, chunk_overlap):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    new_files = [f for f in uploaded_files if f.name not in processed_set]

    if not new_files:
        return

    uploaded_docs = upload_service.upload_files(new_files)
    chunks_to_add = []

    for doc in uploaded_docs:
        doc_chunks = splitter.split_text(doc["content"])

        for i, chunk in enumerate(doc_chunks):
            chunks_to_add.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "filename": doc["filename"],
                        "chunk_index": i,
                        "upload_date": datetime.now().isoformat()
                    }
                )
            )

        processed_set.add(doc["filename"])

    vectorstore.add_documents(chunks_to_add)
