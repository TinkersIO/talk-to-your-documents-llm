import sqlite3
from datetime import datetime
from langchain_classic.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def rebuild_vectorstore(upload_service, vectorstore, processed_set, chunk_size, chunk_overlap):
    conn = sqlite3.connect(upload_service.db_path)
    cur = conn.cursor()
    cur.execute("SELECT filename, filepath FROM documents")
    docs = cur.fetchall()
    conn.close()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks_to_add = []

    for filename, filepath in docs:
        if filename in processed_set:
            continue

        content = upload_service._extract_text(filepath)
        doc_chunks = splitter.split_text(content)

        for i, chunk in enumerate(doc_chunks):
            chunks_to_add.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "filename": filename,
                        "chunk_index": i,
                        "upload_date": datetime.now().isoformat()
                    }
                )
            )

        processed_set.add(filename)

    if chunks_to_add:
        vectorstore.add_documents(chunks_to_add)
