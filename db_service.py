import sqlite3

DB_PATH = "documents.db"

def save_document(content, metadata):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO documents (content, filename, upload_date, file_id)
        VALUES (?, ?, ?, ?)
    """, (
        content,
        metadata.get("filename"),
        metadata.get("upload_date"),
        metadata.get("file_id")
    ))
    conn.commit()
    conn.close()

