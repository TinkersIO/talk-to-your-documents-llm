import os
import asyncio
import sqlite3
import json
from datetime import datetime
from PyPDF2 import PdfReader
from mcp_client import upload_file_via_mcp, save_metadata_via_mcp
from langchain_core.tools import Tool

# ---------------- Helpers ----------------
def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from a PDF file."""
    reader = PdfReader(file_path)
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)

def init_db(db_path: str = "database.db"):
    """Initialize SQLite documents table if it doesn't exist."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
 

    cur.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT,
            filename TEXT,
            path TEXT,
            upload_date TEXT,
            metadata TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_to_db(filename: str, content: str, path: str, metadata: dict, db_path: str = "database.db"):
    """Save document metadata and content to SQLite."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO documents (filename, content, path, upload_date, metadata)
        VALUES (?, ?, ?, ?, ?)
    """, (filename, content, path, datetime.now().isoformat(), json.dumps(metadata)))
    conn.commit()
    conn.close()

# ---------------- Core Upload ----------------
async def _upload_single_file(file, upload_dir: str, write_tool: Tool, sql_tool: Tool, db_path="database.db") -> dict:
    """
    Upload a single file:
    - Save locally
    - Extract text
    - Upload to MCP
    - Save metadata to MCP and SQLite
    """
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.name)

    # Save locally
    with open(file_path, "wb") as f:
        f.write(file.getvalue())

    # Extract content
    if file.type == "application/pdf":
        content = extract_text_from_pdf(file_path)
    else:
        content = file.getvalue().decode("utf-8", errors="ignore")

    metadata = {"uploaded_by": "user"}

    # Upload to MCP
    await upload_file_via_mcp(write_tool, file.name, content.encode("utf-8"))
    await save_metadata_via_mcp(sql_tool, file.name, "User uploaded")

    # Save to SQLite
    save_to_db(file.name, content, file_path, metadata, db_path=db_path)

    return {"filename": file.name, "path": file_path, "content": content}

# ---------------- Public Function ----------------
def upload_files(uploaded_files, write_tool: Tool, sql_tool: Tool, upload_dir: str = "./uploads", db_path="database.db") -> list:
    """
    Upload multiple files (PDF/TXT/CSV):
    - Saves files locally
    - Saves metadata to SQLite and MCP
    Returns list of dicts with filename, path, content
    """
    import nest_asyncio
    nest_asyncio.apply()
    init_db(db_path)

    async def _process_files():
        docs = []
        for file in uploaded_files:
            doc_info = await _upload_single_file(file, upload_dir, write_tool, sql_tool, db_path)
            docs.append(doc_info)
        return docs

    return asyncio.get_event_loop().run_until_complete(_process_files())
