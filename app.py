import streamlit as st
from datetime import datetime
from dotenv import load_dotenv
from mcp_client import MCPClient 
from loaders import FileLoader
from textprocessing import TextProcessor
from vectorstore import VectorStore
from db_service import save_document
# ---------------- Load ENV ----------------
load_dotenv(override=True)

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Talk To My Docs",
    layout="wide"
)

st.title("📄 Talk to Your Docs")

# ---------------- Constants ----------------
MAX_TOP_DOCS = 3

# ---------------- Initialize Components ----------------
file_loader = FileLoader()
text_processor = TextProcessor()
mcp_client = MCPClient()

@st.cache_resource
def get_vectorstore():
    return VectorStore()
vectorstore = get_vectorstore()



# ---------------- File Upload ----------------
st.sidebar.header("📤 Upload Documents")

uploaded_files = st.file_uploader(
    "Upload PDF / DOCX / CSV / XLSX",
    type=["pdf", "docx", "csv", "xlsx"],
    accept_multiple_files=True
)

if uploaded_files:
    with st.spinner("Uploading and indexing documents..."):
        for uploaded_file in uploaded_files:

            # 1️⃣ Upload to MCP (filesystem simulation)
            file_id = mcp_client.upload_file(uploaded_file)
            if not file_id:
                st.warning(f"Failed to upload {uploaded_file.name}")
                continue

            # 2️⃣ Load file content
            text = file_loader.load(uploaded_file)
            if not text.strip():
                continue

            # 3️⃣ Process text into document chunks
            docs = text_processor.process(text, uploaded_file.name)

            upload_time = datetime.now().isoformat()
            for doc in docs:
                # Add metadata
                doc.metadata["upload_date"] = upload_time
                doc.metadata["filename"] = uploaded_file.name
                doc.metadata["file_id"] = file_id

                # 4️⃣ Save to MCP SQLite
                save_document(doc.page_content, doc.metadata)

            # 5️⃣ Add documents to vectorstore
            vectorstore.add_documents(docs)

    st.sidebar.success("✅ Files uploaded, indexed, and saved to MCP successfully!")

# ---------------- Chat Section ----------------
st.header("💬 Chat with your documents")

query = st.text_input("Ask a question from the uploaded documents:")

if query and query.strip():
    with st.spinner("Generating answer..."):

        # 1️⃣ Retrieval: similarity search on vectorstore
        top_docs = vectorstore.similarity_search(query=query, k=MAX_TOP_DOCS)

        if not top_docs:
            st.warning("No relevant documents found.")
            st.stop()

        best_doc = top_docs[1]

        # 3️⃣ Generate ONE clean answer
        answer = mcp_client.generate_answer(query, [best_doc])

    # ---------------- Display Answer ----------------
    st.subheader("Answer")
    st.markdown(f"💡 **{answer}**")

    # -------- Sources --------
    st.subheader(" Sources")
    for doc in [best_doc]:
        upload_date_str = doc.metadata.get("upload_date", "")
        filename = doc.metadata.get("filename", "Unknown")

        try:
            upload_date = datetime.fromisoformat(upload_date_str)
            formatted_date = upload_date.strftime("%d %b %Y, %I:%M %p")
        except Exception:
            formatted_date = upload_date_str

        st.markdown(
            f"- **File:** {filename} | **Uploaded:** {formatted_date}"
        )

else:
    st.info(" Upload documents and ask a question to get started.")
