import streamlit as st
from datetime import datetime
from dotenv import load_dotenv

from loaders import FileLoader
from textprocessing import TextProcessor
from vectorstore import VectorStore
from llm import LLMManager

load_dotenv(override=True)

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Talk To My Docs",
    layout="wide"
)

st.title("📄 Talk to Your Docs")

# ---------------- Initialize Modules ----------------
file_loader = FileLoader()
text_processor = TextProcessor()
llm_manager = LLMManager()

@st.cache_resource(show_spinner=False)
def get_vectorstore():
    return VectorStore()

vectorstore = get_vectorstore()

# ---------------- File Upload ----------------
st.sidebar.header("Upload Documents")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF, DOCX, CSV, XLSX",
    type=["pdf", "docx", "csv", "xlsx"],
    accept_multiple_files=True
)

if uploaded_files:
    with st.spinner("Indexing documents..."):
        for file in uploaded_files:
            text = file_loader.load(file)

            if text.strip():
                docs = text_processor.process(text, file.name)

                upload_time = datetime.now().isoformat()
                for doc in docs:
                    doc.metadata["upload_date"] = upload_time

                vectorstore.add_documents(docs)

    st.sidebar.success("✅ Documents indexed successfully")

# ---------------- Chat Section ----------------
st.header("💬 Chat with your documents")

query = st.text_input("Ask a question from the uploaded documents:")

if query and query.strip():

    # -------- Query Expansion --------
    queries = llm_manager.expand_query(query)

    st.subheader("🔍 Queries Used")
    for i, q in enumerate(queries, 1):
        st.write(f"{i}. {q}")

    # -------- Retrieval --------
    retrieved_docs = []
    for q in queries:
        retrieved_docs.extend(
            vectorstore.similarity_search(q, k=6)
        )

    if not retrieved_docs:
        st.warning("No relevant documents found.")
        st.stop()

    # Remove duplicate documents
    unique_docs = {
        doc.page_content: doc for doc in retrieved_docs
    }.values()

    # -------- Rerank --------
    top_docs = vectorstore.rerank(
        documents=list(unique_docs),
        query=query,
        top_n=5
    )

    # -------- Answer --------
    answer = llm_manager.generate_answer(
        query=query,
        documents=top_docs
    )

    st.subheader("Answer")
    st.write(answer)

    # -------- Sources --------
    st.subheader("Sources")
    from datetime import datetime

for doc in top_docs:
    upload_date_str = doc.metadata.get("upload_date", "")

    try:
        upload_date = datetime.fromisoformat(upload_date_str)

        formatted_date = upload_date.strftime("%d %b %Y, %I:%M %p")
    except Exception:
        formatted_date = upload_date_str  # fallback if parsing fails

    st.markdown(
        f"- **File:** {doc.metadata.get('filename', 'Unknown')} | "
        f"**Upload Date:** {formatted_date}"
    )
else:
    st.info("Enter a question to chat with your documents.")
