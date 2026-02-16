import os
import asyncio
import nest_asyncio
from dotenv import load_dotenv
import streamlit as st
from vectorstore import VectorStore
from llm import get_agent, get_llm
from textprocessing import TextProcessor
from uploads import upload_files


# ---------------- Setup ----------------
load_dotenv(override=True)
nest_asyncio.apply()
st.set_page_config(page_title="📄 Talk To My Docs", layout="wide")
st.title("📄 Talk to Your Documents")

# ---------------- LLM + MCP Agent ----------------
if "agent" not in st.session_state:
    st.session_state.agent = asyncio.run(get_agent())

agent = st.session_state.agent


# ---------------- LLM & Vectorstore ----------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = VectorStore()
vectorstore = st.session_state.vectorstore


# ---------------- Session State ----------------
if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = []

# ---------------- Sidebar Upload ----------------
st.sidebar.header("📤 Upload Documents")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF / TXT / CSV",
    type=["pdf", "txt", "csv"],
    accept_multiple_files=True
)

if uploaded_files:
    st.session_state.uploaded_docs.clear()
    with st.spinner("Processing documents..."):
        
        uploaded_docs = upload_files(
            uploaded_files,
            upload_dir="./uploads"
        )
        st.session_state.uploaded_docs.extend(uploaded_docs)

     
        processor = TextProcessor(chunk_size=500, chunk_overlap=50)
        chunks = []
        for doc in uploaded_docs:
            chunks.extend(processor.process(doc["content"], doc["filename"]))
        vectorstore.add_documents(chunks)

        st.sidebar.success("✅ Documents uploaded, saved & indexed")

# ---------------- Chat Helpers ----------------
def get_top_chunks(query):
    
    processor = TextProcessor(chunk_size=500, chunk_overlap=50)

   
    if "summary" in query.lower() and st.session_state.uploaded_docs:
        doc = st.session_state.uploaded_docs[0]
        return [type("Doc", (), {"page_content": doc["content"], "metadata": {"filename": doc["filename"]}})]

   
    matched_doc = next((d for d in st.session_state.uploaded_docs if d["filename"] in query), None)
    if matched_doc:
        chunks = processor.process(matched_doc["content"], matched_doc["filename"])
        return [c for c in chunks if c.page_content.strip()][:5]

 
    return vectorstore.similarity_search(query, k=5)

# ---------------- Chat Section ----------------
st.header("💬 Ask questions from your documents")
query = st.text_input("Ask a question")

if query:
    if not st.session_state.uploaded_docs:
        st.warning("Please upload documents first.")
    else:
        with st.spinner("Thinking..."):
            top_chunks = get_top_chunks(query)
            if not top_chunks:
                answer = "I don't know"
            else:
                context = "\n\n".join(
                [doc.page_content for doc in top_chunks]
            )
            prompt = f"""
Answer the question ONLY using the context below.
If it has some content summarize it.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{query}
"""

                
            llm = get_llm()
            

            response = llm.invoke(prompt)
            answer = response.content

        # Display Answer
        st.subheader("Answer")
        st.markdown(answer)

        # Display Sources
        st.subheader("Sources")
        for doc in top_chunks:
            filename = doc.metadata.get("filename", "unknown") if hasattr(doc, "metadata") and doc.metadata else "unknown"
            st.markdown(f"- **{filename}**")
