import os
import pdfplumber
import docx
import fitz  # PyMuPDF
import pandas as pd


class FileLoader:
    def load(self, file, filename=None):
        """
        file: Streamlit UploadedFile OR file-like object (opened in binary mode)
        filename: required when reading file from disk
        """

        # ---------- CASE 1: Streamlit UploadedFile ----------
        if hasattr(file, "type"):
            if file.type == "application/pdf":
                return self._load_pdf(file)
            elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                return self._load_docx(file)
            elif file.type == "text/csv":
                return self._load_csv(file)
            elif file.type in (
                "application/vnd.ms-excel",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ):
                return self._load_xlsx(file)
            else:
                return ""

        # ---------- CASE 2: File from disk ----------
        if filename:
            ext = os.path.splitext(filename)[1].lower()

            if ext == ".pdf":
                return self._load_pdf(file)
            elif ext == ".docx":
                return self._load_docx(file)
            elif ext == ".csv":
                return self._load_csv(file)
            elif ext == ".xlsx":
                return self._load_xlsx(file)

        return ""

    # ================= PDF =================
    def _load_pdf(self, file):
        """
        Robust PDF loader:
        1) Try pdfplumber
        2) Fallback to PyMuPDF (fitz)
        """

        # -------- TRY pdfplumber --------
        try:
            file.seek(0)
            text = ""
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ""
            if text.strip():
                return text
        except Exception:
            pass  # fallback

        # -------- FALLBACK: PyMuPDF --------
        try:
            file.seek(0)
            text = ""
            doc = fitz.open(stream=file.read(), filetype="pdf")
            for page in doc:
                text += page.get_text()
            return text
        except Exception as e:
            print("❌ PDF read failed:", e)
            return ""

    # ================= DOCX =================
    def _load_docx(self, file):
        try:
            doc = docx.Document(file)
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception:
            return ""

    # ================= CSV =================
    def _load_csv(self, file):
        try:
            df = pd.read_csv(file)
            return df.to_string(index=False)
        except Exception:
            return ""

    # ================= XLSX =================
    def _load_xlsx(self, file):
        try:
            df = pd.read_excel(file)
            return df.to_string(index=False)
        except Exception:
            return ""
