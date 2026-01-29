from PyPDF2 import PdfReader
import pandas as pd
from docx import Document


class FileLoader:
    def load(self, file):
        """
        Detect file type and load text
        """
        if file.type == "application/pdf":
            return self._load_pdf(file)

        elif file.type.endswith("wordprocessingml.document"):
            return self._load_docx(file)

        elif file.type == "text/csv":
            return self._load_csv(file)

        elif file.type.endswith("spreadsheetml.sheet"):
            return self._load_xlsx(file)

        return ""

    def _load_pdf(self, file):
        reader = PdfReader(file)
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    def _load_docx(self, file):
        doc = Document(file)
        return "\n".join(p.text for p in doc.paragraphs)

    def _load_csv(self, file):
        df = pd.read_csv(file)
        return df.to_string(index=False)

    def _load_xlsx(self, file):
        df = pd.read_excel(file)
        return df.to_string(index=False)


           