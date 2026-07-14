from pathlib import Path
from pypdf import PdfReader

def load_pdf(file_path):
    reader = PdfReader(file_path)
    doc_id = Path(file_path).name

    pages = []

    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()

        if text:
            pages.append({
                "doc_id": doc_id,
                "page": page_num + 1,   # Human-friendly numbering
                "text": text
            })

    return pages