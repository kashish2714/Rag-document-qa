from pypdf import PdfReader

class FileParser:

    def parse(self, file_path):
        if file_path.endswith(".txt"):
            return self.parse_txt(file_path)
        elif file_path.endswith(".pdf"):
            return self.parse_pdf(file_path)
        else:
            raise Exception("Unsupported file type")

    def parse_txt(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def parse_pdf(self, file_path):
        reader = PdfReader(file_path)
        text = ""

        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text()

        return text

    def chunk_text(self, text, chunk_size=300, overlap=50):
        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start = end - overlap

        return chunks
