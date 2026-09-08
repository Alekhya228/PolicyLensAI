import pdfplumber
import os

class PolicyParser:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        
    def extract_text(self):
        """Extracts plain text from the PDF."""
        full_text = []
        with pdfplumber.open(self.pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    full_text.append({"page": i + 1, "text": text})
        return full_text

    def extract_tables(self):
        """Extracts tables from the PDF and converts them to markdown-like strings."""
        tables_data = []
        with pdfplumber.open(self.pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                for table in tables:
                    # Convert list of lists to string
                    table_str = "\n".join([" | ".join([str(cell) if cell else "" for cell in row]) for row in table])
                    tables_data.append({"page": i + 1, "table": table_str})
        return tables_data
        
    def parse_all(self):
        """Returns structured data containing both text and tables."""
        filename = os.path.basename(self.pdf_path)
        
        # Simple extraction for mock dates
        import re
        year_match = re.search(r'(20\d{2})', filename)
        last_updated = year_match.group(1) if year_match else "Unknown"
        
        return {
            "source": filename,
            "last_updated": last_updated,
            "text": self.extract_text(),
            "tables": self.extract_tables()
        }

if __name__ == "__main__":
    # Test the parser
    sample_pdf = "data/raw_pdfs/grading_policy.pdf"
    if os.path.exists(sample_pdf):
        parser = PolicyParser(sample_pdf)
        data = parser.parse_all()
        print(f"Extracted {len(data['text'])} pages of text.")
        print(f"Extracted {len(data['tables'])} tables.")
        if data['tables']:
            print("Sample table:\n", data['tables'][0]['table'])
