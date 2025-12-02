from pypdf import PdfReader
from typing import List, Dict

def load_pdf(path: str) -> List[dict]:
    """
    Read PDF at `path`. Return list of pages:
    [{ "page_no": 1, "text": "..." }, ...]
    """

    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append({"page_no": i, "text": text})
    return pages

# quick test
if __name__ == "__main__":
    from pathlib import Path

    # this file: tutor_agent_project/src/tools/load_pdf.py
    THIS_DIR = Path(__file__).resolve().parent          # → src/tools
    PROJECT_ROOT = THIS_DIR.parent.parent               # → tutor_agent_project

    DATA_DIR = PROJECT_ROOT / "data" / "bookchapters"

    pdf_path = DATA_DIR / "grade8" / "ch_7.pdf"

    pages = load_pdf(str(pdf_path))
    print(f"Loaded {len(pages)} pages; first 200 chars:\n", pages[0]["text"][:200])
    print(f"Loaded {len(pages)} pages; last 200 chars in page 0:\n", pages[0]["text"][200:])
    
    lines = [ln.strip() for ln in pages[0]["text"].splitlines() if ln.strip()]
    
    last_line = lines[-1].strip()
    print("Last non-empty line in page 0:", repr(last_line))

