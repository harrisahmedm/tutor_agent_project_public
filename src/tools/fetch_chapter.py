# src/tools/fetch_chapter.py
"""
fetch_chapter.py

Accepts flexible grade and chapter_id inputs.
Primary accepted forms:
  - grade: "7", "grade7", 7
  - chapter_id: "2", "ch_2", "ch2", "chapter2", "ch_02", 2

Behavior:
  - Normalize inputs to filenames like 'ch_2.pdf' and grade folder like 'grade7'
  - Locate file under data/bookchapters/{grade}/
  - Use load_pdf() to extract text in each page
  - Build page_map mapping chapter-local page -> printed book page (if detected)
  - Build printed_to_chapter reverse mapping
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
import re

from src.tools.load_pdf import load_pdf

def _normalize_grade(grade: Any) -> str:
    """
    Normalize grade input to folder name like 'grade7'.

    Accepts:
      - "7", 7 -> "grade7"
      - "grade7", "Grade 7" -> "grade7"
    """
    if isinstance(grade, int):
        return f"grade{int(grade)}"
    s = str(grade).strip().lower()
    # If it's already like 'grade7' or 'grade 7'
    m = re.search(r'grade\s*0*?(\d+)', s)
    if m:
        return f"grade{int(m.group(1))}"
    # If it's just digits
    m2 = re.search(r'\b(\d+)\b', s)
    if m2:
        return f"grade{int(m2.group(1))}"
    # fallback: return as-is (safe)
    return s.replace(" ", "")


def _normalize_chapter_filename(chapter_id: Any) -> str:
    """
    Normalize chapter identifier to file name 'ch_N.pdf'.

    Accepts:
      - integer 2 -> "ch_2.pdf"
      - "2", "02" -> "ch_2.pdf"
      - "ch2", "ch_2", "chapter2", "chapter_2" -> "ch_2.pdf"
      - "ch_10.pdf" -> "ch_10.pdf" (preserves .pdf)
    """
    if isinstance(chapter_id, int):
        n = int(chapter_id)
        return f"ch_{n}.pdf"
    s = str(chapter_id).strip().lower()
    # If user passed a full filename already
    if s.endswith(".pdf"):
        # Normalize prefix like 'ch7.pdf' -> 'ch_7.pdf'
        s_no_ext = s[:-4]
        m = re.search(r'(\d+)', s_no_ext)
        if m:
            return f"ch_{int(m.group(1))}.pdf"
        return s
    # extract first integer
    m = re.search(r'(\d+)', s)
    if m:
        return f"ch_{int(m.group(1))}.pdf"
    # fallback - append .pdf
    s = s.replace(" ", "_")
    if not s.endswith(".pdf"):
        s = s + ".pdf"
    return s


def _find_printed_page_number_in_text(page_text: str) -> Optional[int]:
    """
    Extract the printed book page number from the last non-empty line
    using the '.indd' anchor pattern observed in NCERT textbook PDFs.

    Pattern examples:
        "Chapter-2.indd   24Chapter-2.indd   24 4/12/2025 ..."
        "Chapter 3.indd   48Chapter 3.indd   48 10-07-2025 ..."

    Logic:
      1. Take the last non-empty line.
      2. Find the FIRST occurrence of ".indd" (case-insensitive).
      3. Extract the substring immediately after it.
      4. Split on whitespace → first token (e.g., "24Chapter", "48Chapter").
      5. Remove any trailing alphabetic characters from that token.
      6. Convert the numeric prefix to int → return it.
      7. If anything fails, return None.
    """

    if not page_text:
        return None

    # Get last non-empty line
    lines = [ln.strip() for ln in page_text.splitlines() if ln.strip()]
    if not lines:
        return None

    last_line = lines[-1]
    lower = last_line.lower()
    anchor = ".indd"

    first_idx = lower.find(anchor)
    if first_idx == -1:
        return None  # cannot extract without anchor

    # Substring immediately after ".indd"
    tail = last_line[first_idx + len(anchor):].strip()
    if not tail:
        return None

    # First whitespace-delimited token (e.g. "24Chapter", "48Chapter")
    parts = tail.split()
    if not parts:
        return None

    token = parts[0]

    # Remove trailing alphabetic characters (e.g., "24Chapter" -> "24")
    m = re.match(r"(\d+)", token)
    if not m:
        return None

    try:
        return int(m.group(1))
    except Exception:
        return None



def fetch_chapter(grade: Any, chapter_id: Any) -> Dict[str, Any]:
    """
    Load a chapter and return a structured chapter_obj.

    Args:
        grade: e.g., "7", "grade7", 7
        chapter_id: e.g., "2", "ch_2", "chapter2"

    Returns:
        chapter_obj dict with keys:
          - grade (normalized, e.g., "grade7")
          - chapter_id (normalized, e.g., "ch_2")
          - path: absolute PDF path
          - pages: list of {"page_no": int, "text": str}
          - num_pages: int
          - full_text: str (concatenated)
          - printed_page_numbers: list indexed 0..N-1 -> printed page number or None
          - map_printed_to_chapter: dict printed_page number -> chapter_page_number (starts with 1)
          - map_chapter_to_printed: dict chapter_page_number (starts with 1) -> printed_page number
    Raises:
        FileNotFoundError if no file found.
    """

    norm_grade = _normalize_grade(grade)
    norm_chapter_file = _normalize_chapter_filename(chapter_id)

    # Resolve repo data path
    base_dir = Path(__file__).resolve().parents[2] / "data" / "bookchapters" # repo_root/data/bookchapters
    grade_dir = base_dir / norm_grade
    candidate_path = grade_dir / norm_chapter_file

    if candidate_path.exists():
        pdf_path = candidate_path.resolve()
    else:
        # Make a clean error message using relative path
        try:
            repo_root = Path(__file__).resolve().parents[2]
            rel = candidate_path.relative_to(repo_root)
        except ValueError:
            rel = candidate_path  # fallback (unlikely)
        raise FileNotFoundError(f"Chapter file not found at {rel}")
        
    # Extract text pages using load_pdf
    pages = load_pdf(str(pdf_path))
    num_pages = len(pages)
    full_text = "\n".join(pg.get("text", "") for pg in pages)

    # Build clean page numbering structures
    printed_page_numbers: List[Optional[int]] = [None] * num_pages
    map_printed_to_chapter: Dict[int, int] = {}
    map_chapter_to_printed: Dict[int, int] = {}

    for chapter_page_num, pg in enumerate(pages, start=1):
        txt = pg.get("text", "") or ""
        printed = _find_printed_page_number_in_text(txt)

        # store in 0-based list
        printed_page_numbers[chapter_page_num - 1] = printed

        if printed is not None:
            # printed page → chapter page (only first occurrence)
            if printed not in map_printed_to_chapter:
                map_printed_to_chapter[printed] = chapter_page_num

            # chapter page → printed page (we always want this)
            map_chapter_to_printed[chapter_page_num] = printed

    chapter_obj: Dict[str, Any] = {
        "grade": norm_grade,
        "chapter_id": norm_chapter_file.replace(".pdf", ""),
        "path": str(pdf_path),
        "pages": pages,
        "num_pages": num_pages,
        "full_text": full_text,
        "printed_page_numbers": printed_page_numbers,
        "map_printed_to_chapter": map_printed_to_chapter,
        "map_chapter_to_printed": map_chapter_to_printed,
    }

    return chapter_obj



# Quick local test when executed directly
if __name__ == "__main__":
    # simple smoke test (only runs when file executed)
    try:
        demo = fetch_chapter("7", "4")
        print("Loaded:", demo["path"])
        print("Normalized grade/chapter:", demo["grade"], demo["chapter_id"])
        print("Num pages:", demo["num_pages"])
        # page number mapping
        demo_map = [(i, demo["printed_page_numbers"][i]) for i in range(0, min(8, demo["num_pages"] + 1))]
        print("Chapter page number index (starts at 0)-> printed page number mapping:", demo_map)
        print("Printed page numebr -> chapter page number (first 5):", list(demo["map_printed_to_chapter"].items())[:5])
    except Exception as e:
        print("fetch_chapter test failed:", e)

