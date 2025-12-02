# load_context.py
import re
import base64
from io import BytesIO
from typing import List, Tuple, Dict, Any, Optional, Any as TAny

import fitz  # PyMuPDF
from PIL import Image

from src.tools.fetch_chapter import fetch_chapter

import os  # For local testing only

import logging
logger = logging.getLogger("tutor_agent")

# For async ToolContext handling
import inspect
import asyncio
from google.genai import types as genai_types
from src.tools.artifacts_helper import save_artifact_robust, load_artifact_robust

# Chapter header lookup tables (MVP, editable)
CHAPTER_HEADER_LOOKUP_G7 = {
    1: "Large Numbers Around Us",
    2: "Arithmetic Expressions",
    3: "A Peek Beyond the Point",
    4: "Expressions using Letter-Numbers",
    5: "Parallel and Intersecting Lines",
    6: "Number Play",
    7: "A Tale of Three Intersecting Lines",
    8: "Working with Fractions",
}
CHAPTER_HEADER_LOOKUP_G8 = {
    1: "A Square and A Cube",
    2: "Power Play",
    3: "A Story of Numbers",
    4: "Quadrilaterals",
    5: "Number Play",
    6: "We Distribute, Yet Things Multiply",
    7: "Proportional Reasoning-1",
}

MAX_IMAGE_BYTES = 100 * 1024

import uuid


def clean_page_text(text: str,
                    chapter_heading: Optional[str] = None,
                    page_no_in_chapter: Optional[int] = None,
                    printed_page_number: Optional[int] = None) -> str:
    """
    Clean a single page's extracted text for parsing.
    """
    if not text:
        return ""

    lines = text.splitlines()
    cleaned_lines: List[str] = []

    for ln in lines:
        stripped = ln.strip()
        if not stripped:
            cleaned_lines.append("")  # preserve blanks for later collapse
            continue
        low = stripped.lower()

        # Remove header line starting "Ganita Prakash | Grade"
        if low.startswith("ganita prakash | grade"):
            continue

        # Remove chapter heading on pages other than the first page of a chapter
        if chapter_heading and page_no_in_chapter is not None and page_no_in_chapter != 1:
            if stripped.lower().startswith(chapter_heading.lower()):
                continue

        # Remove lines containing ".indd"
        if ".indd" in low:
            continue

        # Remove exact printed page number if provided (standalone token)
        if printed_page_number is not None:
            if re.fullmatch(rf"\s*{re.escape(str(printed_page_number))}\s*", stripped):
                continue

        cleaned_lines.append(ln)

    cleaned = "\n".join(cleaned_lines)

    # Remove "Note to the Teacher:" paragraph (case-insensitive)
    cleaned = re.sub(r"(?is)Note\s+to\s+the\s+Teacher:.*?(?:\n\s*\n|\Z)", "", cleaned)

    # Collapse multiple blank lines to a single blank line
    cleaned = re.sub(r"\n\s*\n+", "\n\n", cleaned).strip()
    return cleaned


def _find_fig_positions(clean_text: str) -> List[int]:
    """
    Return start indices of 'Figure it Out' occurrences in the cleaned text.
    """
    return [m.start() for m in re.finditer(r"Figure it Out", clean_text)]


def _parse_numbered_items(text_block: str) -> Dict[int, Tuple[str, int]]:
    """
    Parse numbered items from a text block and return mapping: item_number -> (item_text, start_index)
    """
    pat = re.compile(
        r"(?:^|\n)\s*(\d+)\s*[\.\)]\s+(.+?)(?=(?:\n\s*\d+\s*[\.\)]|\Z))",
        flags=re.DOTALL,
    )

    items: Dict[int, Tuple[str, int]] = {}
    for m in pat.finditer(text_block):
        num = int(m.group(1))
        txt = m.group(2).strip()
        start_idx = m.start()
        items[num] = (txt, start_idx)
    return items


def _collect_blocks_for_item(doc: fitz.Document,
                             start_page: int,
                             first_line: str,
                             char_cap: int,
                             item_no: Optional[int] = None) -> Tuple[str, List[List[float]], bool]:
    """
    Collect consecutive PDF text blocks starting from the block containing `first_line`.
    """
    assembled: List[str] = []
    bboxes: List[List[float]] = []
    truncated = False

    def _normalize_for_search(s: str) -> str:
        s2 = re.sub(r'[\u00A0\u202F\u2007\u200B]', ' ', s)
        s2 = s2.replace('\uFF1A', ':')
        s2 = re.sub(r'\s+', ' ', s2).strip()
        return s2.strip(" \t\n\r:;.-—")

    def process_page(pg_no: int) -> Optional[bool]:
        nonlocal assembled, bboxes, truncated

        page = doc.load_page(pg_no - 1)
        blocks = page.get_text("dict").get("blocks", [])

        text_blocks: List[str] = []
        for block in blocks:
            if block.get("type", 0) != 0:
                text_blocks.append("")
                continue
            bt = ""
            for line in block.get("lines", []):
                bt += "".join(span.get("text", "") for span in line.get("spans", [])) + "\n"
            text_blocks.append(bt)

        start_block_index = None
        start_block_text = ""

        # Attempt 1: exact substring match
        for idx, bt in enumerate(text_blocks):
            if not bt:
                continue
            if first_line and first_line in bt:
                start_block_index = idx
                start_block_text = bt
                break

        # Attempt 2: normalized substring match
        if start_block_index is None and first_line:
            first_norm = _normalize_for_search(first_line)
            if first_norm:
                for idx, bt in enumerate(text_blocks):
                    if not bt:
                        continue
                    if first_norm in _normalize_for_search(bt):
                        start_block_index = idx
                        start_block_text = bt
                        break

        # Attempt 3: numbered-anchor search if item_no is not None
        if start_block_index is None and item_no is not None:
            anchor_pat = re.compile(rf"(?m)^\s*{re.escape(str(item_no))}\s*[\.\)]\s+")
            for idx, bt in enumerate(text_blocks):
                if not bt:
                    continue
                if anchor_pat.search(bt):
                    start_block_index = idx
                    start_block_text = bt
                    break

        if start_block_index is None:
            return None

        start_has_mathtalk = bool(re.search(r"Math Talk", start_block_text))
        start_has_trythis = bool(re.search(r"Try This", start_block_text))

        for idx in range(start_block_index, len(text_blocks)):
            bt = text_blocks[idx]
            if not bt:
                continue

            if re.search(r"^\s*\d+\s*[\.\)]", bt.strip(), flags=re.MULTILINE):
                return True

            if re.search(r"(?i)\.indd", bt):
                return True

            if re.search(r"(?i)Note\s+to\s+the\s+Teacher:", bt):
                return True

            if re.search(r"Activity\s*\d+\s*[:\-]", bt):
                return True

            if re.search(r"Math Talk", bt):
                if not (idx == start_block_index and start_has_mathtalk):
                    return True

            if re.search(r"Try This", bt):
                if not (idx == start_block_index and start_has_trythis):
                    return True

            current_len = sum(len(x) for x in assembled)
            if current_len + len(bt) > char_cap:
                allowed = char_cap - current_len
                sub = bt[:allowed]
                m = re.search(r"(?s)(.*[\.?!])[^\.?!]*\Z", sub)
                if m:
                    assembled.append(m.group(1).strip())
                else:
                    assembled.append(sub.strip())
                bb = blocks[idx].get("bbox", (0, 0, 0, 0))
                bboxes.append([pg_no, float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])])
                truncated = True
                return True

            assembled.append(bt.strip())
            bb = blocks[idx].get("bbox", (0.0, 0.0, 0.0, 0.0))
            bboxes.append([pg_no, float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])])

        return True

    res = process_page(start_page)
    if res is None:
        return "", [], False

    if bboxes:
        last_page = bboxes[-1][0]
        last_y1 = bboxes[-1][3]
        page_h = doc.load_page(last_page - 1).rect.height

        if last_y1 > page_h * 0.7 and last_page < doc.page_count:
            next_page_no = last_page + 1
            page = doc.load_page(next_page_no - 1)
            for block in page.get_text("dict").get("blocks", []):
                if block.get("type", 0) != 0:
                    continue
                bt = ""
                for line in block.get("lines", []):
                    bt += "".join(span.get("text", "") for span in line.get("spans", [])) + "\n"

                if re.search(r"^\s*\d+\s*[\.\)]", bt.strip(), flags=re.MULTILINE):
                    break
                if re.search(r"(?i)\.indd", bt):
                    break
                if re.search(r"(?i)Note\s+to\s+the\s+Teacher:", bt):
                    break
                if re.search(r"Activity\s*\d+\s*[:\-]", bt):
                    break
                if re.search(r"Math Talk", bt):
                    break
                if re.search(r"Try This", bt):
                    break

                current_len = sum(len(x) for x in assembled)
                if current_len + len(bt) > char_cap:
                    allowed = char_cap - current_len
                    sub = bt[:allowed]
                    m = re.search(r"(?s)(.*[\.?!])[^\.?!]*\Z", sub)
                    if m:
                        assembled.append(m.group(1).strip())
                    else:
                        assembled.append(sub.strip())
                    bb = block.get("bbox", (0, 0, 0, 0))
                    bboxes.append([next_page_no, float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])])
                    truncated = True
                    break

                assembled.append(bt.strip())
                bb = block.get("bbox", (0, 0, 0, 0))
                bboxes.append([next_page_no, float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])])

    text = "\n\n".join(assembled).strip()
    text_norm = re.sub(r'[\u00A0\u202F\u2007\u200B]', ' ', text).replace('\uFF1A', ':')

    mact = re.search(r"Activity\s*\d+\s*(?:[:\-])?", text_norm)
    if mact:
        text_norm = text_norm[:mact.start()].rstrip()
        truncated = True

    return text_norm, bboxes, truncated


def _pil_encode_and_fit(pil_img: Image.Image, max_bytes: int) -> Tuple[bytes, str]:
    """
    Encode PIL image to bytes under max_bytes by downscaling and quality steps.
    """
    buf = BytesIO()
    prefer_png = pil_img.mode in ("RGBA", "LA")
    if not prefer_png:
        try:
            colors = pil_img.convert("RGB").getcolors(maxcolors=1000000)
            if colors and len(colors) <= 64:
                prefer_png = True
        except Exception:
            pass

    fmt = "PNG" if prefer_png else "JPEG"
    if fmt == "JPEG":
        pil_img.save(buf, format="JPEG", quality=85, optimize=True)
    else:
        pil_img.save(buf, format="PNG", optimize=True)
    data = buf.getvalue()
    if len(data) <= max_bytes:
        return data, fmt

    w, h = pil_img.size
    scale = 0.85
    min_w, min_h = 600, 800
    while len(data) > max_bytes and w > min_w and h > min_h:
        w = max(int(w * scale), min_w)
        h = max(int(h * scale), min_h)
        small = pil_img.resize((w, h), Image.LANCZOS)
        buf = BytesIO()
        if fmt == "JPEG":
            small.save(buf, format="JPEG", quality=85, optimize=True)
        else:
            small.save(buf, format="PNG", optimize=True)
        data = buf.getvalue()
        pil_img = small

    if len(data) <= max_bytes:
        return data, fmt

    if fmt == "JPEG":
        for q in (75, 65, 60):
            buf = BytesIO()
            pil_img.save(buf, format="JPEG", quality=q, optimize=True)
            data = buf.getvalue()
            if len(data) <= max_bytes:
                return data, "JPEG"

    if fmt == "PNG":
        try:
            pal = pil_img.convert("P", palette=Image.ADAPTIVE, colors=128)
            buf = BytesIO()
            pal.save(buf, format="PNG", optimize=True)
            data = buf.getvalue()
            if len(data) <= max_bytes:
                return data, "PNG"
            pal2 = pil_img.convert("P", palette=Image.ADAPTIVE, colors=64)
            buf = BytesIO()
            pal2.save(buf, format="PNG", optimize=True)
            data = buf.getvalue()
            if len(data) <= max_bytes:
                return data, "PNG"
        except Exception:
            pass

    return data, fmt


def render_pages_as_images(pdf_path: str, page_nos: List[int], max_bytes: int = MAX_IMAGE_BYTES) -> List[Dict[str, Any]]:
    """
    Render full pages as images and return a list (in the same order as page_nos) of dicts.
    """
    images_out: List[Dict[str, Any]] = []
    doc = fitz.open(pdf_path)
    for pg_no in page_nos:
        if not (1 <= pg_no <= doc.page_count):
            continue
        page = doc.load_page(pg_no - 1)
        mat = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=mat)
        png_bytes = pix.tobytes("png")
        pil = Image.open(BytesIO(png_bytes)).convert("RGBA")
        encoded, fmt = _pil_encode_and_fit(pil, max_bytes)
        mime = "image/png" if fmt == "PNG" else "image/jpeg"
        images_out.append({
            "chapter_page_no": pg_no,
            "bbox": [0.0, 0.0, page.rect.width, page.rect.height],
            "width": pil.width,
            "height": pil.height,
            "format": fmt,
            "mime": mime,
            "bytes_b64": base64.b64encode(encoded).decode("ascii"),
            "size_bytes": len(encoded),
        })
    doc.close()
    return images_out


async def load_context(grade: int,
                 chapter_id: int,
                 chapter_page_number: Optional[int] = None,
                 book_page_number: Optional[int] = None,
                 exercise_number: int = None,
                 tool_context: Optional[TAny] = None,
                 *,
                 char_cap: int = 900,
                 max_images_bytes: int = MAX_IMAGE_BYTES) -> Dict[str, Any]:
    """
    Load a textbook exercise and its related images.

    Args:
        grade (int):
            The textbook grade level (e.g., 7 or 8).

        chapter_id (int):
            The numeric chapter identifier inside that grade.

        chapter_page_number (int, optional):
            The page number within the chapter where the exercise appears.
            Supply this OR book_page_number.

        book_page_number (int, optional):
            The printed page number in the book.
            Supply this OR chapter_page_number.

        exercise_number (int):
            The exercise item number to extract (e.g., 1, 2, 3...).

        tool_context (ToolContext, optional):
            When provided by the runner, images are also saved as session-scoped
            artifacts. Each image dict in the output then includes:
                - "artifact_name": the artifact filename
                - "artifact_saved": boolean

        char_cap (int):
            Maximum number of text characters to extract for the problem stem.

        max_images_bytes (int):
            Maximum allowed bytes for each rendered page image. Images are resized
            and re-encoded to stay under this limit.

    Returns:
        dict:
            A dictionary describing the exercise. Important keys:

            - "problem_text":          str, cleaned text of the exercise
            - "exercise_number":       int
            - "chapter_page_no":       int or None
            - "book_page_no":          int or None
            - "images":                list of image dicts (each contains):
                {
                    "image_index": int (0-based index for lightweight UI calls),
                    "chapter_page_no": int,
                    "mime": str,
                    "size_bytes": int,
                    "bytes_b64": str,
                    "artifact_name": str or None,
                    "artifact_saved": bool
                }
            - "first_page_image_present":  bool
            - "second_page_image_present": bool
            - "text_bboxes":           list of bounding-box info
            - "truncated":             bool (true if char_cap forced truncation)
            - "multiple_figureitout_sections": bool


    Description:
    Load the problem text and associated images from local textbook files.

    Returns a dict containing "problem_text" and "images" among other metadata.
    Each image dict now includes an 'image_index' (0-based) to support lightweight UI render calls.
    If a ToolContext is provided and the artifact service is available, images are saved as session-scoped artifacts
    and each image dict is annotated with 'artifact_name' and 'artifact_saved' boolean.
    """
    import time
    t0 = time.time()
    logger.info("load_context start")
    logger.info("DEBUG: load_context called; tool_context=%s", type(tool_context))

    if chapter_page_number is None and book_page_number is None:
        raise ValueError("You must supply at least one of chapter_page_number or book_page_number.")

    chapter = fetch_chapter(grade, chapter_id)
    pdf_path = chapter["path"]
    pages_list = chapter["pages"]
    pages = {pg["page_no"]: pg for pg in pages_list}

    if chapter_page_number is None and book_page_number is not None:
        mapped = chapter.get("map_printed_to_chapter", {}).get(book_page_number)
        if mapped is None:
            raise ValueError(f"Printed page {book_page_number} not found.")
        chapter_page_number = mapped

    if book_page_number is None:
        book_page_number = chapter.get("map_chapter_to_printed", {}).get(chapter_page_number)

    try:
        gnum = int(str(grade).strip())
    except Exception:
        gnum = 7
    header_lookup = CHAPTER_HEADER_LOOKUP_G7 if gnum == 7 else CHAPTER_HEADER_LOOKUP_G8
    chapter_heading = header_lookup.get(chapter_id) if isinstance(chapter_id, int) else None

    candidate_page_nos = [chapter_page_number - 1, chapter_page_number, chapter_page_number + 1]
    cleaned_pages: Dict[int, str] = {}
    for pno in candidate_page_nos:
        if pno >= 1 and pno in pages:
            printed = chapter.get("map_chapter_to_printed", {}).get(pno)
            cleaned_pages[pno] = clean_page_text(
                pages[pno].get("text", "") or "",
                chapter_heading,
                pno,
                printed
            )

    occurrences: List[Dict[str, Any]] = []
    for pno, cleaned in cleaned_pages.items():
        fig_positions = _find_fig_positions(cleaned)
        for spos in fig_positions:
            next_text = cleaned_pages.get(pno + 1, "")
            combined = cleaned[spos:] + "\n\n" + next_text
            combined = re.split(
                r"(?:\n\s*Activity\s*\d+\s*[:\-])|(?:\n\s*Math Talk\b)|(?:\n\s*Try This\b)|(?:\.indd)",
                combined
            )[0]
            items = _parse_numbered_items(combined)
            occurrences.append({"page": pno, "start": spos, "items": items, "combined": combined})

    matches = [occ for occ in occurrences if exercise_number in occ["items"]]

    if not matches:
        return {
            "problem_text": "",
            "exercise_number": exercise_number,
            "found": False,
            "multiple_figureitout_sections": False,
            "chapter_page_no": chapter_page_number,
            "book_page_no": book_page_number,
            "pages_involved": {
                "previous": chapter_page_number - 1 if (chapter_page_number - 1) in pages else None,
                "current": chapter_page_number,
                "next": chapter_page_number + 1 if (chapter_page_number + 1) in pages else None
            },
            "text_bboxes": [],
            "images": [],
            "first_page_image_present": False,
            "second_page_image_present": False,
            "truncated": False,
            "char_cap": char_cap,
        }

    matches_by_page: Dict[int, List[Dict[str, Any]]] = {}
    for m in matches:
        matches_by_page.setdefault(m["page"], []).append(m)

    multiple_flag = sum(len(v) for v in matches_by_page.values()) > 1

    selected_occ = None
    for preferred in (chapter_page_number, chapter_page_number + 1, chapter_page_number - 1):
        if preferred in matches_by_page:
            occs = matches_by_page[preferred]
            selected_occ = occs[0] if len(occs) == 1 else (occs[0] if preferred != chapter_page_number - 1 else occs[-1])
            break
    if selected_occ is None:
        selected_occ = matches[0]

    selected_page = selected_occ["page"]
    item_text, item_start = selected_occ["items"][exercise_number]
    raw_item_text = item_text

    sel_clean = cleaned_pages.get(selected_page, "")
    actual_start_page = selected_page if item_start < len(sel_clean) else selected_page + 1

    first_line = ""
    if raw_item_text:
        for ln in raw_item_text.splitlines():
            s = ln.strip()
            if s:
                first_line = s
                break

    doc = fitz.open(pdf_path)
    problem_text, text_bboxes, truncated_flag = _collect_blocks_for_item(
        doc, actual_start_page, first_line, char_cap, item_no=exercise_number
    )
    if not problem_text:
        problem_text = re.sub(r"\s+", " ", raw_item_text).strip()
        if len(problem_text) > char_cap:
            problem_text = problem_text[:char_cap]
            truncated_flag = True

    def _norm(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "")).strip()

    norm_prob = _norm(problem_text)
    first_page_no = actual_start_page
    found_page = None

    for candidate in (actual_start_page, actual_start_page + 1):
        if candidate not in cleaned_pages:
            continue
        page_text = cleaned_pages[candidate]
        for line in page_text.splitlines():
            m = re.match(rf"^\s*{exercise_number}\.\s*(.*)$", line)
            if not m:
                continue
            suffix = _norm(m.group(1))
            if suffix and norm_prob.startswith(suffix):
                found_page = candidate
                break
        if found_page is not None:
            break

    if found_page is not None:
        first_page_no = found_page
    else:
        if text_bboxes:
            bpg = int(text_bboxes[0][0])
            if bpg in pages:
                first_page_no = bpg

    if first_page_no not in pages:
        first_page_no = actual_start_page

    page_nos_to_render = [first_page_no]

    include_second = False
    items_map = selected_occ["items"]
    next_tuple = items_map.get(exercise_number + 1)

    sel_clean_for_occ = cleaned_pages.get(selected_occ["page"], "")
    if next_tuple:
        next_start = next_tuple[1]
        if next_start >= len(sel_clean_for_occ):
            include_second = True
    else:
        include_second = True

    if not include_second and text_bboxes:
        for bb in text_bboxes:
            if int(bb[0]) == first_page_no + 1:
                include_second = True
                break

    second_page_no = first_page_no + 1
    if include_second and second_page_no in pages:
        page_nos_to_render.append(second_page_no)

    t1 = time.time()
    logger.info("before render_pages_as_images: %.2fs elapsed", t1 - t0)

    try:
        images = render_pages_as_images(pdf_path, page_nos_to_render, max_bytes=max_images_bytes)
    except Exception:
        logger.exception(
            "render_pages_as_images failed for pdf_path=%s page_nos=%s", pdf_path, page_nos_to_render
        )
        images = []

    t2 = time.time()
    logger.info("after render_pages_as_images: took %.2fs", t2 - t1)

    normalized_images: List[Dict[str, Any]] = []
    for idx, img in enumerate(images):
        # ensure bytes_b64 exists
        if "bytes_b64" not in img and "bytes" in img:
            try:
                img["bytes_b64"] = base64.b64encode(img["bytes"]).decode("ascii")
            except Exception:
                img["bytes_b64"] = ""

        img.setdefault("mime", ("image/png" if img.get("format","PNG").upper()=="PNG" else "image/jpeg"))
        if "size_bytes" not in img:
            try:
                img["size_bytes"] = len(base64.b64decode(img["bytes_b64"])) if img.get("bytes_b64") else 0
            except Exception:
                img["size_bytes"] = 0

        if img.get("size_bytes", 0) > max_images_bytes:
            logger.warning(
                "[load_context] image size %d > max_images_bytes %d for chapter_page_no=%s",
                img.get("size_bytes", 0),
                max_images_bytes,
                img.get("chapter_page_no"),
            )

        # Add stable image_index for lightweight UI render calls
        img["image_index"] = idx

        # safe metadata for logs
        safe_img = {
            "image_index": img.get("image_index"),
            "chapter_page_no": img.get("chapter_page_no"),
            "format": img.get("format"),
            "mime": img.get("mime"),
            "width": img.get("width"),
            "height": img.get("height"),
            "size_bytes": img.get("size_bytes", 0),
        }
        logger.debug("Prepared image metadata: %s", safe_img)

        normalized_images.append(img)

    images = normalized_images

    first_present = len(images) >= 1
    second_present = len(images) >= 2

    for img in images:
        printed = chapter.get("map_chapter_to_printed", {}).get(img["chapter_page_no"])
        if printed is not None:
            img["book_page_no"] = printed

    # Persist artifacts via ToolContext when available (annotate with artifact_name + artifact_saved)
    if tool_context is not None:
        for img in images:
            img["artifact_saved"] = False
            try:
                b64 = img.get("bytes_b64")
                if not b64:
                    img["artifact_name"] = None
                    img["artifact_saved"] = False
                    continue
                raw = base64.b64decode(b64)
                artifact_name = f"loadctx_{uuid.uuid4().hex}.png"

                # Preferred: create a genai_types.Part with inline_data
                artifact_part = None
                try:
                    inline = genai_types.InlineData(data=raw, mime_type=img.get("mime", "image/png"))
                    artifact_part = genai_types.Part(inline_data=inline)
                except Exception:
                    # Last-resort: attach bytes as text-like fallback (rare)
                    try:
                        artifact_part = genai_types.Part(inline_data={"data": raw, "mime_type": img.get("mime", "image/png")})
                    except Exception:
                        artifact_part = None

                if artifact_part is not None:
                    try:
                        saved_ok = await save_artifact_robust(tool_context, artifact_name, artifact_part)
                        img["artifact_saved"] = bool(saved_ok)
                    except Exception:
                        logger.exception("save_artifact_robust failed for %s", artifact_name)
                        img["artifact_saved"] = False
                else:
                    img["artifact_saved"] = False

                img["artifact_name"] = artifact_name
            except Exception:
                logger.exception("Failed saving a load_context image as artifact (continuing)")
                img.setdefault("artifact_name", None)
                img.setdefault("artifact_saved", False)

    # --- begin map artifact creation (inserted) ---
    import json
    map_id = None
    try:
        # Create a short stable map id
        map_id = f"loadmap_{uuid.uuid4().hex[:12]}"

        map_obj = {
            "map_id": map_id,
            "images": []
        }

        for img in images:
            # include minimal metadata per image; keep bytes fallback for reliability
            map_obj["images"].append({
                "image_index": img.get("image_index"),
                "artifact_name": img.get("artifact_name"),
                "mime": img.get("mime"),
                "chapter_page_no": img.get("chapter_page_no"),
                "size_bytes": img.get("size_bytes"),
                "caption": img.get("caption", ""),
                # Store fallback base64 only if artifact save did not succeed
                "bytes_b64_fallback": img.get("bytes_b64") if not img.get("artifact_saved") else None,
                # Keep original bytes_b64 only for diagnostics when artifact not saved
                "bytes_b64": img.get("bytes_b64") if not img.get("artifact_saved") else None,
            })

        # Save the small JSON map as a session-scoped artifact if tool_context supports it
        if tool_context is not None:
            try:
                part = genai_types.Part(text=json.dumps(map_obj))
                try:
                    await save_artifact_robust(tool_context, map_id, part)
                except Exception:
                    logger.exception("save_artifact_robust failed for map_id=%s", map_id)
            except Exception:
                logger.exception("Failed to create or save map artifact for load_context (inner)")
    except Exception:
        logger.exception("Failed to create map metadata for load_context")
    # --- end map artifact creation ---

    # --- begin populate session cache (inserted) ---
    try:
        # best-effort: put map metadata into agent module's _SESSION_IMAGE_CACHE
        try:
            from src.agents.tutor_agent import agent as _agent_mod  # guarded import
        except Exception:
            _agent_mod = None

        if _agent_mod is not None and map_id:
            try:
                cache = getattr(_agent_mod, "_SESSION_IMAGE_CACHE", None)
                if cache is None:
                    # create and attach cache if missing
                    _agent_mod._SESSION_IMAGE_CACHE = {}
                    cache = _agent_mod._SESSION_IMAGE_CACHE
                # store minimal map payload (images + timestamp)
                cache[map_id] = {
                    "images": map_obj.get("images", []),
                    "timestamp": time.time(),
                }
                logger.info("Populated _SESSION_IMAGE_CACHE with map_id=%s (images=%d)", map_id, len(map_obj.get("images", [])))
            except Exception:
                logger.exception("Failed to populate _SESSION_IMAGE_CACHE for map_id=%s", map_id)
    except Exception:
        # keep non-fatal: do not fail the whole tool if caching fails
        logger.debug("Session cache population skipped due to import/caching error.")
    # --- end populate session cache ---

    doc.close()

    logger.info("load_context is just about to return: total %.2fs elapsed", time.time() - t0)

    return {
        "problem_text": problem_text,
        "exercise_number": exercise_number,
        "found": True,
        "multiple_figureitout_sections": multiple_flag,
        "chapter_page_no": chapter_page_number,
        "book_page_no": book_page_number,
        "pages_involved": {
            "previous": first_page_no - 1 if (first_page_no - 1) in pages else None,
            "current": first_page_no,
            "next": first_page_no + 1 if (first_page_no + 1) in pages else None
        },
        "text_bboxes": text_bboxes,
        "images": images,
        "map_id": map_id,
        "first_page_image_present": first_present,
        "second_page_image_present": second_present,
        "truncated": truncated_flag,
        "char_cap": char_cap,
    }


# Helper to save images from load_context output for local testing
def save_images_from_load_context_output(out: dict, out_dir: str = "data/images"):
    os.makedirs(out_dir, exist_ok=True)
    saved = []
    for i, img in enumerate(out.get("images", []), start=1):
        b64 = img["bytes_b64"]
        data = base64.b64decode(b64)
        fmt = img.get("format", "PNG").upper()
        ext = "png" if fmt == "PNG" else "jpg"
        path = os.path.join(out_dir, f"exercise_img_{i}.{ext}")
        with open(path, "wb") as f:
            f.write(data)
        try:
            pil = Image.open(BytesIO(data))
            pil.save(path)
        except Exception:
            pass
        saved.append(path)
    return saved


if __name__ == "__main__":
    # quick local test (sync runner will call async function)
    import asyncio, json
    out = asyncio.run(load_context(grade=8, chapter_id=7, book_page_number=170, exercise_number=2))
    print(json.dumps({k: v for k, v in out.items() if k != "images"}, indent=2)[:2000])
    saved_paths = save_images_from_load_context_output(out, out_dir="data/images")
    print("Saved:", saved_paths)
