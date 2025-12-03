"""
session_reviewer_helpers.py

Minimal helper utilities used by the Session Reviewer agent to
fetch session transcripts and image metadata/bytes produced by
the Streamlit session (or stored artifacts).

Intended usage:
- The Session Reviewer agent imports these functions directly and
  calls them synchronously (no ADK tool registration required).

Functions provided:
- fetch_transcript_text(session_id: str, byte_limit: int) -> tuple[int, str] | None
- get_image_info_from_map(map_id: str, indices: list[int] | None) -> list[dict]
- fetch_image_bytes(path_or_entry) -> bytes | None

Notes / assumptions:
- Streamlit (or other code) must already have written the transcript
  and image artifacts to disk or to st.session_state/_SESSION_IMAGE_CACHE.
- The helpers try several common paths; adapt the search paths to match
  the project's exact artifact locations if needed.
"""

from __future__ import annotations
import os
import json
import logging
import re
from typing import Optional, Tuple, List, Dict, Any

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Try optional imports for PDF->text extraction and image handling
try:
    from pdfminer.high_level import extract_text as pdfminer_extract_text  # type: ignore
    _HAS_PDFMINER = True
except Exception:
    _HAS_PDFMINER = False

# Optional: allow reading Streamlit session cache if running in same process
try:
    import streamlit as st  # type: ignore
    _HAS_STREAMLIT = True
except Exception:
    st = None  # type: ignore
    _HAS_STREAMLIT = False


# ---------------------------
# Helper: fetch_transcript_text
# ---------------------------
def fetch_transcript_text(session_id: str, byte_limit: int) -> Optional[Tuple[int, str]]:
    """
    Locate and return transcript text for a session.

    Returns:
        (byte_count, text) if found
        None if transcript not found or unreadable

    Behavior:
    - Looks for a text transcript (.txt) and PDF transcript (.pdf) in a few common locations:
      1) streamlit session_state artifacts if available (st.session_state['transcripts'] or similar)
      2) ./data/sessions/{session_id}/transcript.txt or transcript.pdf
      3) ./data/artifacts/{session_id}_transcript.txt or .pdf
      4) ./artifacts/{session_id}/transcript.*
      5) current working directory fallback: transcript_{session_id}.txt/.pdf
    - If a PDF is found and pdfminer is installed, attempts text extraction.
    - Always reports byte_count = len(text.encode('utf-8')) when text is available.
    - Caller should check byte_count against configured TRANSCRIPT_LIMIT_BYTES.
    """
    try:
        # 0) Try to read an in-process transcript stored in Streamlit session_state (if available)
        if _HAS_STREAMLIT:
            try:
                # Common keys that might contain transcript text or path
                # This is heuristic — adjust to your app's actual keys if needed.
                ss = st.session_state
                # Try a direct text field
                for key in ("transcript_text", "transcript", "session_transcript"):
                    if key in ss and isinstance(ss[key], str) and ss[key].strip():
                        text = ss[key]
                        byte_count = len(text.encode("utf-8"))
                        return (byte_count, text)
                # Try a stored PDF/text path
                for key in ("transcript_path", "transcript_pdf_path", "transcript_file"):
                    if key in ss and isinstance(ss[key], str) and os.path.exists(ss[key]):
                        path = ss[key]
                        return _read_transcript_from_path(path)
            except Exception as e:
                logger.debug("Streamlit session_state lookup failed: %s", e)

        # 1) Candidate paths to search (ordered)
        candidates = [
            os.path.join("data", "sessions", session_id, "transcript.txt"),
            os.path.join("data", "sessions", session_id, "transcript.pdf"),
            os.path.join("data", "artifacts", f"{session_id}_transcript.txt"),
            os.path.join("data", "artifacts", f"{session_id}_transcript.pdf"),
            os.path.join("artifacts", session_id, "transcript.txt"),
            os.path.join("artifacts", session_id, "transcript.pdf"),
            os.path.join(".", f"transcript_{session_id}.txt"),
            os.path.join(".", f"transcript_{session_id}.pdf"),
        ]

        # also try a map-id style: maybe Streamlit saved under "session_{id}.pdf"
        candidates += [
            os.path.join("data", "sessions", f"session_{session_id}.pdf"),
            os.path.join("data", "sessions", f"{session_id}.pdf"),
        ]

        for path in candidates:
            if path and os.path.exists(path) and os.path.isfile(path):
                # Found a file. If it's .txt, read directly. If .pdf, extract text if possible.
                return _read_transcript_from_path(path)

        # 2) Fallback: there may be a central transcripts folder
        for folder in ("data/transcripts", "transcripts", "uploads/transcripts"):
            if os.path.isdir(folder):
                # try files that contain session_id
                for fname in os.listdir(folder):
                    if session_id in fname:
                        p = os.path.join(folder, fname)
                        if os.path.isfile(p):
                            return _read_transcript_from_path(p)

        logger.debug("Transcript not found for session_id=%s", session_id)
        return None

    except Exception as exc:
        logger.exception("Unexpected error in fetch_transcript_text for session_id=%s: %s", session_id, exc)
        return None


def _read_transcript_from_path(path: str) -> Optional[Tuple[int, str]]:
    """
    Internal helper: read text from a .txt or .pdf transcript path.
    Returns (byte_count, text) or None on failure.
    """
    try:
        lower = path.lower()
        if lower.endswith(".txt") or lower.endswith(".md"):
            with open(path, "r", encoding="utf-8") as fh:
                text = fh.read()
            byte_count = len(text.encode("utf-8"))
            return (byte_count, text)
        elif lower.endswith(".pdf"):
            # try to extract text via pdfminer if available
            if _HAS_PDFMINER:
                try:
                    text = pdfminer_extract_text(path)
                    byte_count = len(text.encode("utf-8"))
                    return (byte_count, text)
                except Exception as e:
                    logger.warning("pdfminer failed to extract text from %s: %s", path, e)
                    # fallback: return the pdf byte size but empty text
                    try:
                        size = os.path.getsize(path)
                        return (size, "")
                    except Exception:
                        return None
            else:
                # pdfminer not available: return PDF file size and empty text (caller must handle)
                try:
                    size = os.path.getsize(path)
                    logger.debug("pdfminer not installed; returning pdf byte size only for %s", path)
                    return (size, "")
                except Exception as e:
                    logger.warning("Could not stat pdf %s: %s", path, e)
                    return None
        else:
            # unknown extension: attempt to read as text
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    text = fh.read()
                byte_count = len(text.encode("utf-8"))
                return (byte_count, text)
            except Exception:
                # as fallback, return file size
                try:
                    size = os.path.getsize(path)
                    return (size, "")
                except Exception:
                    return None
    except Exception as exc:
        logger.exception("Error reading transcript path %s: %s", path, exc)
        return None


# ---------------------------
# Helper: get_image_info_from_map
# ---------------------------
def get_image_info_from_map(map_id: str, indices: Optional[List[int]] = None) -> List[Dict[str, Any]]:
    """
    Given a map_id (the identifier that maps to stored images), return image metadata list.

    The function tries these sources (in order):
    - In-process Streamlit session cache: st.session_state.get("_SESSION_IMAGE_CACHE") or similar.
      The expected structure is a mapping of map_id -> list of image entries (dicts or objects).
    - JSON file on disk under typical artifact paths:
         ./data/session_images/{map_id}.json
         ./data/artifacts/{map_id}.json
         ./artifacts/{map_id}.json
      The JSON is expected to contain a list of images with fields (filename, path, mime, size, ...).
    - If none found, returns an empty list.

    Each returned dict has keys (best-effort):
    - index: int
    - filename: str
    - mime: str (if available)
    - byte_size: int (if available)
    - path: str (filesystem path if available)
    - is_student_uploaded: bool
    - extra: any additional metadata
    """
    results: List[Dict[str, Any]] = []
    try:
        # 0) Try Streamlit in-process cache
        if _HAS_STREAMLIT:
            try:
                ss = st.session_state
                # Common cache key heuristics
                cache_keys = ["_SESSION_IMAGE_CACHE", "session_image_cache", "image_cache", "images_map"]
                for key in cache_keys:
                    if key in ss:
                        cache = ss[key]
                        # We're expecting something like {map_id: [entries...]} or entries directly
                        if isinstance(cache, dict):
                            if map_id in cache:
                                entries = cache[map_id]
                            else:
                                # maybe the cache itself is a list of entries and map_id is unused
                                entries = cache if isinstance(cache, list) else []
                        elif isinstance(cache, list):
                            entries = cache
                        else:
                            entries = []
                        # convert entries to normalized dicts
                        for idx, ent in enumerate(entries):
                            if indices and idx not in indices:
                                continue
                            normalized = _normalize_cache_image_entry(ent, idx)
                            results.append(normalized)
                        if results:
                            return results
            except Exception as e:
                logger.debug("Streamlit image cache lookup failed: %s", e)

        # 1) Try JSON artifact locations on disk
        candidate_paths = [
            os.path.join("data", "session_images", f"{map_id}.json"),
            os.path.join("data", "artifacts", f"{map_id}.json"),
            os.path.join("artifacts", f"{map_id}.json"),
            os.path.join("data", "sessions", map_id, "images.json"),
            os.path.join("data", "sessions", map_id, "image_map.json"),
        ]
        for p in candidate_paths:
            if os.path.exists(p) and os.path.isfile(p):
                try:
                    with open(p, "r", encoding="utf-8") as fh:
                        raw = json.load(fh)
                    # raw might be dict { "images": [...] } or a list
                    if isinstance(raw, dict) and "images" in raw and isinstance(raw["images"], list):
                        entries = raw["images"]
                    elif isinstance(raw, list):
                        entries = raw
                    elif isinstance(raw, dict):
                        # try to find a list value inside
                        entries = []
                        for v in raw.values():
                            if isinstance(v, list):
                                entries = v
                                break
                    else:
                        entries = []
                    for idx, ent in enumerate(entries):
                        if indices and idx not in indices:
                            continue
                        normalized = _normalize_cache_image_entry(ent, idx)
                        results.append(normalized)
                    if results:
                        return results
                except Exception as e:
                    logger.exception("Failed to parse image map JSON %s: %s", p, e)

        # 2) No map found — attempt a generic images folder lookup for session_id style maps
        # If map_id looks like a session id, try a folder
        fallback_folders = [
            os.path.join("data", "sessions", map_id),
            os.path.join("data", "sessions", map_id, "images"),
            os.path.join("uploads", "images", map_id),
        ]
        for folder in fallback_folders:
            if os.path.isdir(folder):
                files = sorted(os.listdir(folder))
                for idx, fname in enumerate(files):
                    if indices and idx not in indices:
                        continue
                    p = os.path.join(folder, fname)
                    try:
                        stat = os.stat(p)
                        entry = {
                            "index": idx,
                            "filename": fname,
                            "mime": _guess_mime_from_filename(fname),
                            "byte_size": stat.st_size,
                            "path": p,
                            "is_student_uploaded": False,
                        }
                        results.append(entry)
                    except Exception:
                        continue
                if results:
                    return results

        # nothing discovered
        return results
    except Exception as exc:
        logger.exception("Unexpected error in get_image_info_from_map(map_id=%s): %s", map_id, exc)
        return results


def _normalize_cache_image_entry(ent: Any, idx: int) -> Dict[str, Any]:
    """
    Normalize a single cache entry (which may be a dict, tuple, or custom object)
    into a predictable dict shape.
    """
    try:
        if isinstance(ent, dict):
            filename = ent.get("filename") or ent.get("name") or ent.get("file")
            path = ent.get("path") or ent.get("file_path")
            mime = ent.get("mime") or ent.get("content_type")
            size = ent.get("size") or ent.get("byte_size") or (os.path.getsize(path) if path and os.path.exists(path) else None)
            is_student = bool(ent.get("is_student_uploaded") or ent.get("uploaded_by") == "student")
            return {
                "index": ent.get("index", idx),
                "filename": filename or (os.path.basename(path) if path else f"image_{idx}"),
                "mime": mime or _guess_mime_from_filename(filename or ""),
                "byte_size": size or (len(ent.get("bytes")) if ent.get("bytes") else None),
                "path": path,
                "is_student_uploaded": is_student,
                "extra": {k: v for k, v in ent.items() if k not in ("filename", "path", "mime", "size", "bytes")}
            }
        # If it's a tuple (path, meta) or similar
        if isinstance(ent, (list, tuple)) and len(ent) >= 1:
            path = ent[0]
            fname = os.path.basename(path) if isinstance(path, str) else f"image_{idx}"
            size = None
            try:
                size = os.path.getsize(path)
            except Exception:
                size = None
            return {
                "index": idx,
                "filename": fname,
                "mime": _guess_mime_from_filename(fname),
                "byte_size": size,
                "path": path,
                "is_student_uploaded": False,
                "extra": {}
            }
        # unknown type: stringify
        return {
            "index": idx,
            "filename": f"image_{idx}",
            "mime": None,
            "byte_size": None,
            "path": None,
            "is_student_uploaded": False,
            "extra": {"raw": repr(ent)}
        }
    except Exception as exc:
        logger.exception("Error normalizing cache entry: %s", exc)
        return {
            "index": idx,
            "filename": f"image_{idx}",
            "mime": None,
            "byte_size": None,
            "path": None,
            "is_student_uploaded": False,
            "extra": {"error": str(exc)}
        }


def _guess_mime_from_filename(fname: str) -> Optional[str]:
    if not fname:
        return None
    fname = fname.lower()
    if fname.endswith(".png"):
        return "image/png"
    if fname.endswith(".jpg") or fname.endswith(".jpeg"):
        return "image/jpeg"
    if fname.endswith(".gif"):
        return "image/gif"
    if fname.endswith(".webp"):
        return "image/webp"
    return None


# ---------------------------
# Helper: fetch_image_bytes
# ---------------------------
def fetch_image_bytes(path_or_entry: Any) -> Optional[bytes]:
    """
    Given a path string or a normalized get_image_info_from_map() entry,
    return image bytes or None.

    Accepts:
    - path string (filesystem)
    - dict entry with 'path' or 'bytes' keys (as returned by get_image_info_from_map)
    - an object with .read() (file-like) or .getbuffer()

    Does not raise on missing files; logs and returns None.
    """
    try:
        if path_or_entry is None:
            return None
        # If passed a dict-like entry
        if isinstance(path_or_entry, dict):
            # If bytes are already embedded
            if "bytes" in path_or_entry and isinstance(path_or_entry["bytes"], (bytes, bytearray)):
                return bytes(path_or_entry["bytes"])
            # If an in-memory buffer is present as 'buffer'
            if "buffer" in path_or_entry and hasattr(path_or_entry["buffer"], "tobytes"):
                try:
                    return path_or_entry["buffer"].tobytes()
                except Exception:
                    pass
            # If a filesystem path is present
            p = path_or_entry.get("path")
            if p and isinstance(p, str) and os.path.exists(p):
                try:
                    with open(p, "rb") as fh:
                        return fh.read()
                except Exception:
                    logger.exception("Failed to read image bytes from path %s", p)
                    return None
            # nothing else found
            return None

        # If passed a path string
        if isinstance(path_or_entry, str):
            if os.path.exists(path_or_entry) and os.path.isfile(path_or_entry):
                try:
                    with open(path_or_entry, "rb") as fh:
                        return fh.read()
                except Exception as e:
                    logger.exception("Failed to read image bytes from path %s: %s", path_or_entry, e)
                    return None
            else:
                # maybe a base64 string? not supported here
                logger.debug("fetch_image_bytes: path string does not exist: %s", path_or_entry)
                return None

        # If passed a file-like object
        if hasattr(path_or_entry, "read"):
            try:
                data = path_or_entry.read()
                # if returns str, encode
                if isinstance(data, str):
                    return data.encode("utf-8")
                return data
            except Exception as e:
                logger.exception("Failed to read from file-like object: %s", e)
                return None

        # Otherwise unsupported type
        logger.debug("fetch_image_bytes received unsupported type: %s", type(path_or_entry))
        return None

    except Exception as exc:
        logger.exception("Unexpected error in fetch_image_bytes: %s", exc)
        return None
