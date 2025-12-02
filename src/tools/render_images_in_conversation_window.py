# src/tools/render_images_in_conversation_window.py
import uuid
import base64
from typing import List, Dict, Any, Optional
import logging
logger = logging.getLogger("tutor_agent")

# For async ToolContext handling
import inspect
import asyncio


async def render_images_in_conversation_window(images: List[Dict[str, Any]], tool_context: Optional[Any] = None) -> Dict[str, Any]:
    """
    Display one or more images in the conversation window.

    Args:
        images (List[dict]): A list where EACH item is a dict representing ONE image.
        tool_context (optional): ADK ToolContext injected by runner (if available). If provided,
                                 this function will attempt to fetch image bytes for entries that
                                 only include an 'artifact_name' by calling tool_context.load_artifact(...).

    REQUIRED FIELD (for each image):
        - EITHER "bytes" (raw bytes) OR "bytes_b64" (base64 string).
          At least one of these must be provided OR the image dict must include 'artifact_name'
          and a ToolContext must be available so bytes can be loaded.

    OPTIONAL FIELDS:
        - "chapter_page_no" (int): Page number this image came from.
        - "format" (str): Usually "PNG" or "JPEG".
        - "mime" (str): MIME type such as "image/png" or "image/jpeg".
        - "caption" (str): Caption to be displayed with the image.
        - "size_bytes" (int): Size in bytes.
        - "artifact_name" (str): artifact reference previously saved by load_context via ToolContext.save_artifact

    Returns:
        dict:
            {
                "render_images": [...]
            }
    """
    logger.info("render_images_in_conversation_window called with %d images", len(images))
    logger.info("DEBUG: render_images_in_conversation_window called; tool_context=%s", type(tool_context))

    out = {"render_images": []}
    if not images:
        return out

    # helper to try loading artifact bytes via ToolContext (async-aware)
    async def _try_load_artifact_bytes_async(artifact_name: str) -> Optional[bytes]:
        if not artifact_name or tool_context is None:
            return None
        load_fn = getattr(tool_context, "load_artifact", None)
        if not callable(load_fn):
            # some ADK versions may expose a different method name; try alternatives
            load_fn = getattr(tool_context, "get_artifact", None) or getattr(tool_context, "load_artifact_bytes", None)
        if not callable(load_fn):
            return None
        try:
            # try keyword call then fall back to positional
            try:
                res = load_fn(name=artifact_name)
            except TypeError:
                try:
                    res = load_fn(artifact_name)
                except TypeError:
                    res = load_fn(artifact_name, "session")
            # If result is awaitable, await it
            if inspect.isawaitable(res):
                try:
                    res = await res
                except Exception:
                    logger.exception("Awaiting load_artifact coroutine failed for %s", artifact_name)
                    return None
            # If the API returns an object with .content or .data, extract it
            if isinstance(res, bytes):
                return res
            if hasattr(res, "content"):
                return getattr(res, "content")
            if isinstance(res, dict) and res.get("content"):
                return res.get("content")
            # fallback: if res is string base64, decode it
            if isinstance(res, str):
                try:
                    return base64.b64decode(res)
                except Exception:
                    return None
        except Exception:
            logger.exception("Failed to load artifact %s via tool_context", artifact_name)
        return None

    for img in images:
        # produce a stable id
        img_id = str(uuid.uuid4())

        # Choose bytes_b64: prefer raw bytes if present, else reuse existing base64
        b64: Optional[str] = None
        if img.get("bytes") is not None:
            try:
                b64 = base64.b64encode(img["bytes"]).decode("ascii")
            except Exception:
                b64 = None
        if b64 is None and img.get("bytes_b64"):
            b64 = img.get("bytes_b64")

        # If bytes not present but artifact reference exists and tool_context available, try to load it
        if b64 is None and img.get("artifact_name"):
            try:
                raw = await _try_load_artifact_bytes_async(img.get("artifact_name"))
                if raw:
                    try:
                        b64 = base64.b64encode(raw).decode("ascii")
                        # if size not set, set it
                        if "size_bytes" not in img or not img.get("size_bytes"):
                            try:
                                img["size_bytes"] = len(raw)
                            except Exception:
                                pass
                    except Exception:
                        b64 = None
            except Exception:
                b64 = None

        mime = img.get("mime")
        if not mime:
            fmt = (img.get("format") or "").upper()
            mime = "image/jpeg" if fmt == "JPEG" else "image/png"

        out["render_images"].append({
            "id": img_id,
            "chapter_page_no": img.get("chapter_page_no"),
            "bytes_b64": b64,
            "mime": mime,
            "caption": img.get("caption", ""),
            "size_bytes": img.get("size_bytes"),
            "artifact_name": img.get("artifact_name"),
        })

    logger.info("render_images_in_conversation_window returning %d images", len(out["render_images"]))
    return out


