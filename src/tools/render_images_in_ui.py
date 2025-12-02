import json
import base64
import inspect
import uuid
import logging
from typing import Any, Dict, List, Optional

# use the robust loader from artifacts_helper
from src.tools.artifacts_helper import load_artifact_robust

logger = logging.getLogger("tutor_agent")

# The tool is async to match typical ToolContext async APIs.
async def render_images_in_ui(
    map_id: Optional[str] = None,
    indices: Optional[List[int]] = None,
    tool_context: Optional[Any] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Resolve image bytes for UI rendering.

    Args:
        map_id (str | None):
            Optional short id produced by load_context(). If provided, the
            tool will load the small JSON "map" artifact and resolve images
            from the artifact store.

        indices (list[int] | None):
            List of 0-based image indices to render. If None, all images in
            the map (or session cache) are considered.

        tool_context (ToolContext | None):
            Runner-provided context used to load artifacts. If not provided
            only session-cache fallback will be available.

    Returns:
        dict: {"render_images": [ { "id": str,
                                    "chapter_page_no": Optional[int],
                                    "bytes_b64": str,
                                    "mime": Optional[str],
                                    "caption": Optional[str],
                                    "size_bytes": Optional[int]
                                  }, ... ]}

    Notes:
        - Primary resolution path: load map artifact named `map_id` using
          tool_context.load_artifact().
        - Fallback: attempt to read session cache `_SESSION_IMAGE_CACHE` from
          src.agent and choose the most recent entry.
        - If an image refers to an artifact, this tool will load that artifact
          and prefer inline_data or file_data (base64) when present. If an
          artifact cannot be loaded, an optional bytes_b64 fallback saved in
          the map metadata will be used.
    """
    
    logger.info(
        "render_images_in_ui called with map_id=%s, indices=%s, tool_context=%s",
        map_id, indices, type(tool_context)
    )
    
    out = {"render_images": []}
    try:
        images_meta = None
        # Normalize requested indices
        if indices is None:
            indices = []

        # 1) Try map_id resolution via tool_context
        if map_id:
            # Attempt to load the saved JSON map artifact via robust helper.
            map_text = None
            try:
                # helper handles tool_context None and multiple call signatures
                res = await load_artifact_robust(tool_context, map_id)
                if res is None:
                    map_text = None
                elif isinstance(res, str):
                    map_text = res
                else:
                    # genai types.Part-like object: prefer .text, else try inline_data/file_data
                    map_text = getattr(res, "text", None)
                    if not map_text and hasattr(res, "inline_data") and getattr(res, "inline_data", None):
                        inline = getattr(res, "inline_data")
                        if hasattr(inline, "data") and isinstance(inline.data, (str, bytes)):
                            map_text = inline.data if isinstance(inline.data, str) else base64.b64decode(inline.data).decode("utf-8", errors="ignore")
                    if not map_text and hasattr(res, "file_data") and getattr(res, "file_data", None):
                        fd = getattr(res, "file_data")
                        map_text = getattr(fd, "data", None) or getattr(fd, "file_uri", None)
            except Exception:
                logger.exception("render_images_in_ui: failed to load map artifact %s", map_id)
                map_text = None

            if map_text:
                try:
                    map_obj = json.loads(map_text)
                    # Build dict: index -> meta (use image_index as canonical key)
                    images_meta = {int(img["image_index"]): img for img in map_obj.get("images", [])}
                    logger.info("render_images_in_ui: resolved map_id=%s with %d images", map_id, len(images_meta))
                except Exception:
                    logger.exception("render_images_in_ui: failed to parse map JSON from artifact %s", map_id)
                    images_meta = None

        # 2) Fallback to session cache if map not resolved
        if images_meta is None:
            try:
                # Attempt to import session cache from agent (if available).
                # Agent will create and maintain `_SESSION_IMAGE_CACHE` mapping.
                from src.agents.tutor_agent import agent as _agent_mod  # guarded import (agent module path)
                cache = getattr(_agent_mod, "_SESSION_IMAGE_CACHE", None)
                if isinstance(cache, dict) and cache:
                    # choose the most recent entry by timestamp if values have 'timestamp'
                    latest_entry = None
                    latest_ts = None
                    for v in cache.values():
                        ts = v.get("timestamp") if isinstance(v, dict) else None
                        if ts is None:
                            latest_entry = v
                            break
                        if latest_ts is None or ts > latest_ts:
                            latest_ts = ts
                            latest_entry = v
                    if latest_entry:
                        imgs = latest_entry.get("images") or []
                        images_meta = {int(img["image_index"]): img for img in imgs}
                        logger.info("render_images_in_ui: resolved via session cache with %d images", len(images_meta))
            except Exception:
                # Import failure or no cache present. Proceed with what we have.
                images_meta = None

        # If still nothing, return empty
        if not images_meta:
            logger.info("render_images_in_ui: no image metadata found for map_id=%s", map_id)
            return out

        # If indices not provided, render all
        if not indices:
            indices = sorted(images_meta.keys())

        # 3) For each requested index, try to load artifact, else use fallback bytes
        for raw_idx in indices:
            try:
                idx = int(raw_idx)
            except Exception:
                continue
            meta = images_meta.get(idx)
            if not meta:
                continue

            artifact_name = meta.get("artifact_name")
            bytes_b64 = None
            mime = meta.get("mime")
            size_bytes = meta.get("size_bytes")
            chapter_page_no = meta.get("chapter_page_no")
            caption = meta.get("caption", "")

            # If artifact_name available, attempt to load via robust helper
            if artifact_name:
                try:
                    art_part = await load_artifact_robust(tool_context, artifact_name)
                    if art_part is not None:
                        # case: art_part is a plain base64/text string
                        if isinstance(art_part, str):
                            # assume base64 string if it looks like base64 otherwise treat as text
                            bytes_b64 = art_part
                        else:
                            # prefer .inline_data
                            inline = getattr(art_part, "inline_data", None)
                            if inline and getattr(inline, "data", None):
                                raw = inline.data
                                if isinstance(raw, (bytes, bytearray)):
                                    bytes_b64 = base64.b64encode(raw).decode("ascii")
                                elif isinstance(raw, str):
                                    bytes_b64 = raw
                                mime = getattr(inline, "mime_type", mime)
                            else:
                                fd = getattr(art_part, "file_data", None)
                                if fd and getattr(fd, "data", None):
                                    data = fd.data
                                    if isinstance(data, str):
                                        bytes_b64 = data
                                    elif isinstance(data, (bytes, bytearray)):
                                        bytes_b64 = base64.b64encode(data).decode("ascii")
                                    mime = getattr(fd, "mime_type", mime)
                except Exception:
                    logger.exception("render_images_in_ui: failed to load artifact %s for index %s", artifact_name, idx)

            # 4) fallback to bytes_b64 stored in map metadata
            if not bytes_b64:
                bytes_b64 = meta.get("bytes_b64_fallback") or meta.get("bytes_b64") or None

            if not bytes_b64:
                logger.info("render_images_in_ui: no bytes for index %s (map_id=%s)", idx, map_id)
                continue

            out["render_images"].append({
                "id": f"{map_id or 'session'}_{idx}_{uuid.uuid4().hex[:8]}",
                "chapter_page_no": chapter_page_no,
                "bytes_b64": bytes_b64,
                "mime": mime,
                "caption": caption or "",
                "size_bytes": size_bytes,
            })

    except Exception:
        logger.exception("render_images_in_ui: unexpected error")
    return out
