# src/agents/session_reviewer_agent/agent.py
"""
Session Reviewer agent (minimal).

Behavior:
- Exports a small LlmAgent instance `_AGENT` for registration as an AgentTool.
- Main entrypoint: `review_session_manifest(manifest, tool_context=None, timeout_seconds=30)`
  - Expected manifest keys: map_id (optional) and problem_text (recommended fallback).
  - Attempts to fetch full transcript via session_id from tool_context; if absent or unreadable,
    falls back to manifest['problem_text'].
  - Fetches image metadata via map_id using helpers.
  - Builds a concise prompt and calls the model; expects strictly parseable JSON in response.
- Designed for in-process Agent→Agent (A2A) usage via ADK Runner.
"""

from __future__ import annotations
import json
import time
import logging
from typing import Optional, Dict, Any, List

# ADK imports (expected to be available in the environment)
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.artifacts import InMemoryArtifactService

# Local helpers for transcript and image access
from src.tools.session_reviewer_helpers import (
    fetch_transcript_text,
    get_image_info_from_map,
    fetch_image_bytes,
)

logger = logging.getLogger("session_reviewer_agent")
logger.propagate = False
if not logger.handlers:
    logger.setLevel(logging.INFO)
    try:
        fh = logging.FileHandler("logs/session_reviewer_agent.log", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(fh)
    except Exception:
        # If file logging is not available, proceed with default handlers.
        pass

# -------------------------
# Agent instruction
# -------------------------
SYSTEM_INSTRUCTION = """
You are an expert tutor and a concise session reviewer for a mathematics tutoring session.

If the session transcript is available and <= 60 KB:
 - Use the transcript for your analysis; use the provided problem_text and up to 2 images only as supporting context that sets the main problem for the tutoring session.
 - Assess two things: (A) how effectively the tutor correctly identified the exact problem; (B) how effective the tutoring was for helping the student reach understanding.
 - Be specific and critical: explicitly call out any incorrect steps, missed checks, or places where the tutor could have asked a better question.
 - Produce ONLY valid JSON with exactly these fields:
   - session_summary: 2-4 sentences, concise and factual.
   - strengths: list of **up to 3** short bullet strings (max 12 words each).
   - weaknesses: list of **up to 3** short bullet strings (max 12 words each).
   - actionable_steps: list of **up to 3** short, concrete recommendations for the tutor (max 12 words each).
   - student_message: one encouraging sentence (max 18 words).
 - Do NOT invent facts, do not exaggerate, and do not include additional commentary outside the JSON.
 - Prefer concise, critical feedback over generic praise.

If the transcript is missing or > 60 KB:
 - Return the fallback JSON where string fields contain the single string "not available: session transcript may be too long"
   and list fields contain a one-element list with that string.

"""

# -------------------------
# LlmAgent instance
# -------------------------
_AGENT = LlmAgent(
    model="gemini-2.5-flash",
    name="session_reviewer_agent",
    description="A small agent that reviews tutoring sessions and returns a short JSON review.",
    instruction=SYSTEM_INSTRUCTION,
    tools=[],  # I/O handled by Python helpers; no model-visible tools required
)

# -------------------------
# Limits and config
# -------------------------
TRANSCRIPT_LIMIT_BYTES = 60 * 1024  # conservative transcript size limit
MAX_IMAGES = 2  # include metadata for up to this many images in prompt

# -------------------------
# Prompt builder
# -------------------------
def _build_prompt(transcript_text: str, image_metadatas: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    parts.append("You are an expert tutoring coach. Produce the requested JSON summary.")
    parts.append("\n--- TRANSCRIPT START ---\n")
    parts.append(transcript_text or "(No transcript text available.)")
    parts.append("\n--- TRANSCRIPT END ---\n")

    if image_metadatas:
        parts.append("Images attached (descriptions):")
        for i, m in enumerate(image_metadatas, start=1):
            caption = m.get("caption") or m.get("filename") or ""
            size = m.get("byte_size")
            mime = m.get("mime")
            parts.append(f"Image {i}: {caption} (mime={mime or 'unknown'}, size_bytes={size or 'unknown'})")
    else:
        parts.append("No images available.")

    parts.append(
        "\nReturn ONLY valid JSON with keys: session_summary, strengths, weaknesses, actionable_steps, student_message.\n"
    )
    return "\n".join(parts)

# -------------------------
# Core function
# -------------------------
def review_session_manifest(manifest: Dict[str, Any], tool_context: Optional[Any] = None, timeout_seconds: int = 30) -> Any:
    """
    Review a session based on manifest.

    Manifest (expected):
      - map_id: identifier for image map (optional)
      - problem_text: short problem text (recommended fallback if transcript missing)

    Returns:
      - dict with JSON fields defined in SYSTEM_INSTRUCTION on success
      - short string error message on failure
    """
    try:
        # Defensive normalization: accept several shapes for 'manifest'
        # If caller passed a JSON string or other wrapped object, normalize to dict
        if isinstance(manifest, str):
            try:
                parsed = json.loads(manifest)
                if isinstance(parsed, dict):
                    manifest = parsed
                else:
                    # keep as fallback dict
                    manifest = {"raw_text": str(manifest)}
            except Exception:
                manifest = {"raw_text": manifest}
        elif isinstance(manifest, dict):
            # Unwrap common wrapper keys: 'request' or 'text'
            if "request" in manifest:
                req = manifest.get("request")
                if isinstance(req, str):
                    try:
                        parsed = json.loads(req)
                        if isinstance(parsed, dict):
                            # replace manifest with parsed request and preserve any top-level session_id
                            session_id_val = manifest.get("session_id")
                            manifest = parsed
                            if session_id_val:
                                manifest["session_id"] = session_id_val
                    except Exception:
                        # keep the raw string accessible
                        manifest["raw_request"] = req
                elif isinstance(req, dict):
                    # inline the request dict (but preserve outer session_id if present)
                    session_id_val = manifest.get("session_id")
                    manifest = req.copy()
                    if session_id_val:
                        manifest["session_id"] = session_id_val
            elif "text" in manifest and isinstance(manifest.get("text"), str):
                # possibly the payload is JSON inside 'text'
                try:
                    parsed = json.loads(manifest.get("text"))
                    if isinstance(parsed, dict):
                        session_id_val = manifest.get("session_id")
                        manifest = parsed
                        if session_id_val:
                            manifest["session_id"] = session_id_val
                except Exception:
                    # leave manifest as-is (but keep 'raw_text' for diagnostics)
                    manifest["raw_text"] = manifest.get("text")    
                
        
        # Read manifest fields (only these two are expected)
        map_id = manifest.get("map_id")
        problem_text = manifest.get("problem_text") or ""

        # Resolve session_id from tool_context if ADK provides it (preferred)
        session_id: Optional[str] = None
        if tool_context is not None:
            try:
                if hasattr(tool_context, "session_id"):
                    session_id = getattr(tool_context, "session_id")
                elif isinstance(tool_context, dict) and "session_id" in tool_context:
                    session_id = tool_context.get("session_id")
            except Exception:
                session_id = None

        # 1) Attempt to fetch transcript text via helpers if session_id available
        transcript_res = None
        if session_id:
            try:
                transcript_res = fetch_transcript_text(session_id, TRANSCRIPT_LIMIT_BYTES)
            except Exception:
                transcript_res = None

        # 2) Choose text for prompt: transcript if available and small, otherwise manifest problem_text
        if transcript_res is None:
            if not problem_text:
                logger.info("No transcript found and no problem_text provided in manifest.")
                return "Unable to review: transcript not found."
            byte_count = len(problem_text.encode("utf-8"))
            transcript_text = problem_text
        else:
            byte_count, transcript_text = transcript_res
            if byte_count is None or byte_count == 0:
                if not problem_text:
                    logger.info("Transcript unreadable and no fallback problem_text provided.")
                    return "Unable to review: transcript not found."
                byte_count = len(problem_text.encode("utf-8"))
                transcript_text = problem_text

        # 3) Size check
        if byte_count > TRANSCRIPT_LIMIT_BYTES:
            logger.info("Transcript too large (%d bytes).", byte_count)
            return "Unable to review: transcript may be too long."

        # 4) Fetch image metadata via helpers (no image bytes embedded)
        image_infos = []
        if map_id:
            try:
                image_infos = get_image_info_from_map(map_id, indices=None) or []
                if len(image_infos) > MAX_IMAGES:
                    image_infos = image_infos[:MAX_IMAGES]
            except Exception:
                logger.exception("Failed to fetch image info for map_id=%s", map_id)
                image_infos = []

        image_metadatas: List[Dict[str, Any]] = []
        for info in image_infos:
            image_metadatas.append({
                "filename": info.get("filename"),
                "mime": info.get("mime"),
                "byte_size": info.get("byte_size"),
                "caption": info.get("caption") or "",
            })

        # 5) Build prompt and run agent
        prompt = _build_prompt(transcript_text, image_metadatas)

        runner_session_id = session_id or (map_id if isinstance(map_id, str) else "session_reviewer")
        session_svc = InMemorySessionService()
        artifact_svc = InMemoryArtifactService()
        runner = Runner(
            agent=_AGENT,
            app_name="session_reviewer_oneoff",
            session_service=session_svc,
            artifact_service=artifact_svc,
        )

        response_text: Optional[str] = None
        start = time.time()
        for event in runner.run(user_id="session_reviewer", session_id=runner_session_id, new_message=prompt):
            try:
                cont = getattr(event, "content", None)
                if cont and getattr(cont, "text", None):
                    response_text = cont.text
                cand_attr = getattr(event, "candidates", None)
                if cand_attr:
                    for cand in cand_attr:
                        try:
                            t = getattr(cand, "text", None)
                            if isinstance(t, str) and t.strip():
                                response_text = t
                        except Exception:
                            continue
                if time.time() - start > timeout_seconds:
                    logger.info("Runner timed out after %ds", timeout_seconds)
                    break
            except Exception:
                continue

        # 6) Validate model output
        if not response_text or not isinstance(response_text, str) or not response_text.strip():
            logger.info("Reviewer produced no output.")
            return "Unable to review: reviewer produced no output."

        parsed_json = None
        try:
            stripped = response_text.strip()
            if stripped.startswith("{"):
                parsed_json = json.loads(stripped)
            else:
                start_idx = stripped.find("{")
                end_idx = stripped.rfind("}")
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    candidate = stripped[start_idx:end_idx + 1]
                    parsed_json = json.loads(candidate)
        except Exception as exc:
            logger.exception("Failed to parse JSON from reviewer response: %s. Raw response: %s", exc, response_text)
            return "Unable to review: reviewer parsing failed."

        required_keys = {"session_summary", "strengths", "weaknesses", "actionable_steps", "student_message"}
        if not parsed_json or not required_keys.issubset(set(parsed_json.keys())):
            logger.info("Reviewer returned JSON missing required keys.")
            return "Unable to review: reviewer returned unexpected JSON."

        return parsed_json

    except Exception as exc:
        logger.exception("Unexpected error in review_session_manifest: %s", exc)
        return "Unable to review: internal error."

# Convenience wrapper
def review_session_tool(manifest: Dict[str, Any], tool_context: Optional[Any] = None) -> Any:
    return review_session_manifest(manifest, tool_context=tool_context)

__all__ = ["_AGENT", "review_session_manifest", "review_session_tool"]
