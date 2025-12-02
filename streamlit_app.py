# streamlit_app.py

import os
import queue
import threading
import time
import re
import io
import base64
from datetime import datetime
import logging
import uuid
from typing import Optional, Tuple
import json

import streamlit as st
import markdown as md
from weasyprint import HTML, CSS

# Page config and titles
st.set_page_config(
    page_title="A Specialized Mathematics Tutor Agent for Students of Grades 7 & 8",
    layout="centered",
)
st.title("A Specialized Mathematics Tutor Agent for Students of Grades 7 & 8")

# Reduce server max upload size to 10 MB
# st.set_option("server.maxUploadSize", 10)  # value in MB
# Unfortunately, Streamlit does not currently support dynamic setting of maxUploadSize via st.set_option.


# Obtain module-level logger
# Use a dedicated logger for the Streamlit UI so logs are easy to separate from agent/tool logs.

import pathlib
import shutil

LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)

# --- Clear existing files in logs directory once per Streamlit session (keep it tidy for each run) ---
# This avoids deleting active log files on Streamlit hot-reload.
if not st.session_state.get("logs_cleaned", False):
    try:
        pdir = pathlib.Path(LOGS_DIR)
        for p in pdir.iterdir():
            try:
                # remove regular files and simple subdirectories (be conservative)
                if p.is_file():
                    p.unlink()
                elif p.is_dir():
                    shutil.rmtree(p)
            except Exception:
                # ignore problems deleting individual files (locked, permissions, etc)
                pass
    except Exception:
        # ignore any top-level cleanup failures
        pass
    # mark cleanup done for this Streamlit session/process
    st.session_state["logs_cleaned"] = True
# ------------------------------------------------------------------------

logger = logging.getLogger("streamlit_app")
logger.propagate = False
if not logger.handlers:
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(LOGS_DIR, "streamlit_app.log"), encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(fh)

# Lightweight renderer logger for image-related diagnostics
render_logger = logging.getLogger("streamlit_renderer")
render_logger.propagate = False
if not render_logger.handlers:
    render_logger.setLevel(logging.INFO)
    fh2 = logging.FileHandler(os.path.join(LOGS_DIR, "streamlit_render.log"), encoding="utf-8")
    fh2.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    render_logger.addHandler(fh2)


# Import the ADK-backed call_agent from tutor_agent module
from src.agents.tutor_agent.agent import call_agent

REDACT_AGENT_TEXT = False  # set to True to redact large base64 blobs from agent text for UI safety

# -------------------------
# Helper: process agent tool outputs and function calls to extract images
# -------------------------
def _maybe_get_response_dict(obj):
    """Return a plain dict if obj is dict-like or has a .response attribute."""
    try:
        if obj is None:
            return None
        if isinstance(obj, dict):
            return obj
        if hasattr(obj, "response"):
            return getattr(obj, "response")
        if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
            try:
                return obj.to_dict()
            except Exception:
                pass
    except Exception:
        pass
    return None


def _decode_b64_to_bytes(b64s: str):
    try:
        return base64.b64decode(b64s)
    except Exception:
        return None


def process_agent_outputs(agent_text: str, tool_outputs_list: list, function_calls_list: list):
    """
    Inspect tool_outputs_list and function_calls_list for entries emitted by
    render_images_in_ui, decode base64 image bytes and return:

        (decoded_images_list, safe_text)

    - decoded_images_list: list of raw bytes objects for st.image(...)
    - safe_text: input agent_text with very large base64-like sequences redacted
    """
    images = []

    try:
        # -------------------------
        # 1) Inspect explicit tool outputs (preferred canonical entries)
        # -------------------------
        for entry in tool_outputs_list or []:
            try:
                if not isinstance(entry, dict):
                    continue

                tool_name = entry.get("tool_name") or entry.get("name") or ""
                tool_output = entry.get("output")

                # Normalize "response"-style wrappers (e.g., function_response.response)
                resp_wrapper = None
                try:
                    # If tool_output is an object-like with .response or is a dict with 'response'
                    if isinstance(tool_output, dict) and "response" in tool_output:
                        resp_wrapper = tool_output.get("response")
                    elif hasattr(tool_output, "response"):
                        resp_wrapper = getattr(tool_output, "response")
                except Exception:
                    resp_wrapper = None

                # If this is the load_context result (or similar) and it contains images,
                # log the presence but DO NOT decode them here.
                if (tool_name == "load_context" or (isinstance(tool_output, dict) and "function_response" in (tool_output.keys() or []))) or resp_wrapper:
                    payload = resp_wrapper if resp_wrapper else tool_output
                    if isinstance(payload, dict):
                        imgs = payload.get("images") or payload.get("render_images") or []
                        if imgs and len(imgs) > 0:
                            try:
                                render_logger.info(
                                    "Found %d image(s) in %s output (detected in function_response/load_context). Not decoding until render_images_in_ui tool is called.",
                                    len(imgs), tool_name or "unknown"
                                )
                            except Exception:
                                render_logger.info("Found images in load_context-like output (count unknown).")
                    # continue processing other entries (do not decode load_context images)
                # If this is the render_images_in_ui tool output, decode its payload now.
                if tool_name == "render_images_in_ui":
                    out = tool_output
                    if isinstance(out, dict) and "render_images" in out:
                        imgs_list = out.get("render_images") or []
                    elif isinstance(out, list):
                        imgs_list = out
                    else:
                        imgs_list = []
                        if hasattr(out, "response") and isinstance(getattr(out, "response"), dict):
                            imgs_list = getattr(out, "response").get("render_images") or []
                        elif isinstance(out, dict):
                            imgs_list = out.get("images") or out.get("render_images") or []

                    for im in imgs_list or []:
                        if not isinstance(im, dict):
                            continue
                        b64 = im.get("bytes_b64") or im.get("bytes64") or im.get("b64") or im.get("data")
                        if not b64:
                            render_logger.info("render_images_in_ui entry missing base64 string (tool_output). Skipping.")
                            continue
                        decoded = _decode_b64_to_bytes(b64)
                        if decoded:
                            images.append(decoded)
                            render_logger.info(
                                "Decoded image from tool_output (tool=%s, page=%s, size=%s)",
                                tool_name, im.get("chapter_page_no"), im.get("size_bytes")
                            )
                        else:
                            render_logger.info("Failed to decode base64 from render_images_in_ui tool_output entry")
                    continue

            except Exception:
                render_logger.exception("Error while inspecting a tool_outputs entry")

        # -------------------------
        # 2) Inspect function_calls_list for explicit render_images_in_ui calls (arguments may contain the payload)
        # -------------------------
        for fc in function_calls_list or []:
            try:
                if not isinstance(fc, dict):
                    continue
                fc_tool_name = fc.get("tool_name") or fc.get("name") or ""
                if fc_tool_name != "render_images_in_ui":
                    if fc_tool_name == "load_context":
                        args = fc.get("arguments") or fc.get("raw_arguments") or fc.get("args")
                        imgs_count = 0
                        try:
                            if isinstance(args, dict):
                                imgs = args.get("images") or args.get("response", {}).get("images") or []
                                imgs_count = len(imgs) if isinstance(imgs, list) else 0
                            elif isinstance(args, str):
                                import json as _json
                                try:
                                    parsed = _json.loads(args)
                                    imgs = parsed.get("images") or parsed.get("response", {}).get("images") or []
                                    imgs_count = len(imgs) if isinstance(imgs, list) else 0
                                except Exception:
                                    imgs_count = 0
                        except Exception:
                            imgs_count = 0
                        if imgs_count:
                            render_logger.info(
                                "Function call 'load_context' included %d image(s) in its arguments/response. Not decoding here.",
                                imgs_count
                            )
                    continue

                args = fc.get("arguments") or fc.get("raw_arguments") or fc.get("args")
                payload = None
                if isinstance(args, dict):
                    payload = args
                elif isinstance(args, str):
                    try:
                        import json as _json
                        payload = _json.loads(args)
                    except Exception:
                        payload = None

                if payload:
                    imgs = payload.get("render_images") or payload.get("images") or []
                    for im in imgs or []:
                        if not isinstance(im, dict):
                            continue
                        b64 = im.get("bytes_b64") or im.get("bytes64") or im.get("b64") or im.get("data")
                        if not b64:
                            render_logger.info("render_images_in_ui function_call arg missing base64 string. Skipping.")
                            continue
                        decoded = _decode_b64_to_bytes(b64)
                        if decoded:
                            images.append(decoded)
                            render_logger.info("Decoded image from function_call args (page=%s, size=%s)",
                                               im.get("chapter_page_no"), im.get("size_bytes"))
                        else:
                            render_logger.info("Failed to decode base64 from function_call args for render_images_in_ui")
            except Exception:
                render_logger.exception("Error while inspecting function_calls_list for render_images payloads")

    except Exception:
        render_logger.exception("Unexpected error extracting images from agent outputs")

    # Sanitize agent_text: redact large base64-like sequences for UI safety
    safe_text = re.sub(r"([A-Za-z0-9+/]{100,}={0,2})", "[image-omitted]", agent_text or "")

    txt = agent_text or ""
    if REDACT_AGENT_TEXT:
        safe_text = re.sub(r"([A-Za-z0-9+/]{1000,}={0,2})",
                       "[image-omitted]",
                       txt)
    else:
        safe_text = txt

    return images, safe_text


# -------------------------
# Initialize session state
# -------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "history" not in st.session_state:
    st.session_state.history = []  # each message: {"role","text","images","ts"}
if "agent_images" not in st.session_state:
    st.session_state.agent_images = []
if "show_uploader" not in st.session_state:
    st.session_state.show_uploader = False
if "pending_uploads" not in st.session_state:
    st.session_state.pending_uploads = []
if "awaiting_agent" not in st.session_state:
    st.session_state.awaiting_agent = False
# Ensure a module-level queue exists for worker -> main-thread handoff
if "agent_result_queue" not in st.session_state:
    st.session_state.agent_result_queue = queue.Queue()

# helper: current timestamp string
def now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# initial greeting if empty history
if not st.session_state.history:
    # Actual greeting
    greeting = "Hello! I'm a specialized mathematics tutor agent for grades 7 and 8. What's your name and grade?"
    # For debugging
    greeting_test = "Hello! How can I help you?"
    st.session_state.history.append({"role": "tutor", "text": greeting, "images": [], "ts": now_ts()})


# ---- Upload / Option B helpers (integrated) ----

# Constants for Option B
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB per upload
MAX_UPLOAD_CACHE_BYTES = 30 * 1024 * 1024  # 30 MB total for uploads in cache
UPLOAD_MAP_PREFIX = "uploadmap_"


def _make_map_id() -> str:
    return f"{UPLOAD_MAP_PREFIX}{uuid.uuid4().hex[:12]}"


def _map_image_entry_from_file(raw_bytes: bytes, mime: str = "image/png", chapter_page_no: Optional[int] = None) -> dict:
    """
    Build a single image metadata dict suitable for the _SESSION_IMAGE_CACHE/map usage.
    Tagged with source="upload".
    """
    size_bytes = len(raw_bytes)
    b64 = base64.b64encode(raw_bytes).decode("ascii")
    now = time.time()
    entry = {
        "image_index": 0,
        "artifact_name": None,  # No artifact storage in Option B
        "mime": mime,
        "chapter_page_no": chapter_page_no,
        "size_bytes": size_bytes,
        "caption": "student_solution",
        "bytes_b64_fallback": b64,
        "bytes_b64": b64,
        "source": "upload",
        "timestamp": now,
    }
    return entry


def _populate_upload_maps(map_id: str, map_obj: dict) -> None:
    """
    Store map in streamlit session_state and also attempt to populate the agent module session cache.
    """
    # store in streamlit session for UI-level access (safe local storage)
    st.session_state.setdefault("UPLOAD_MAPS", {})[map_id] = map_obj

    # also populate the shared agent session cache if the agent module is importable in-process
    try:
        from src.agents.tutor_agent import agent as _agent_mod  # guarded import
        if getattr(_agent_mod, "_SESSION_IMAGE_CACHE", None) is None:
            _agent_mod._SESSION_IMAGE_CACHE = {}
        _agent_mod._SESSION_IMAGE_CACHE[map_id] = {"images": map_obj.get("images", []), "timestamp": time.time()}
    except Exception:
        logger.debug("Agent module not importable while populating upload maps; continuing without agent cache population.")


def _evict_upload_cache_if_needed(max_bytes: int = MAX_UPLOAD_CACHE_BYTES) -> None:
    """
    Evict upload-sourced entries from the agent _SESSION_IMAGE_CACHE until upload-only total size <= max_bytes.
    Eviction policy: oldest upload maps (by their stored timestamp) removed first.
    """
    try:
        from src.agents.tutor_agent import agent as _agent_mod  # guarded import
    except Exception:
        # No agent module available; nothing to evict
        _evict_st_uploads_if_needed(max_bytes)
        return

    cache = getattr(_agent_mod, "_SESSION_IMAGE_CACHE", None)
    if not isinstance(cache, dict) or not cache:
        _evict_st_uploads_if_needed(max_bytes)
        return

    # Gather upload-sourced maps with their sizes and timestamps
    upload_entries = []
    total_upload_bytes = 0
    for mid, payload in list(cache.items()):
        imgs = payload.get("images", []) if isinstance(payload, dict) else []
        upload_imgs = [img for img in imgs if img.get("source") == "upload"]
        if not upload_imgs:
            continue
        entry_size = sum((img.get("size_bytes") or 0) for img in upload_imgs)
        entry_ts = payload.get("timestamp") or (min((img.get("timestamp") or time.time()) for img in upload_imgs) if upload_imgs else time.time())
        upload_entries.append({"map_id": mid, "size": entry_size, "timestamp": entry_ts})
        total_upload_bytes += entry_size

    # If under threshold, nothing to do
    if total_upload_bytes <= max_bytes:
        return

    # Sort entries by timestamp (oldest first), evict until under limit
    upload_entries.sort(key=lambda x: x["timestamp"])
    bytes_to_free = total_upload_bytes - max_bytes
    freed = 0
    for e in upload_entries:
        mid = e["map_id"]
        try:
            if mid in cache:
                del cache[mid]
                freed += e["size"]
            # also delete from st.session_state UPLOAD_MAPS if present
            if mid in st.session_state.get("UPLOAD_MAPS", {}):
                del st.session_state["UPLOAD_MAPS"][mid]
        except Exception:
            logger.exception("Failed to evict upload map %s", mid)
        if freed >= bytes_to_free:
            break

    logger.info("Evicted upload cache entries: freed %d bytes (threshold %d)", freed, bytes_to_free)


def _evict_st_uploads_if_needed(max_bytes: int = MAX_UPLOAD_CACHE_BYTES) -> None:
    """
    Evict upload maps stored only in st.session_state if the total upload size exceeds max_bytes.
    Useful when agent module is not importable.
    """
    uploads = st.session_state.get("UPLOAD_MAPS", {})
    if not isinstance(uploads, dict) or not uploads:
        return
    entries = []
    total = 0
    for mid, obj in uploads.items():
        imgs = obj.get("images", []) if isinstance(obj, dict) else []
        upload_imgs = [img for img in imgs if img.get("source") == "upload"]
        if not upload_imgs:
            continue
        size = sum((img.get("size_bytes") or 0) for img in upload_imgs)
        ts = min((img.get("timestamp") or time.time()) for img in upload_imgs) if upload_imgs else time.time()
        entries.append({"map_id": mid, "size": size, "timestamp": ts})
        total += size

    if total <= max_bytes:
        return

    entries.sort(key=lambda x: x["timestamp"])
    bytes_to_free = total - max_bytes
    freed = 0
    for e in entries:
        mid = e["map_id"]
        try:
            if mid in st.session_state.get("UPLOAD_MAPS", {}):
                del st.session_state["UPLOAD_MAPS"][mid]
                freed += e["size"]
        except Exception:
            logger.exception("Failed to evict st.session_state upload map %s", mid)
        if freed >= bytes_to_free:
            break

    logger.info("Evicted in-session upload maps: freed %d bytes (threshold %d)", freed, bytes_to_free)


def handle_upload_option_b_fileobj(fobj, chapter_page_no: Optional[int] = None) -> Tuple[Optional[str], Optional[str]]:
    """
    Handle the uploaded file object using Option B:
      - Validate file size (<= MAX_UPLOAD_BYTES)
      - Convert to base64 and build a small map object
      - Store in st.session_state and attempt to populate the agent's session cache
    Returns: (map_id, error_message) where error_message is None on success.
    """
    if fobj is None:
        return None, "No file provided."

    try:
        raw = fobj.read()
    except Exception as e:
        logger.exception("Failed to read uploaded file")
        return None, f"Failed to read uploaded file: {e}"

    size_bytes = len(raw)
    if size_bytes <= 0:
        return None, "Uploaded file is empty."

    if size_bytes > MAX_UPLOAD_BYTES:
        return None, f"Uploaded file too large ({size_bytes} bytes). Max allowed is {MAX_UPLOAD_BYTES} bytes."

    mime = getattr(fobj, "type", "image/jpeg")
    entry = _map_image_entry_from_file(raw, mime=mime, chapter_page_no=chapter_page_no)
    map_id = _make_map_id()
    map_obj = {"map_id": map_id, "images": [entry]}

    _populate_upload_maps(map_id, map_obj)
    _evict_upload_cache_if_needed(MAX_UPLOAD_CACHE_BYTES)

    # keep a copy in pending_uploads (raw bytes) so it attaches to next message like previous behavior
    st.session_state.pending_uploads = [raw]

    return map_id, None


# -------------------------
# Layout: make center column wider: [1,11,1]
# -------------------------
left_col, center_col, right_col = st.columns([1, 11, 1])

with center_col:

    # -------------------------
    # RED WARNING BANNER (copyright / deployment warning)
    # -------------------------
    st.markdown(
        """
<div style="
    background-color:#ffcccc;
    border-left: 6px solid #cc0000;
    padding: 12px;
    border-radius: 4px;
    margin-bottom: 20px;">
<b>⚠️ WARNING:</b> This application uses <i>locally supplied</i> NCERT textbook files.
<br>It is intended for <b>local testing and academic evaluation only</b>.
<br><b>Public deployment is strictly prohibited.</b>
</div>
""",
        unsafe_allow_html=True,
    )

    # Conversation Window (render into a reusable placeholder so we can update it mid-run)
    st.subheader("Conversation")
    chat_placeholder = st.empty()  # placeholder we can re-render into

    def render_chat():
        """Render the conversation from st.session_state.history into the placeholder."""
        with chat_placeholder.container():
            for msg in st.session_state.history:
                role = msg.get("role", "tutor")
                text = msg.get("text", "")
                images = msg.get("images", [])
                ts = msg.get("ts", "")

                display_role = "assistant" if role == "tutor" else "user"
                with st.chat_message(display_role):
                    # timestamp
                    if ts:
                        st.write(f"*{ts}*")
                    # LaTeX handling (rendered in UI)
                    latex_blocks = re.findall(r"\$\$(.*?)\$\$", text, flags=re.DOTALL)
                    text_without_latex = re.sub(r"\$\$(.*?)\$\$", " ", text, flags=re.DOTALL).strip()
                    if text_without_latex:
                        st.write(text_without_latex)
                    if latex_blocks:
                        for expr in latex_blocks:
                            try:
                                st.latex(expr.strip())
                            except Exception:
                                st.write("$$" + expr.strip() + "$$")
                    # images attached to message
                    for img in images:
                        if isinstance(img, bytes):
                            try:
                                st.image(img)
                            except Exception:
                                st.write("[image could not be displayed]")
                        elif isinstance(img, str) and img.startswith("http"):
                            try:
                                st.image(img)
                            except Exception:
                                st.write("[image url could not be displayed]")
                        else:
                            # fallback display (e.g. text caption)
                            st.write(img)

    # initial render
    render_chat()
    # scroll to bottom initially
    st.components.v1.html("<script>window.scrollTo(0, document.body.scrollHeight);</script>", height=0)


    # -------------------------
    # Student Response Window
    # -------------------------
    st.subheader("Your response")

    # Input + attach button ("+")
    input_col, attach_col = st.columns([20, 1])
    with input_col:
        quick_input = st.chat_input("Type a message and press Enter to send")
    with attach_col:
        attach_pressed = st.button("+", key="attach_button")
        if attach_pressed:
            st.session_state.show_uploader = not st.session_state.show_uploader

    # Helper lines
    st.write("_Use Enter to send. Use Shift+Enter to insert a newline._")
    st.write("_Use the + button to upload your solution as one image file (attached to next message)._")
    st.write("_Maximum upload size is 10 MB per upload. Only one upload is allowed per message._")
    st.write("_Streamlit does not allow max upload size in runtime, so it may display a higher limit._")

    # inline uploader toggled by Attach; single-file only
    if st.session_state.show_uploader:
        uploaded = st.file_uploader(
            "Upload one image file (png/jpg/jpeg) — this will be attached to your next message.",
            accept_multiple_files=False,
            type=["png", "jpg", "jpeg"],
            key="uploader_inline_single",
        )
        if uploaded:
            # handle via Option B: validate, store map, populate cache and set pending_uploads
            map_id, err = handle_upload_option_b_fileobj(uploaded, chapter_page_no=None)
            if err:
                st.error(err)
            else:
                st.success(f"Uploaded and stored locally as map_id: **{map_id}**")
                # show the uploaded image preview (use the file object again)
                try:
                    uploaded.seek(0)
                    st.image(uploaded.read(), caption=uploaded.name, use_column_width=True)
                except Exception:
                    # fallback: use pending_uploads bytes preview if available
                    if st.session_state.pending_uploads:
                        try:
                            st.image(st.session_state.pending_uploads[0], caption="Uploaded image (preview)")
                        except Exception:
                            pass

                st.write("You can now send your message (Enter). The uploaded image will be attached to the message and sent to the agent.")
                st.code(json.dumps({"role": "user", "type": "student_solution_uploaded", "map_id": map_id}))
                # Attempt to auto-notify agent if an in-process handler exists
                try:
                    from src.agents.tutor_agent import agent as _agent_mod  # guarded import
                    if hasattr(_agent_mod, "handle_student_solution_message"):
                        try:
                            _agent_mod.handle_student_solution_message({"type": "student_solution_uploaded", "map_id": map_id})
                            st.info("Agent notified (in-process handler invoked).")
                        except Exception:
                            logger.exception("Failed to call agent handler synchronously.")
                            st.warning("Agent handler invocation failed; please paste the JSON above to the agent input manually.")
                    else:
                        st.info("Agent handler not available for auto-notify. Please send the JSON above to the agent input.")
                except Exception:
                    st.info("Agent not importable in this process. Please notify the agent manually with the printed map_id.")


    # -------------------------
    # Interact with student (thread + queue handoff; safe Streamlit usage)
    # -------------------------
    def _agent_worker(user_prompt: str, user_id: str, session_id: str, result_q: "queue.Queue"):
        """
        Background thread target: call the blocking agent and PUT result into result_q.
        DO NOT call streamlit APIs from this function.
        Puts either:
          - (agent_text, new_session_id)
          - or (agent_text, new_session_id, tool_result)
          - or (agent_text, new_session_id, tool_outputs_list, function_calls_list)
        into result_q for main thread consumption.
        """
        try:
            # call_agent may return different shapes depending on tool outputs
            res = call_agent(user_prompt, user_id=user_id, session_id=session_id)
            # ensure we always put a tuple/list into the queue
            if isinstance(res, tuple) or isinstance(res, list):
                result_q.put(res)
            else:
                # single value -> treat as agent_text only
                result_q.put((str(res), session_id))
        except Exception as e:
            result_q.put((f"[Agent error] {e}", session_id))

    if quick_input:
        # Collect attachments (safe to access st here — main thread)
        attached = st.session_state.pending_uploads.copy() if st.session_state.pending_uploads else []
        # clear pending uploads so attachments don't persist across messages
        st.session_state.pending_uploads = []
        st.session_state.show_uploader = False

        # Append student's message immediately (main thread)
        st.session_state.history.append(
            {"role": "student", "text": quick_input, "images": attached, "ts": now_ts()}
        )

        # Immediately re-render chat so student's message appears right away
        render_chat()
        # scroll to bottom so student sees their message
        st.components.v1.html("<script>window.scrollTo(0, document.body.scrollHeight);</script>", height=0)

        # Build prompt snippet
        N = 12
        snippet = "\n".join(f"{m['role'].capitalize()}: {m['text']}" for m in st.session_state.history[-N:])
        user_prompt = snippet

        # Mark waiting state
        st.session_state.awaiting_agent = True

        # Start worker thread; pass the queue object (do NOT pass st.session_state into thread)
        worker = threading.Thread(
            target=_agent_worker,
            args=(user_prompt, "student", st.session_state.session_id, st.session_state.agent_result_queue),
            daemon=True,
        )
        worker.start()

        # Main thread: show spinner and poll the result queue
        with st.spinner("Agent thinking..."):
            # poll until a result arrives
            while st.session_state.get("awaiting_agent", False):
                try:
                    item = st.session_state.agent_result_queue.get(timeout=0.12)

                    # Normalize the returned item into (agent_text, new_session_id, tool_outputs_list, function_calls_list)
                    agent_text = ""
                    new_session_id = st.session_state.session_id
                    tool_outputs_list = []
                    function_calls_list = []

                    if isinstance(item, (list, tuple)):
                        if len(item) == 1:
                            agent_text = str(item[0])
                        elif len(item) == 2:
                            agent_text, new_session_id = item
                        elif len(item) == 3:
                            # legacy 3-tuple: (agent_text, session_id, tool_outputs_list)
                            agent_text, new_session_id, tool_part = item[0], item[1], item[2]
                            if isinstance(tool_part, list):
                                tool_outputs_list = tool_part
                            elif tool_part is None:
                                tool_outputs_list = []
                            else:
                                tool_outputs_list = [tool_part]
                        else:
                            # len >= 4: (agent_text, session_id, tool_outputs_list, function_calls_list)
                            agent_text = item[0]
                            new_session_id = item[1]
                            tool_part = item[2]
                            func_part = item[3] if len(item) >= 4 else None

                            # normalize tool_part
                            if isinstance(tool_part, list):
                                tool_outputs_list = tool_part
                            elif tool_part is None:
                                tool_outputs_list = []
                            else:
                                tool_outputs_list = [tool_part]

                            # normalize func_part
                            if isinstance(func_part, list):
                                function_calls_list = func_part
                            elif func_part is None:
                                function_calls_list = []
                            else:
                                function_calls_list = [func_part]
                    else:
                        # unexpected shape: treat as raw agent_text
                        agent_text = str(item)

                    # update session id
                    st.session_state.session_id = new_session_id

                    # Use helper to extract images and sanitized agent text
                    try:
                        tutor_images, safe_text = process_agent_outputs(agent_text, tool_outputs_list, function_calls_list)
                    except Exception:
                        render_logger.exception("Error in process_agent_outputs helper")
                        tutor_images = []

                        txt = agent_text or ""
                        if REDACT_AGENT_TEXT:
                            # Conservative redaction: only remove extremely long base64-like blobs (1000+ chars)
                            safe_text = re.sub(r"([A-Za-z0-9+/]{1000,}={0,2})",
                                            "[image-omitted]",
                                            txt)
                        else:
                            safe_text = txt
                        
                    # Append tutor reply to history including any images decoded
                    st.session_state.history.append(
                        {"role": "tutor", "text": safe_text, "images": tutor_images, "ts": now_ts()}
                    )

                    # clear awaiting flag and re-render
                    st.session_state.awaiting_agent = False
                    render_chat()
                    # scroll to bottom so student sees the reply
                    st.components.v1.html("<script>window.scrollTo(0, document.body.scrollHeight);</script>", height=0)

                except queue.Empty:
                    # no result yet; continue polling
                    continue
                except Exception:
                    logger.exception("Streamlit main loop exception while handling agent result")
                    st.session_state.awaiting_agent = False
                    break

    # -------------------------
    # Transcript (PDF with WeasyPrint)
    # -------------------------
    st.markdown("---")
    st.subheader("Transcript")
    st.write(
        "Download the conversation transcript (PDF). This preserves Markdown formatting including lists, bold text, and newlines."
    )

    def build_transcript_markdown(history):
        parts = []
        for m in history:
            ts = m.get("ts", "")
            role = m.get("role", "tutor").capitalize()
            text = m.get("text", "")
            # if images present, note them in transcript text (images are not embedded in PDF)
            imgs = m.get("images", [])
            if imgs:
                text = text + "\n\n" + f"_[{len(imgs)} image(s) attached]_"
            parts.append(f"### [{ts}] {role}\n\n{text}\n\n")
        return "\n".join(parts)

    transcript_md = build_transcript_markdown(st.session_state.history)

    def markdown_to_html(md_text: str) -> tuple[str, str]:
        html_body = md.markdown(md_text, extensions=["extra", "sane_lists", "nl2br"])
        css = """
        body { font-family: 'Helvetica', Arial, sans-serif; margin: 30px; font-size: 12pt; color: #111; }
        h1, h2, h3 { color: #0b4f6c; }
        pre { background: #f4f4f4; padding: 8px; border-radius: 4px;}
        code { background: #f4f4f4; padding: 2px 4px; border-radius: 3px; }
        ul { margin-left: 20px; }
        ol { margin-left: 20px; }
        """
        full = f"<html><head><meta charset='utf-8'></head><body>{html_body}</body></html>"
        return full, css

    html_body, css_text = markdown_to_html(transcript_md)

    def make_pdf_bytes_weasy(html: str, css_text: str) -> bytes:
        css = CSS(string=css_text)
        pdf = HTML(string=html).write_pdf(stylesheets=[css])
        return pdf

    pdf_bytes = make_pdf_bytes_weasy(html_body, css_text)

    st.download_button(
        label="Download Transcript (PDF)",
        data=pdf_bytes,
        file_name="tutor_transcript.pdf",
        mime="application/pdf",
    )

    st.markdown("---")

    # Quit Session button
    if st.button("Quit Session", key="quit_session"):
        os._exit(0)

    st.caption(f"session_id: {st.session_state.session_id}")

# end center column

# end file
