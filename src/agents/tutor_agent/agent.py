# src/agents/tutor_agent/agent.py
# Ensure environment variables from .env are loaded BEFORE ADK/GenAI clients are created.
from dotenv import load_dotenv
import os
load_dotenv()  # loads GOOGLE_API_KEY from .env into environment variables, if present

"""
Minimal ADK LlmAgent with function-tool registrations for 
`load_context` and `render_images_in_ui` (hybrid session-cache approach).
"""

from typing import Optional, Tuple
import uuid
import inspect
import asyncio
import tempfile
import base64

# Logging + pretty-print for debugging runner events
import logging
from pprint import pformat, pprint

# ensure logs directory exists
try:
    os.makedirs("logs", exist_ok=True)
except Exception:
    pass

logger = logging.getLogger("tutor_agent")
# avoid double-propagation to root handlers
logger.propagate = False
if not logger.handlers:
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler("logs/tutor_agent_events.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(fh)

# small convenience to also echo compact info to stdout for quick debugging
def _log_event_brief(event):
    """Write a short event summary to stdout for quick debugging."""
    try:
        name = getattr(event, "name", None) or getattr(event, "type", None)
        # print a one-line summary to terminal
        print(f"[ADK EVENT] {name} - attrs: {list(getattr(event, '__dict__', {}).keys())}")
    except Exception:
        print("[ADK EVENT] summary print failed")


from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# new import for artifact service
from google.adk.artifacts import InMemoryArtifactService

#-------------------------------
# Import the function tools
#-------------------------------
# load_context tool
from src.tools.load_context import load_context

# Old tool for rendering images in the UI.
# from src.tools.render_images_in_conversation_window import render_images_in_conversation_window

# New hybrid UI tool
from src.tools.render_images_in_ui import render_images_in_ui

# -------------------------
# Prompt for the tutor agent
# -------------------------
SYSTEM_INSTRUCTION = """You are tutor agent — an intelligent, calm, empathetic, encouraging, step-by-step virtual tutor for school mathematics problems of grades 7 and 8. 
You must follow these instructions precisely.

1. Start the conversation by greeting the student warmly and asking for their name and grade.

2. Ask the student which question they need help with. 
    You are only authorized to help students with questions that appear in the “Figure it Out” sections of their textbook.  
    - If the student asks about any other type of question, politely decline:  
     “Sorry, I am not authorized to help you for any other type of questions.”

3. If the question belongs to a Figure it Out section, ask the student for:
   - chapter number  
   - page number (either chapter page number or full-book printed page number)  
   - exercise number  

   Once these are provided, call the tool `load_context` with these parameters and the grade exactly.
   (Note: If the student provides a page number ambigously, you may ask them to clarify whether it is the chapter page number or the printed book page number.
   You may also make an intelligent guess based on typical chapter and book page number ranges.)  
   The tool will return:  
   - the problem text  
   - any associated images  
   - metadata including a short `map_id` that indexes the images

4. Wait until the `load_context` tool returns. 
    When the tool result is available, first intelligently extract the full problem text (ignoring irrelevant text) from the `problem_text` attribute returned by the tool. 
    Then clearly state the extracted problem so the student can read it.

5. Next, call the `render_images_in_ui` tool exactly once, passing the `map_id` returned by `load_context` and the list of image indices to display (i.e. more precisely, the list of the image_index attributes (int) for each entry in the "images" key). 
    Example call:
       render_images_in_ui(map_id=<map_id>, indices=[0,1])
    Immediately after calling `render_images_in_ui`, inform the student that the images are being displayed.
    **Important:** You must display images returned by the `load_context` tool ONLY by calling `render_images_in_ui`. You must NOT embed any image bytes or base64 into your generated text responses for the student.

6. Ask the student to look at the extracted problem text and the displayed images, and confirm whether this is the correct question. 
    **You MUST wait for an explicit confirmation token from the student.** 
    The student must reply with exactly the word `YES` (case-insensitive) to confirm (acceptable replies: `YES`, `Yes`, `yes`). 
    Do NOT proceed to any tutoring steps unless the student replies exactly with this confirmation token. Do not accept any other tokens or freeform confirmations (e.g., "y", "ok", "sure", "that is it").

7. If the student does not confirm, ask them for corrected details and call `load_context` again with the new parameters. 
    Repeat this loop until the question is correctly confirmed.

8. Once the question is confirmed with the `YES` token, intelligently use the images referenced by the `map_id` and the confirmed problem text to begin your tutoring session.

9. Begin your tutoring session strictly in Socratic mode.
   - Never give away any answer directly at any step.
   - Always probe the student with questions that help them produce the reasoning themselves.
   - Adapt difficulty to the student’s responses.
   - If the student attaches an image of their attempted solution, examine it carefully and respond with constructive, gentle feedback on their approach, identifying any mistakes or misunderstandings and guiding them through corrective reasoning.


10. You must end the tutoring session only when one of the following occurs:
   A. The student arrives at the final answers, expresses understanding, and says they have no further questions about this problem.
   B. The student indicates they want to stop or no longer need help.

11. If the student asks a question about a different Figure it Out problem in the middle of an already started session on a confirmed problem, tell them politely that this will count as a new tutoring session and they must quit the current session to start a new one.

12. Never assist with non–Figure it Out problems and never invent or hallucinate exercises. Only use the problem text and images returned by load_context.
"""

# For debugging
TEST_INSTRUCTION_1 = """
For the next user message:

1. Immediately call the `load_context` tool with the EXACT parameters given by the user.
2. After the tool returns, return ONLY the `problem_text` attribute.
3. Do NOT generate any extra text.
"""

# For debugging: updated to require the new render_images_in_ui tool
TEST_INSTRUCTION = """
You are running a strict tool-integration test. Follow these steps exactly and then STOP.

1) Call the tool function **by name**: 
    load_context(grade=8, chapter_id=7, chapter_page_number=12, exercise_number=2). 
    Wait for that tool call to return.

2) From the returned object from load_context, extract ONLY and store (do not print):
   - problem_text (string)
   - exercise_number (int or null)
   - chapter_page_no (int or null)
   - book_page_no (int or null)
   - a list of the image_index attributes (int) for each entry in the "images" key
   - map_id (the short id returned by load_context to reference images)

3) Immediately and explicitly call the tool function **by name**:
   render_images_in_ui(map_id=<map_id>, indices=<the list of the image_index attributes from step 2>)
   Wait until that tool call returns. Do not attempt to inline images in text; do not include base64 bytes.

4) After both tool calls have completed, output exactly two lines ONLY (no additional text, no explanation):
   Line 1: a single-line JSON object with keys {"problem_text","exercise_number","chapter_page_no","book_page_no","load_images_count","map_id"} where "load_images_count" is the length of the list of the image_index attributes (int) for each entry in the "images" key.
   Line 2: a single-line JSON object with keys {"render_images_returned_count","render_images_returned_meta"} where "render_images_returned_count" is the length of the list of dictionaries returned by the render_images_in_ui tool and "render_images_returned_meta" is a list of the sub-dictionaries {id, chapter_page_no, mime, size_bytes} returned by the render_images_in_ui tool.

End. Do not generate any other content. The tool names are exact and must be used as shown.
"""




# -------------------------
# Create LlmAgent and register tools
# -------------------------
# ADK will automatically wrap native Python functions as FunctionTools when you pass them in `tools`.
_AGENT = LlmAgent(
    model="gemini-2.5-flash",           # adjust model name if desired
    name="tutor_agent",
    description="Tutors middle school students in mathematics (grades 7-8).",
    instruction=SYSTEM_INSTRUCTION,
    tools=[
        load_context,
        render_images_in_ui,
        # legacy approach (commented out): render_images_in_conversation_window
        # render_images_in_conversation_window,
    ],               
)
logger.info("DEBUG: registered tools for agent: %s", [ getattr(t, "name", str(t)) for t in _AGENT.tools ])


# -------------------------
# Runner & Session service
# -------------------------
_SESSION_SERVICE = InMemorySessionService()

# Create an Artifact service for storing artifacts (images/files). InMemory is fine for local dev.
_ARTIFACT_SERVICE = InMemoryArtifactService()

# Pass artifact_service into Runner so tools can call tool_context.save_artifact/load_artifact
_RUNNER = Runner(
    agent=_AGENT,
    app_name="tutor_app",
    session_service=_SESSION_SERVICE,
    artifact_service=_ARTIFACT_SERVICE,   
)

# Session-image cache for the hybrid approach (maps map_id -> {timestamp, images})
# Tools or runner may populate this if artifact service is unavailable. Keys are map_id strings.
_SESSION_IMAGE_CACHE: dict = {}

# -------------------------
# Minimal handler: process student-solution messages (UI -> agent)
# -------------------------
def handle_student_solution_message(message: dict) -> dict:
    """
    Minimal synchronous handler to process a student-solution uploaded via the UI.

    Expected input:
        {"type": "student_solution_uploaded", "map_id": "<map_id>", ...}

    Behavior:
      - Calls render_images_in_ui(map_id=..., indices=[0]) to fetch the normalized
        render payload (tool output shape: {"render_images": [...]})
      - Decodes the first image's bytes and writes a temporary file
      - Returns a small dict acknowledging the processed upload with temp file path and metadata

    This function is intentionally synchronous so UI code that imports the agent module
    can call it directly (best-effort). It uses asyncio.run where appropriate.
    """
    map_id = None
    try:
        map_id = message.get("map_id")
    except Exception:
        pass

    if not map_id:
        return {"status": "error", "reason": "missing map_id"}

    # Call render_images_in_ui (async-capable) and wait for result synchronously
    try:
        coro = render_images_in_ui(map_id=map_id, indices=[0], tool_context=None)
        if asyncio.iscoroutine(coro):
            try:
                # Try to run in existing loop if none running
                loop = None
                try:
                    loop = asyncio.get_event_loop()
                except Exception:
                    loop = None

                if loop and loop.is_running():
                    # If an event loop is already running (rare in sync module), run coroutine in a thread-safe manner
                    fut = asyncio.run_coroutine_threadsafe(coro, loop)
                    out = fut.result(timeout=15)
                else:
                    out = asyncio.run(coro)
            except Exception as e:
                # Fallback: try creating a new event loop
                try:
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    out = new_loop.run_until_complete(coro)
                    new_loop.close()
                except Exception as e2:
                    return {"status": "error", "reason": f"render_images_in_ui invocation failed: {e2}"}
        else:
            out = coro
    except Exception as e:
        return {"status": "error", "reason": f"render_images_in_ui call failed: {e}"}

    if not isinstance(out, dict):
        return {"status": "error", "reason": "render_images_in_ui returned unexpected type"}

    imgs = out.get("render_images", []) or out.get("images", [])
    if not imgs:
        return {"status": "error", "reason": "no images found in render_images_in_ui result"}

    first = imgs[0]
    b64 = first.get("bytes_b64") or first.get("bytes") or first.get("data")
    # if bytes are not base64 string, try other fields
    if isinstance(b64, (bytes, bytearray)):
        raw = bytes(b64)
    else:
        try:
            if isinstance(b64, str):
                raw = base64.b64decode(b64)
            else:
                raw = None
        except Exception as e:
            return {"status": "error", "reason": f"base64 decode failed: {e}"}

    if not raw:
        return {"status": "error", "reason": "failed to obtain image bytes from render payload"}

    try:
        fd, tmp_path = tempfile.mkstemp(prefix="student_upload_", suffix=".png")
        os.write(fd, raw)
        os.close(fd)
    except Exception as e:
        return {"status": "error", "reason": f"failed to write temp file: {e}"}

    # Return minimal metadata; do not return raw bytes
    return {
        "status": "ok",
        "map_id": map_id,
        "temp_path": tmp_path,
        "mime": first.get("mime"),
        "size_bytes": first.get("size_bytes") or len(raw),
    }

def _create_session_if_needed(user_id: str = "student", session_id: Optional[str] = None) -> str:
    """
    Ensure a session exists. If session_id is provided, return it.
    Create a new session otherwise. ADK's create_session may be async in some versions,
    so we attempt to call it safely.
    """
    if session_id:
        return session_id

    session_id = str(uuid.uuid4())

    # create session; handle both sync and async create_session implementations
    try:
        create_fn = getattr(_SESSION_SERVICE, "create_session", None)
        if create_fn is None:
            # If no create_session exposed (unlikely), just return new id
            return session_id

        if inspect.iscoroutinefunction(create_fn):
            # Run async create_session in a new event loop if possible
            try:
                asyncio.run(create_fn(app_name="tutor_app", user_id=user_id, session_id=session_id))
            except RuntimeError:
                # If an event loop is already running, schedule it via create_task and wait briefly
                coro = create_fn(app_name="tutor_app", user_id=user_id, session_id=session_id)
                loop = asyncio.get_event_loop()
                loop.run_until_complete(coro)
        else:
            # sync call
            create_fn(app_name="tutor_app", user_id=user_id, session_id=session_id)
    except Exception:
        # If session creation fails for any reason, return the generated id — Runner may still accept it.
        pass

    return session_id

def call_agent(
    user_text: str,
    user_id: str = "student",
    session_id: Optional[str] = None,
):
    """
    Send user_text to the ADK LlmAgent using Runner.run and return a 4-tuple:
        (agent_text, session_id, tool_outputs_list, function_calls_list)

    - agent_text: best textual agent reply captured (str)
    - session_id: session identifier used/created for the run (str)
    - tool_outputs_list: list of dicts recording tool outputs, each:
        {"tool_name": str, "output": Any, "timestamp": ISO8601 str}
    - function_calls_list: list of dicts recording model function-call requests, each:
        {"tool_name": str, "arguments": Any, "raw_arguments": str, "timestamp": ISO8601 str}

    The function iterates ADK runner events and collects:
      * function_call requests emitted by the model (so UI or logs can inspect intent)
      * tool outputs returned by the runner (so UI can render images / content)
      * a small per-event diagnostic that is written to logs/event_attrs.json
      * when a function_response object is seen, write a recursive attribute dump to a JSON file.
    """
    import json
    import time
    from datetime import datetime

    # Ensure a session exists
    session_id = _create_session_if_needed(user_id=user_id, session_id=session_id)

    # Build content object expected by Runner.run
    content = types.Content(role="user", parts=[types.Part(text=user_text)])

    # main text collection: accumulate parts then join at the end
    agent_text_parts = []
    tool_outputs_list = []
    function_calls_list = []

    # Diagnostic capture of per-event attribute names and small previews
    events_debug = []

    # Serializer helper: recursively convert objects to JSON-safe dict up to `max_depth`
    def _serialize_obj(obj, max_depth=3, _seen=None):
        """
        Recursively serialize `obj` into JSON-serializable structure.
        Stops at max_depth and avoids recursion via _seen set.
        """
        if _seen is None:
            _seen = set()

        def _short_str(s, limit=1000):
            if not isinstance(s, str):
                return s
            if len(s) > limit:
                return f"<str {len(s)} chars - truncated to {limit}> " + s[:limit]
            return s

        try:
            # primitive types
            if obj is None or isinstance(obj, (bool, int, float)):
                return obj
            if isinstance(obj, str):
                return _short_str(obj, limit=1000)
            if isinstance(obj, (list, tuple)):
                if max_depth <= 0:
                    return f"<list len={len(obj)}>"
                out = []
                for it in obj:
                    out.append(_serialize_obj(it, max_depth=max_depth-1, _seen=_seen))
                return out
            if isinstance(obj, dict):
                if max_depth <= 0:
                    return {"_dict_len": len(obj)}
                od = {}
                for k, v in list(obj.items())[:200]:  # cap dict items to avoid huge dumps
                    try:
                        key = str(k)
                    except Exception:
                        key = repr(k)
                    od[key] = _serialize_obj(v, max_depth=max_depth-1, _seen=_seen)
                if len(obj) > 200:
                    od["_truncated_items"] = len(obj) - 200
                return od

            # prevent cycles
            oid = id(obj)
            if oid in _seen:
                return f"<circular_ref id={oid}>"
            _seen.add(oid)

            # objects with __dict__
            if hasattr(obj, "__dict__"):
                if max_depth <= 0:
                    return {"_class": obj.__class__.__name__}
                d = {}
                for k, v in list(getattr(obj, "__dict__", {}).items())[:200]:
                    if callable(v) or k.startswith("_"):
                        continue
                    try:
                        d[k] = _serialize_obj(v, max_depth=max_depth-1, _seen=_seen)
                    except Exception:
                        d[k] = repr(v)
                return {"_class": obj.__class__.__name__, "_attrs": d}

            # fallback: inspect non-callable public attrs via dir()
            if max_depth <= 0:
                return {"_repr": repr(obj)}
            summary = {"_class": obj.__class__.__name__}
            try:
                attrs = [a for a in dir(obj) if not a.startswith("_")]
                cnt = 0
                attr_map = {}
                for a in attrs:
                    if cnt >= 200:
                        attr_map["_truncated_attrs"] = len(attrs) - 200
                        break
                    try:
                        val = getattr(obj, a)
                    except Exception:
                        cnt += 1
                        continue
                    if callable(val):
                        continue
                    try:
                        attr_map[a] = _serialize_obj(val, max_depth=max_depth-1, _seen=_seen)
                    except Exception:
                        attr_map[a] = repr(val)
                    cnt += 1
                summary["_attrs"] = attr_map
            except Exception:
                summary["_repr"] = repr(obj)
            return summary
        except Exception as e:
            try:
                return {"_error": repr(e)}
            except Exception:
                return {"_error": "serialize failure"}

    # ====== Auto-render fallback helper (optional) ======
    # Set this flag to True to enable the fallback. Keep False to disable.
    ENABLE_AUTO_RENDER = False

    def _auto_invoke_render_for_loadctx(tool_outputs_list, function_calls_list, timeout_sec: int = 10):
        """
        If the most recent load_context tool output (in tool_outputs_list) contains
        an 'images' list and the model did NOT propose render_images_in_ui,
        call the local render_images_in_ui(...) tool function to produce
        a normalized 'render_images' payload and append it to tool_outputs_list.

        This is a best-effort fallback for debugging/robustness; it does not alter
        the model's function_calls_list. Use ENABLE_AUTO_RENDER to toggle.
        """
        if not ENABLE_AUTO_RENDER:
            return None

        try:
            # If model already proposed a render call, do nothing.
            if any(fc.get("tool_name") == "render_images_in_ui" for fc in function_calls_list):
                return None

            # Find the most recent load_context tool output
            candidate = None
            for entry in reversed(tool_outputs_list):
                if entry.get("tool_name") == "load_context":
                    candidate = entry
                    break
            if not candidate:
                return None

            out = candidate.get("output")
            imgs = None

            # Extract images from common shapes: dict or FunctionResponse-like object
            try:
                if isinstance(out, dict):
                    imgs = out.get("images") or out.get("render_images")
                else:
                    # If runner stored a FunctionResponse-like object, try common attrs
                    if hasattr(out, "response"):
                        try:
                            resp = getattr(out, "response")
                            if isinstance(resp, dict):
                                imgs = resp.get("images") or resp.get("render_images")
                        except Exception:
                            pass
                    # also handle object with 'output' or 'output' key
                    if imgs is None and hasattr(out, "output"):
                        try:
                            oo = getattr(out, "output")
                            if isinstance(oo, dict):
                                imgs = oo.get("images") or oo.get("render_images")
                        except Exception:
                            pass
            except Exception:
                imgs = None

            if not imgs or not isinstance(imgs, (list, tuple)) or len(imgs) == 0:
                return None

            # Call the local render_images_in_ui tool (async aware)
            import asyncio, inspect, time as _time
            logger.info("DEBUG: Auto-invoking render_images_in_ui (fallback).")
            rr = None
            try:
                coro = render_images_in_ui(map_id=None, indices=[i for i in range(len(imgs))], tool_context=None)
                if inspect.isawaitable(coro):
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            # run in event loop thread-safe; wait up to timeout_sec seconds
                            fut = asyncio.run_coroutine_threadsafe(coro, loop)
                            rr = fut.result(timeout=timeout_sec)
                        else:
                            rr = loop.run_until_complete(coro)
                    except Exception:
                        # fallback to asyncio.run (may re-create loop)
                        try:
                            rr = asyncio.run(coro)
                        except Exception:
                            rr = None
                else:
                    rr = coro
            except Exception:
                logger.exception("DEBUG: auto-render call failed when executing coroutine")
                rr = None

            # If we got a normalized render payload, append a synthetic tool_output entry
            if isinstance(rr, dict):
                ts_local = (_time.datetime.utcnow().isoformat() + "Z") if hasattr(_time, "datetime") else None
                tool_outputs_list.append({
                    "tool_name": "render_images_in_ui",
                    "output": rr,
                    "timestamp": ts_local,
                    "source_attr": "auto_invoke_fallback",
                })
                logger.info("DEBUG: Auto-invoke produced render_images payload with %d items", len(rr.get("render_images", [])))
                # return the payload for caller if desired
                return rr

        except Exception:
            logger.exception("DEBUG: _auto_invoke_render_for_loadctx encountered an error")
        return None
    # ====== end helper ======
    
    # Iterate over events produced by the ADK Runner
    try:
        for event in _RUNNER.run(user_id=user_id, session_id=session_id, new_message=content):
            # Generic timestamp for this event
            ts = datetime.utcnow().isoformat() + "Z"

            # Keep the previous brief console trace behavior
            try:
                _log_event_brief(event)
            except Exception:
                pass

            # --------------------------
            # Diagnostic: record attribute names and compact part summaries
            # --------------------------
            try:
                ev_attr_names = sorted(set(list(getattr(event, "__dict__", {}).keys()) + [
                    k for k in dir(event) if not k.startswith("_") and not callable(getattr(event, k))
                ]))
            except Exception:
                ev_attr_names = []

            parts_summary = []
            try:
                cont = getattr(event, "content", None)
                if cont and getattr(cont, "parts", None):
                    for p in cont.parts:
                        p_summary = {}
                        try:
                            p_summary["part_attrs"] = list(getattr(p, "__dict__", {}).keys())
                        except Exception:
                            p_summary["part_attrs"] = []
                        try:
                            if hasattr(p, "function_call") and getattr(p, "function_call") is not None:
                                fc = getattr(p, "function_call")
                                p_summary["function_call_name"] = getattr(fc, "name", None)
                                args = getattr(fc, "arguments", None)
                                if isinstance(args, str):
                                    p_summary["function_call_args_len"] = len(args)
                                else:
                                    p_summary["function_call_args_type"] = type(args).__name__
                        except Exception:
                            pass
                        try:
                            if hasattr(p, "inline_data") and getattr(p, "inline_data", None) is not None:  # slight safe-guard
                                inline = getattr(p, "inline_data")
                                p_summary["inline_mime"] = getattr(inline, "mime_type", None)
                                data = getattr(inline, "data", None)
                                if isinstance(data, str):
                                    p_summary["inline_data_b64_len"] = len(data)
                        except Exception:
                            pass
                        try:
                            if hasattr(p, "file_data") and getattr(p, "file_data", None) is not None:
                                fd = getattr(p, "file_data")
                                p_summary["file_mime"] = getattr(fd, "mime_type", None)
                                data = getattr(fd, "data", None)
                                if isinstance(data, str):
                                    p_summary["file_data_b64_len"] = len(data)
                        except Exception:
                            pass
                        parts_summary.append(p_summary)
            except Exception:
                parts_summary = ["(error summarizing parts)"]

            events_debug.append({
                "ts": ts,
                "attr_names": ev_attr_names,
                "content_parts_summary": parts_summary,
            })

            # --------------------------
            # 1) Capture explicit tool outputs (preferred canonical attributes)
            # --------------------------
            try:
                tool_name = getattr(event, "tool_name", None)
                tool_output_val = None
                source_attr = None
                for attr in ("tool_output", "tool_result", "tool_response", "tool_return"):
                    if hasattr(event, attr):
                        source_attr = attr
                        tool_output_val = getattr(event, attr)
                        break

                if tool_name is not None and tool_output_val is not None:
                    entry = {
                        "tool_name": tool_name,
                        "output": tool_output_val,
                        "timestamp": ts,
                        "source_attr": source_attr,
                    }
                    try:
                        entry["repr"] = pformat(tool_output_val)
                    except Exception:
                        entry["repr"] = repr(tool_output_val)
                    tool_outputs_list.append(entry)
                    try:
                        logger.info("[tool_output] %s @ %s (source=%s)", tool_name, ts, source_attr)
                        if tool_name in ("load_context", "render_images_in_ui"):
                            logger.info("Tool output details (repr): %s", entry.get("repr"))
                    except Exception:
                        logger.info("[tool_output] %s @ %s (details unavailable)", tool_name, ts)
                    for h in logger.handlers:
                        try:
                            h.flush()
                        except Exception:
                            pass
            except Exception:
                logger.exception("Failed to record explicit tool_output event.")

            # --------------------------
            # 2) Inspect event.content.parts for function_call, inline/file data, assistant text,
            #    and function_response which contains the tool's returned payload.
            #    Capture any textual output produced by the model into agent_text_parts.
            # --------------------------
            try:
                cont = getattr(event, "content", None)
                if cont and getattr(cont, "parts", None):
                    for part in cont.parts:
                        # function_call extraction
                        try:
                            if hasattr(part, "function_call") and getattr(part, "function_call") is not None:
                                fc = getattr(part, "function_call")
                                fc_name = getattr(fc, "name", None)
                                fc_args_raw = getattr(fc, "arguments", None)
                                fc_args_parsed = None
                                if isinstance(fc_args_raw, str):
                                    try:
                                        fc_args_parsed = json.loads(fc_args_raw)
                                    except Exception:
                                        fc_args_parsed = fc_args_raw
                                else:
                                    fc_args_parsed = fc_args_raw

                                function_calls_list.append({
                                    "tool_name": fc_name,
                                    "arguments": fc_args_parsed,
                                    "raw_arguments": fc_args_raw,
                                    "timestamp": ts,
                                })
                                logger.info("DEBUG: model function_call proposal: %s", fc_name)
                                logger.info(f"[function_call] {fc_name} @ {ts} (from content.parts)")
                                logger.debug("Function call details: %s", {"name": fc_name, "args": fc_args_parsed})
                        except Exception:
                            logger.exception("Failed while extracting function_call from content.parts.")

                        # capture function_response if present on the part and dump its attributes
                        try:
                            if hasattr(part, "function_response") and getattr(part, "function_response") is not None:
                                fr = getattr(part, "function_response")
                                # append synthetic tool_outputs entry (for downstream)
                                fr_entry = {
                                    "tool_name": getattr(fr, "tool_name", None) or getattr(fr, "name", None) or "function_response",
                                    "output": fr,
                                    "timestamp": ts,
                                    "source_attr": "content.parts.function_response",
                                }
                                try:
                                    fr_entry["repr"] = pformat(fr)
                                except Exception:
                                    fr_entry["repr"] = repr(fr)
                                tool_outputs_list.append(fr_entry)
                                logger.info("[tool_output] function_response captured (content.parts) @ %s", ts)

                                # --- diagnostic dump: serialize function_response up to depth 3 ---
                                try:
                                    dump = _serialize_obj(fr, max_depth=6)
                                    fname = f"logs/function_response_dump_{int(time.time()*1000)}.json"
                                    os.makedirs("logs", exist_ok=True)
                                    with open(fname, "w", encoding="utf-8") as ef:
                                        json.dump(dump, ef, indent=2)
                                    logger.info("Wrote function_response diagnostic to %s", fname)
                                    for h in logger.handlers:
                                        try:
                                            h.flush()
                                        except Exception:
                                            pass
                                    # optional auto-invoke fallback (debug/robustness)
                                    _auto_invoke_render_for_loadctx(tool_outputs_list, function_calls_list)
                                except Exception:
                                    logger.exception("Failed to write function_response diagnostic (content.parts).")
                        except Exception:
                            logger.exception("Failed while extracting function_response from content.parts.")

                        # inline_data/file_data -> synthetic tool output entries
                        try:
                            if hasattr(part, "inline_data") and getattr(part, "inline_data") is not None:
                                inline = getattr(part, "inline_data")
                                inline_summary = {
                                    "mime_type": getattr(inline, "mime_type", None),
                                }
                                data = getattr(inline, "data", None)
                                if isinstance(data, str):
                                    inline_summary["data_b64_len"] = len(data)
                                tool_outputs_list.append({
                                    "tool_name": "inline_data",
                                    "output": inline_summary,
                                    "raw": inline,
                                    "timestamp": ts,
                                })
                                logger.info("[inline_data] captured in content.parts @ %s", ts)

                            if hasattr(part, "file_data") and getattr(part, "file_data") is not None:
                                fd = getattr(part, "file_data")
                                fd_summary = {
                                    "mime_type": getattr(fd, "mime_type", None),
                                }
                                data = getattr(fd, "data", None)
                                if isinstance(data, str):
                                    fd_summary["data_b64_len"] = len(data)
                                tool_outputs_list.append({
                                    "tool_name": "file_data",
                                    "output": fd_summary,
                                    "raw": fd,
                                    "timestamp": ts,
                                })
                                logger.info("[file_data] captured in content.parts @ %s", ts)
                        except Exception:
                            logger.exception("Failed while inspecting content.parts for inline/file data.")

                        # textual assistant parts: append the observed text chunk
                        try:
                            if hasattr(part, "text") and part.text:
                                # Append text chunk preserving order
                                try:
                                    agent_text_parts.append(part.text.strip())
                                except Exception:
                                    # fallback: convert to str then append
                                    agent_text_parts.append(str(part.text).strip())
                                logger.debug("Appended agent text chunk (len=%d)", len(agent_text_parts[-1]))
                        except Exception:
                            logger.exception("Failed while extracting agent text from content.parts.")

            except Exception:
                logger.exception("Failed to inspect event.content.parts for function_call parts.")

            # --------------------------
            # 3) Inspect event.candidates[*].content.parts for function_call (Patch 1)
            #    and for candidate-embedded tool outputs if present. Also capture
            #    candidate-level function_response if present and dump it.
            #    Also extract candidate textual content into agent_text_parts if present.
            # --------------------------
            try:
                candidates = getattr(event, "candidates", None)
                if candidates:
                    for cand in candidates:
                        try:
                            # If candidate has top-level text attribute, append it
                            try:
                                cand_top_text = getattr(cand, "text", None)
                                if isinstance(cand_top_text, str) and cand_top_text.strip():
                                    agent_text_parts.append(cand_top_text.strip())
                                    logger.debug("Appended agent text from candidate.text (len=%d)", len(cand_top_text.strip()))
                            except Exception:
                                pass

                            cand_cont = getattr(cand, "content", None)
                            # candidate-embedded parts: function_call and file/inline data
                            if cand_cont and getattr(cand_cont, "parts", None):
                                for part in cand_cont.parts:
                                    # candidate textual parts: append
                                    try:
                                        if hasattr(part, "text") and part.text:
                                            try:
                                                agent_text_parts.append(part.text.strip())
                                            except Exception:
                                                agent_text_parts.append(str(part.text).strip())
                                            logger.debug("Appended agent text from candidate.content.parts (len=%d)", len(agent_text_parts[-1]))
                                    except Exception:
                                        pass

                                    # function_call inside candidate
                                    try:
                                        if hasattr(part, "function_call") and getattr(part, "function_call") is not None:
                                            fc = getattr(part, "function_call")
                                            fc_name = getattr(fc, "name", None)
                                            fc_args_raw = getattr(fc, "arguments", None)
                                            fc_args_parsed = None
                                            if isinstance(fc_args_raw, str):
                                                try:
                                                    fc_args_parsed = json.loads(fc_args_raw)
                                                except Exception:
                                                    fc_args_parsed = fc_args_raw
                                            else:
                                                fc_args_parsed = fc_args_raw

                                            function_calls_list.append({
                                                "tool_name": fc_name,
                                                "arguments": fc_args_parsed,
                                                "raw_arguments": fc_args_raw,
                                                "timestamp": ts,
                                            })
                                            logger.info("DEBUG: model function_call proposal: %s", fc_name)
                                    except Exception:
                                        logger.exception("Failed while extracting function_call from event.candidates content.")

                                    # candidate-level function_response
                                    try:
                                        if hasattr(part, "function_response") and getattr(part, "function_response") is not None:
                                            fr = getattr(part, "function_response")
                                            fr_entry = {
                                                "tool_name": getattr(fr, "tool_name", None) or getattr(fr, "name", None) or "function_response",
                                                "output": fr,
                                                "timestamp": ts,
                                                "source_attr": "candidate.content.parts.function_response",
                                            }
                                            try:
                                                fr_entry["repr"] = pformat(fr)
                                            except Exception:
                                                fr_entry["repr"] = repr(fr)
                                            tool_outputs_list.append(fr_entry)
                                            logger.info("[tool_output] function_response captured (candidate) @ %s", ts)

                                            # --- diagnostic dump for candidate function_response ---
                                            try:
                                                dump = _serialize_obj(fr, max_depth=6)
                                                fname = f"logs/function_response_dump_{int(time.time()*1000)}.json"
                                                os.makedirs("logs", exist_ok=True)
                                                with open(fname, "w", encoding="utf-8") as ef:
                                                    json.dump(dump, ef, indent=2)
                                                logger.info("Wrote function_response diagnostic to %s", fname)
                                                for h in logger.handlers:
                                                    try:
                                                        h.flush()
                                                    except Exception:
                                                        pass
                                                # optional auto-invoke fallback (debug/robustness)
                                                _auto_invoke_render_for_loadctx(tool_outputs_list, function_calls_list)
                                            except Exception:
                                                logger.exception("Failed to write function_response diagnostic (candidate).")
                                    except Exception:
                                        logger.exception("Failed while extracting function_response from candidate part.")

                                    # candidate inline/file data -> synthetic tool outputs
                                    try:
                                        if hasattr(part, "inline_data") and getattr(part, "inline_data") is not None:
                                            inline = getattr(part, "inline_data")
                                            inline_summary = {"mime_type": getattr(inline, "mime_type", None)}
                                            data = getattr(inline, "data", None)
                                            if isinstance(data, str):
                                                inline_summary["data_b64_len"] = len(data)
                                            tool_outputs_list.append({
                                                "tool_name": "inline_data",
                                                "output": inline_summary,
                                                "raw": inline,
                                                "timestamp": ts,
                                            })
                                            logger.info("[inline_data] captured in candidate content @ %s", ts)

                                        if hasattr(part, "file_data") and getattr(part, "file_data") is not None:
                                            fd = getattr(part, "file_data")
                                            fd_summary = {"mime_type": getattr(fd, "mime_type", None)}
                                            data = getattr(fd, "data", None)
                                            if isinstance(data, str):
                                                fd_summary["data_b64_len"] = len(data)
                                            tool_outputs_list.append({
                                                "tool_name": "file_data",
                                                "output": fd_summary,
                                                "raw": fd,
                                                "timestamp": ts,
                                            })
                                            logger.info("[file_data] captured in candidate content @ %s", ts)
                                    except Exception:
                                        logger.exception("Failed while inspecting candidate parts for inline/file data.")

                            # Some runtimes may attach tool output into the candidate itself.
                            # Check a few well-known attrs on the candidate.
                            for cand_attr in ("tool_output", "tool_result", "tool_response", "tool_return"):
                                if hasattr(cand, cand_attr):
                                    try:
                                        cand_tool_output = getattr(cand, cand_attr)
                                        cand_tool_name = getattr(cand, "tool_name", None) or getattr(cand, "name", None)
                                        entry = {
                                            "tool_name": cand_tool_name,
                                            "output": cand_tool_output,
                                            "timestamp": ts,
                                            "source_attr": f"candidate.{cand_attr}",
                                        }
                                        try:
                                            entry["repr"] = pformat(cand_tool_output)
                                        except Exception:
                                            entry["repr"] = repr(cand_tool_output)
                                        tool_outputs_list.append(entry)
                                        logger.info("[tool_output] %s @ %s (source=candidate.%s)", cand_tool_name, ts, cand_attr)
                                    except Exception:
                                        logger.exception("Failed while extracting tool output from candidate.")
                        except Exception:
                            # continue to next candidate if this one fails
                            continue
            except Exception:
                logger.exception("Failed to inspect event.candidates for function_call or tool output.")

            # --------------------------
            # 4) If this event is marked final assistant response, break loop early
            # --------------------------
            try:
                if hasattr(event, "is_final_response") and event.is_final_response():
                    logger.debug("Event marks final assistant response; breaking loop.")
                    break
            except Exception:
                # ignore and keep iterating
                pass

    except Exception as e:
        # Log runner-level errors and return what we have
        logger.exception("Runner raised exception during call_agent: %s", e)

    # Optional: log stored session messages if session service exposes a getter (best-effort)
    try:
        get_sess = getattr(_SESSION_SERVICE, "get_session", None) or getattr(_SESSION_SERVICE, "get_messages", None)
        if get_sess:
            try:
                sess_data = get_sess(session_id)
                logger.info("Session snapshot (type %s): %s", type(sess_data).__name__, pformat(sess_data))
                for h in logger.handlers:
                    try:
                        h.flush()
                    except Exception:
                        pass
            except Exception:
                logger.debug("Session introspection not supported or failed.")
    except Exception:
        pass

    # Write diagnostic events report so we can inspect exact event shapes produced by this Runner
    try:
        out_path = "logs/event_attrs.json"
        os.makedirs("logs", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as ef:
            json.dump(events_debug, ef, indent=2)
        logger.info("Wrote event attribute diagnostics to %s", out_path)
        for h in logger.handlers:
            try:
                h.flush()
            except Exception:
                pass
    except Exception:
        logger.exception("Failed to write event attribute diagnostics")

    # join accumulated agent text parts into a single string for return
    try:
        agent_text = "\n".join([p for p in agent_text_parts if isinstance(p, str) and p.strip()]) if agent_text_parts else ""
    except Exception:
        agent_text = str(agent_text_parts)

    # Return the 4-tuple: agent_text, session id, tool outputs, function calls
    return agent_text, session_id, tool_outputs_list, function_calls_list
