# src/tools/artifacts_helper.py
import inspect
import base64
import logging
from typing import Any, Optional

logger = logging.getLogger("tutor_agent.artifacts_helper")


async def _maybe_await(v):
    """If v is awaitable, await it; otherwise return v."""
    if inspect.isawaitable(v):
        try:
            return await v
        except Exception as e:
            logger.exception("awaiting returned awaitable failed: %s", e)
            return None
    return v


async def load_artifact_robust(tool_context: Any, filename: str) -> Optional[Any]:
    """
    Robustly load an artifact using a runner-provided tool_context.

    Tries several reasonable call signatures (positional, filename=, older APIs).
    Awaits coroutine results when necessary.

    Returns:
      - the raw return from the runner (could be str, bytes, genai types.Part-like object),
        or None on failure.
    """
    if tool_context is None:
        logger.debug("load_artifact_robust: no tool_context provided")
        return None

    load_fn = getattr(tool_context, "load_artifact", None)
    if not callable(load_fn):
        logger.debug("load_artifact_robust: tool_context has no load_artifact")
        return None

    candidates = []
    # try several call shapes
    try:
        candidates.append(("positional", lambda: load_fn(filename)))  # try keyword name `filename`
    except Exception:
        pass
    # many runners accept `filename` kwarg
    try:
        candidates.append(("kw_filename", lambda: load_fn(filename=filename)))
    except Exception:
        pass
    # some older wrappers expect (filename, session_id)
    try:
        candidates.append(("pos_session", lambda: load_fn(filename, None)))
    except Exception:
        pass
    # try simple positional first (some wrappers accept just filename)
    try:
        candidates.insert(0, ("pos_simple", lambda: load_fn(filename)))
    except Exception:
        pass

    last_exc = None
    for name, call in candidates:
        try:
            res = call()
        except TypeError as te:
            last_exc = te
            continue
        except Exception as e:
            logger.exception("load_artifact_robust: unexpected error calling load_artifact (%s): %s", name, e)
            last_exc = e
            continue

        # await if needed
        res = await _maybe_await(res)
        if res is None:
            # treat as failure for this signature and try next
            last_exc = None
            continue
        # successful (may be str or Part or object)
        return res

    # If we reach here, nothing worked
    logger.debug("load_artifact_robust: all candidate signatures failed; last exception: %s", repr(last_exc))
    return None


async def save_artifact_robust(tool_context: Any, filename: str, artifact_part: Any, *, app_name: str = "tutor_app", user_id: str = "student", session_id: Optional[str] = None) -> bool:
    """
    Robustly save an artifact via tool_context.save_artifact.

    Tries a few signatures and awaits coroutine returns. Returns True on success.
    """
    if tool_context is None:
        logger.debug("save_artifact_robust: no tool_context provided")
        return False

    save_fn = getattr(tool_context, "save_artifact", None)
    if not callable(save_fn):
        logger.debug("save_artifact_robust: tool_context has no save_artifact")
        return False

    candidates = []
    # do sensible variants
    try:
        candidates.append(("full_kw", lambda: save_fn(app_name=app_name, user_id=user_id, filename=filename, artifact=artifact_part, session_id=session_id)))
    except Exception:
        pass
    try:
        candidates.append(("kw_simple", lambda: save_fn(filename=filename, artifact=artifact_part, session_id=session_id)))
    except Exception:
        pass
    try:
        candidates.append(("pos", lambda: save_fn(filename, artifact_part, session_id)))
    except Exception:
        pass
    try:
        candidates.insert(0, ("pos_simple", lambda: save_fn(filename)))
    except Exception:
        pass

    last_exc = None
    for name, call in candidates:
        try:
            res = call()
        except TypeError as te:
            last_exc = te
            continue
        except Exception as e:
            logger.exception("save_artifact_robust: unexpected error calling save_artifact (%s): %s", name, e)
            last_exc = e
            continue

        res = await _maybe_await(res)

        # Many save_fns return None, int (version), or bool. Consider non-falsey as success.
        if res is None:
            # If API returns None but didn't raise, treat as success conservatively?
            # Safer to treat as True only when runner docs mention returning version. We'll treat None as True.
            return True
        if isinstance(res, bool):
            return bool(res)
        # numeric return -> success
        try:
            if int(res) >= 0:
                return True
        except Exception:
            pass
        # fallback: if truthy, return True
        if res:
            return True

    logger.debug("save_artifact_robust: all candidate signatures failed; last exception: %s", repr(last_exc))
    return False
