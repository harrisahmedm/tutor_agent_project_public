# scripts/test_call_agent_tool_outputs.py
import os
import json
from pprint import pformat
from pathlib import Path
from src.agents.tutor_agent.agent import call_agent

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Adjust this prompt so it provides the chapter/page/exercise in one shot.
# Use values you know exist in your local NCERT PDF dataset.
# Example prompt (change numbers to match a known exercise in your files):
prompt = (
    "Please call the load_context tool with these exact parameters: grade=8, chapter_id=7, "
    "chapter_page_number=12, exercise_number=2. "
    "Return only the content of the problem_text attribute of what the tool returns, and nothing else."
)

print("Calling call_agent()... (this may take a few seconds)")
assistant_text, session_id, tool_outputs_list, function_calls_list = call_agent(prompt, user_id="test_user")

print("Call completed.")
print("Assistant text (trimmed):", (assistant_text[:500] + "...") if assistant_text else "<empty>")

# Save tool_outputs_list to a JSON file with a safe serializer fallback
outpath = Path("logs/tool_outputs.json")
serializable = []
for entry in tool_outputs_list:
    # We will try direct JSON serialization; if it fails, fall back to pformat() for the problematic fields.
    try:
        json.dumps(entry)
        serializable.append(entry)
    except Exception:
        # Build a serializable summary
        s = {}
        s["tool_name"] = entry.get("tool_name")
        # Keep 'output' if simple, else use pformat
        try:
            json.dumps(entry.get("output"))
            s["output"] = entry.get("output")
        except Exception:
            s["output_repr"] = pformat(entry.get("output"))
        # copy timestamp
        s["timestamp"] = entry.get("timestamp")
        # include any repr we added earlier
        if "repr" in entry:
            s["repr"] = entry["repr"]
        # include raw if present (repr)
        if "raw" in entry:
            try:
                json.dumps(entry["raw"])
                s["raw"] = entry["raw"]
            except Exception:
                s["raw_repr"] = pformat(entry["raw"])
        serializable.append(s)

with outpath.open("w", encoding="utf-8") as f:
    json.dump({"session_id": session_id, "assistant_text": assistant_text, "tool_outputs": serializable, "function_calls": function_calls_list}, f, indent=2, ensure_ascii=False)

print(f"Wrote tool outputs summary to {outpath.resolve()}")
print("Top-level tool names seen:", [e.get("tool_name") for e in tool_outputs_list])

