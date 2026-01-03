from __future__ import annotations

import json
import time
from dataclasses import dataclass
import trace
from typing import Any, Callable, Dict, List, Optional, Tuple

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool


def _coerce_text(content: Any) -> str:
    """Best-effort to coerce provider-specific content blocks to a user-facing string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if "text" in item and isinstance(item["text"], str):
                    parts.append(item["text"])
                elif "content" in item and isinstance(item["content"], str):
                    parts.append(item["content"])
        return "\n".join([p for p in parts if p]).strip()

    return str(content).strip()


def _json_default(o: Any) -> Any:
    """json.dumps fallback for odd types (numpy/pandas/bytes)."""
    if isinstance(o, (bytes, bytearray)):
        return f"<bytes:{len(o)}>"
    try:
        import numpy as np  # type: ignore
        if isinstance(o, (np.integer, np.floating)):
            return o.item()
    except Exception:
        pass
    return str(o)


def _json_safe_tool_result(tool_name: str, tool_result: Any) -> Any:
    """
    Make tool_result safe for json.dumps() before sending to the model.
    Important: remove binary plot bytes.
    """
    if not isinstance(tool_result, dict):
        return tool_result

    if tool_name == "python_viz":
        images = tool_result.get("images") or []
        safe = dict(tool_result)
        safe.pop("images", None)
        safe["image_count"] = len(images)
        if isinstance(safe.get("stdout"), str) and len(safe["stdout"]) > 4000:
            safe["stdout"] = safe["stdout"][:4000] + "\n...(truncated)..."
        return safe

    return tool_result


@dataclass
class TraceEvent:
    t: float
    kind: str  # "tool_start" | "tool_end" | "model"
    name: str
    payload: Dict[str, Any]


OnEvent = Callable[[Dict[str, Any]], None]


def run_tool_calling_loop(
    model,
    tools: List[BaseTool],
    system_prompt: str,
    user_prompt: str,
    max_steps: int = 50,
    on_event: Optional[OnEvent] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Streaming tool-calling loop.

    Visible outputs (added to outputs AND emitted live):
      - {"type":"assistant_text","text":...}
      - {"type":"plot","image_bytes":...}

    Status outputs (emitted live ONLY; not stored in outputs):
      - {"type":"status","text":...}

    Trace (returned at end, for debug expander).
    """
    tool_map = {t.name: t for t in tools}
    model = model.bind_tools(tools)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    outputs: List[Dict[str, Any]] = []
    trace: List[TraceEvent] = []

    def emit_visible(event: Dict[str, Any]) -> None:
        outputs.append(event)
        if on_event:
            on_event(event)

    def emit_status(text: str) -> None:
        if on_event:
            on_event({"type": "status", "text": text})

    for step in range(1, max_steps + 1):
        emit_status(f"Step {step}/{max_steps}: asking model…")

        t0 = time.time()
        ai: AIMessage = model.invoke(messages)

        trace.append(
            TraceEvent(
                t=t0,
                kind="model",
                name="invoke",
                payload={
                    "step": step,
                    "ai_content_preview": _coerce_text(ai.content)[:500],
                    "has_tool_calls": bool(getattr(ai, "tool_calls", None)),
                },
            )
        )

        messages.append(ai)

        # Show assistant text if present (often empty in tool-only steps)
        text = _coerce_text(ai.content)
        if text:
            emit_visible({"type": "assistant_text", "text": text})

        tool_calls = getattr(ai, "tool_calls", None) or []
        if not tool_calls:
            emit_status("Done.")
            break

        for call in tool_calls:
            tool_name = call.get("name")
            tool_args = call.get("args") or {}
            tool_id = call.get("id")

            if tool_name not in tool_map:
                tool_result = {"ok": False, "error": f"Unknown tool: {tool_name}"}

                safe_for_model = _json_safe_tool_result(tool_name or "unknown", tool_result)
                messages.append(
                    ToolMessage(
                        content=json.dumps(safe_for_model, ensure_ascii=False, default=_json_default),
                        tool_call_id=tool_id,
                    )
                )

                safe_for_trace = _json_safe_tool_result(tool_name, tool_result)

                trace.append(
                    TraceEvent(
                        t=time.time(),
                        kind="tool_end",
                        name=tool_name,
                        payload={"args": tool_args, "output": safe_for_trace},
                    )
                )

                emit_status(f"Tool error: unknown tool `{tool_name}`")
                continue

            emit_status(f"Step {step}: running tool `{tool_name}`…")

            tool = tool_map[tool_name]
            trace.append(TraceEvent(t=time.time(), kind="tool_start", name=tool_name, payload={"args": tool_args}))

            try:
                tool_result = tool.invoke(tool_args)
            except Exception as e:
                tool_result = {"ok": False, "error": str(e)}

            safe_for_trace = _json_safe_tool_result(tool_name, tool_result)

            trace.append(
                TraceEvent(
                    t=time.time(),
                    kind="tool_end",
                    name=tool_name,
                    payload={"args": tool_args, "output": safe_for_trace},
                )
            )


            emit_status(f"Step {step}: finished `{tool_name}`")

            # Feed tool result back to the model (JSON-safe)
            safe_for_model = _json_safe_tool_result(tool_name, tool_result)
            messages.append(
                ToolMessage(
                    content=json.dumps(safe_for_model, ensure_ascii=False, default=_json_default),
                    tool_call_id=tool_id,
                )
            )

            # Visible plots: stream immediately
            if tool_name == "python_viz" and isinstance(tool_result, dict):
                if tool_result.get("ok"):
                    for img_bytes in tool_result.get("images") or []:
                        emit_visible({"type": "plot", "image_bytes": img_bytes})

    trace_out = [{"t": e.t, "kind": e.kind, "name": e.name, "payload": e.payload} for e in trace]
    return outputs, trace_out
