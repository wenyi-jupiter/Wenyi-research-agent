"""LLM interaction logger.

Monkey-patches QwenProvider.invoke and ChatOpenAI._agenerate to capture every
LLM call (input messages + output) to per-task JSONL files under ``llm_logs/``.

Usage:
    # At server startup (once):
    from agent_engine.llm_logger import install_patches
    install_patches()

    # Per-task (before execution):
    from agent_engine.llm_logger import begin_task_logging, finish_task_logging
    begin_task_logging(task_id, user_request)
    ...run task...
    finish_task_logging()
"""

from __future__ import annotations

import contextlib
import json
import logging
import time
import traceback
from contextvars import ContextVar
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LOG_DIR = _PROJECT_ROOT / "llm_logs"

# ── Per-task context (async-safe via contextvars) ──
_task_log_file: ContextVar[Path | None] = ContextVar("_task_log_file", default=None)
_task_call_seq: ContextVar[int] = ContextVar("_task_call_seq", default=0)
_task_id_var: ContextVar[str | None] = ContextVar("_task_id_var", default=None)
# Explicit caller set by agents before LLM calls (avoids deep-stack detection failure)
_caller_context: ContextVar[str | None] = ContextVar("_caller_context", default=None)

_patches_installed = False


@contextlib.contextmanager
def set_caller_context(caller: str):
    """Context manager to set the current LLM caller for logging.

    Use this in agents before calling model.ainvoke() or provider.invoke() when
    the call chain is deep (e.g. executor -> LangChain -> ChatOpenAI) and
    stack-based detection would fail.
    """
    token = _caller_context.set(caller)
    try:
        yield
    finally:
        _caller_context.reset(token)


def begin_task_logging(task_id: str, user_request: str = "") -> Path:
    """Start logging for a new task. Returns the JSONL log file path."""
    LOG_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"llm_interactions_{ts}_{task_id[:12]}.jsonl"

    _task_log_file.set(log_file)
    _task_call_seq.set(0)
    _task_id_var.set(task_id)

    header = {
        "type": "task_start",
        "task_id": task_id,
        "user_request": user_request,
        "timestamp": datetime.now().isoformat(),
    }
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(header, ensure_ascii=False, default=str) + "\n")

    logger.info("[LLM-Log] Task %s logging to %s", task_id, log_file)
    return log_file


def finish_task_logging() -> Path | None:
    """Finish logging for the current task, write summary, and return summary path."""
    log_file = _task_log_file.get()
    if log_file is None or not log_file.exists():
        return None
    summary_path = _write_summary(log_file)
    _task_log_file.set(None)
    _task_call_seq.set(0)
    _task_id_var.set(None)
    return summary_path


# ── Internal helpers ──

def _serialize_message(msg: Any) -> dict:
    from langchain_core.messages import (
        AIMessage, HumanMessage, SystemMessage, ToolMessage,
    )
    if isinstance(msg, SystemMessage):
        role = "system"
    elif isinstance(msg, HumanMessage):
        role = "user"
    elif isinstance(msg, AIMessage):
        role = "assistant"
    elif isinstance(msg, ToolMessage):
        role = "tool"
    elif isinstance(msg, dict):
        return msg
    else:
        role = type(msg).__name__

    content = ""
    if hasattr(msg, "content"):
        content = msg.content if isinstance(msg.content, str) else str(msg.content)

    result: dict[str, Any] = {"role": role, "content_length": len(content)}

    if len(content) <= 5000:
        result["content"] = content
    else:
        result["content_preview"] = (
            content[:2000]
            + f"\n... [{len(content)} chars total] ...\n"
            + content[-2000:]
        )

    if hasattr(msg, "tool_calls") and msg.tool_calls:
        result["tool_calls"] = []
        for tc in msg.tool_calls:
            if hasattr(tc, "name"):
                result["tool_calls"].append({
                    "name": tc.name,
                    "args": getattr(tc, "args", {}),
                    "id": getattr(tc, "id", ""),
                })
            elif isinstance(tc, dict):
                result["tool_calls"].append(tc)

    if hasattr(msg, "tool_call_id"):
        result["tool_call_id"] = msg.tool_call_id

    return result


def _detect_caller() -> str:
    # Prefer explicit caller set via set_caller_context() — avoids deep-stack
    # detection failure when executor uses model.ainvoke() -> ChatOpenAI._agenerate
    # through many LangChain/LangGraph layers.
    explicit = _caller_context.get()
    if explicit:
        return explicit

    stack = traceback.extract_stack(limit=50)
    for frame in reversed(stack):
        fname = frame.filename.replace("\\", "/")
        if "planner.py" in fname:
            return "planner"
        if "executor.py" in fname:
            return "executor"
        if "critic.py" in fname:
            return "critic"
        if "reporter.py" in fname:
            return "reporter"
    return "unknown"


def _log_entry(entry: dict) -> None:
    log_file = _task_log_file.get()
    if log_file is None:
        return

    seq = _task_call_seq.get() + 1
    _task_call_seq.set(seq)

    entry["seq"] = seq
    entry["task_id"] = _task_id_var.get()
    entry["timestamp"] = datetime.now().isoformat()

    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    except OSError:
        pass

    caller = entry.get("caller", "?")
    inp_tok = entry.get("output", {}).get("input_tokens", 0)
    out_tok = entry.get("output", {}).get("output_tokens", 0)
    n_msgs = len(entry.get("input_messages", []))
    tool_calls = entry.get("output", {}).get("tool_calls", [])
    tool_info = f" | tools:{len(tool_calls)}" if tool_calls else ""
    logger.info(
        "[LLM #%d] %s | msgs:%d | tokens:%d+%d=%d%s",
        seq, caller, n_msgs, inp_tok, out_tok, inp_tok + out_tok, tool_info,
    )


# ── Monkey-patches ──

def install_patches() -> None:
    """Install LLM logging patches (idempotent)."""
    global _patches_installed
    if _patches_installed:
        return
    _patches_installed = True

    _patch_qwen_provider()
    _patch_langchain_model()
    logger.info("[LLM-Log] Patches installed")


def _patch_qwen_provider() -> None:
    from agent_engine.llm.qwen import QwenProvider

    _original_invoke = QwenProvider.invoke

    @wraps(_original_invoke)
    async def logged_invoke(self, messages, tools=None, **kwargs):
        if _task_log_file.get() is None:
            return await _original_invoke(self, messages, tools=tools, **kwargs)

        caller = _detect_caller()
        input_msgs = [_serialize_message(m) for m in messages]
        start = time.time()

        try:
            response = await _original_invoke(self, messages, tools=tools, **kwargs)
            elapsed = time.time() - start

            content = response.content
            _log_entry({
                "caller": caller,
                "method": "QwenProvider.invoke",
                "model": self.model,
                "elapsed_s": round(elapsed, 2),
                "input_messages": input_msgs,
                "input_message_count": len(input_msgs),
                "output": {
                    "content_length": len(content),
                    "content_preview": (
                        content[:3000] if len(content) <= 3000
                        else content[:1500]
                        + f"\n...[{len(content)} chars]...\n"
                        + content[-1500:]
                    ),
                    "tool_calls": response.tool_calls,
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "total_tokens": response.total_tokens,
                    "finish_reason": response.finish_reason,
                },
            })
            return response

        except Exception as e:
            _log_entry({
                "caller": caller,
                "method": "QwenProvider.invoke",
                "model": self.model,
                "elapsed_s": round(time.time() - start, 2),
                "input_messages": input_msgs,
                "input_message_count": len(input_msgs),
                "output": {"error": str(e)},
            })
            raise

    QwenProvider.invoke = logged_invoke


def _patch_langchain_model() -> None:
    from langchain_openai import ChatOpenAI

    _original_agenerate = ChatOpenAI._agenerate

    @wraps(_original_agenerate)
    async def logged_agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        if _task_log_file.get() is None:
            return await _original_agenerate(
                self, messages, stop=stop, run_manager=run_manager, **kwargs
            )

        caller = _detect_caller()
        input_msgs = [_serialize_message(m) for m in messages]
        start = time.time()

        try:
            result = await _original_agenerate(
                self, messages, stop=stop, run_manager=run_manager, **kwargs
            )
            elapsed = time.time() - start

            content = ""
            tool_calls_data: list[dict] = []
            if result.generations:
                gen = result.generations[0]
                msg = gen.message if hasattr(gen, "message") else None
                if msg:
                    content = msg.content if isinstance(msg.content, str) else str(msg.content)
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            if hasattr(tc, "name"):
                                tool_calls_data.append({
                                    "name": tc.name,
                                    "args": getattr(tc, "args", {}),
                                    "id": getattr(tc, "id", ""),
                                })
                            elif isinstance(tc, dict):
                                tool_calls_data.append(tc)

            llm_output = result.llm_output or {}
            token_usage = llm_output.get("token_usage", {})
            inp_tok = token_usage.get("prompt_tokens", 0) or token_usage.get("input_tokens", 0)
            out_tok = token_usage.get("completion_tokens", 0) or token_usage.get("output_tokens", 0)

            _log_entry({
                "caller": caller,
                "method": "ChatOpenAI._agenerate",
                "model": getattr(self, "model_name", "unknown"),
                "elapsed_s": round(elapsed, 2),
                "input_messages": input_msgs,
                "input_message_count": len(input_msgs),
                "output": {
                    "content_length": len(content),
                    "content_preview": (
                        content[:3000] if len(content) <= 3000
                        else content[:1500]
                        + f"\n...[{len(content)} chars]...\n"
                        + content[-1500:]
                    ),
                    "tool_calls": tool_calls_data,
                    "input_tokens": inp_tok,
                    "output_tokens": out_tok,
                    "total_tokens": inp_tok + out_tok,
                },
            })
            return result

        except Exception as e:
            _log_entry({
                "caller": caller,
                "method": "ChatOpenAI._agenerate",
                "model": getattr(self, "model_name", "unknown"),
                "elapsed_s": round(time.time() - start, 2),
                "input_messages": input_msgs,
                "input_message_count": len(input_msgs),
                "output": {"error": str(e)},
            })
            raise

    ChatOpenAI._agenerate = logged_agenerate


# ── Summary writer ──

def _write_summary(log_file: Path) -> Path:
    entries = []
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("type") == "task_start":
                continue
            entries.append(obj)

    summary_file = log_file.with_name(
        log_file.name.replace("llm_interactions_", "llm_summary_").replace(".jsonl", ".txt")
    )

    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("LLM Interaction Summary")
    lines.append(f"Total calls: {len(entries)}")
    lines.append(f"Log file: {log_file}")
    lines.append("=" * 80)

    total_in = 0
    total_out = 0
    caller_stats: dict[str, dict[str, int]] = {}

    for entry in entries:
        seq = entry.get("seq", "?")
        caller = entry.get("caller", "?")
        method = entry.get("method", "?")
        elapsed = entry.get("elapsed_s", 0)
        n_msgs = entry.get("input_message_count", 0)
        output = entry.get("output", {})
        inp_tok = output.get("input_tokens", 0)
        out_tok = output.get("output_tokens", 0)
        content_len = output.get("content_length", 0)
        tool_calls = output.get("tool_calls", [])
        error = output.get("error")

        total_in += inp_tok
        total_out += out_tok

        if caller not in caller_stats:
            caller_stats[caller] = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "errors": 0}
        caller_stats[caller]["calls"] += 1
        caller_stats[caller]["input_tokens"] += inp_tok
        caller_stats[caller]["output_tokens"] += out_tok
        if error:
            caller_stats[caller]["errors"] += 1

        lines.append("")
        lines.append(f"--- Call #{seq} ---")
        lines.append(f"  Caller: {caller} ({method})")
        lines.append(f"  Model: {entry.get('model', '?')}")
        lines.append(f"  Time: {elapsed:.1f}s")
        lines.append(f"  Input: {n_msgs} messages")
        lines.append(f"  Tokens: {inp_tok} in + {out_tok} out = {inp_tok + out_tok} total")

        if error:
            lines.append(f"  ERROR: {error[:200]}")
        else:
            lines.append(f"  Output: {content_len} chars")
            if tool_calls:
                tool_names = [tc.get("name", "?") for tc in tool_calls]
                lines.append(f"  Tool calls: {', '.join(tool_names)}")

    lines.append("")
    lines.append("=" * 80)
    lines.append("TOTALS")
    lines.append("=" * 80)
    lines.append(f"  Total calls: {len(entries)}")
    lines.append(f"  Total input tokens: {total_in:,}")
    lines.append(f"  Total output tokens: {total_out:,}")
    lines.append(f"  Total tokens: {total_in + total_out:,}")
    lines.append("")
    lines.append("  Per-caller breakdown:")
    for caller, stats in sorted(caller_stats.items()):
        tot = stats["input_tokens"] + stats["output_tokens"]
        lines.append(
            f"    {caller}: {stats['calls']} calls, "
            f"{stats['input_tokens']:,}+{stats['output_tokens']:,}={tot:,} tokens"
        )
        if stats["errors"]:
            lines.append(f"      ({stats['errors']} errors)")

    summary_text = "\n".join(lines)
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(summary_text)

    logger.info("[LLM-Log] Summary written to %s", summary_file)
    return summary_file
