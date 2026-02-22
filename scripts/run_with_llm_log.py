# -*- coding: utf-8 -*-
"""Run a task with full LLM interaction logging.

Captures every LLM call (input messages + output) to a timestamped JSON-lines file.
Works by monkey-patching the QwenProvider.invoke method and the LangChain model
returned by get_langchain_model.

Usage:
    cd d:\agent_demo
    python scripts/run_with_llm_log.py --request "Search a PDF and summarize"
"""

import asyncio
import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from functools import wraps
from typing import TextIO

# Add project source to path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

os.chdir(str(_PROJECT_ROOT))

# NOTE: On Windows with non-UTF-8 console (e.g. CP936), passing CJK text via
# --request on the command line may produce mojibake.  Use --request_file instead
# (reads UTF-8 file) together with $env:PYTHONUTF8=1 / python -X utf8.

# Force UTF-8 stdout/stderr on Windows to avoid GBK encoding errors
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


class _SafeStream:
    """Best-effort stream wrapper that ignores transient OSError on write/flush."""

    def __init__(self, stream: TextIO):
        self._stream = stream

    def write(self, data: str) -> int:
        try:
            return self._stream.write(data)
        except OSError:
            return len(data)

    def flush(self) -> None:
        try:
            self._stream.flush()
        except OSError:
            pass

    @property
    def encoding(self) -> str:
        return getattr(self._stream, "encoding", "utf-8")

    def isatty(self) -> bool:
        return getattr(self._stream, "isatty", lambda: False)()


# Wrap std streams before logging.basicConfig so StreamHandler binds safe streams.
sys.stdout = _SafeStream(sys.stdout)
sys.stderr = _SafeStream(sys.stderr)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)


class _TeeStream:
    """Mirror writes to multiple text streams."""

    def __init__(self, *streams: TextIO):
        self._streams = streams

    def write(self, data: str) -> int:
        alive_streams = []
        for s in self._streams:
            try:
                s.write(data)
                alive_streams.append(s)
            except OSError:
                # On Windows, redirected/closed handles can sporadically throw
                # OSError(22). Best effort logging should not abort the run.
                continue
        if alive_streams:
            self._streams = tuple(alive_streams)
        return len(data)

    def flush(self) -> None:
        alive_streams = []
        for s in self._streams:
            try:
                s.flush()
                alive_streams.append(s)
            except OSError:
                continue
        if alive_streams:
            self._streams = tuple(alive_streams)

    @property
    def encoding(self) -> str:
        return getattr(self._streams[0], "encoding", "utf-8")

    def isatty(self) -> bool:
        return any(getattr(s, "isatty", lambda: False)() for s in self._streams)

# ── Log file setup ──
LOG_DIR = Path("d:/agent_demo/llm_logs")
LOG_DIR.mkdir(exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOG_DIR / f"llm_interactions_{timestamp}.jsonl"
SUMMARY_FILE = LOG_DIR / f"llm_summary_{timestamp}.txt"

_call_seq = 0  # Global sequence counter


def _serialize_message(msg) -> dict:
    """Serialize a LangChain message to a dict."""
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

    result = {"role": role, "content_length": len(content)}

    # Truncate content for logging (keep first/last 2000 chars)
    if len(content) <= 5000:
        result["content"] = content
    else:
        result["content_preview"] = content[:2000] + f"\n... [{len(content)} chars total] ...\n" + content[-2000:]

    # Include tool calls if present
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

    # Include tool_call_id for ToolMessage
    if hasattr(msg, "tool_call_id"):
        result["tool_call_id"] = msg.tool_call_id

    return result


def _log_interaction(entry: dict):
    """Append a log entry to the JSONL file."""
    global _call_seq
    _call_seq += 1
    entry["seq"] = _call_seq
    entry["timestamp"] = datetime.now().isoformat()

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

    # Print summary to console
    caller = entry.get("caller", "?")
    inp_tokens = entry.get("output", {}).get("input_tokens", 0)
    out_tokens = entry.get("output", {}).get("output_tokens", 0)
    n_msgs = len(entry.get("input_messages", []))
    has_tools = bool(entry.get("output", {}).get("tool_calls"))
    tool_info = f" | tools: {len(entry['output']['tool_calls'])}" if has_tools else ""
    print(
        f"  [LLM #{_call_seq}] {caller} | "
        f"msgs: {n_msgs} | "
        f"tokens: {inp_tokens}+{out_tokens}={inp_tokens+out_tokens}"
        f"{tool_info}"
    )


def patch_qwen_provider():
    """Monkey-patch QwenProvider.invoke to log inputs/outputs."""
    from agent_engine.llm.qwen import QwenProvider

    _original_invoke = QwenProvider.invoke

    @wraps(_original_invoke)
    async def logged_invoke(self, messages, tools=None, **kwargs):
        # Determine caller from call stack
        import traceback
        stack = traceback.extract_stack(limit=10)
        caller = "unknown"
        for frame in reversed(stack):
            fname = frame.filename.replace("\\", "/")
            if "planner.py" in fname:
                caller = "planner"
                break
            elif "critic.py" in fname:
                caller = "critic"
                break
            elif "reporter.py" in fname:
                caller = "reporter"
                break
            elif "executor.py" in fname:
                caller = "executor"
                break

        # Serialize input
        input_msgs = [_serialize_message(m) for m in messages]

        start_time = time.time()
        try:
            response = await _original_invoke(self, messages, tools=tools, **kwargs)
            elapsed = time.time() - start_time

            # Log the interaction
            entry = {
                "caller": caller,
                "method": "QwenProvider.invoke",
                "model": self.model,
                "elapsed_s": round(elapsed, 2),
                "input_messages": input_msgs,
                "input_message_count": len(input_msgs),
                "output": {
                    "content_length": len(response.content),
                    "content_preview": response.content[:3000] if len(response.content) <= 3000 else response.content[:1500] + f"\n...[{len(response.content)} chars]...\n" + response.content[-1500:],
                    "tool_calls": response.tool_calls,
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "total_tokens": response.total_tokens,
                    "finish_reason": response.finish_reason,
                },
            }
            _log_interaction(entry)
            return response

        except Exception as e:
            elapsed = time.time() - start_time
            entry = {
                "caller": caller,
                "method": "QwenProvider.invoke",
                "model": self.model,
                "elapsed_s": round(elapsed, 2),
                "input_messages": input_msgs,
                "input_message_count": len(input_msgs),
                "output": {"error": str(e)},
            }
            _log_interaction(entry)
            raise

    QwenProvider.invoke = logged_invoke
    print(f"[Patch] QwenProvider.invoke patched for logging")


def patch_langchain_model():
    """Monkey-patch ChatOpenAI._agenerate to log all LangChain model calls.

    Since model.bind_tools() returns a RunnableBinding (Pydantic model) which
    doesn't allow setting attributes, we patch at the ChatOpenAI class level.
    ChatOpenAI._agenerate is the core async method that all ainvoke calls
    eventually flow through.
    """
    from langchain_openai import ChatOpenAI

    _original_agenerate = ChatOpenAI._agenerate

    @wraps(_original_agenerate)
    async def logged_agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        # Determine caller from call stack
        import traceback
        stack = traceback.extract_stack(limit=15)
        caller = "unknown"
        for frame in reversed(stack):
            fname = frame.filename.replace("\\", "/")
            if "executor.py" in fname:
                caller = "executor"
                break
            elif "planner.py" in fname:
                caller = "planner"
                break
            elif "critic.py" in fname:
                caller = "critic"
                break
            elif "reporter.py" in fname:
                caller = "reporter"
                break

        # Serialize input messages
        input_msgs = [_serialize_message(m) for m in messages]

        start_time = time.time()
        try:
            result = await _original_agenerate(self, messages, stop=stop, run_manager=run_manager, **kwargs)
            elapsed = time.time() - start_time

            # Extract response from ChatResult
            content = ""
            tool_calls_data = []
            if result.generations and len(result.generations) > 0:
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

            # Extract tokens from llm_output
            inp_tok = 0
            out_tok = 0
            llm_output = result.llm_output or {}
            token_usage = llm_output.get("token_usage", {})
            inp_tok = (
                token_usage.get("prompt_tokens", 0)
                or token_usage.get("input_tokens", 0)
            )
            out_tok = (
                token_usage.get("completion_tokens", 0)
                or token_usage.get("output_tokens", 0)
            )

            entry = {
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
                        else content[:1500] + f"\n...[{len(content)} chars]...\n" + content[-1500:]
                    ),
                    "tool_calls": tool_calls_data,
                    "input_tokens": inp_tok,
                    "output_tokens": out_tok,
                    "total_tokens": inp_tok + out_tok,
                },
            }
            _log_interaction(entry)
            return result

        except Exception as e:
            elapsed = time.time() - start_time
            entry = {
                "caller": caller,
                "method": "ChatOpenAI._agenerate",
                "model": getattr(self, "model_name", "unknown"),
                "elapsed_s": round(elapsed, 2),
                "input_messages": input_msgs,
                "input_message_count": len(input_msgs),
                "output": {"error": str(e)},
            }
            _log_interaction(entry)
            raise

    ChatOpenAI._agenerate = logged_agenerate
    print(f"[Patch] ChatOpenAI._agenerate patched for logging")


def write_summary():
    """Read the JSONL log and write a human-readable summary."""
    entries = []
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    lines = []
    lines.append(f"=" * 80)
    lines.append(f"LLM Interaction Summary")
    lines.append(f"Total calls: {len(entries)}")
    lines.append(f"Log file: {LOG_FILE}")
    lines.append(f"=" * 80)

    total_input_tokens = 0
    total_output_tokens = 0
    caller_stats = {}

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

        total_input_tokens += inp_tok
        total_output_tokens += out_tok

        if caller not in caller_stats:
            caller_stats[caller] = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "errors": 0}
        caller_stats[caller]["calls"] += 1
        caller_stats[caller]["input_tokens"] += inp_tok
        caller_stats[caller]["output_tokens"] += out_tok
        if error:
            caller_stats[caller]["errors"] += 1

        lines.append("")
        lines.append(f"─── Call #{seq} ───")
        lines.append(f"  Caller: {caller} ({method})")
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

        # Show input message roles
        input_msgs = entry.get("input_messages", [])
        if input_msgs:
            roles = [m.get("role", "?") for m in input_msgs]
            lines.append(f"  Message roles: {' → '.join(roles)}")

    lines.append("")
    lines.append(f"{'=' * 80}")
    lines.append(f"TOTALS")
    lines.append(f"{'=' * 80}")
    lines.append(f"  Total calls: {len(entries)}")
    lines.append(f"  Total input tokens: {total_input_tokens:,}")
    lines.append(f"  Total output tokens: {total_output_tokens:,}")
    lines.append(f"  Total tokens: {total_input_tokens + total_output_tokens:,}")
    lines.append("")
    lines.append("  Per-caller breakdown:")
    for caller, stats in sorted(caller_stats.items()):
        tot = stats["input_tokens"] + stats["output_tokens"]
        lines.append(
            f"    {caller}: {stats['calls']} calls, "
            f"{stats['input_tokens']:,}+{stats['output_tokens']:,}={tot:,} tokens"
        )
        if stats["errors"]:
            lines.append(f"      ({stats['errors']} errors)"
        )

    summary_text = "\n".join(lines)
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print(f"\n{summary_text}")
    return summary_text


async def main():
    parser = argparse.ArgumentParser(description="Run a task with LLM interaction logging.")
    parser.add_argument(
        "--request",
        default="用 web_search 搜索一个公开 PDF 链接，并用 fetch_url 抓取其中一个 PDF，"
                "最后总结 PDF 文本中的关键一句话，并给出引用 URL。",
        help="User request text to run through the agent graph.",
    )
    parser.add_argument(
        "--request_file",
        default=None,
        help="Optional UTF-8 text file containing the user request (avoids shell quoting issues).",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Override max_steps in initial state. Defaults to config value (100).",
    )
    parser.add_argument(
        "--max_tool_calls",
        type=int,
        default=None,
        help="Optional override for max_tool_calls in initial state.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=None,
        help="Optional override for max_tokens in initial state.",
    )
    parser.add_argument(
        "--console_log_file",
        default=None,
        help=(
            "Optional UTF-8 mirror file for full stdout/stderr output. "
            "Use this instead of shell redirection on Windows to avoid mojibake."
        ),
    )
    args = parser.parse_args()

    console_mirror_fp = None
    if args.console_log_file:
        try:
            console_mirror_fp = open(
                args.console_log_file, "w", encoding="utf-8", errors="replace"
            )
            sys.stdout = _TeeStream(sys.stdout, console_mirror_fp)
            sys.stderr = _TeeStream(sys.stderr, console_mirror_fp)
            print(f"[Setup] Console mirror file: {args.console_log_file}")
        except Exception as e:
            print(f"[WARN] Could not enable --console_log_file={args.console_log_file!r}: {e}")

    print(f"[Setup] Log file: {LOG_FILE}")
    print(f"[Setup] Summary file: {SUMMARY_FILE}")

    # Apply patches BEFORE importing anything that creates providers
    patch_qwen_provider()
    patch_langchain_model()

    # Now import and run
    from agent_engine.agents.graph import AgentOrchestrator
    from agent_engine.agents.graph import create_initial_state
    from agent_engine.tools.builtin import (
        code_execute, fetch_url, list_directory, read_file, web_search, write_file,
        sec_edgar_financials, sec_edgar_filings, search_document,
    )
    # Trigger tool registration
    _ = (code_execute, fetch_url, list_directory, read_file, web_search, write_file,
         sec_edgar_financials, sec_edgar_filings, search_document)

    user_request = args.request
    if args.request_file:
        req_arg = Path(args.request_file)
        if not req_arg.is_absolute():
            candidates = [
                _PROJECT_ROOT / req_arg,
                Path.cwd() / req_arg,
                Path(__file__).resolve().parent / req_arg.name,
            ]
        else:
            candidates = [req_arg]
        loaded = False
        for cand in candidates:
            try:
                with open(cand, "r", encoding="utf-8") as f:
                    user_request = f.read().strip() or user_request
                print(f"[Setup] Loaded request from {cand} ({len(user_request)} chars)")
                loaded = True
                break
            except FileNotFoundError:
                continue
            except Exception as e:
                print(f"[WARN] Could not read {cand}: {e}")
                continue
        if not loaded:
            print(f"[WARN] Could not read --request_file={args.request_file!r} (tried: {candidates})")

    print(f"\n{'=' * 60}")
    print(f"Running task: {user_request}")
    print(f"{'=' * 60}\n")

    orchestrator = AgentOrchestrator()

    start = time.time()
    try:
        # Build state manually so we can override budgets for testing.
        initial_state = create_initial_state(user_request=user_request)
        if args.max_steps is not None:
            initial_state["max_steps"] = int(args.max_steps)
        if args.max_tool_calls is not None:
            initial_state["max_tool_calls"] = int(args.max_tool_calls)
        if args.max_tokens is not None:
            initial_state["max_tokens"] = int(args.max_tokens)

        result = await orchestrator.graph.ainvoke(
            initial_state,
            config={"recursion_limit": 120},
        )
    except Exception as e:
        print(f"\n[ERROR] Task failed: {e}")
        import traceback
        traceback.print_exc()
        result = {"error": str(e)}

    elapsed = time.time() - start
    print(f"\n[Done] Task completed in {elapsed:.1f}s")

    # Save task result
    result_file = LOG_DIR / f"task_result_{timestamp}.json"
    try:
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        print(f"[Done] Task result saved to {result_file}")
    except Exception as e:
        print(f"[WARN] Could not save task result: {e}")

    # Write summary
    write_summary()

    print(f"\n[Files]")
    print(f"  LLM log (JSONL):  {LOG_FILE}")
    print(f"  Summary (text):   {SUMMARY_FILE}")
    print(f"  Task result:      {result_file}")

    if console_mirror_fp:
        console_mirror_fp.close()


if __name__ == "__main__":
    asyncio.run(main())
