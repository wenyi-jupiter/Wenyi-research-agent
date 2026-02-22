"""Code execution tool with sandboxing."""

import asyncio
import subprocess
import sys
import tempfile
from pathlib import Path

from agent_engine.tools.registry import tool


@tool(
    name="code_execute",
    description="Execute Python code in a sandboxed environment. Returns stdout, stderr, and return code.",
    tags=["code", "python", "execute"],
)
async def code_execute(
    code: str,
    timeout: float = 30.0,
    capture_output: bool = True,
) -> dict:
    """Execute Python code.

    Args:
        code: Python code to execute.
        timeout: Maximum execution time in seconds.
        capture_output: Capture stdout and stderr.

    Returns:
        Dictionary with execution results.

    Note:
        This is a basic implementation. For production, consider:
        - Using Docker containers for isolation
        - Resource limits (memory, CPU)
        - Network restrictions
        - File system sandboxing
    """
    # Create a temporary file for the code
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False,
        encoding="utf-8",
    ) as f:
        f.write(code)
        temp_path = f.name

    try:
        # Run in subprocess with timeout
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            temp_path,
            stdout=asyncio.subprocess.PIPE if capture_output else None,
            stderr=asyncio.subprocess.PIPE if capture_output else None,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout,
            )

            return {
                "success": process.returncode == 0,
                "return_code": process.returncode,
                "stdout": stdout.decode("utf-8") if stdout else "",
                "stderr": stderr.decode("utf-8") if stderr else "",
            }

        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return {
                "success": False,
                "return_code": -1,
                "error": f"Execution timed out after {timeout} seconds",
                "stdout": "",
                "stderr": "",
            }

    except Exception as e:
        return {
            "success": False,
            "return_code": -1,
            "error": str(e),
            "stdout": "",
            "stderr": "",
        }

    finally:
        # Clean up temp file
        Path(temp_path).unlink(missing_ok=True)


@tool(
    name="shell_execute",
    description="Execute a shell command. Use with caution - only for trusted commands.",
    tags=["shell", "command", "execute"],
)
async def shell_execute(
    command: str,
    working_dir: str | None = None,
    timeout: float = 60.0,
    shell: bool = True,
) -> dict:
    """Execute a shell command.

    Args:
        command: Command to execute.
        working_dir: Working directory for the command.
        timeout: Maximum execution time in seconds.
        shell: Use shell execution.

    Returns:
        Dictionary with execution results.

    Warning:
        This tool can execute arbitrary commands. Use with extreme caution
        and only in trusted environments.
    """
    try:
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=working_dir,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout,
            )

            return {
                "success": process.returncode == 0,
                "return_code": process.returncode,
                "stdout": stdout.decode("utf-8") if stdout else "",
                "stderr": stderr.decode("utf-8") if stderr else "",
                "command": command,
            }

        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return {
                "success": False,
                "return_code": -1,
                "error": f"Command timed out after {timeout} seconds",
                "stdout": "",
                "stderr": "",
                "command": command,
            }

    except Exception as e:
        return {
            "success": False,
            "return_code": -1,
            "error": str(e),
            "stdout": "",
            "stderr": "",
            "command": command,
        }


@tool(
    name="python_eval",
    description="Evaluate a Python expression and return the result. For simple calculations and data transformations.",
    tags=["python", "eval", "calculate"],
)
async def python_eval(expression: str) -> dict:
    """Evaluate a Python expression.

    Args:
        expression: Python expression to evaluate.

    Returns:
        Dictionary with result or error.

    Note:
        This uses eval() with restricted builtins for safety.
        Only basic operations are allowed.
    """
    # Restricted builtins for safer eval
    safe_builtins = {
        "abs": abs,
        "all": all,
        "any": any,
        "bin": bin,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "filter": filter,
        "float": float,
        "format": format,
        "hex": hex,
        "int": int,
        "len": len,
        "list": list,
        "map": map,
        "max": max,
        "min": min,
        "oct": oct,
        "ord": ord,
        "pow": pow,
        "range": range,
        "reversed": reversed,
        "round": round,
        "set": set,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
        "True": True,
        "False": False,
        "None": None,
    }

    try:
        # Add math functions
        import math

        safe_builtins.update({
            "math": math,
            "pi": math.pi,
            "e": math.e,
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "log": math.log,
            "log10": math.log10,
            "exp": math.exp,
            "ceil": math.ceil,
            "floor": math.floor,
        })

        result = eval(expression, {"__builtins__": safe_builtins}, {})

        return {
            "success": True,
            "expression": expression,
            "result": result,
            "type": type(result).__name__,
        }

    except SyntaxError as e:
        return {
            "success": False,
            "expression": expression,
            "error": f"Syntax error: {e}",
        }
    except NameError as e:
        return {
            "success": False,
            "expression": expression,
            "error": f"Name error: {e}",
        }
    except Exception as e:
        return {
            "success": False,
            "expression": expression,
            "error": str(e),
        }
