"""File operation tools."""

import os
from pathlib import Path

import aiofiles

from agent_engine.tools.registry import tool


def _safe_path(path: str, base_dir: str | None = None) -> Path:
    """Ensure path is safe and within allowed directory.

    Args:
        path: The path to check.
        base_dir: Base directory to restrict access to.

    Returns:
        Resolved Path object.

    Raises:
        ValueError: If path is outside allowed directory.
    """
    resolved = Path(path).resolve()

    if base_dir:
        base = Path(base_dir).resolve()
        if not str(resolved).startswith(str(base)):
            raise ValueError(f"Path {path} is outside allowed directory")

    return resolved


@tool(
    name="read_file",
    description="Read the contents of a file. Returns the file content as text.",
    tags=["file", "read", "io"],
)
async def read_file(
    path: str,
    encoding: str = "utf-8",
    max_size: int = 1_000_000,
) -> dict:
    """Read a file's contents.

    Args:
        path: Path to the file to read.
        encoding: File encoding (default utf-8).
        max_size: Maximum file size to read in bytes.

    Returns:
        Dictionary with file content or error.
    """
    try:
        file_path = _safe_path(path)

        if not file_path.exists():
            return {"error": f"File not found: {path}", "path": path}

        if not file_path.is_file():
            return {"error": f"Not a file: {path}", "path": path}

        file_size = file_path.stat().st_size
        if file_size > max_size:
            return {
                "error": f"File too large: {file_size} bytes (max {max_size})",
                "path": path,
                "size": file_size,
            }

        async with aiofiles.open(file_path, mode="r", encoding=encoding) as f:
            content = await f.read()

        return {
            "path": str(file_path),
            "content": content,
            "size": len(content),
            "encoding": encoding,
        }

    except UnicodeDecodeError as e:
        return {"error": f"Encoding error: {e}", "path": path}
    except Exception as e:
        return {"error": str(e), "path": path}


@tool(
    name="write_file",
    description="Write content to a file. Creates the file if it doesn't exist.",
    tags=["file", "write", "io"],
)
async def write_file(
    path: str,
    content: str,
    encoding: str = "utf-8",
    create_dirs: bool = True,
) -> dict:
    """Write content to a file.

    Args:
        path: Path to the file to write.
        content: Content to write.
        encoding: File encoding (default utf-8).
        create_dirs: Create parent directories if needed.

    Returns:
        Dictionary with result or error.
    """
    try:
        file_path = _safe_path(path)

        if create_dirs:
            file_path.parent.mkdir(parents=True, exist_ok=True)

        async with aiofiles.open(file_path, mode="w", encoding=encoding) as f:
            await f.write(content)

        return {
            "path": str(file_path),
            "size": len(content),
            "success": True,
        }

    except Exception as e:
        return {"error": str(e), "path": path, "success": False}


@tool(
    name="list_directory",
    description="List the contents of a directory. Returns files and subdirectories.",
    tags=["file", "directory", "io"],
)
async def list_directory(
    path: str,
    recursive: bool = False,
    include_hidden: bool = False,
    max_items: int = 1000,
) -> dict:
    """List directory contents.

    Args:
        path: Path to the directory.
        recursive: List recursively.
        include_hidden: Include hidden files/directories.
        max_items: Maximum items to return.

    Returns:
        Dictionary with directory contents or error.
    """
    try:
        dir_path = _safe_path(path)

        if not dir_path.exists():
            return {"error": f"Directory not found: {path}", "path": path}

        if not dir_path.is_dir():
            return {"error": f"Not a directory: {path}", "path": path}

        items: list[dict] = []
        count = 0

        if recursive:
            iterator = dir_path.rglob("*")
        else:
            iterator = dir_path.iterdir()

        for item in iterator:
            if count >= max_items:
                break

            if not include_hidden and item.name.startswith("."):
                continue

            try:
                stat = item.stat()
                items.append({
                    "name": item.name,
                    "path": str(item),
                    "type": "directory" if item.is_dir() else "file",
                    "size": stat.st_size if item.is_file() else None,
                    "modified": stat.st_mtime,
                })
                count += 1
            except (PermissionError, OSError):
                continue

        return {
            "path": str(dir_path),
            "items": items,
            "count": len(items),
            "truncated": count >= max_items,
        }

    except Exception as e:
        return {"error": str(e), "path": path}


@tool(
    name="delete_file",
    description="Delete a file or empty directory.",
    tags=["file", "delete", "io"],
)
async def delete_file(path: str) -> dict:
    """Delete a file or empty directory.

    Args:
        path: Path to delete.

    Returns:
        Dictionary with result or error.
    """
    try:
        file_path = _safe_path(path)

        if not file_path.exists():
            return {"error": f"Path not found: {path}", "path": path, "success": False}

        if file_path.is_file():
            os.remove(file_path)
        elif file_path.is_dir():
            os.rmdir(file_path)  # Only removes empty directories

        return {"path": str(file_path), "success": True}

    except OSError as e:
        return {"error": str(e), "path": path, "success": False}
    except Exception as e:
        return {"error": str(e), "path": path, "success": False}
