"""Process lifecycle management: PID file, port check, graceful shutdown."""

import atexit
import logging
import os
import signal
import socket
import sys
from pathlib import Path

from agent_engine.config import get_settings

logger = logging.getLogger(__name__)

# PID file path (relative to project root)
_THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = _THIS_FILE.parent.parent.parent  # src/agent_engine/lifecycle.py -> project root
PID_FILE = PROJECT_ROOT / ".server.pid"


def check_port_available(host: str, port: int) -> bool:
    """Check if a port is available for binding.

    Args:
        host: Host address to check.
        port: Port number to check.

    Returns:
        True if port is available, False if already in use.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.settimeout(1)
    try:
        sock.bind((host, port))
        sock.close()
        return True
    except OSError:
        sock.close()
        return False


def get_existing_pid() -> int | None:
    """Read PID from PID file if it exists and process is alive.

    Returns:
        PID if a valid running process exists, None otherwise.
    """
    if not PID_FILE.exists():
        return None

    try:
        pid = int(PID_FILE.read_text().strip())
    except (ValueError, OSError):
        _cleanup_pid_file()
        return None

    # Check if process is still running
    if _is_process_alive(pid):
        return pid

    # Stale PID file
    _cleanup_pid_file()
    return None


def _is_process_alive(pid: int) -> bool:
    """Check if a process with the given PID is still running."""
    if sys.platform == "win32":
        # Windows: use ctypes to check process
        import ctypes
        kernel32 = ctypes.windll.kernel32
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if handle:
            kernel32.CloseHandle(handle)
            return True
        return False
    else:
        # Unix: send signal 0 to check
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


def write_pid_file() -> None:
    """Write current process PID to PID file."""
    pid = os.getpid()
    try:
        PID_FILE.write_text(str(pid))
        logger.info(f"PID file written: {PID_FILE} (PID={pid})")
    except OSError as e:
        logger.warning(f"Failed to write PID file: {e}")


def _cleanup_pid_file() -> None:
    """Remove PID file if it exists."""
    try:
        if PID_FILE.exists():
            PID_FILE.unlink()
            logger.info(f"PID file removed: {PID_FILE}")
    except OSError as e:
        logger.warning(f"Failed to remove PID file: {e}")


def ensure_single_instance(host: str, port: int) -> None:
    """Ensure only one instance of the server is running.

    Checks both PID file and port availability.
    Raises SystemExit if another instance is detected.

    Args:
        host: Host address.
        port: Port number.
    """
    # Check PID file first
    existing_pid = get_existing_pid()
    if existing_pid:
        logger.error(
            f"Another instance is already running (PID={existing_pid}). "
            f"PID file: {PID_FILE}. "
            f"Kill the existing process or remove the PID file to start a new instance."
        )
        sys.exit(1)

    # Check port availability
    if not check_port_available(host, port):
        logger.error(
            f"Port {port} is already in use on {host}. "
            f"Another server may be running without a PID file. "
            f"Free the port before starting."
        )
        sys.exit(1)

    # All clear — write PID file
    write_pid_file()

    # Register cleanup on exit
    atexit.register(_cleanup_pid_file)

    # Register signal handlers for graceful cleanup
    _register_signal_handlers()


def _register_signal_handlers() -> None:
    """Register signal handlers for graceful shutdown."""
    def _signal_handler(signum, frame):
        sig_name = signal.Signals(signum).name
        logger.info(f"Received signal {sig_name}, initiating shutdown...")
        _cleanup_pid_file()
        sys.exit(0)

    # SIGTERM (kill / docker stop)
    signal.signal(signal.SIGTERM, _signal_handler)

    # SIGINT is already handled by uvicorn, but we register cleanup via atexit


def request_shutdown() -> None:
    """Request a graceful shutdown of the current process.

    Sends SIGINT to self, which uvicorn handles for graceful shutdown.
    """
    pid = os.getpid()
    logger.info(f"Shutdown requested for PID={pid}")
    if sys.platform == "win32":
        # Windows: raise KeyboardInterrupt via CTRL_C_EVENT
        os.kill(pid, signal.CTRL_C_EVENT)
    else:
        os.kill(pid, signal.SIGINT)
