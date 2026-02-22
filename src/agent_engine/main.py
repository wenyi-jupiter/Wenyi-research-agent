"""FastAPI application entry point."""

import logging
import uvicorn
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from agent_engine.api.routes import router as api_router
from agent_engine.api.websocket import router as ws_router
from agent_engine.config import get_settings
from agent_engine.tools.builtin import (
    code_execute,
    fetch_url,
    list_directory,
    read_file,
    web_search,
    write_file,
)

logger = logging.getLogger(__name__)

# ---------- Shared engine reference for cleanup ----------
_db_engine = None


def get_shared_db_engine():
    """Get or create the shared database engine (singleton)."""
    global _db_engine
    if _db_engine is None:
        from sqlalchemy.ext.asyncio import create_async_engine
        settings = get_settings()
        _db_engine = create_async_engine(settings.database_url, echo=False)
    return _db_engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management.

    Startup:
        - Register built-in tools
        - Initialize database engine
        - Write PID file (for non-reload worker)
    Shutdown:
        - Close all WebSocket connections
        - Dispose database engine (close connection pool)
        - Clean up PID file
    """
    from agent_engine.lifecycle import write_pid_file, _cleanup_pid_file

    settings = get_settings()

    # --- Startup ---
    logger.info("Starting Agent Engine...")

    # Register built-in tools (auto-register via decorators)
    _ = (code_execute, fetch_url, list_directory, read_file, web_search, write_file)
    logger.info("Built-in tools registered")

    # Install LLM interaction logging patches
    from agent_engine.llm_logger import install_patches
    install_patches()

    # Initialize database engine (warm up connection pool)
    try:
        engine = get_shared_db_engine()
        logger.info(f"Database engine initialized: {settings.database_url.split('@')[-1]}")
    except Exception as e:
        logger.warning(f"Database engine init warning: {e}")

    # Write PID file for this worker
    write_pid_file()
    logger.info("Application startup complete")

    yield

    # --- Shutdown ---
    logger.info("Shutting down Agent Engine...")

    # 1. Close all WebSocket connections
    from agent_engine.api.websocket import manager
    await manager.close_all()
    logger.info("All WebSocket connections closed")

    # 2. Dispose database engine (close connection pool)
    if _db_engine is not None:
        await _db_engine.dispose()
        logger.info("Database connection pool closed")

    # 3. Clean up PID file
    _cleanup_pid_file()
    logger.info("Application shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="Agent Engine",
    description="Multi-Agent Collaboration Execution Engine based on LangGraph",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(api_router, prefix="/api/v1")
app.include_router(ws_router, prefix="/ws")


# Serve frontend static files
# Path: src/agent_engine/main.py -> need to go up to project root
_this_file = Path(__file__).resolve()
PROJECT_ROOT = _this_file.parent.parent.parent  # src/agent_engine/main.py -> agent_demo
frontend_path = PROJECT_ROOT / "frontend"


@app.get("/")
async def root():
    """Serve the frontend UI (no cache to ensure latest code)."""
    index_file = frontend_path / "index.html"
    if index_file.exists():
        return FileResponse(
            str(index_file),
            media_type="text/html",
            headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
        )
    return {
        "name": "Agent Engine",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/api/v1/health",
    }


# Mount static files after defining routes
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")


@app.get("/api")
async def api_info():
    """API info endpoint."""
    return {
        "name": "Agent Engine API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/api/v1/health",
    }


def run():
    """Run the application."""
    import socket as _socket

    settings = get_settings()

    # Pre-flight checks before starting uvicorn
    from agent_engine.lifecycle import ensure_single_instance
    ensure_single_instance(settings.api_host, settings.api_port)

    logger.info(f"Starting server on {settings.api_host}:{settings.api_port}")

    # Pre-bind socket with SO_REUSEADDR to handle Windows ghost ports
    # (ports stuck in TIME_WAIT / CLOSE_WAIT after process death).
    # Uvicorn's bind_socket also sets SO_REUSEADDR, but on Windows it can
    # race with the kernel cleanup.  Pre-binding guarantees the address is
    # reserved before uvicorn's async startup.
    sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    sock.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
    sock.bind((settings.api_host, settings.api_port))
    sock.set_inheritable(True)

    config = uvicorn.Config(
        app,  # Pass app object directly instead of string import
        host=settings.api_host,
        port=settings.api_port,
        reload=False,  # Cannot use reload with app object
        log_level="info",
    )
    config.load()  # Initialize lifespan, logging, etc.
    server = uvicorn.Server(config)

    # Use asyncio.run() to pass the pre-bound socket.
    # Flush stdout/stderr to ensure logs appear in pipe/terminal captures.
    import asyncio
    import sys

    async def _serve():
        await server.serve(sockets=[sock])

    try:
        asyncio.run(_serve())
    finally:
        sys.stdout.flush()
        sys.stderr.flush()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
