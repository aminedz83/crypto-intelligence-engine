"""FastAPI application factory."""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import api_router
from app.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.health.router import router as health_router

settings = get_settings()
configure_logging("DEBUG" if settings.debug else "INFO")
log = get_logger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting %s (env=%s)", settings.app_name, settings.environment)
    # Hard guard: this codebase has no live-execution path. Refuse to boot in
    # production if someone flips the flag expecting real trading.
    if settings.is_production and settings.live_trading_enabled:
        raise RuntimeError(
            "live_trading_enabled=True but this build supports paper trading only."
        )
    yield
    log.info("Shutting down %s", settings.app_name)


def _frontend_dir(cfg) -> Path:
    """Resolve the frontend directory. FRONTEND_DIR overrides; otherwise the
    repo's frontend/ (main.py lives at backend/app/main.py → parents[2] = repo)."""
    if cfg.frontend_dir:
        return Path(cfg.frontend_dir)
    return Path(__file__).resolve().parents[2] / "frontend"


def _mount_frontend(app: FastAPI, cfg) -> None:
    """Serve the vanilla frontend from FastAPI.

    /            -> frontend/index.html
    /css, /js, /assets -> static files
    /api/v1, /health   -> unchanged (routers are registered first, so they win)

    If the frontend is not present (e.g. the API-only Docker image), this is a
    no-op and the backend serves the API alone — Phase 1 behaviour is preserved.
    No fictitious route or data is added.
    """
    frontend = _frontend_dir(cfg)
    index = frontend / "index.html"
    if not index.exists():
        log.info("Frontend not found at %s — serving API only.", frontend)
        return
    for sub in ("css", "js", "assets"):
        directory = frontend / sub
        if directory.is_dir():
            app.mount(f"/{sub}", StaticFiles(directory=str(directory)), name=f"static-{sub}")

    @app.get("/", include_in_schema=False)
    async def serve_index() -> FileResponse:
        return FileResponse(str(index), media_type="text/html")

    log.info("Serving frontend from %s", frontend)


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description="Real-time crypto intelligence & paper-trading platform (Phase 1).",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health_router)
    app.include_router(api_router, prefix=settings.api_prefix)
    # Mount the frontend LAST so /health and /api/v1 routes always take priority.
    _mount_frontend(app, settings)
    return app


app = create_app()
