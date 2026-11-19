"""Authenticated single-process ASGI application for personal deployments."""

import fcntl
import hashlib
import json
import os
import secrets
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlsplit

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .store import Store
from .worker import Worker

STATIC = Path(__file__).parent / "static"


class AnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ticker: str = Field(pattern=r"^[A-Z0-9^][A-Z0-9.^=\-]{0,31}$")
    trade_date: date
    asset_type: Literal["stock", "crypto"] = "stock"
    analysts: list[Literal["market", "news", "social", "fundamentals"]] = Field(
        default=["market", "news", "fundamentals"], min_length=1, max_length=4,
    )
    rounds: int = Field(default=1, ge=1, le=3, strict=True)

    @field_validator("trade_date")
    @classmethod
    def validate_date(cls, value):
        if value > date.today() or value < date(2000, 1, 1):
            raise ValueError("请选择 2000 年至今天之间的日期")
        return value

    @field_validator("analysts")
    @classmethod
    def unique_analysts(cls, value):
        return list(dict.fromkeys(value))


def create_app(data_dir=None):
    load_dotenv()
    username = os.getenv("TRADINGAGENTS_WEB_USER", "admin")
    password = os.getenv("TRADINGAGENTS_WEB_PASSWORD", "")
    if len(password) < 16 or not username:
        raise RuntimeError("Set TRADINGAGENTS_WEB_PASSWORD (at least 16 characters) and a username")
    root = Path(data_dir or os.getenv("TRADINGAGENTS_WEB_DATA_DIR", "~/.tradingagents/web")).expanduser().resolve()
    timeout = int(os.getenv("TRADINGAGENTS_WEB_JOB_TIMEOUT", "1800"))
    if not 60 <= timeout <= 7200:
        raise RuntimeError("TRADINGAGENTS_WEB_JOB_TIMEOUT must be between 60 and 7200 seconds")

    @asynccontextmanager
    async def lifespan(app):
        root.mkdir(parents=True, exist_ok=True)
        with (root / "worker.lock").open("w") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("Only one web process may use this data directory") from exc
            app.state.store = Store(root / "jobs.sqlite3")
            worker = Worker(app.state.store, root, timeout)
            worker.start()
            try:
                yield
            finally:
                worker.stop()

    security = HTTPBasic()

    def authenticate(credentials: Annotated[HTTPBasicCredentials, Depends(security)]):
        valid_user = secrets.compare_digest(credentials.username.encode(), username.encode())
        valid_password = secrets.compare_digest(
            hashlib.sha256(credentials.password.encode()).digest(),
            hashlib.sha256(password.encode()).digest(),
        )
        if not (valid_user and valid_password):
            raise HTTPException(401, "登录信息不正确", headers={"WWW-Authenticate": 'Basic realm="TradingAgents"'})

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    @app.middleware("http")
    async def guard(request: Request, call_next):
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            origin = request.headers.get("origin")
            if (request.headers.get("sec-fetch-site") == "cross-site"
                    or (origin and urlsplit(origin).netloc != request.headers.get("host"))):
                return JSONResponse({"detail": "跨站请求被拒绝"}, status_code=403)
            if request.headers.get("content-type", "").split(";")[0] != "application/json":
                return JSONResponse({"detail": "需要 JSON 请求"}, status_code=415)
            # Bound actual streamed bytes, including chunked requests.
            body = bytearray()
            async for chunk in request.stream():
                body.extend(chunk)
                if len(body) > 8192:
                    return JSONResponse({"detail": "请求过大"}, status_code=413)
            request._body = bytes(body)
        response = await call_next(request)
        response.headers.update({
            "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY", "Referrer-Policy": "no-referrer",
            "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'",
        })
        return response

    @app.get("/healthz")
    def health():
        return {"status": "ok"}

    @app.get("/", dependencies=[Depends(authenticate)])
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/static/{name}", dependencies=[Depends(authenticate)])
    def static(name: Literal["app.js", "style.css"]):
        return FileResponse(STATIC / name)

    @app.get("/api/config", dependencies=[Depends(authenticate)])
    def config():
        from tradingagents.default_config import DEFAULT_CONFIG
        from tradingagents.llm_clients.api_key_env import get_api_key_env

        provider = DEFAULT_CONFIG["llm_provider"]
        key = get_api_key_env(provider)
        return {
            "provider": provider, "deep_model": DEFAULT_CONFIG["deep_think_llm"],
            "quick_model": DEFAULT_CONFIG["quick_think_llm"],
            "configured": bool(os.getenv(key)) if key and provider != "openai_compatible" else True,
            "timeout_seconds": timeout,
        }

    @app.get("/api/jobs", dependencies=[Depends(authenticate)])
    def jobs():
        return app.state.store.list()

    @app.post("/api/jobs", status_code=202, dependencies=[Depends(authenticate)])
    def submit(payload: AnalysisRequest):
        if not config()["configured"]:
            raise HTTPException(503, "请先在服务端配置模型 API 密钥")
        try:
            return app.state.store.create(payload.model_dump(mode="json"))
        except OverflowError as exc:
            raise HTTPException(429, str(exc)) from exc

    def get_job(job_id):
        job = app.state.store.get(job_id)
        if job is None:
            raise HTTPException(404, "任务不存在")
        return job

    @app.get("/api/jobs/{job_id}", dependencies=[Depends(authenticate)])
    def detail(job_id: str):
        return get_job(job_id)

    @app.get("/api/jobs/{job_id}/report", dependencies=[Depends(authenticate)])
    def report(job_id: str):
        job = get_job(job_id)
        if job["status"] != "completed":
            raise HTTPException(409, "报告尚未生成")
        path = root / "reports" / job["id"] / "complete_report.md"
        if not path.is_file():
            raise HTTPException(404, "报告文件不存在，请检查数据卷")
        return FileResponse(path, media_type="text/markdown", filename=f"{job['request']['ticker']}-report.md")

    @app.get("/api/jobs/{job_id}/evidence", dependencies=[Depends(authenticate)])
    def report_evidence(job_id: str):
        job = get_job(job_id)
        if job["status"] != "completed":
            raise HTTPException(409, "报告尚未生成")
        path = root / "reports" / job["id"] / "evidence.json"
        if not path.is_file():
            return {"available": False, "claims": [], "records": []}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("invalid evidence document")
            semantic = data.get("semantic_validation") or {}
            records = data.get("records") or []
            if not isinstance(semantic, dict) or not isinstance(records, list):
                raise ValueError("invalid evidence fields")
            claims = semantic.get("claims") or []
            if not isinstance(claims, list) or any(not isinstance(c, dict) for c in claims):
                raise ValueError("invalid claims")
            if any(not isinstance(record, dict) for record in records):
                raise ValueError("invalid records")
        except (OSError, UnicodeError, ValueError):
            raise HTTPException(500, "证据数据无法读取")
        claim_fields = ("claim_id", "claim_type", "claim", "decision", "reason_flags", "reasoning", "missing_information", "evidence_quotes", "failure_reason")
        record_fields = ("id", "tool", "target_ticker", "requested_ticker", "source_ticker", "source_url", "available_at", "period_end", "fetched_at")
        return {"available": bool(claims), "claims": [{k: c.get(k) for k in claim_fields} for c in claims],
                "records": [{k: record.get(k) for k in record_fields} for record in records]}

    return app


def main():
    import uvicorn

    uvicorn.run("tradingagents.web.app:create_app", factory=True,
                host=os.getenv("TRADINGAGENTS_WEB_HOST", "127.0.0.1"),
                port=int(os.getenv("PORT", "8000")), workers=1)
