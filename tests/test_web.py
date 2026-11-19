"""Private web security, durable queue and worker lifecycle without paid APIs."""

import json
import subprocess
import time
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from tradingagents.web.app import create_app
from tradingagents.web.store import Store
from tradingagents.web.worker import Worker

PASSWORD = "test-only-password-123456789"
PAYLOAD = {"ticker": "AAPL", "trade_date": "2026-01-02", "analysts": ["market"], "rounds": 1}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADINGAGENTS_WEB_PASSWORD", PASSWORD)
    monkeypatch.setenv("TRADINGAGENTS_WEB_USER", "admin")
    monkeypatch.setattr(Worker, "start", lambda self: self.store.recover())
    monkeypatch.setattr(Worker, "stop", lambda self: None)
    with TestClient(create_app(tmp_path)) as session:
        session.auth = ("admin", PASSWORD)
        yield session


def test_password_required(monkeypatch):
    monkeypatch.setenv("TRADINGAGENTS_WEB_PASSWORD", "short")
    with pytest.raises(RuntimeError, match="16 characters"):
        create_app()


def test_auth_and_security_headers(client):
    for path in ["/", "/api/config", "/api/jobs", "/static/app.js", "/api/jobs/missing/report"]:
        assert client.get(path, auth=("admin", "wrong")).status_code == 401
    assert client.post("/api/jobs", json=PAYLOAD, auth=("admin", "wrong")).status_code == 401
    response = client.get("/")
    assert response.status_code == 200
    assert "no-store" in response.headers["cache-control"]
    assert "frame-ancestors 'none'" in response.headers["content-security-policy"]
    assert client.get("/healthz", auth=None).json() == {"status": "ok"}
    assert client.get("/docs").status_code == 404


def test_config_does_not_expose_secrets(client, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "server-secret-should-never-be-shared")
    response = client.get("/api/config")
    assert response.json()["configured"]
    assert "server-secret" not in response.text
    assert PASSWORD not in response.text


@pytest.mark.parametrize("changes", [
    {"ticker": "../../etc/passwd"}, {"ticker": "<script>"},
    {"trade_date": "2099-01-01"}, {"analysts": []}, {"analysts": ["unknown"]},
    {"rounds": 4}, {"rounds": True}, {"backend_url": "http://internal"},
])
def test_request_validation(client, changes):
    assert client.post("/api/jobs", json={**PAYLOAD, **changes}).status_code == 422
    assert client.get("/api/jobs").json() == []


def test_csrf_and_size_limits(client):
    assert client.post("/api/jobs", json=PAYLOAD, headers={"origin": "https://evil.test"}).status_code == 403
    assert client.post("/api/jobs", json=PAYLOAD, headers={"sec-fetch-site": "cross-site"}).status_code == 403
    assert client.post("/api/jobs", content=json.dumps(PAYLOAD), headers={"content-type": "text/plain"}).status_code == 415
    assert client.post("/api/jobs", content="x" * 9000, headers={"content-type": "application/json"}).status_code == 413


def test_queue_capacity_and_reports(client, tmp_path):
    for _ in range(10):
        response = client.post("/api/jobs", json=PAYLOAD)
        assert response.status_code == 202
    assert client.post("/api/jobs", json=PAYLOAD).status_code == 429
    job = response.json()
    path = f"/api/jobs/{job['id']}/report"
    assert client.get(path).status_code == 409
    directory = tmp_path / "reports" / job["id"]
    directory.mkdir(parents=True)
    (directory / "complete_report.md").write_text("# Example\n<script>alert(1)</script>")
    client.app.state.store.finish(job["id"], signal="Hold")
    response = client.get(path)
    assert response.status_code == 200
    assert "attachment" in response.headers["content-disposition"]
    assert response.headers["x-content-type-options"] == "nosniff"
    assert client.get("/api/jobs/unknown").status_code == 404


def test_report_evidence_auth_legacy_and_invalid_data(client, tmp_path):
    job = client.post("/api/jobs", json=PAYLOAD).json()
    path = f"/api/jobs/{job['id']}/evidence"
    assert client.get(path, auth=("admin", "wrong")).status_code == 401
    assert client.get(path).status_code == 409
    directory = tmp_path / "reports" / job["id"]
    directory.mkdir(parents=True)
    client.app.state.store.finish(job["id"], signal="Hold")
    assert client.get(path).json() == {"available": False, "claims": [], "records": []}
    (directory / "evidence.json").write_text(json.dumps({
        "records": [{"id": "E1", "source_url": "https://example.com", "content": "private body"}],
        "semantic_validation": {"claims": [{"claim_id": "C1", "claim": "example", "decision": "review", "reasoning": "missing"}]},
    }))
    data = client.get(path).json()
    assert data["claims"][0]["reasoning"] == "missing"
    assert data["records"][0]["source_url"] == "https://example.com"
    assert "content" not in data["records"][0]
    (directory / "evidence.json").write_text('{invalid')
    assert client.get(path).status_code == 500


def test_restart_recovery(tmp_path):
    store = Store(tmp_path / "jobs.sqlite3")
    running = store.create(PAYLOAD)
    store.claim()
    queued = store.create(PAYLOAD)
    reopened = Store(tmp_path / "jobs.sqlite3")
    reopened.recover()
    assert reopened.get(running["id"])["status"] == "failed"
    assert reopened.claim()["id"] == queued["id"]


def test_single_process_lock(client, tmp_path):
    with pytest.raises(RuntimeError, match="Only one"), TestClient(create_app(tmp_path)):
        pass


def test_worker_timeout_and_error_redaction(tmp_path, monkeypatch):
    store = Store(tmp_path / "jobs.sqlite3")
    job = store.create(PAYLOAD)
    process = MagicMock()
    process.communicate.side_effect = [subprocess.TimeoutExpired("runner", 60), (None, None)]
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: process)
    Worker(store, tmp_path, 60).execute(job)
    process.kill.assert_called_once()
    assert store.get(job["id"])["status"] == "failed"
    assert "时间上限" in store.get(job["id"])["error"]
    process.communicate.side_effect = RuntimeError("SECRET_API_KEY")
    Worker(store, tmp_path, 60).execute(job)
    assert "SECRET" not in store.get(job["id"])["error"]


def test_worker_completes_and_stops(tmp_path, monkeypatch):
    store = Store(tmp_path / "jobs.sqlite3")
    job = store.create(PAYLOAD)
    directory = tmp_path / "reports" / job["id"]
    directory.mkdir(parents=True)
    (directory / "complete_report.md").write_text("# Research")
    (directory / "result.json").write_text('{"signal": "Hold"}')
    process = MagicMock(returncode=0)
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: process)
    worker = Worker(store, tmp_path, 60)
    worker.start()
    deadline = time.monotonic() + 3
    while store.get(job["id"])["status"] != "completed" and time.monotonic() < deadline:
        time.sleep(0.01)
    worker.stop()
    assert store.get(job["id"])["signal"] == "Hold"
    assert not worker.thread.is_alive()
