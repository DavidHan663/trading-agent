"""Durable bounded job queue. One worker owns this database at a time."""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from uuid import uuid4


def now():
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, path):
        self.path = path
        with self.connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, request TEXT NOT NULL, status TEXT NOT NULL,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                signal TEXT, error TEXT)""")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def decode(row):
        if row is None:
            return None
        item = dict(row)
        item["request"] = json.loads(item["request"])
        return item

    def create(self, request):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            count = db.execute(
                "SELECT count(*) FROM jobs WHERE status IN ('queued','running')"
            ).fetchone()[0]
            if count >= 10:
                raise OverflowError("任务队列已满，请等待现有任务完成。")
            job_id = uuid4().hex
            stamp = now()
            db.execute(
                "INSERT INTO jobs VALUES (?, ?, 'queued', ?, ?, NULL, NULL)",
                (job_id, json.dumps(request), stamp, stamp),
            )
        return self.get(job_id)

    def get(self, job_id):
        with self.connect() as db:
            return self.decode(db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone())

    def list(self):
        with self.connect() as db:
            return [self.decode(row) for row in db.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT 100"
            )]

    def recover(self):
        with self.connect() as db:
            db.execute(
                "UPDATE jobs SET status='failed', error=?, updated_at=? WHERE status='running'",
                ("服务曾中断，此任务未完成。请重新提交。", now()),
            )

    def claim(self):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT * FROM jobs WHERE status='queued' ORDER BY created_at LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            db.execute("UPDATE jobs SET status='running', updated_at=? WHERE id=?", (now(), row['id']))
        return self.get(row["id"])

    def finish(self, job_id, *, signal=None, error=None):
        with self.connect() as db:
            db.execute(
                "UPDATE jobs SET status=?, signal=?, error=?, updated_at=? WHERE id=?",
                ("failed" if error else "completed", signal, error, now(), job_id),
            )
