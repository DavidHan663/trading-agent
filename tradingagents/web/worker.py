"""Sequential subprocess worker with a hard timeout and clean shutdown."""

import json
import subprocess
import sys
import threading


class Worker:
    def __init__(self, store, data_dir, timeout):
        self.store = store
        self.data_dir = data_dir
        self.timeout = timeout
        self.stop_event = threading.Event()
        self.process = None
        self.thread = threading.Thread(target=self.loop, daemon=True)

    def start(self):
        self.store.recover()
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=5)
        if self.thread.is_alive() and self.process is not None:
            self.process.kill()
            self.thread.join(timeout=5)

    def loop(self):
        while not self.stop_event.is_set():
            job = self.store.claim()
            if job is None:
                self.stop_event.wait(0.5)
                continue
            self.execute(job)

    def execute(self, job):
        directory = self.data_dir / "reports" / job["id"]
        directory.mkdir(parents=True, exist_ok=True)
        try:
            # Provider exceptions may contain credentials. Do not capture or expose
            # raw SDK output in the browser, database, or default service logs.
            self.process = subprocess.Popen(
                [sys.executable, "-m", "tradingagents.web.runner", str(directory)],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                text=True,
            )
            self.process.communicate(json.dumps(job["request"]), timeout=self.timeout)
            if self.process.returncode:
                raise RuntimeError("Analysis process failed")
            result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
            if not (directory / "complete_report.md").is_file():
                raise RuntimeError("Report missing")
            self.store.finish(job["id"], signal=result["signal"])
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.communicate()
            self.store.finish(job["id"], error="分析超过时间上限，已停止。请减少分析师或讨论轮数后重试。")
        except Exception:
            self.store.finish(job["id"], error="分析未完成，请检查服务端模型、密钥和数据源配置后重试。")
        finally:
            self.process = None
