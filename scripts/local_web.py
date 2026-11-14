"""Start a private browser workspace on this computer; no cloud account needed."""

import argparse
import base64
import errno
import os
import secrets
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path
from urllib.parse import quote

PROJECT = Path(__file__).resolve().parents[1]
LOCAL = PROJECT / ".local-web"


def password():
    LOCAL.mkdir(mode=0o700, exist_ok=True)
    path = LOCAL / "password"
    if not path.exists():
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            pass
        else:
            with os.fdopen(descriptor, "w") as stream:
                stream.write(secrets.token_urlsafe(32))
    return path.read_text().strip()


def ready(port, secret):
    token = base64.b64encode(f"admin:{secret}".encode()).decode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/jobs", headers={"Authorization": f"Basic {token}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=1) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def open_browser(port, secret):
    # Random credentials only authorize this loopback-only service. The browser
    # caches Basic auth, so subsequent navigation can use the clean local URL.
    webbrowser.open(f"http://admin:{quote(secret, safe='')}@127.0.0.1:{port}/")


def main():
    parser = argparse.ArgumentParser(description="在本机打开 TradingAgents 网页")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    secret = password()
    port_file = LOCAL / "port"
    if port_file.exists():
        previous_port = int(port_file.read_text())
        if ready(previous_port, secret):
            print(f"网页已在运行：http://127.0.0.1:{previous_port}", flush=True)
            if not args.no_browser:
                open_browser(previous_port, secret)
            return
    for port in range(8765, 8786):
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError as exc:
                if exc.errno == errno.EADDRINUSE:
                    continue
                raise RuntimeError("系统未允许打开本地网页端口，请检查本机网络权限。") from exc
            break
    else:
        raise RuntimeError("本地端口已占用，请关闭旧的网页程序后重试。")

    environment = os.environ.copy()
    environment.update({
        "TRADINGAGENTS_WEB_USER": "admin", "TRADINGAGENTS_WEB_PASSWORD": secret,
        "TRADINGAGENTS_WEB_DATA_DIR": str(LOCAL / "data"),
        "TRADINGAGENTS_RESULTS_DIR": str(LOCAL / "engine" / "logs"),
        "TRADINGAGENTS_CACHE_DIR": str(LOCAL / "engine" / "cache"),
        "TRADINGAGENTS_MEMORY_LOG_PATH": str(LOCAL / "engine" / "memory.md"),
    })
    port_file.write_text(str(port))
    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "tradingagents.web.app:create_app", "--factory",
         "--host", "127.0.0.1", "--port", str(port), "--workers", "1"],
        cwd=PROJECT, env=environment,
    )
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError("网页启动失败，请检查上方提示。")
            if ready(port, secret):
                print(f"\n网页已准备好：http://127.0.0.1:{port}", flush=True)
                print("使用期间请保留此窗口；退出请按 Control+C。", flush=True)
                if not args.no_browser:
                    open_browser(port, secret)
                break
            time.sleep(0.2)
        else:
            raise RuntimeError("网页启动超时，请重试。")
        process.wait()
    except KeyboardInterrupt:
        print("\n正在关闭网页程序…", flush=True)
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    main()
