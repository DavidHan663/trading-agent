# 可选 Docker 部署

本地日常使用直接按 [README](README.md) 双击启动，无需 Docker。这里保留唯一一套网页容器配置。

## 启动

需要 Docker 和 Compose。配置本机 `.env`：模型服务商、API 密钥、有效模型 ID，以及至少 16 位的随机 `TRADINGAGENTS_WEB_PASSWORD`。

```bash
docker compose -f compose.web.yml up -d --build
```

打开 `http://127.0.0.1:8000`，账号默认 `admin`，密码为环境文件中的值。数据保存在 `research_data` 卷；不要执行 `down -v`，以免删除数据。每个数据目录只支持一个服务进程。

## 可选域名 HTTPS

在服务器上将域名指向服务器 IP，开放 TCP 80/443，在 `.env` 设置 `TRADINGAGENTS_DOMAIN=你的域名`，再运行：

```bash
docker compose -f compose.web.yml --profile https up -d --build
```

Caddy 负责证书与 HTTPS。公网访问应使用 HTTPS；应用 8000 端口仍仅绑定服务器本机。

## 状态与备份

```bash
docker compose -f compose.web.yml ps
docker compose -f compose.web.yml logs --tail 50 web
```

备份时先停止服务，再导出持久化目录：

```bash
docker compose -f compose.web.yml --profile https stop
docker compose -f compose.web.yml run --rm --no-deps --entrypoint tar web czf - -C /home/appuser .tradingagents > research-backup.tar.gz
docker compose -f compose.web.yml --profile https up -d
```

镜像按 `requirements-web.lock` 固定版本安装。修改依赖后执行 `uv lock`，再通过 `uv export --locked --no-dev --extra web --no-emit-project --no-hashes --output-file requirements-web.lock` 同步导出文件。
