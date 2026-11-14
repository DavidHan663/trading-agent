# 私人研究工作台部署

这是单用户版本：中文网页、HTTP Basic 登录、持久化任务队列、后台分析、历史报告及 Markdown 下载。沿用原有 Python 分析引擎，不连接券商或自动下单。

## 只想用网页，不买服务器（推荐）

macOS 下双击项目根目录的 **`启动网页.command`** 即可。首次启动自动准备依赖，之后自动打开浏览器。保留启动窗口；退出请按 Control+C。无需 Docker、云服务器或域名。

网页仅监听本机 `127.0.0.1`，默认端口 8765（占用时自动选其他端口）。启动器会生成本机随机访问密码并自动登录，重开浏览器可以再次双击启动入口。终端窗口只负责保持程序运行，日常研究全在网页完成。

在 `.env` 中配置你使用的模型密钥、服务商和模型 ID（见下一节示例）；没有配置时仍可打开页面，但不能完成真实分析。此模式的历史、报告、缓存和本机访问密码全部保存在项目的 `.local-web/`，删除该目录会清除这些本地数据。备份时先退出程序，再复制整个 `.local-web/`。

这种模式不支付服务器或域名费用；调用云端模型仍按服务商计费，市场数据也需要网络连接。电脑休眠或关闭本地程序会中断任务。需要手机访问或全天候运行时，再选择下面的远程部署方案。

## 本地试用

需要 Python 3.10+，支持 Linux 和 macOS。Windows 请通过 Docker 或 WSL 运行。

```bash
pip install '.[web]'
cp .env.example .env  # 已有 .env 时不要覆盖，补齐下面的变量即可
```

在 `.env` 中配置登录密码及你有权限使用的模型。模型 ID 必须以服务商实际提供的为准；仓库默认模型不保证在你的账号可用。

```dotenv
TRADINGAGENTS_WEB_USER=admin
TRADINGAGENTS_WEB_PASSWORD=替换为至少16位的随机密码
TRADINGAGENTS_LLM_PROVIDER=openai
OPENAI_API_KEY=替换为你的密钥
TRADINGAGENTS_DEEP_THINK_LLM=替换为可用模型ID
TRADINGAGENTS_QUICK_THINK_LLM=替换为可用模型ID
TRADINGAGENTS_LLM_MAX_RETRIES=2
TRADINGAGENTS_MAX_TOKENS=4096
```

可通过 `python -c 'import secrets; print(secrets.token_urlsafe(32))'` 生成密码。不要使用文档中的示例文字作为实际密码。

```bash
tradingagents-web
```

打开 <http://127.0.0.1:8000>，在浏览器登录框输入账号密码。也可直接运行：

```bash
python -m uvicorn tradingagents.web.app:create_app --factory --host 127.0.0.1 --port 8000 --workers 1
```

网页只展示服务商、模型名称和密钥是否已配置，API 密钥不会传给浏览器。默认数据源沿用 yfinance；各数据源的额度、访问权限及可用性需使用自己的网络和账户验证。

## Docker 部署

服务器需要 Docker 和 Compose。复制项目（包括隐藏配置文件，但排除本地 `.venv`、缓存和报告），在服务器单独配置 `.env`，权限建议设置为 `600`。

镜像按 `requirements-web.lock` 安装固定版本。修改依赖后，先更新 `uv.lock`，再执行 `uv export --locked --no-dev --extra web --no-emit-project --no-hashes --output-file requirements-web.lock` 同步部署依赖。

先仅开放本机端口验证：

```bash
docker compose -f compose.web.yml up -d --build
docker compose -f compose.web.yml ps
curl --fail http://127.0.0.1:8000/healthz
```

远程试用可以使用 SSH 端口转发：`ssh -L 8000:127.0.0.1:8000 用户名@服务器地址`，然后在自己电脑打开 <http://127.0.0.1:8000>。

## 通过域名上线

将域名的 DNS 指向服务器，允许入站 TCP 80/443（可选 UDP 443），并在 `.env` 设置：

```dotenv
TRADINGAGENTS_DOMAIN=research.example.com
```

```bash
docker compose -f compose.web.yml --profile https up -d --build
docker compose -f compose.web.yml --profile https logs --tail 50 proxy web
```

访问 `https://你的域名`。Caddy 负责申请和续期证书。应用的 8000 端口仅绑定服务器本机；公网访问应走 HTTPS，因为浏览器登录使用 HTTP Basic。

## 运行边界与维护

- 每个数据目录仅允许运行一个 Web 进程，文件锁会阻止重复启动。不要增加 workers 或横向扩容。单用户首版不提供注册、支付或多人数据隔离。
- 最多保留 10 个待执行或运行中的任务，依次分析，避免数据源全局配置冲突与无界并发费用。页面展示最近 100 条，旧记录仍保留在数据库中。
- 超时默认 1800 秒，通过 `TRADINGAGENTS_WEB_JOB_TIMEOUT` 设置（60–7200 秒）；超时会终止分析进程。轮数和输出 token 上限可以降低消耗，但不能保证固定费用。
- 正在运行的任务遇到重启会标记为失败，需手动重新提交；排队任务会继续执行。已完成报告持久化到 `research_data` 卷。关闭网页不影响任务。
- 默认数据位于 `~/.tradingagents/web`，包括 SQLite 数据库与报告。使用 `TRADINGAGENTS_WEB_DATA_DIR` 可修改网页数据目录。分析引擎的缓存、日志与记忆也在 `~/.tradingagents`，Docker 已持久化整个目录。
- 默认不记录模型 SDK 原始输出，防止异常内容泄露密钥。失败时先检查服务端配置，再通过原命令行在受控环境诊断。网页错误信息不会包含原始服务商异常。
- 登录由浏览器管理，退出请关闭浏览器私人窗口或清除该站点的身份信息。建议在私人窗口使用。密码修改后需重启服务；没有联网暴力破解限流，公网使用应保留随机长密码，或加上 VPN / 访问网关。

查看服务状态：`docker compose -f compose.web.yml ps`。更新：重新复制代码后执行 `docker compose -f compose.web.yml --profile https up -d --build`。普通停止使用 `down`；不要加 `-v`，它会删除数据卷。

备份时先停服务以确保 SQLite 与报告一致，再导出整个持久化目录：

```bash
docker compose -f compose.web.yml --profile https stop
docker compose -f compose.web.yml run --rm --no-deps --entrypoint tar web czf - -C /home/appuser .tradingagents > research-backup.tar.gz
docker compose -f compose.web.yml --profile https up -d
```

备份包含私人研究资料，应妥善保存。恢复到空数据卷时，使用相同服务镜像执行 `tar xzf - -C /home/appuser`，从备份文件通过标准输入导入，再启动服务。生产更新前保留上一版代码和备份。

## 验收

```bash
pip install '.[dev]'
pytest -q
```

上线后用真实账户完成一次：登录 → 新建分析 → 等待完成 → 查看并下载报告 → 重启服务 → 确认历史仍在。自动化测试隔离外部 API，不证明真实模型或市场数据可用。
