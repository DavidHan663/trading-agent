# Trading Agent Research

基于 LangGraph 的多智能体金融研究项目，提供本地中文网页、研究报告和固定输入评估。无需购买服务器或域名。

## 核心设计

- 角色协作：4 类分析师、多空研究员、研究经理、交易员、3 类风险评估员与投资组合经理。
- 工作流：分析师并行研究，汇总报告后进行多空辩论和风险讨论；工具调用与消息状态在分析子图中隔离。
- 数据约束：行情快照、技术指标校验、分析日期过滤、结构化决策与历史记忆。
- 本地网页：后台任务、超时停止、历史记录、报告查看与下载。
- 评估基础：30 个合成案例、冻结证据回放、执行记录和人工审核指标。尚未建立真实模型质量基线。

## 本地使用

需要 Python 3.10+ 和可用的模型账号。复制环境模板，并填写自己的密钥及账号实际可用的模型 ID；已有 `.env` 时不要覆盖。

```bash
cp .env.example .env
```

macOS 双击 **`启动网页.command`**，程序会准备依赖并自动打开浏览器。保留启动窗口；退出按 Control+C。

网页仅监听本机，默认地址为 `http://127.0.0.1:8765`，端口占用时自动更换。本地密码、报告及缓存保存在 `.local-web/`，不会上传到 GitHub。电脑休眠或关闭程序可能中断分析；云端模型调用仍按服务商计费。

也可手动安装并启动（Linux / macOS）：

```bash
pip install -e '.[web]'
python scripts/local_web.py
```

Windows 建议通过 WSL 或 [Docker](DEPLOYMENT.md) 使用。原命令行入口 `tradingagents` 仍保留。

## 测试与评估

```bash
pip install -e '.[dev]'
pytest -q
python -m tradingagents.evaluation.run
```

最后一条命令默认离线运行，不调用模型，也不产生质量评分。真实模型评估、人工审核和成本统计见 [评估说明](tradingagents/evaluation/README.md)。运行结果位于 `eval_results/`，不会上传到 GitHub。

## 代码结构

```text
tradingagents/
  agents/       角色、提示词与决策结构
  graph/        状态图、子图、路由与检查点
  dataflows/    数据源、指标与时间约束
  llm_clients/  模型服务适配
  evaluation/   固定案例、回放与审核
  web/          本地网页与任务后台
tests/          自动化测试
scripts/        启动和凭据检查
cli/            原命令行兼容入口
```

依赖以 `pyproject.toml` 定义，`uv.lock` 用于版本锁定；可选容器部署使用 `requirements-web.lock` 导出依赖，仅保留一套网页 Docker 配置。

## 来源与许可

本项目基于 [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents) 扩展，保留原有分析引擎及 Apache-2.0 许可。本仓库增加了本地网页工作台、一键启动和固定输入评估基础设施。许可见 [LICENSE](LICENSE)。

研究报告由模型生成，可能存在错误或缺失，仅供研究参考；当前不连接券商或自动执行交易。
