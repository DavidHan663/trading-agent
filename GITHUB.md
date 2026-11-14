# 将本地项目上传到 GitHub

Git 管理本地版本，GitHub 存放远程仓库。这个目录已初始化本地 Git，但尚未提交、绑定远程仓库或上传。

## 先创建一个空仓库

1. 登录 GitHub，点击右上角 `+` → `New repository`。
2. 名称例如 `tradingagents-research`，初期建议选 `Private`。
3. 不勾选添加 README、`.gitignore` 或 License，因为本地已有这些文件。
4. 创建后复制仓库的 HTTPS 或 SSH 地址。仓库地址可以提供给协作者，密码和访问令牌不要发送。

## 通过 GitHub Desktop 上传（适合不想使用命令行）

安装官方 GitHub Desktop 并登录自己的账号，选择 `File → Add Local Repository`，选择本项目文件夹。检查文件列表不包含 `.env`、`.local-web`、`.venv`、报告或评估输出，填写提交说明并提交。

如果还没有在网页创建仓库，可在 Desktop 里选择 `Publish repository`，保留私人仓库选项；如果已创建，设置 Repository Settings 中的远程地址为该仓库，然后 Push。登录和支付等账号操作由你自己完成。

## 通过终端上传

进入项目目录，先检查待上传文件：

```bash
cd /Users/yanghan/Project/TradingAgents-main
.venv/bin/python scripts/check_secrets.py
git status --short
```

扫描工具只输出文件名、行号和命中类型，不打印密钥。它是启发式检查，不能保证没有敏感内容。已将 `.env`、本地访问密码、虚拟环境、报告和评估运行结果排除；示例环境文件可上传，但不能填入真实密钥。

如果首次提交提示缺少作者信息，请使用自己的信息配置，建议用 GitHub 邮箱设置中提供的隐私邮箱；以下尖括号内容需要替换：

```bash
git config user.name "<你的显示名称>"
git config user.email "<你的GitHub隐私邮箱>"
```

再提交并上传，将仓库地址替换为你刚创建的真实地址：

```bash
git add .
git diff --cached --stat
git commit -m "Add research workspace and frozen evaluation baseline"
git remote add origin https://github.com/你的用户名/tradingagents-research.git
git push -u origin main
```

HTTPS 推送需要使用 GitHub 支持的认证方式，不能使用账号密码作为 Git 密码。也可以用已经配置好的 SSH。不要把 Token 写进远程仓库 URL 或项目文件。

之后每次修改：检查 → `git add` → `git commit` → `git push`。不要上传已有真实密钥的文件后再删除，那样密钥仍可能留在提交历史里。

## 开源来源

项目包含 TauricResearch/TradingAgents 的开源代码。保留现有 LICENSE、版权及来源说明，并在介绍中明确自己的扩展部分；初始化 Git 不会恢复原上游的提交历史。

官方流程参考：[Adding locally hosted code to GitHub](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github)。
