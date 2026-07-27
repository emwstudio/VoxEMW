# VoxEMW —— 峰哥反指提示器（峰哥说啥我反着来）

定时巡检峰哥微博动态 → **DeepSeek** 以突发主播人设逐条生成「峰哥说啥我反着来」播报稿 →
**VoxCPM2** 用峰哥本人的声音语音播报 → 警报样式 Web 页按时间线展示。
支持语音提问：对着页面麦克风问「峰哥今天说长鑫了吗」，几秒内语音答复。

按「五块积木」组织，每块在 `configs/alerter.yaml` 里声明、可替换：

| 积木 | 默认实现 | 运行位置 |
|------|---------|---------|
| VAD | `webspeech`（浏览器 Web Speech API 端点检测） | web/ 页面 |
| STT | `webspeech`（浏览器语音识别，zh-CN） | web/ 页面 |
| LLM | `deepseek`（写播报稿 + 存档语义选号） | 服务端（AutoDL） |
| TTS | `voxcpm2`（峰哥音色克隆） | 服务端（AutoDL） |
| 人设 | `file`（`personas/newsanchor.md`，蒸馏的突发主播人设） | 服务端 |

## 架构

```
Kimi Code 会话 cron（本机，每 5 分钟）──Kimi WebBridge 操作浏览器──> 峰哥微博
   │  POST /api/briefing {posts:[{time,text}]}（主页 ≤2 条时自动补实时搜索，防限流隐藏）
   ▼
voxemw.server（AutoDL，aiohttp 单端口 :8000，SSH 隧道到本机）
   ├─ briefing.py  当天过滤 + 正文指纹去重 + 存档 + 人设 prompt + 选号器（纯逻辑）
   ├─ tts.py       VoxCPM2 峰哥音色 → wav（单线程池，torch.compile 约束）
   └─ data/        posts_archive.json（当天动态存档+播报稿）+ seen_posts.json（去重指纹）
   ▲ 轮询 /api/alerts、/api/posts/today、POST /api/ask
web/ 警报页（无构建）：微博时间线（播报稿+原文引文+▶播放）、语音答复弹窗、
   播放中卡片警戒条纹+⚠ALERT+🚨闪烁（禁连播）
```

三条链路：

- **定时短报**：每 5 分钟巡检 → 新动态逐条生成播报稿（不重复播报，重启不丢）→
  进时间线 + 自动播报最新一条
- **语音问答**：页面麦克风 → webspeech 转文字 → `POST /api/ask` → 服务端用 LLM
  在当天存档里语义选相关动态（错字/近义/「最新一条」都行）→ 约 8 秒出语音，
  答复走右上角弹窗（不进时间线）；存档为空才回退 agent 轮询（2 分钟 cron）
- **时间线**：刷新页面 = 当天全部动态按发布时间倒序列出（静默不出声），
  每条 ▶ 点播播报稿

## 快速开始

### 前置（本机，一次性）

1. 装 Kimi WebBridge 守护进程：
   `curl -fsSL https://cdn.kimi.com/webbridge/install.sh | bash`
2. Chrome/Edge 装「Kimi WebBridge」扩展，图标显示 Connected，浏览器登录微博
3. 向 Kimi Code 要两个定时任务（会话内 cron，关会话即停）：
   5 分钟巡检 + 2 分钟语音指令兜底，采集流程见 `skills/fengge-alerter/SKILL.md`

### 服务端（AutoDL）

1. 开实例：RTX 4090D（24GB），镜像选 **Miniconda**，系统盘 ≥ 50GB
2. 同步仓库到实例。音色默认用峰哥本人素材 `assets/fengge/ref.wav` + `ref.txt`
   （仓库自带）；换音色在 `configs/alerter.yaml` 的 `voices` 加同名条目
3. `cp .env.example .env.local`，填入 `DEEPSEEK_API_KEY`
4. `bash scripts/autodl_setup.sh`（幂等：conda → venv → torch → voxcpm →
   hf-mirror 下载 VoxCPM2 → nohup 起服务）
5. 本机开 SSH 隧道：`ssh -CNg -L 8000:127.0.0.1:8000 root@<主机> -p <端口>`
6. 浏览器开 `http://localhost:8000`

排障：`tail -f logs/alerter.log`；改配置/换音色后
`pkill -f voxemw.server` 再重跑 `scripts/autodl_setup.sh`。
SSH 网关抽风时的备用通道：实例 JupyterLab（AutoPanel 的 `/jupyter/` 路径）——
`scripts/_jupyter_term_run.py '<命令>' <超时秒>` 经网页终端 websocket 执行命令。

## API

- `GET  /api/blocks` — 五积木声明（不下发任何密钥）
- `POST /api/briefing` `{posts:[{time,text}], query?, task_id?}` —
  定时流程：逐条新动态生成播报稿+告警，全旧返回 `{"skipped":"no_new_posts"}`；
  语音查询：围绕 query 生成答复；当天无相关内容也有兜底答复并核销任务
- `GET  /api/posts/today` — 当天存档（按发布时间倒序，含播报稿），时间线数据源
- `POST /api/backfill` — 给存档里缺播报稿的当天动态补生成（迁移用，不产生告警）
- `GET  /api/alerts?since=<id>` / `GET /api/alerts/<id>/audio` — 告警与 wav
- `POST /api/ask` `{query}` — 语音提问（服务端快路径直接答；存档为空才入队
  等 agent，见 `GET /api/tasks/pending`）
- `GET  /api/voices`、`POST /api/tts` — 调试（时间线 ▶ 按钮也用它）

错误统一 `{"error": "..."}`（400 参数 / 502 LLM / 500 TTS / 503 无音频）。

## 采集与去重规则（血泪教训，已固化进 skill）

- 指纹 = 正文 sha1：时间标签漂移（N分钟前→N小时前）不影响去重
- 采集时结尾互动计数（转发/评论/赞、转发微博内嵌的统计行）必须循环剥干净，
  否则计数上涨会被当成新动态重复播报
- 微博会限流隐藏峰哥部分动态（主页不可见、搜索可见）：主页抓到 ≤2 条时
  必须补实时搜索（s.weibo.com/realtime）按作者过滤合并
- 正文原文原样提交，任何二次加工都会污染指纹

## 换积木

- **换音色**：素材放 `assets/<id>/ref.wav` + `ref.txt`，`voices` 加同名条目，重启
- **换人设**：改 `blocks.persona.path` 指向别的 prompt 文件
- **换 LLM/TTS/VAD/STT 实现**：改 `blocks` 对应段的 `impl`（VAD/STT 目前只有
  webspeech 一种实现，留了扩展位）

## 本地开发（macOS，无 GPU）

```bash
python3 -m venv .venv && .venv/bin/python -m pip install pytest pyyaml numpy aiohttp
.venv/bin/python -m pytest tests/ -v
.venv/bin/python -m voxemw.server --no-tts   # 音频接口 503，其余全可用
```

## 目录

- `configs/alerter.yaml` — 唯一配置（五积木 blocks + alerter + voices + server）
- `voxemw/` — `config.py`（YAML+.env+人设文件加载）、`llm.py`（DeepSeek 客户端）、
  `briefing.py`（过滤/去重/存档/prompt/选号纯逻辑）、`tts.py`（VoxCPM2 封装）、
  `server.py`（aiohttp）
- `personas/newsanchor.md` — 突发主持人设（LLM system prompt）
- `skills/newsanchor-perspective/` — 蒸馏的主持人设 skill 完整版（含选音色指南）
- `skills/fengge-alerter/` — Agent 侧微博采集 playbook（cron 用，含防限流/清洗规则）
- `web/` — 警报样式静态页（时间线 + 答复弹窗 + 播放警报动效）
- `scripts/autodl_setup.sh` — AutoDL 一键部署（幂等）
- `scripts/_jupyter_term_run.py` — Jupyter 网页终端命令执行器（SSH 备用通道）
- `assets/` — 音色克隆素材（默认 `fengge/`，仓库自带；素材本身不入库）
- `data/` — 运行时存档与去重指纹（不入库）
- `tests/` — 纯逻辑单测（不 import torch，本机可跑）

## 合规

反指短报是**娱乐内容，不构成任何建议**；据此操作盈亏自负。
音色克隆素材由使用者本人提供；AI 生成内容需标注，不得用于冒充、欺诈。
微博内容版权归原博主，采集仅用于个人提醒，不要二次分发。

## 相关链接

- VoxCPM2：https://huggingface.co/openbmb/VoxCPM2
- DeepSeek API：https://platform.deepseek.com
- Kimi WebBridge：https://www.kimi.com/zh-cn/features/webbridge
