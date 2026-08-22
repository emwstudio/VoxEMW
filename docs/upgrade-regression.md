# 上游 speech-to-speech 升级回归方案（Mac 本地版）

升级钉死的上游 commit（见 `scripts/mac_setup.sh`）时按此单执行。
原则：**现役 `.venv-mac` 永不动**，新环境验证通过再扶正；任何阶段红灯即回滚。

## 我们的自定义接触面（升级风险点）

- `voxemw/pipeline/launch.py`：复刻上游 `s2s_pipeline` 启动流程（怕上游改启动结构）
  + monkeypatch（flex_attention 兼容 / torch.hub 离线兜底——怕上游改被 patch 的类名/签名）
- `voxemw/pipeline/backends.py`：BackendSpec 注册表接入 qwen3asr（怕上游改 BackendSpec 接口）
- 自定义 handler：`stt_qwen3asr.py`（BaseSTTHandler + speculative_turns）
- orchestrator 依赖的 realtime 协议行为：`session.update` 结构、
  `response.create` 被拒（conversation_already_has_active_response）重试、
  `conversation.item.create` deferred 队列、`response.done` 计数、GA/beta 双名音频事件
- RTC 音轨链路（sched.feed_audio / flush 打断清空）

## 阶段 0：隔离准备

- 记录当前钉版 commit（mac_setup.sh 里），回滚锚点
- `uv venv --python 3.12 .venv-next`，按 mac_setup.sh 的依赖段装到新环境
  （新版 speech-to-speech 换目标 commit），禁止复制 .venv-mac（shebang 硬编码路径）

## 阶段 1：静态对账

- diff 上游新旧版 `s2s_pipeline` serve 流程 vs `launch.py` 复刻段：BackendSpec 接口、
  `module_kwargs` 字段、realtime router 初始化参数
- 检查 `BaseSTTHandler` 接口变更，以及 launch.py 各 monkeypatch 的目标符号是否还在
- `.venv-next` 跑全部单测（`python -m pytest tests/`，无需模型）

## 阶段 2：启动冒烟

- `VOXEMW_CONFIG` 不变，用 `.venv-next` 起 pipeline + orchestrator：
  pipeline ws 就绪（Uvicorn running）/ orchestrator :8000 返 200 /
  pipeline.log 见 `Qwen3-ASR loaded ... hotwords=...`

## 阶段 3：脚本化端到端

- `.venv-mac/bin/python scripts/smoke_mac.py`：文本注入，断言回文本 + 音频 delta，
  首音频 ~2s 量级
- `.venv-mac/bin/python scripts/smoke_mac.py --wav <16k 中文.wav> --expect 味真足`：
  真音频链路，断言 STT 转写含热词
- pipeline.log 观测：STT 单段耗时、DeepSeek 首句、Qwen3-TTS TTFA，
  闭嘴→首音 ~2-2.5s 基线，偏差 >20% 红灯

## 阶段 4：人工交互矩阵（浏览器 http://localhost:8000）

| 用例 | 验收点 |
|---|---|
| 普通问答 ×3 | 转写准（黑话）、首音及时 |
| 说话中打断 | 音频即停、能接新话 |
| 长回复（>30s） | 不卡、字幕流式不积压 |

## 回滚

进程换回 `.venv-mac` 重启（2 分钟）；绿灯后 `mv .venv-mac .venv-old &&
mv .venv-next .venv-mac`，确认一天无异常再删 `.venv-old`。
