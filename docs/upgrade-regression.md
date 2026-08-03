# 上游 speech-to-speech 升级回归方案

升级 huggingface/speech-to-speech 时按此单执行，预估 1.5-2 小时。
原则：**现役 `.venv` 永不动**，新环境验证通过再扶正；任何阶段红灯即回滚。

## 我们的自定义接触面（升级风险点）

- `voxemw/pipeline/launch.py`：monkeypatch `get_stt_handler` / `get_tts_handler` 工厂
  （怕上游改签名）+ 复刻上游 `main()` 启动流程（怕上游改启动结构）
- 自定义 handler 三件套：`stt_sensevoice.py` / `stt_qwen3asr.py`（BaseSTTHandler）、
  `tts_voxcpm.py`（BaseHandler + cancel_scope + speculative_turns + voices 热切换）
- orchestrator 依赖的 realtime 协议行为：`session.update` 结构、
  `response.create` 被拒（conversation_already_has_active_response）重试、
  `conversation.item.create` deferred 队列、`response.done` 计数、GA/beta 双名音频事件
- 截帧打分两阶段注入（垫场→打分）、打断链路（speech_started → avatar reset）

## 阶段 0：隔离准备（15 分钟）

- `pip show speech-to-speech` 记录当前版本，写入 `requirements.txt` 钉死（回滚锚点）
- 新建 `.venv-next`（`python -m venv`），装 `requirements.txt` + 新版 speech-to-speech
- 禁止复制 `.venv`（shebang 硬编码路径，复制即坏）

## 阶段 1：静态对账（30 分钟）

- diff 上游新版 `main()` vs `launch.py` 复刻段：工厂签名、`module_kwargs` 字段、
  realtime router 初始化参数
- 检查 `BaseSTTHandler` / `BaseHandler` / `SpeculativeTurnTracker` 接口变更
- `.venv-next` 跑通全部单测（`python -m pytest tests/`，无需 GPU）

## 阶段 2：启动冒烟（15 分钟）

- `.venv-next` 起三进程：pipeline ws 就绪 / avatar 就绪 / `/api/personas` 200

## 阶段 3：脚本化端到端（30 分钟）

- `measure_latency.py`：转写→首音频 ~2.0s 基线，偏差 >20% 红灯
- `diag_sync.py`：speech/idle 帧 tag 正确、说话期间无 idle 帧（speech_active 门控）、
  爬坡帧序正常、句尾 speech 帧积压 < 50
- VoxCPM 音色热切换（session.update 的 voice override）

## 阶段 4：人工交互矩阵（30 分钟，浏览器）

| 用例 | 验收点 |
|---|---|
| 普通问答 ×3 | 转写准、首音及时、口型齐 |
| 说话中打断 | 音频即停、嘴型归位、能接新话 |
| 长回复（>30s） | 不卡、?debug=1 帧队列不见底 |
| 截帧打分 | 暗号「让我好好看看你」→ 垫场→打分两阶段走完 |
| 待机/倾听微动 | 眨眼轻摇头正常、说话无缝接口型 |

## 回滚

进程换回 `.venv` 重启（2 分钟）；绿灯后 `mv .venv .venv-old && mv .venv-next .venv`，
确认一天无异常再删 `.venv-old`。
