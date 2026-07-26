# VoxEMW —— 搭积木语音助手

基于 [huggingface/speech-to-speech](https://github.com/huggingface/speech-to-speech) 管线的
实时语音对话助手：你对着麦克风说话，助手用指定角色的口吻和音色实时语音回复。

每个环节都是可替换的「积木」：VAD / STT / LLM / TTS / 人设，全部在一份 YAML
里声明，`launch.py` 渲染成 speech-to-speech 的 CLI 参数启动。默认组合：
Qwen3-ASR（STT）+ DeepSeek API（LLM）+ VoxCPM2 音色克隆（TTS）+ 网红人设
（大胃袋良子 / 峰哥亡命天涯），部署目标 AutoDL 单卡 RTX 4090D。

## 架构

```
浏览器（web/ 无构建静态页）
  │  麦克风 16kHz PCM ──► ws://<host>:8765/v1/realtime（OpenAI Realtime 协议）
  │                      连接后客户端发 session.update 注入人设 instructions
  ▼
GPU 实例（AutoDL RTX 4090D）
  launch.py ── 读 configs/autodl-4090.yaml ──► 渲染 argv ──► exec speech-to-speech
  └─ speech-to-speech 管线（vendor/，打 patches/register-handlers.patch）
       mic ─► silero VAD ─► Qwen3-ASR-1.7B（STT，自定义后端）
       文本 ─► DeepSeek API（LLM，chat-completions，多轮历史服务端维护）
       语音 ◄── openbmb/VoxCPM2（TTS，自定义后端，Ultimate Cloning + 流式输出）
  python3 -m http.server 8000 --directory web   （静态页 + personas.json）
```

- **管线全程 16kHz**；服务端 VAD 判停，支持打断（barge-in）
- **人设注入**：realtime 模式下 `--init_chat_prompt` 不生效，人设 instructions 由
  Web 客户端连接后用 `session.update` 注入（服务端深合并，LLM 每轮读取包装为
  system message）；多轮历史由服务端 Chat 对象维护（`chat_size`，默认 30）
- **音色热切换**：`tts.voices` 里的每个音色（key = 人设 id）启动时预编码
  prompt cache，前端切人设时经 `session.update` 的 `audio.output.voice`
  即时换音色，无需重启
- **自定义后端**：`--stt qwen3asr` / `--tts voxcpm`（另有备选 `--tts omnivoice`），由
  `patches/register-handlers.patch`（唯一事实源）注册进 vendor 管线；
  `extensions/` 是 handler 的人类可读副本

## 五块积木

| 积木 | 默认 | 可换（注册表见 `voxemw/backends.py`） |
|------|------|--------------------------------------|
| VAD | silero | 上游仅此一种，只能调参 |
| STT | qwen3asr（Qwen3-ASR-1.7B，自定义） | whisper / faster-whisper / parakeet-tdt / paraformer |
| LLM | chat-completions（DeepSeek API） | responses-api / transformers（本地） |
| TTS | voxcpm（openbmb/VoxCPM2，自定义，Ultimate Cloning + 流式） | omnivoice / qwen3 / kokoro / pocket / chatTTS / facebookMMS |
| 人设 | personas/liangzi.md | personas/*.md 任意增删 |

## 快速开始（AutoDL）

1. 开实例：RTX 4090D（24GB），镜像选 **Miniconda**，系统盘 ≥ 50GB
2. 同步仓库到实例（如 `/root/voxemw`），并放入音色素材
   `assets/<角色>/ref.wav` + `ref.txt`（10–30s 清晰单人声 + 逐字台词；
   默认配置需要 `assets/liangzi/` 和 `assets/fengge/` 两份）
3. `cp .env.example .env.local`，填入 `DEEPSEEK_API_KEY`
4. `bash scripts/autodl_setup.sh`（幂等：conda py312 → venv → torch 2.8 cu128 →
   打 patch → 装 speech-to-speech + voxcpm → 生成 personas.json →
   hf-mirror 下载模型到数据盘 → nohup 起服务）
5. 本机开 SSH 隧道（AutoDL 不开公网端口）：

   ```bash
   ssh -CNg -L 8000:127.0.0.1:8000 -L 8765:127.0.0.1:8765 root@<实例主机> -p <SSH端口>
   ```

6. 浏览器打开 `http://localhost:8000`，选角色 → 连接 → 说话

排障：`tail -f logs/s2s.log`；改配置后 `pkill -f speech_to_speech` 再重跑
`scripts/autodl_setup.sh`（切换已有音色不用重启，新增音色/改配置才需要）。

也可用配套 skill（`~/.agents/skills/autodl`）通过官方 API 开/关/释放实例、保存镜像。

## 斗地主模式

在聊天管线同一套语音积木上，还有一个「语音斗地主」：你 + 良子/峰哥两个人机
一桌打牌，你出牌用鼠标点牌，bot 的出牌决策和台词走 DeepSeek（非法决策由
引擎校验、带错误重试一次、内置策略兜底）。

- **固定地主**：`configs/doudizhu.yaml` 里 `game.fixed_landlord: you`——你永远
  地主，良子/峰哥永远农民，抱团怼你（敌我铁律写在 prompt 里：农民之间只捧不损）
- **对话由出牌驱动**：当前版本默认关麦（`web/doudizhu.js` 顶部
  `MIC_ENABLED = false`），bot 台词完全跟着出牌轮替走——点评上一家 + 报自己
  的牌；想恢复语音插话/口令，改回 `true` 即可（服务端 STT/VAD 链路一直保留）
- **台词机制**：每句必带经典口头禅（池子按人设建，一局内不重复，开新局重置）；
  先报动作再接话（不要必以「不要」开头、出牌先报牌名，服务端 `_normalize_say`
  硬兜底）；剩牌数由服务端按引擎真值校正（`_correct_count_claims`，模型说错
  直接改对）
- **节奏与反馈**：bot 说完话牌才落桌（带飞入动画）→ 下一家高亮；音效分三种
  （出牌「啪」/ 不要低嘟 / 轮到你双嘀）；完局弹大字横幅「本局结束 · 地主/农民赢」

与聊天管线**互斥**（24G 显存装不下两份 STT+TTS）。开局：
`bash scripts/start_game.sh`（会先停聊天管线，再起 `doudizhu/server.py`，
日志在 `logs/game.log`）。

SSH 隧道在原有基础上**加 8766 端口**：

```bash
ssh -CNg -L 8000:127.0.0.1:8000 -L 8765:127.0.0.1:8765 -L 8766:127.0.0.1:8766 root@<实例主机> -p <SSH端口>
```

浏览器打开 `http://localhost:8000/doudizhu.html`（静态页由同一个
`http.server 8000` 提供），点「上桌」开局。前端改了要硬刷新
（`doudizhu.html` 里脚本带版本号防缓存）。
回聊天模式：`pkill -f "doudizhu.serve[r]"`，再按老方式起 `launch.py`

代码在 `doudizhu/`（纯逻辑 `cards/engine/heuristic` + bot/chat 层 + voice/server
接入层），配置 `configs/doudizhu.yaml`，纯逻辑单测 `tests/test_doudizhu.py`，
实例侧无头整局验证脚本 `scripts/e2e_doudizhu_test.py`。

## 换积木

编辑 `configs/autodl-4090.yaml` 对应段的 `backend` 和参数（每段注释里写了可选
backend 和换法示例，如 LLM 换本地 vLLM / 本地 transformers）。参数名 → CLI flag
的映射以 `voxemw/backends.py` 注册表为准，flag 逐一核对自 vendor 的
`arguments_classes/`。渲染结果可先干跑确认：

```bash
python launch.py --config configs/autodl-4090.yaml --dry-run
```

改完重启服务生效（`pkill -f speech_to_speech && python launch.py`）。

## 加角色

1. 造人设：可用 huashu-nuwa skill 蒸馏（「蒸馏XX」→ 生成思维框架 SKILL.md），
   或手写
2. 在 `personas/` 新建 `<id>.md`：YAML frontmatter（`name` 显示名、
   `ref_wav`、`ref_text` 音色素材路径）+ 人设正文（作为 instructions 全文注入）。
   参照 `personas/liangzi.md`
3. `python scripts/build_personas.py` 重新生成 `web/personas.json`
4. 刷新页面，角色下拉里就有了

**音色热切换**：人设热切换会同时换 instructions（说话方式）和音色——前端
`session.update` 把 `audio.output.voice` 设为人设 id，服务端 VoxCPM handler
按名字选用 `tts.voices` 里预编码的 prompt cache（Ultimate Cloning：参考音频
同时作续写 prompt + 克隆 reference，配 ref.txt 逐字台词）。给新角色配音色：

1. 音色素材放到 `assets/<id>/ref.wav` + `ref.txt`
2. 在 `configs/autodl-4090.yaml` 的 `tts.voices` 加同名 key（key 必须 = persona id）：

   ```yaml
   tts:
     ref_audio: assets/liangzi/ref.wav   # 默认音色（voice 未命中时用）
     ref_text: assets/liangzi/ref.txt
     voices:
       liangzi: { ref_audio: assets/liangzi/ref.wav, ref_text: assets/liangzi/ref.txt }
       fengge:  { ref_audio: assets/fengge/ref.wav,  ref_text: assets/fengge/ref.txt }
   ```

3. 重启服务一次（新音色只在启动时预编码；之后页面里随时切，不用再重启）

`tts.ref_audio/ref_text` 仍是默认音色；voice 名在 `voices` 里没匹配到时回退到它。

## 本地开发（macOS，无 GPU）

本机只写代码 + 跑纯逻辑单测（不 import torch/transformers）：

```bash
python3 -m venv .venv && .venv/bin/python -m pip install pytest pyyaml
.venv/bin/python -m pytest tests/ -v
```

`launch.py --dry-run` 也可在本机验证配置渲染（需 `DEEPSEEK_API_KEY` 占位）。

## 目录

- `configs/` — 积木配置（YAML，六段：vad/stt/llm/tts/persona/server）
- `voxemw/backends.py` — 后端注册表（YAML 参数名 → CLI flag）
- `launch.py` — 启动器：读 YAML → 渲染 argv → exec speech-to-speech
- `personas/` — 人设积木（frontmatter + 正文）；源素材在 `skills/`
- `scripts/build_personas.py` — personas/*.md → web/personas.json
- `scripts/autodl_setup.sh` — AutoDL 一键部署（幂等）
- `scripts/start_game.sh` — 停聊天管线、起斗地主服务
- `scripts/smoke_doudizhu_*.py` / `e2e_doudizhu_test.py` — 本地假 LLM 冒烟 / 实例整局验证
- `web/` — 无构建静态前端（realtime ws 客户端 + 角色切换；`doudizhu.html/js` 牌桌 UI）
- `doudizhu/` — 语音斗地主（纯逻辑引擎 + DeepSeek bot + 语音接入层）
- `vendor/speech-to-speech/` — 上游管线（pinned commit，`patches/` 打补丁）
- `patches/register-handlers.patch` — 自定义后端注册（唯一事实源）
- `extensions/` — 自定义 handler 的人类可读副本
- `assets/` — 音色克隆素材（自行提供，不入库）
- `tests/` — 纯逻辑单测

## 合规

音色克隆素材由使用者本人提供，仅限娱乐演示；AI 生成内容需标注，不得用于
冒充、欺诈。人设均为基于公开言论的娱乐扮演，非本人观点；涉及健康/心理等
真实求助请以专业渠道为准。

## 相关链接

- 上游管线：https://github.com/huggingface/speech-to-speech
- VoxCPM2：https://huggingface.co/openbmb/VoxCPM2
- OmniVoice（备选 TTS）：https://huggingface.co/k2-fsa/OmniVoice
- Qwen3-ASR：https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf
