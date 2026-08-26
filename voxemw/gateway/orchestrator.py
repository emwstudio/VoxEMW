"""Orchestrator：浏览器唯一入口，编排语音管线（s2s）与 RTC 音频轨。

架构：
    浏览器 ←→ 本进程（aiohttp，:8000）
              └→ s2s realtime ws（:8765，voxemw.pipeline.launch 起的语音管线）

职责：
- 下行：s2s 的 TTS 音频 delta → AudioPacer（RTC 音频轨，Opus 48k）；
  WS 只留控制/转写事件（音频 delta 剥掉 base64 音频体省带宽）
- 上行：浏览器麦克风音频/控制消息 → 转发 s2s
- persona：浏览器发 {"type": "vox.persona", "id": ...} 切换人设，
  本进程把人设正文/音色经 session.update 注入 s2s（instructions + voice）
- 打断：s2s 报 speech_started → flush 音频队列；若回复播了一半，
  把已听前缀补写回上下文（heard_prefix，防模型忘记自己说到哪）
- 单用户单会话：新浏览器连接顶掉旧会话（s2s 只有 1 个管线槽位，
  换网络产生的僵尸会话被新连接立即踢掉，无需等超时/刷新两次）

浏览器侧协议：
  /ws 文本帧（JSON）：
    → {"type": "vox.persona", "id": "<persona_id>"}   切换人设
    → OpenAI Realtime 事件原样透传（input_audio_buffer.append / response.cancel 等）
    ← OpenAI Realtime 事件透传（transcription / response.done 等）
    ← {"type": "vox.status", "persona": "<id>",
       "rtc": {"enabled": bool, "ice_servers": [...]}}
  POST /rtc/offer：WebRTC 信令，body {"sdp", "type"} → answer
  GET  /rtc/ice ：下发 ICE 配置
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import os
import re
import sys
from pathlib import Path

from voxemw.gateway.phoneme import analyze as lip_analyze

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger(__name__)


def audio_level(pcm: bytes) -> float:
    """音频块的响度（0..1，RMS 归一化 ×5 增益）——随音频事件下发给前端，
    驱动星空/光环的能量动画。服务端算是因为客户端 WebAudio 分析在
    RTC 重连/自动播放挂起/WebKit  quirks 下不可靠（2026-08-23 实测）。"""
    import numpy as np

    samples = np.frombuffer(pcm, dtype=np.int16)
    if samples.size == 0:
        return 0.0
    rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)) / 32768.0)
    return round(min(rms * 5.0, 1.0), 3)

# s2s 事件 → 编排动作（纯函数分类，便于单测）
AUDIO_DELTA_EVENTS = {"response.output_audio.delta", "response.audio.delta"}  # GA / beta 名都收
INTERRUPT_EVENTS = {
    "input_audio_buffer.speech_started",  # 用户开口（打断）：清音频队列
}


def build_session_update(persona_id: str, persona_text: str) -> dict:
    """注入人设的 session.update：instructions = 人设正文，voice = persona id
    （TTS voices 表 key）。"""
    return {
        "type": "session.update",
        "session": {
            "type": "realtime",
            "instructions": persona_text,
            "audio": {
                "input": {
                    "turn_detection": {
                        "type": "server_vad",
                        "interrupt_response": True,
                    }
                },
                "output": {"voice": persona_id},
            },
        },
    }


def classify_s2s_event(event: dict) -> tuple[bool, bool, bytes | None]:
    """分类 s2s 下行事件。返回 (relay_to_browser, is_interrupt, audio_pcm|None)。"""
    etype = event.get("type", "")
    pcm = None
    if etype in AUDIO_DELTA_EVENTS:
        delta = event.get("delta")
        if delta:
            pcm = base64.b64decode(delta)
    return True, etype in INTERRUPT_EVENTS, pcm


_PUNCT = "。！？；，、…—.!?;,"


def heard_prefix(transcript: str, audio_seconds: float, played_seconds: float) -> str:
    """打断时估算用户实际听到的文本前缀（纯函数，便于单测）。

    播放进度占已生成音频的比例 ≈ 听到的文本比例（中文语速在一条回复内
    足够均匀）。不足 2 字不值得注入（上游会把整条回复从上下文回滚，
    零前缀=保持回滚）。截断处回退到最近的标点，避免半个词留在上下文里。
    """
    if not transcript or audio_seconds <= 0 or played_seconds <= 0:
        return ""
    n = int(len(transcript) * min(1.0, played_seconds / audio_seconds))
    if n < 2:
        return ""
    cut = transcript[:n]
    if n < len(transcript):
        for i in range(len(cut) - 1, 0, -1):
            if cut[i] in _PUNCT:
                cut = cut[: i + 1]
                break
    return cut


_NORM_RE = re.compile(r"[\s，。！？、,.!?…~—「」『』\"'：:；;（）()【】\[\]]+")


def is_echo(user_transcript: str, recent_assistant: list[str]) -> bool:
    """回声回合判定（纯函数，便于单测）：转写出的「用户话」其实是助手
    自己的声音被麦克风收回去（外放泄漏）。

    规则：去标点空白后，候选 ≥4 字 且 与任一近期助手文本存在 ≥4 字的
    「连续」匹配块 且 总覆盖率 ≥50%。连续块要求是关键——回声重放的是连续
    音频，必然带出长连续块；而真实短答只会撞上零散文案（「我吃了」撞
    「你吃了吗」就 2-3 字），2026-08-23 实测三轮误杀后加这道。
    短句（<4 字）永不判回声——「你好啊」这种真实短句撞车概率太高。
    已知误杀面：用户紧接着整句复读良子的话逗他，会被当回声掐掉——接受。"""
    import difflib

    candidate = _NORM_RE.sub("", user_transcript or "")
    if len(candidate) < 4:
        return False
    for past in recent_assistant:
        p = _NORM_RE.sub("", past or "")
        if not p:
            continue
        blocks = difflib.SequenceMatcher(None, candidate, p).get_matching_blocks()
        longest = max((m.size for m in blocks), default=0)
        coverage = sum(m.size for m in blocks) / len(candidate)
        if longest >= 4 and coverage >= 0.5:
            return True
    return False


def is_vocabulary_recitation(transcript: str, hotwords: list[str]) -> bool:
    """热词表背诵判定（纯函数，便于单测）：噪音/杂声被热词先验脑补成
    「良子，大胃袋，味真足。」这种整段词表复读（2026-08-23 实测出现）。

    规则：候选去标点空白后逐一剥掉热词（长词优先），命中 ≥2 个且剥完
    无残余 = 背诵。真实短句安全：单热词（「大胃袋」）或带残余
    （「味真足啊」剩「啊」）都放行。"""
    candidate = _NORM_RE.sub("", transcript or "")
    if len(candidate) < 2:
        return False
    hits = 0
    for word in sorted(set(hotwords), key=len, reverse=True):
        w = _NORM_RE.sub("", str(word))
        if w and w in candidate:
            hits += 1
            candidate = candidate.replace(w, "")
    return hits >= 2 and not candidate


_EMO_SENT_RE = re.compile(r"[^。！？!?…~]{2,}[。！？!?…~]+")
_EMO_LABELS = ("happy", "angry", "sad", "surprised", "neutral")


async def _classify_emotion(client, llm: dict, sentence: str) -> str | None:
    """一句话情绪分类（复用 llm 配置，非思考模式，max_tokens 8）。失败静默 None。"""
    try:
        r = await client.post("/chat/completions", json={
            "model": llm["model"],
            "messages": [
                {"role": "system", "content":
                 "你是情绪分类器。输入是直播主播说的一句中文口语，判断说话人的情绪，"
                 "只回复一个标签：happy（开心/得意/兴奋/炫耀）、angry（生气/急眼/怼人）、"
                 "sad（难过/委屈/走心/温情）、surprised（惊讶/震惊/懵）、"
                 "neutral（平静叙述/以上都不是）。只输出标签本身，任何多余字都不要。"},
                {"role": "user", "content": sentence},
            ],
            "max_tokens": 8,
            "temperature": 0,
            "stream": False,
            "reasoning_effort": "none",
        })
        text = (r.json()["choices"][0]["message"]["content"] or "").strip().lower()
        for label in _EMO_LABELS:
            if label in text:
                return label
    except Exception as e:
        logger.debug("情绪分类失败（忽略）: %r", e)
    return None


class Session:
    """一个浏览器连接 ↔ 一路 s2s 的编排。"""

    def __init__(self, browser_ws, s2s_url: str,
                 personas: dict, default_persona: str,
                 rtc_pacer=None, rtc_ice_servers: list | None = None,
                 hotwords: list | None = None,
                 emo_llm: dict | None = None):
        self.browser = browser_ws
        self.s2s_url = s2s_url
        # WebRTC 音频轨：下行音频走 RTC，WS 只留控制/转写
        self.pacer = rtc_pacer
        self._rtc_ice_servers = rtc_ice_servers or []
        self.personas = personas
        self.persona_id = default_persona
        self._hotwords = hotwords or []  # STT 热词表（背诵判定用）
        self.s2s = None
        self._assistant_speaking = False  # 本轮回复有音频在播（打断回报判定用）
        self._resp_had_content = False  # 本轮回复是否有任何文本/音频产出（空回复兜底用）
        self._empty_nudged = False      # 本轮是否已追问过（防追问死循环）
        self._reply_transcript = ""     # 本轮回复的转写文本（打断回报估算用）
        self._reply_audio_samples = 0   # 本轮回复已生成音频采样数（同上）
        self._assistant_history: list[str] = []  # 近 2 轮完整回复（回声判定用）
        self._suppress_ghost = False    # 回声回合压制中：丢弃该回合全部 response 事件
        self._playback_watch = None     # 播放清空监听任务（response.done ≠ 播完）
        self._lip_carry = b""           # 音素分析：跨 PCM 块的帧尾留存
        # 情绪分类（数字人表情驱动）：转写按句切分，整句异步判情绪发 vox.emotion。
        # 复用 llm 配置的独立小调用，不阻塞语音链路，失败静默
        self._emo_llm = emo_llm
        self._emo_client = None
        self._emo_buf = ""
        self._emo_seq = 0
        self._emo_tasks: set = set()

    async def _lip_watch(self) -> None:
        """把 pacer 实际播出的音素帧转发浏览器（vox.lip）——口型对齐播出时刻。"""
        while True:
            await asyncio.sleep(0.03)
            if self.pacer is None:
                continue
            frames = self.pacer.pop_played_lip()
            if frames:
                try:
                    await self.browser.send_str(json.dumps({"type": "vox.lip", "frames": frames}))
                except Exception:
                    return

    async def _notify_playback_done(self) -> None:
        """等 pacer 里的音频真正播完，再通知前端（vox.playback_done）。
        上限 90s 兜底（RTC 未建连时队列不会被消费，防永远等不到）。"""
        for _ in range(900):
            if self.pacer is None or self.pacer.buffered_audio_seconds <= 0.02:
                break
            await asyncio.sleep(0.1)
        try:
            await self.browser.send_str(json.dumps({"type": "vox.playback_done"}))
        except Exception:
            pass

    async def _connect_s2s_with_retry(self):
        """连管线，槽位占用时自动重试。

        管线只有 1 个会话槽：上个会话断开后 SESSION_END 要等 handler 链
        排空才释放（TTS 还在生成时能拖到 ~10s），这期间新连接被 accept 后
        秒拒（session_limit_reached + 1008）。表现就是「刷新/首点必连不上」。
        槽位释放在几秒级，此处重试几次即可透明吸收（2026-08-25 实测）。
        """
        import websockets

        for attempt in range(6):
            try:
                s2s = await websockets.connect(self.s2s_url, max_size=16 * 1024 * 1024)
                try:
                    # 判活：正常会话首条必是 session.created；秒拒则拿到 error/被关
                    raw = await asyncio.wait_for(s2s.recv(), timeout=3)
                    if json.loads(raw).get("type") == "session.created":
                        if attempt:
                            logger.info("s2s 第 %d 次重试连上", attempt + 1)
                        # session.created 前端无 handler，吞掉无影响（日志留痕）
                        return s2s
                    logger.info("s2s 秒拒（槽位释放中），1.5s 后第 %d 次重试", attempt + 2)
                    await s2s.close()
                except (TimeoutError, websockets.exceptions.ConnectionClosed):
                    logger.info("s2s 连接后未就绪（槽位释放中），1.5s 后第 %d 次重试", attempt + 2)
            except Exception as e:
                logger.info("s2s 连接失败: %r，1.5s 后第 %d 次重试", e, attempt + 2)
            await asyncio.sleep(1.5)
        return None

    async def run(self) -> None:
        s2s = await self._connect_s2s_with_retry()
        if s2s is None:
            logger.warning("s2s 多次重试仍未连上，放弃本会话")
            return
        async with s2s:
            self.s2s = s2s
            await self._apply_persona(self.persona_id)
            await self._send_status()
            tasks = [
                asyncio.create_task(self._browser_to_s2s()),
                asyncio.create_task(self._s2s_to_browser()),
                asyncio.create_task(self._lip_watch()),
            ]
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            for task in done:
                logger.info("session 退出: task=%s exc=%r",
                            task.get_coro().__qualname__, task.exception())

    async def close(self) -> None:
        """关闭本会话（新浏览器连接顶掉旧连接时调用）。
        断开 s2s 释放管线槽位；转发协程随连接关闭自行退出。"""
        logger.info("close() 被调用: s2s=%s", self.s2s)
        if self.s2s is not None:
            try:
                await self.s2s.close()
            except Exception:
                pass

    async def _send_status(self) -> None:
        await self.browser.send_str(json.dumps({
            "type": "vox.status",
            "persona": self.persona_id,
            "rtc": {"enabled": self.pacer is not None, "ice_servers": self._rtc_ice_servers},
        }))

    # ── 情绪分类（数字人表情驱动）──

    def _emo_feed(self, delta: str) -> None:
        """流式转写按句切分，整句送去判情绪。"""
        if self._emo_llm is None:
            return
        self._emo_buf += delta
        pos = 0
        for m in _EMO_SENT_RE.finditer(self._emo_buf):
            pos = m.end()
            self._emo_classify(m.group(0))
        self._emo_buf = self._emo_buf[pos:]

    def _emo_flush(self) -> None:
        """response.done：不足一句的尾巴也判（短回复全靠它）。"""
        tail = self._emo_buf.strip()
        self._emo_buf = ""
        if self._emo_llm is not None and len(tail) >= 2:
            self._emo_classify(tail)

    def _emo_classify(self, sentence: str) -> None:
        self._emo_seq += 1
        task = asyncio.create_task(self._emo_classify_send(self._emo_seq, sentence))
        self._emo_tasks.add(task)
        task.add_done_callback(self._emo_tasks.discard)

    def _emo_http(self):
        if self._emo_client is None:
            import httpx
            self._emo_client = httpx.AsyncClient(
                base_url=self._emo_llm["base_url"],
                headers={"Authorization": f"Bearer {self._emo_llm['api_key']}"},
                timeout=10)
        return self._emo_client

    async def _emo_classify_send(self, seq: int, sentence: str) -> None:
        emo = await _classify_emotion(self._emo_http(), self._emo_llm, sentence)
        if not emo:
            return
        try:
            await self.browser.send_str(json.dumps(
                {"type": "vox.emotion", "seq": seq, "emotion": emo, "text": sentence[:40]}))
        except Exception:
            pass

    async def _apply_persona(self, persona_id: str) -> None:
        persona = self.personas[persona_id]
        self.persona_id = persona_id
        await self.s2s.send(json.dumps(build_session_update(persona_id, persona["text"])))

    # ── 两条转发协程 ──

    async def _browser_to_s2s(self) -> None:
        async for message in self.browser:
            if message.type.name != "TEXT":
                continue  # 二进制帧（历史截帧协议）已废弃，直接忽略
            try:
                event = json.loads(message.data)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "vox.persona":
                pid = event.get("id")
                if pid in self.personas:
                    await self._apply_persona(pid)
                    await self._send_status()
                continue
            await self.s2s.send(message.data)

    async def _s2s_to_browser(self) -> None:
        async for raw in self.s2s:
            if not isinstance(raw, str):
                continue
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                await self.browser.send_str(raw)
                continue
            relay, is_interrupt, pcm = classify_s2s_event(event)
            etype = event.get("type", "")
            # ── 回声回合压制：外放泄漏把助手自己的话收成「用户说」──
            if self._suppress_ghost:
                if etype == "response.done" or is_interrupt:
                    self._suppress_ghost = False  # 幽灵回合结束/真人新开口：解除
                elif etype.startswith("response."):
                    continue  # 幽灵回合事件全丢：不上屏、不出声、不计数
            if etype == "conversation.item.input_audio_transcription.completed":
                heard = event.get("transcript", "")
                if is_vocabulary_recitation(heard, self._hotwords):
                    logger.info("热词背诵压制（噪音被词表脑补）：%r，掐掉", heard[:30])
                    self._suppress_ghost = True
                    await self.s2s.send(json.dumps({"type": "response.cancel"}))
                    continue  # 转写不上屏
                if is_echo(heard, self._assistant_history + [self._reply_transcript]):
                    logger.info("回声回合压制：%r 与近期助手文本重合，掐掉", heard[:30])
                    self._suppress_ghost = True
                    await self.s2s.send(json.dumps({"type": "response.cancel"}))
                    continue  # 转写不上屏
            # 空回复兜底追踪：本轮有任何文本/音频产出即视为有内容
            if etype == "response.created":
                if self._reply_transcript:
                    self._assistant_history = (
                        self._assistant_history + [self._reply_transcript])[-2:]
                self._resp_had_content = False
                self._reply_transcript = ""     # 本轮回复转写（打断回报用）
                self._reply_audio_samples = 0   # 本轮回复已生成音频采样
                self._emo_buf = ""              # 情绪判定缓冲（新一轮）
            if etype == "response.output_audio_transcript.delta":
                self._reply_transcript += event.get("delta", "")
                self._emo_feed(event.get("delta", ""))
            if pcm is not None:
                self._reply_audio_samples += len(pcm) // 2
                self._assistant_speaking = True
            if pcm is not None or (
                etype in ("response.output_audio_transcript.delta",
                          "response.output_text.delta",
                          "response.output_audio_transcript.done")
                and (event.get("delta") or event.get("transcript"))
            ):
                self._resp_had_content = True
            elif etype == "response.done":
                self._assistant_speaking = False
                self._emo_flush()  # 尾巴判情绪
                # 生成完毕 ≠ 播放完毕：pacer 队列里还有没播的音频。
                # 通知前端播放真正清空（或即将清空）的时刻，前端据此收「说话中」
                if self.pacer is None:
                    await self.browser.send_str(json.dumps({"type": "vox.playback_done"}))
                elif self._playback_watch is None or self._playback_watch.done():
                    self._playback_watch = asyncio.create_task(self._notify_playback_done())
                status = (event.get("response") or {}).get("status")
                if (not self._resp_had_content and not self._empty_nudged
                        and status in (None, "completed")):
                    # LLM 偶发只吐 1 个 token → 清理后无声。
                    # 追问一次让模型重答，把抽风变成一句话的事
                    self._empty_nudged = True
                    logger.warning("空回复兜底：本轮无文本/音频产出，追问重答")
                    await self.s2s.send(json.dumps({
                        "type": "conversation.item.create",
                        "item": {"type": "message", "role": "user", "content": [{
                            "type": "input_text",
                            "text": "（你刚才的回复是空的，用一句符合你人设的话接上——比如假装清了清嗓子——然后正常回答我刚才的问题。别提这条提示）"}]}}))
                    await self.s2s.send(json.dumps({"type": "response.create"}))
            if is_interrupt:
                was_speaking = self._assistant_speaking  # 判定要在状态翻转前取
                self._assistant_speaking = False
                self._emo_buf = ""  # 打断：尾巴不判
                self._empty_nudged = False  # 新一轮对话，重置追问名额
                if was_speaking and self.pacer is not None:
                    # 打断回报：上游会把整条回复从上下文回滚，但用户实际已经
                    # 听到了一段——把已听前缀作为 assistant 消息补写回上下文
                    #（须在下方 flush 清计数前读）
                    played_s = self.pacer.reply_played_seconds
                    prefix = heard_prefix(self._reply_transcript,
                                          self._reply_audio_samples / 16000, played_s)
                    if prefix:
                        logger.info("打断回报：已播 %.1fs，已听前缀 %d 字补写回上下文",
                                    played_s, len(prefix))
                        await self.s2s.send(json.dumps({
                            "type": "conversation.item.create",
                            "item": {"type": "message", "role": "assistant",
                                     "content": [{"type": "output_text", "text": prefix}]}}))
                if self.pacer is not None:
                    # 打断：清 RTC 音频队列。response.done 后 was_speaking 已归 False，
                    # 打断回报不触发，但 flush 照样会清掉没播完的尾巴——无日志时
                    # 「她的话被谁吃了」无从查证（2026-08-25「三遍只说两遍」实为
                    # 用户抢话打断，排查全靠这条）
                    dropped_s = self.pacer.buffered_audio_seconds
                    self.pacer.flush()
                    if dropped_s > 0.1:
                        logger.info("打断清队：丢弃未播音频 %.1fs（response.done 后抢话）",
                                    dropped_s)
            lip_frames = []
            if pcm is not None:
                lip_frames, self._lip_carry = lip_analyze(pcm, self._lip_carry)
            if pcm is not None and self.pacer is not None:
                self.pacer.feed_audio(pcm, lip_frames)  # RTC 音频轨（音素帧随队播出）
            if relay:
                if pcm is not None:
                    # 音频走 RTC 音轨，WS 只留事件本身（剥掉 base64 音频体省带宽）；
                    # 附带响度（lvl）供前端驱动能量动画。口型音素（lip）只在无 pacer
                    # 时挂事件（生成时刻）；有 pacer 时走 vox.lip 播出侧通道对齐播放
                    event = {k: v for k, v in event.items() if k != "delta"}
                    event["lvl"] = audio_level(pcm)
                    if lip_frames and self.pacer is None:
                        event["lip"] = lip_frames
                    await self.browser.send_str(json.dumps(event))
                else:
                    await self.browser.send_str(raw)


def _build_emo_llm(config: dict) -> dict | None:
    """情绪分类的 LLM 配置（复用 llm 节）。key 缺失时功能静默关闭。"""
    llm_cfg = config.get("llm") or {}
    try:
        from voxemw.config import resolve_api_key
        api_key = resolve_api_key(llm_cfg)
    except SystemExit:
        return None
    return {
        "base_url": str(llm_cfg.get("base_url", "https://api.deepseek.com")).rstrip("/"),
        "model": str(llm_cfg.get("model_name", "deepseek-chat")),
        "api_key": api_key,
    }


def create_app(config: dict):
    from aiohttp import web

    server = config.get("server") or {}
    personas = config["personas"]["resolved"]
    default_persona = config["personas"]["default"]
    # STT 热词表（热词背诵压制用；配置里允许 list 或逗号串）
    stt_hotwords = (config.get("stt") or {}).get("hotwords") or []
    if isinstance(stt_hotwords, str):
        stt_hotwords = [w.strip() for w in stt_hotwords.split(",") if w.strip()]

    s2s_url = f"ws://{server.get('s2s_host', '127.0.0.1')}:{server.get('s2s_port', 8765)}/v1/realtime"
    emo_llm = _build_emo_llm(config)

    async def index(_request):
        return web.FileResponse(REPO_ROOT / "web" / "index.html")

    async def api_personas(_request):
        return web.json_response({
            "default": default_persona,
            "list": [
                {
                    "id": pid,
                    "name": p["name"],
                    "label": p.get("label") or p["name"],
                }
                for pid, p in personas.items()
            ],
        })

    # ── WebRTC 音频轨 ──
    rtc_cfg = config.get("rtc") or {}
    rtc_manager = None
    rtc_ice_servers: list = []
    if rtc_cfg.get("enabled", False):
        from voxemw.gateway.rtc import RTCManager

        rtc_manager = RTCManager(rtc_cfg)
        rtc_ice_servers = rtc_manager.browser_ice_servers
        logger.info("WebRTC 音频轨启用（Opus 48k）")

    # 单用户产品：新浏览器连接顶掉旧会话（换网络/僵尸会话不再需要刷新两次）
    current_session: dict = {"session": None}

    async def api_rtc_offer(request):
        if rtc_manager is None:
            return web.json_response({"error": "rtc 未启用"}, status=404)
        session = current_session["session"]
        if session is None or session.pacer is None:
            return web.json_response({"error": "无活跃会话，先连 /ws"}, status=409)
        session.pacer.flush()  # 新 RTC 连接从干净的队列起步（重连不播陈年积压）
        offer = await request.json()
        try:
            answer = await rtc_manager.handle_offer(offer, session.pacer)
        except Exception:
            # 排障：把失败的 offer SDP 落日志（Safari/老 WebKit 的 m-section 顺序差异）
            logger.exception("RTC offer 处理失败，SDP 前 800 字: %s",
                             str(offer.get("sdp"))[:800])
            raise
        return web.json_response(answer)

    async def api_rtc_ice(request):
        # 前端每次建连现取 ICE 配置
        if rtc_manager is None:
            return web.json_response({"ice_servers": []})
        return web.json_response({"ice_servers": rtc_manager.browser_ice_servers})

    async def api_rtc_debug(request):
        # 前端 12s 后回传的 ICE 诊断（候选/连接状态）
        body = await request.json()
        logger.info("RTC 前端诊断: %s", json.dumps(body, ensure_ascii=False))
        return web.json_response({"ok": True})

    async def api_pacer_debug(_request):
        """排障观测点：当前会话音频调度器的喂入/播放计数。
        fed>0 而 played 不动 = RTC 音频轨没在消费；fed=0 = 音频没喂进这条会话。"""
        session = current_session["session"]
        p = session.pacer if session is not None else None
        if p is None:
            return web.json_response({"pacer": None})
        return web.json_response({
            "buffered_seconds": round(p.buffered_audio_seconds, 2),
            "samples_fed": p._audio_samples_fed,
            "samples_played": p._audio_samples_played,
        })

    async def ws_handler(request):
        ws = web.WebSocketResponse(max_msg_size=16 * 1024 * 1024)
        await ws.prepare(request)
        old = current_session["session"]
        if old is not None:
            logger.info("新连接到达，顶掉旧会话（释放管线槽位）")
            await old.close()
        # ?alead=毫秒：新回复音频压后量（云时代等 avatar 渲染用，纯语音默认 0，可调）
        try:
            lead = float(request.query.get("alead", "0")) / 1000.0
        except ValueError:
            lead = 0.0
        pacer = None
        if rtc_manager is not None:
            from voxemw.gateway.audio_pacer import AudioPacer
            pacer = AudioPacer(audio_lead=lead)
        session = Session(ws, s2s_url, personas, default_persona,
                          rtc_pacer=pacer, rtc_ice_servers=rtc_ice_servers,
                          hotwords=stt_hotwords, emo_llm=emo_llm)
        current_session["session"] = session
        try:
            await session.run()
        finally:
            if current_session["session"] is session:
                current_session["session"] = None
        return ws

    app = web.Application()

    @web.middleware
    async def _no_cache(request, handler):
        # 前端 JS/HTML 迭代频繁，禁缓存防浏览器跑旧版（新旧协议不匹配会静默失声）
        resp = await handler(request)
        if request.path == "/" or request.path.startswith("/static"):
            resp.headers["Cache-Control"] = "no-cache, must-revalidate"
        return resp

    app.middlewares.append(_no_cache)
    app.router.add_get("/", index)
    app.router.add_get("/api/personas", api_personas)
    app.router.add_post("/rtc/offer", api_rtc_offer)
    app.router.add_get("/rtc/ice", api_rtc_ice)
    app.router.add_post("/rtc/debug", api_rtc_debug)
    app.router.add_get("/rtc/pacer", api_pacer_debug)
    app.router.add_get("/ws", ws_handler)
    app.router.add_static("/static", REPO_ROOT / "web")
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="VoxEMW 编排入口（浏览器 ↔ s2s + RTC 音频轨）")
    parser.add_argument("--config", default=os.environ.get("VOXEMW_CONFIG", "configs/assistant.yaml"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    from aiohttp import web

    from voxemw.config import load_config, load_dotenv

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    load_dotenv(REPO_ROOT / ".env.local")
    config = load_config(config_path)

    server = config.get("server") or {}
    host = str(server.get("host", "0.0.0.0"))
    port = int(server.get("port", 8000))

    # 可选 LAN TLS 入口（iPhone/iPad 用：iOS 的 getUserMedia 只在 https 下可用）。
    # 证书由 scripts/make_lan_tls.sh 生成，环境变量缺省/文件不存在则只开 http。
    tls_cert = os.environ.get("VOX_TLS_CERT", "")
    tls_key = os.environ.get("VOX_TLS_KEY", "")
    tls_port = int(os.environ.get("VOX_TLS_PORT", "9443"))
    ssl_ctx = None
    if tls_cert and tls_key and Path(tls_cert).is_file() and Path(tls_key).is_file():
        import ssl

        ssl_ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        ssl_ctx.load_cert_chain(tls_cert, tls_key)

    async def serve() -> None:
        runner = web.AppRunner(create_app(config))
        await runner.setup()
        await web.TCPSite(runner, host, port).start()
        logger.info("orchestrator 就绪: http://%s:%d", host, port)
        if ssl_ctx is not None:
            # LAN 入口必须绑 0.0.0.0（http 入口可以只绑回环）
            await web.TCPSite(runner, "0.0.0.0", tls_port, ssl_context=ssl_ctx).start()
            logger.info("LAN TLS 入口就绪: https://0.0.0.0:%d（iPhone 用这个）", tls_port)
        await asyncio.Event().wait()  # 常驻

    asyncio.run(serve())


if __name__ == "__main__":
    main()
