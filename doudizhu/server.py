"""斗地主语音服务器：游戏引擎 + DeepSeek bot + Qwen3-ASR + VoxCPM2 的编排。

协议（ws，JSON；音频 int16 16kHz PCM 的 base64）：
  客户端 -> 服务端：
    {"type":"audio","pcm":...}        麦克风流
    {"type":"play","cards":["S3"]}    点牌出牌
    {"type":"pass"}                   不要（按钮）
    {"type":"bid","call":true}        叫/不叫地主（按钮）
    {"type":"new_game"}               完局后再来一局
  服务端 -> 客户端：
    {"type":"hello","names":{...},"voices":[...]}
    {"type":"state","state":{...},"events":[...]}
    {"type":"stt","text":...}         你说话的转写
    {"type":"subtitle","who":...,"text":...}
    {"type":"tts_start"/"tts"/"tts_end","voice":...,"pcm":...}
    {"type":"error","message":...}

单客户端（一个真人）；与 speech-to-speech 聊天管线互斥（显存），启动前先停掉它。
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
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from doudizhu import chat  # noqa: E402
from doudizhu.bot import decide_and_act, find_used_phrases, react  # noqa: E402
from doudizhu.deepseek import chat_complete  # noqa: E402
from doudizhu.engine import Game, IllegalMove  # noqa: E402
from doudizhu.persona import load_persona  # noqa: E402

logger = logging.getLogger("doudizhu.server")

USER_SEAT = "you"
BOT_SEATS = ("liangzi", "fengge")
SEATS = (USER_SEAT, *BOT_SEATS)

# 事件 -> 哪个 bot 接话吐槽。只留完局总结（游戏已结束、不打乱轮替）；
# 炸弹/报单的插嘴去掉了——那是别的 bot 在轮次外抢话，用户反馈「乱搭话」
_REACT_EVENTS = ("finish",)

# LLM 台词里的动作/语气括注（全角半角都算）：字幕保留，TTS 前剥掉，
# 不然 VoxCPM 会把「（笑）」之类的也念出来
_PAREN_RE = re.compile(r"[（(][^（）()]*[）)]")


def load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


class GameServer:
    def __init__(self, config: dict):
        self.cfg = config
        self.names = {USER_SEAT: "你"}
        self.bots = {}
        for seat in BOT_SEATS:
            persona = load_persona(REPO_ROOT / "personas" / f"{seat}.md")
            self.bots[seat] = persona
            self.names[seat] = persona.name

        llm_cfg = config["llm"]
        api_key = os.environ.get(llm_cfg.get("api_key_env", "DEEPSEEK_API_KEY")) or os.environ.get(
            "LLM_API_KEY"
        )
        if not api_key:
            sys.exit("ERROR: 缺少 LLM API key（DEEPSEEK_API_KEY 或 LLM_API_KEY）")
        self.llm = partial(
            chat_complete,
            llm_cfg["base_url"],
            api_key,
            llm_cfg["model"],
        )

        self.game = Game(seats=SEATS)
        # 固定地主（configs: game.fixed_landlord）；None 则轮流叫地主
        self.fixed_landlord = (config.get("game") or {}).get("fixed_landlord")
        self._engine_lock = asyncio.Lock()
        self._send_lock = asyncio.Lock()
        self._tts_q: asyncio.Queue = asyncio.Queue()
        self._driver_task: asyncio.Task | None = None
        # 本局各 bot 已用过的口头禅（bot.py 的池），开局时重置——台词每句带梗且不重复
        self._used_phrases: dict[str, list[str]] = {seat: [] for seat in BOT_SEATS}
        self.ws = None

        # TTS 专用单线程池：VoxCPM2 optimize 用 mode="reduce-overhead"（cudagraph
        # trees），其容器挂在首次编译所在线程的 TLS 上，换线程调用编译产物会
        # AssertionError(_is_key_in_tls)。加载（含 warmup 编译）和每次合成必须
        # 在同一个线程——池子只养一个 worker，提交的第一个任务就是加载。
        self._tts_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts")

        # 语音模型在 load_models() 里加载（重，启动一次）
        self.stt = None
        self.tts = None
        self.vad = None

    # ------------------------------------------------------------------
    # 模型
    # ------------------------------------------------------------------

    def _load_tts(self, tts_cfg: dict) -> None:
        """在 TTS 专用线程里加载 + warmup（只应由 _tts_pool 的第一个任务调用）。"""
        from doudizhu.voice import TTS

        self.tts = TTS(**tts_cfg)
        for seat, persona in self.bots.items():
            ref_wav = (REPO_ROOT / persona.ref_wav).resolve()
            ref_text_path = (REPO_ROOT / persona.ref_text).resolve()
            ref_text = ref_text_path.read_text(encoding="utf-8").strip()
            self.tts.add_voice(seat, str(ref_wav), ref_text)

    def load_models(self) -> None:
        from doudizhu.voice import STT, VADSegmenter

        stt_cfg = self.cfg["stt"]
        self.stt = STT(**stt_cfg)
        self._tts_pool.submit(self._load_tts, dict(self.cfg["tts"])).result()
        self.vad = VADSegmenter(**self.cfg.get("vad", {}))
        logger.info("All models loaded. voices=%s", self.tts.voices)

    # ------------------------------------------------------------------
    # ws 收发
    # ------------------------------------------------------------------

    async def send(self, msg: dict) -> None:
        if self.ws is None:
            return
        try:
            async with self._send_lock:
                await self.ws.send(json.dumps(msg, ensure_ascii=False))
        except Exception as e:
            logger.warning("ws send failed: %s", e)

    async def broadcast_state(self, events: list[dict]) -> None:
        await self.send({
            "type": "state",
            "state": self.game.state_for(USER_SEAT),
            "events": events,
        })

    # ------------------------------------------------------------------
    # TTS 输出队列（串行，一个说完下一个说）
    # ------------------------------------------------------------------

    async def tts_consumer(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            seat, text = await self._tts_q.get()
            try:
                if seat is None:  # 关闭信号
                    return
                await self.send({"type": "subtitle", "who": seat, "text": text})
                spoken = _PAREN_RE.sub("", text).strip()
                if not spoken:  # 整句都是括注：只出字幕，不发声
                    continue
                await self.send({"type": "tts_start", "voice": seat})
                t0 = loop.time()
                total_pcm = 0
                chunk_q: asyncio.Queue = asyncio.Queue()

                def produce():
                    try:
                        for chunk in self.tts.stream_chunks(spoken, seat):
                            pcm = (np.clip(chunk, -1, 1) * 32767).astype(np.int16).tobytes()
                            loop.call_soon_threadsafe(chunk_q.put_nowait, pcm)
                    except Exception:
                        logger.exception("TTS synth failed (%s)", seat)
                    finally:
                        loop.call_soon_threadsafe(chunk_q.put_nowait, None)

                fut = loop.run_in_executor(self._tts_pool, produce)
                while True:
                    pcm = await chunk_q.get()
                    if pcm is None:
                        break
                    total_pcm += len(pcm)
                    await self.send({"type": "tts", "voice": seat,
                                     "pcm": base64.b64encode(pcm).decode("ascii")})
                await fut
                await self.send({"type": "tts_end", "voice": seat})
                # 发完≠播完：合成是十几倍速，浏览器按 1 倍速顺序播。
                # 按音频时长（int16 16kHz 单声道 = 32000 B/s）减去发送已耗时间，
                # 等客户端差不多播完再 task_done——不然 drive_bots 的 join
                # 提前放行，下一个 bot 牌比嘴快
                remaining = total_pcm / 32000.0 - (loop.time() - t0)
                if remaining > 0:
                    await asyncio.sleep(remaining)
            finally:
                self._tts_q.task_done()  # 配合 drive_bots 的 join：说完才轮到下一家

    def speak(self, seat: str, text: str) -> None:
        text = (text or "").strip()
        if text and self.tts is not None:
            self._tts_q.put_nowait((seat, text))

    # ------------------------------------------------------------------
    # 游戏驱动
    # ------------------------------------------------------------------

    async def human_action(self, fn) -> None:
        """真人动作（点牌/按钮/口令）：应用到引擎并继续驱动 bot。"""
        try:
            async with self._engine_lock:
                events = fn()
            await self.broadcast_state(events)
            self._kick_driver()
        except IllegalMove as e:
            await self.send({"type": "error", "message": str(e)})

    def _kick_driver(self) -> None:
        if self._driver_task is None or self._driver_task.done():
            self._driver_task = asyncio.create_task(self.drive_bots())

    async def drive_bots(self) -> None:
        """轮到 bot 就连续行动，直到轮到真人或完局。"""
        while True:
            # 节奏：等上一句语音说完，下一个 bot 再行动（不然牌比嘴快）
            await self._tts_q.join()
            await asyncio.sleep(0.5)  # 说完再留半秒喘气，给客户端一点渲染节奏
            async with self._engine_lock:
                game = self.game
                if game.phase == "finished" or game.phase == "idle":
                    return
                actor = game.bid_turn if game.phase == "bidding" else game.turn
                if actor == USER_SEAT:
                    await self.broadcast_state([])
                    return
                events, say = await asyncio.to_thread(
                    decide_and_act, game, actor, self.bots[actor], self.llm, self.names,
                    self._used_phrases[actor],
                )
                self._used_phrases[actor].extend(find_used_phrases(actor, say))
            self.speak(actor, say)
            await self._react_to(events, actor)
            if self.game.phase == "finished":
                # 完局：牌和结算横幅立刻上桌，不等语音
                await self.broadcast_state(events)
                return
            # 节奏铁律：这句话说完，牌才落到桌面，下一家的高亮跟着这一帧来。
            # 发完≠播完（tts_consumer 里按音频时长等过），join 返回≈客户端播完
            await self._tts_q.join()
            await self.broadcast_state(events)

    async def _react_to(self, events: list[dict], actor: str) -> None:
        """炸弹/报单/胜负时，让相关 bot 补一句嘴炮。"""
        interesting = [e for e in events if e["type"] in _REACT_EVENTS]
        if not interesting:
            return
        e = interesting[0]
        if e["type"] == "finish":
            desc = (
                f"本局结束，{'地主' if e['winner'] == 'landlord' else '农民'}赢了"
                + ("，春天！" if e.get("spring") else "")
            )
            win_seat = e["win_seat"]
            speaker = win_seat if win_seat in self.bots else actor
        elif e["type"] == "bomb":
            desc = f"{self.names.get(e['seat'])} 扔了个{'王炸' if e['combo_type'] == 'rocket' else '炸弹'}！"
            speaker = next(s for s in BOT_SEATS if s != e["seat"])
        else:  # last_card
            desc = f"{self.names.get(e['seat'])} 只剩最后一张牌了！"
            speaker = next(s for s in BOT_SEATS if s != e["seat"])
        line = await asyncio.to_thread(
            react, self.game, speaker, self.bots[speaker], self.llm, self.names, desc
        )
        if line:
            self.speak(speaker, line)

    # ------------------------------------------------------------------
    # 语音输入
    # ------------------------------------------------------------------

    async def on_utterance(self, audio: np.ndarray) -> None:
        """一段完整语音：STT -> 口令/闲聊 -> 动作或 bot 回话。

        全程开麦：任何时刻你说的话都转写——是口令（不要/叫地主）就在
        轮到你的对应阶段生效，否则当闲聊路由给 bot 接话。
        """
        text = await asyncio.to_thread(self.stt.transcribe, audio)
        text = (text or "").strip()
        if not text:
            return
        logger.info("STT: %s", text)
        await self.send({"type": "stt", "text": text})

        async with self._engine_lock:
            result = await asyncio.to_thread(
                chat.handle_user_text,
                self.game, USER_SEAT, text, self.bots, self.llm, self.names,
            )
        kind = result.get("kind")
        if kind == "command":
            if result.get("applied"):
                await self.broadcast_state(result["events"])
                self._kick_driver()
            else:
                await self.send({"type": "error", "message": result.get("reason", "口令不适用")})
        elif kind == "chat":
            self.speak(result["bot"], result["reply"])

    async def _start_game(self) -> None:
        """开一局（上桌/再来一局共用）：发牌、广播、驱动 bot，并重置口头禅记录。"""
        self._used_phrases = {seat: [] for seat in BOT_SEATS}
        async with self._engine_lock:
            events = self.game.start(fixed_landlord=self.fixed_landlord)
        await self.broadcast_state(events)
        self._kick_driver()

    # ------------------------------------------------------------------
    # 连接生命周期
    # ------------------------------------------------------------------

    async def handler(self, ws) -> None:
        if self.ws is not None:
            await ws.send(json.dumps({"type": "error", "message": "已有一桌在进行中，稍后再来"}))
            await ws.close()
            return
        self.ws = ws
        logger.info("client connected: %s", ws.remote_address)
        # 清掉上一桌残留：若上个客户端在合成中途断线，关闭信号/台词会留在队列里，
        # 新消费者一上来读到 (None, None) 就直接退出（整桌哑掉）。
        # 注意必须逐个 task_done：put_nowait 入队时 unfinished_tasks +1，
        # 只 get 不 done 会让计数永远 >0，drive_bots 的 _tts_q.join() 永久卡死
        while not self._tts_q.empty():
            self._tts_q.get_nowait()
            self._tts_q.task_done()
        consumer = asyncio.create_task(self.tts_consumer())
        try:
            await self.send({"type": "hello", "names": self.names,
                             "voices": self.tts.voices if self.tts else []})
            # 上桌即开一局
            await self._start_game()

            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                t = msg.get("type")
                if t == "audio":
                    utterance = self.vad.feed(base64.b64decode(msg["pcm"]))
                    if utterance is not None:
                        asyncio.create_task(self.on_utterance(utterance))
                elif t == "play":
                    cards = msg.get("cards") or []
                    await self.human_action(lambda c=cards: self.game.play(USER_SEAT, c))
                elif t == "pass":
                    await self.human_action(lambda: self.game.pass_turn(USER_SEAT))
                elif t == "bid":
                    call = bool(msg.get("call"))
                    await self.human_action(lambda c=call: self.game.bid(USER_SEAT, c))
                elif t == "new_game":
                    await self._start_game()
        except Exception as e:
            logger.info("client session ended: %s", e)
        finally:
            self.ws = None
            consumer.cancel()
            # driver 可能正挂在 _tts_q.join() 上，不取消的话下一桌 _kick_driver
            # 会看到旧任务未结束而不再创建，bot 就再也不动了
            if self._driver_task is not None:
                self._driver_task.cancel()
            while not self._tts_q.empty():
                self._tts_q.get_nowait()
                self._tts_q.task_done()  # 同 handler 开头：不 done 会让 join 卡死
            logger.info("client disconnected")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(REPO_ROOT / "configs/doudizhu.yaml"))
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    load_env_file(REPO_ROOT / ".env.local")
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    server = GameServer(config)
    server.load_models()

    import websockets

    host = config["server"]["host"]
    port = int(config["server"]["port"])
    logger.info("斗地主服务就绪: ws://%s:%s", host, port)

    async def serve():
        async with websockets.serve(server.handler, host, port, max_size=50 * 1024 * 1024):
            await asyncio.Future()

    asyncio.run(serve())


if __name__ == "__main__":
    main()
