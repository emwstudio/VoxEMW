"""峰哥反指提示器服务：aiohttp 单进程，同端口供警报页 + /api/*。

路由：
  GET  /                       -> web/index.html（静态页同域）
  GET  /api/blocks             -> YAML 五积木声明（绝不下发 api_key）
  POST /api/briefing           {posts, query?, task_id?} -> 生成反指短报+告警
  GET  /api/alerts?since=<id>  -> 增量告警列表（不含 audio）
  GET  /api/alerts/<id>/audio  -> audio/wav（--no-tts 时 503）
  POST /api/ask                {query} -> 语音查询入队，返回 task_id
  GET  /api/tasks/pending      -> 待处理查询（由 briefing 的 task_id 核销）
  GET  /api/voices             -> [{id, name}]（调试）
  POST /api/tts                {text, voice} -> audio/wav（调试）

TTS 线程约束（torch.compile cudagraph TLS）：模型加载和每次合成必须在同一个
线程——专用单线程池，启动时把加载作为第一个任务提交。

用法：
  python -m voxemw.server [--config configs/alerter.yaml] [--no-tts]
  --no-tts：跳过 TTS 模型加载（本地无 GPU 调试用），音频相关接口返回 503。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial
from pathlib import Path

from aiohttp import web

from voxemw.briefing import (
    build_messages,
    build_select_messages,
    dedupe_posts,
    filter_today,
    merge_archive,
    parse_briefing,
    parse_selection,
    post_fingerprint,
)
from voxemw.config import REPO_ROOT, load_config, load_dotenv, resolve_api_key
from voxemw.llm import chat_complete
from voxemw.tts import clean_for_tts, to_wav_bytes

logger = logging.getLogger("voxemw.server")

WEB_DIR = REPO_ROOT / "web"
SEEN_PATH = REPO_ROOT / "data" / "seen_posts.json"  # 博文指纹去重持久化
SEEN_CAP = 2000  # 指纹上限，超出留最新
ARCHIVE_PATH = REPO_ROOT / "data" / "posts_archive.json"  # 微博本地存档（语音查询数据源）
ARCHIVE_CAP = 5000  # 存档条目上限，超出留最新
MAX_TTS_CHARS = 500  # 单条口播稿上限，防误传长文烧卡


class AlerterServer:
    def __init__(self, config: dict, no_tts: bool = False):
        self.cfg = config
        self.blocks = config["blocks"]
        self.alerter_cfg = config.get("alerter") or {}
        self.persona_text = config["persona_text"]
        self.voice_specs = config["voices"]

        llm_cfg = self.blocks["llm"]
        api_key = resolve_api_key(llm_cfg)
        self.llm = partial(
            chat_complete,
            llm_cfg["base_url"],
            api_key,
            llm_cfg["model"],
            temperature=float(llm_cfg.get("temperature", 0.7)),
            max_tokens=int(llm_cfg.get("max_tokens", 500)),
            timeout=float(llm_cfg.get("timeout", 30)),
        )

        # 内存状态：告警列表 + 待处理语音查询
        self.alerts: list[dict] = []
        self._next_alert_id = 1
        self.tasks: list[dict] = []

        # 博文指纹去重：启动加载，每次变更写盘
        self.seen: set[str] = set()
        self._seen_order: list[str] = []
        self._load_seen()

        # 微博本地存档：语音查询的数据源，启动加载，每次变更写盘
        self.archive: list[dict] = []
        self._load_archive()

        self.tts = None
        self._tts_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts")
        if not no_tts:
            # 加载（含 warmup 编译）必须是池子的第一个任务，后续合成同线程复用
            self._tts_pool.submit(self._load_tts).result()
        else:
            logger.warning("--no-tts：跳过 TTS 模型加载，音频接口返回 503")

    def _load_tts(self) -> None:
        """在 TTS 专用线程里加载 + 预建各音色 prompt cache。"""
        from voxemw.tts import TTS

        # blocks.tts 里 impl 是积木声明，不是 TTS 构造参数
        tts_cfg = {k: v for k, v in self.blocks["tts"].items() if k != "impl"}
        self.tts = TTS(**tts_cfg)
        for voice_id, spec in self.voice_specs.items():
            self.tts.add_voice(voice_id, spec["ref_audio"], spec["ref_text"])
        logger.info("TTS ready. voices=%s sample_rate=%s", self.tts.voices, self.tts.sample_rate)

    # ------------------------------------------------------------------
    # 去重指纹持久化
    # ------------------------------------------------------------------

    def _load_seen(self) -> None:
        try:
            data = json.loads(SEEN_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if isinstance(data, list):
            self._seen_order = [str(x) for x in data][-SEEN_CAP:]
            self.seen = set(self._seen_order)
            logger.info("已加载 %d 条博文指纹: %s", len(self.seen), SEEN_PATH)

    def _save_seen(self) -> None:
        SEEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = SEEN_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._seen_order, ensure_ascii=False), encoding="utf-8")
        tmp.replace(SEEN_PATH)

    def _mark_seen(self, posts: list[dict]) -> None:
        """记入新博文指纹并写盘，超 cap 留最新。"""
        changed = False
        for p in posts:
            fp = post_fingerprint(p)
            if fp not in self.seen:
                self.seen.add(fp)
                self._seen_order.append(fp)
                changed = True
        if len(self._seen_order) > SEEN_CAP:
            self._seen_order = self._seen_order[-SEEN_CAP:]
            self.seen = set(self._seen_order)
            changed = True
        if changed:
            self._save_seen()

    # ------------------------------------------------------------------
    # 微博本地存档
    # ------------------------------------------------------------------

    def _load_archive(self) -> None:
        try:
            data = json.loads(ARCHIVE_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if isinstance(data, list):
            self.archive = [e for e in data if isinstance(e, dict) and e.get("fp")][-ARCHIVE_CAP:]
            logger.info("已加载 %d 条微博存档: %s", len(self.archive), ARCHIVE_PATH)

    def _save_archive(self) -> None:
        ARCHIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = ARCHIVE_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.archive, ensure_ascii=False), encoding="utf-8")
        tmp.replace(ARCHIVE_PATH)

    def _archive_posts(self, posts: list[dict]) -> None:
        """把博文并入存档并写盘。新条目/时间戳刷新都要落盘，直接每次写。"""
        merge_archive(self.archive, posts)
        if len(self.archive) > ARCHIVE_CAP:
            self.archive = self.archive[-ARCHIVE_CAP:]
        self._save_archive()

    def _set_archive_briefing(self, post: dict, briefing: str) -> None:
        """把某条博文的反指播报稿写进存档（时间线卡片展示用）。"""
        fp = post_fingerprint(post)
        for e in self.archive:
            if e.get("fp") == fp:
                e["briefing"] = briefing
                self._save_archive()
                return

    # ------------------------------------------------------------------
    # 路由处理
    # ------------------------------------------------------------------

    async def handle_blocks(self, request: web.Request) -> web.Response:
        """下发五积木声明；llm/tts/persona 只给 impl 等无密信息，绝不下发 api_key。"""
        safe: dict = {}
        for name in ("vad", "stt"):
            safe[name] = self.blocks.get(name) or {}
        for name in ("llm", "tts", "persona"):
            spec = dict(self.blocks.get(name) or {})
            spec.pop("api_key_env", None)
            spec.pop("api_key", None)
            safe[name] = spec
        return _json({"blocks": safe})

    async def handle_briefing(self, request: web.Request) -> web.Response:
        try:
            body = await request.json()
        except Exception:
            return _error(400, "请求体必须是 JSON")
        if not isinstance(body, dict):
            return _error(400, "请求体必须是 JSON 对象")
        posts = body.get("posts")
        if not isinstance(posts, list) or not all(isinstance(p, dict) for p in posts):
            return _error(400, "posts 必须是 [{time, text}] 数组")
        query = (body.get("query") or "").strip() or None
        task_id = (body.get("task_id") or "").strip() or None

        # ① 只留当天微博，并全部并入本地存档（语音查询的数据源）
        posts = filter_today(posts)
        if posts:
            self._archive_posts(posts)
        # ② 定时流程（无 query/task_id）做去重；语音查询不去重
        if not query and not task_id:
            posts = dedupe_posts(posts, self.seen)
            if not posts:
                return _json({"skipped": "no_new_posts"})

        if posts and not query and not task_id:
            # ③ 定时巡检：每条新博文单独生成反指播报稿，存档 + 各自成告警
            results = []
            for p in posts:
                try:
                    briefing = await self._gen_briefing([p], None)
                except Exception as e:
                    logger.exception("LLM generate failed")
                    return _error(502, f"LLM 调用失败: {e}")
                # ④ 指纹记 seen + 播报稿写进存档（时间线卡片展示用）
                self._mark_seen([p])
                self._set_archive_briefing(p, briefing)
                alert = await self._make_alert(briefing, None)
                results.append({"alert_id": alert["id"], "briefing": briefing})
            return _json({"alerts": results})

        if posts:
            # ③ 语音查询：围绕 query 整体生成一段答复
            try:
                briefing = await self._gen_briefing(posts, query)
            except Exception as e:
                logger.exception("LLM generate failed")
                return _error(502, f"LLM 调用失败: {e}")
        else:
            # 语音查询兜底：当天没有（相关）微博也要给用户一个交代，否则指令石沉大海
            if not query and not task_id:
                return _json({"skipped": "no_today_posts"})
            # 语音指令原文可能很长（整句话），兜底文案里只引用短主题
            topic = query if query and len(query) <= 12 else None
            briefing = (
                f"b友播报：截至发稿，峰哥今天还没发过和「{topic}」相关的动态。纯属娱乐，不构成任何建议。"
                if topic
                else "b友播报：截至发稿，峰哥今天还没发过相关动态。纯属娱乐，不构成任何建议。"
            )

        # ⑤ 生成告警（有 TTS 同步合成 wav；--no-tts 时 audio=None）
        alert = await self._make_alert(briefing, query)

        # ⑥ 语音查询任务核销
        if task_id:
            self.tasks = [t for t in self.tasks if t["task_id"] != task_id]

        # ⑦ 返回（不下发 audio 本体）
        return _json({"alert_id": alert["id"], "briefing": briefing})

    async def _gen_briefing(self, posts: list[dict], query: str | None) -> str:
        """LLM 生成反指短报；失败抛异常。"""
        max_chars = int(self.alerter_cfg.get("max_briefing_chars", 150))
        messages = build_messages(posts, query, self.persona_text, max_chars=max_chars)
        raw = await asyncio.to_thread(self.llm, messages)
        briefing = parse_briefing(raw)
        if not briefing:
            raise ValueError(f"LLM 返回为空或无法解析: {raw[:200]!r}")
        return briefing

    async def _make_alert(self, briefing: str, query: str | None) -> dict:
        """生成告警（同步 TTS；--no-tts 或合成失败时 audio=None）并入列。"""
        audio = None
        if self.tts is not None:
            loop = asyncio.get_running_loop()
            try:
                audio = await loop.run_in_executor(
                    self._tts_pool, self._synthesize_wav, briefing, next(iter(self.voice_specs))
                )
            except Exception:
                logger.exception("TTS synth failed，告警保留文本")
        alert = {
            "id": self._next_alert_id,
            "ts": datetime.now().isoformat(timespec="seconds"),
            "query": query,
            "briefing": briefing,
            "audio": audio,
        }
        self._next_alert_id += 1
        self.alerts.append(alert)
        return alert

    async def handle_alerts(self, request: web.Request) -> web.Response:
        try:
            since = int(request.query.get("since", "0"))
        except ValueError:
            return _error(400, "since 必须是整数")
        alerts = [
            {"id": a["id"], "ts": a["ts"], "query": a["query"], "briefing": a["briefing"]}
            for a in self.alerts
            if a["id"] > since
        ]
        return _json({"alerts": alerts})

    async def handle_alert_audio(self, request: web.Request) -> web.Response:
        try:
            alert_id = int(request.match_info["id"])
        except ValueError:
            return _error(400, "告警 id 必须是整数")
        alert = next((a for a in self.alerts if a["id"] == alert_id), None)
        if alert is None:
            return _error(404, f"告警不存在: {alert_id}")
        if alert["audio"] is None:
            return _error(503, "该告警无音频（--no-tts 模式或合成失败）")
        return web.Response(body=alert["audio"], content_type="audio/wav")

    async def handle_ask(self, request: web.Request) -> web.Response:
        try:
            body = await request.json()
        except Exception:
            return _error(400, "请求体必须是 JSON")
        query = (body.get("query") or "").strip() if isinstance(body, dict) else ""
        if not query:
            return _error(400, "query 不能为空")

        # 快路径：LLM 在本地存档里语义选相关动态 → 同步生成答复，不等 agent 轮询
        # （语音识别错字/近义表达都能处理；「最新一条」类走确定性取尾，不赌 LLM）
        today = datetime.now().date().isoformat()
        archived = [
            {"time": e["time"], "text": e["text"]}
            for e in self.archive
            if e.get("date") == today
        ]
        if archived:
            if re.search(r"最新|最近|刚发|刚更|一条|几条", query):
                hits = archived[-1:]
            else:
                try:
                    sel_raw = await asyncio.to_thread(
                        self.llm, build_select_messages(archived, query)
                    )
                    hits = [archived[i] for i in parse_selection(sel_raw, len(archived))]
                except Exception as e:
                    logger.exception("LLM select failed (ask fast path)")
                    return _error(502, f"LLM 调用失败: {e}")
            if hits:
                try:
                    briefing = await self._gen_briefing(hits, query)
                except Exception as e:
                    logger.exception("LLM generate failed (ask fast path)")
                    return _error(502, f"LLM 调用失败: {e}")
                alert = await self._make_alert(briefing, query)
                return _json({"alert_id": alert["id"], "briefing": briefing})
            # LLM 判定当天没有相关动态：直接给兜底答复，不用等 agent
            topic = query if len(query) <= 12 else None
            briefing = (
                f"b友播报：截至发稿，峰哥今天还没发过和「{topic}」相关的动态。纯属娱乐，不构成任何建议。"
                if topic
                else "b友播报：截至发稿，峰哥今天还没发过相关动态。纯属娱乐，不构成任何建议。"
            )
            alert = await self._make_alert(briefing, query)
            return _json({"alert_id": alert["id"], "briefing": briefing})

        # 慢路径：存档为空（服务刚起/还没轮询过），入队等 agent 采集补档
        task = {
            "task_id": uuid.uuid4().hex[:12],
            "query": query,
            "ts": datetime.now().isoformat(timespec="seconds"),
        }
        self.tasks.append(task)
        return _json({"task_id": task["task_id"]})

    async def handle_tasks_pending(self, request: web.Request) -> web.Response:
        return _json({"tasks": list(self.tasks)})

    async def handle_posts_today(self, request: web.Request) -> web.Response:
        """本地存档里的当天微博（语音查询数据源/页面时间线），按发布时间倒序。"""
        today = datetime.now().date().isoformat()
        posts = [e for e in self.archive if e.get("date") == today]
        posts.sort(key=lambda e: e.get("ts") or "", reverse=True)
        return _json({
            "posts": [
                {"time": e["time"], "text": e["text"], "ts": e.get("ts"),
                 "briefing": e.get("briefing")}
                for e in posts
            ]
        })

    async def handle_backfill(self, request: web.Request) -> web.Response:
        """给存档里缺播报稿的当天博文补生成（一次性迁移用，不产生告警）。"""
        today = datetime.now().date().isoformat()
        todo = [e for e in self.archive if e.get("date") == today and not e.get("briefing")]
        done = 0
        for e in todo:
            try:
                e["briefing"] = await self._gen_briefing(
                    [{"time": e["time"], "text": e["text"]}], None
                )
                done += 1
            except Exception:
                logger.exception("backfill failed: %s", e.get("fp"))
        if done:
            self._save_archive()
        return _json({"backfilled": done, "remaining": len(todo) - done})

    async def handle_voices(self, request: web.Request) -> web.Response:
        voices = [
            {"id": voice_id, "name": (spec.get("name") or voice_id)}
            for voice_id, spec in self.voice_specs.items()
        ]
        return _json({"voices": voices})

    async def handle_tts(self, request: web.Request) -> web.Response:
        if self.tts is None:
            return _error(503, "TTS 未加载（--no-tts 模式）")
        try:
            body = await request.json()
        except Exception:
            return _error(400, "请求体必须是 JSON")
        text = (body.get("text") or "").strip() if isinstance(body, dict) else ""
        voice = (body.get("voice") or "").strip() if isinstance(body, dict) else ""
        if not text:
            return _error(400, "text 不能为空")
        if not clean_for_tts(text):
            return _error(400, "text 清洗后为空（全是括号括注/标记符号）")
        if len(text) > MAX_TTS_CHARS:
            return _error(400, f"text 超过 {MAX_TTS_CHARS} 字上限")
        if voice and voice not in self.voice_specs:
            return _error(400, f"未知音色: {voice}（可选：{sorted(self.voice_specs)}）")
        voice = voice or next(iter(self.voice_specs))

        loop = asyncio.get_running_loop()
        try:
            wav_bytes = await loop.run_in_executor(
                self._tts_pool, self._synthesize_wav, text, voice
            )
        except Exception as e:
            logger.exception("TTS synth failed (%s)", voice)
            return _error(500, f"TTS 合成失败: {e}")
        return web.Response(body=wav_bytes, content_type="audio/wav")

    def _synthesize_wav(self, text: str, voice: str) -> bytes:
        """只应由 _tts_pool 的工作线程调用（与加载同线程）。"""
        audio, sr = self.tts.synthesize(text, voice)
        if audio.size == 0:
            raise ValueError("合成结果为空")
        return to_wav_bytes(audio, sr)


def _json(data: dict, status: int = 200) -> web.Response:
    return web.json_response(
        data, status=status, dumps=lambda o: json.dumps(o, ensure_ascii=False)
    )


def _error(status: int, message: str) -> web.Response:
    return _json({"error": message}, status=status)


def build_app(server: AlerterServer) -> web.Application:
    app = web.Application()
    app.router.add_get("/api/blocks", server.handle_blocks)
    app.router.add_post("/api/briefing", server.handle_briefing)
    app.router.add_get("/api/alerts", server.handle_alerts)
    app.router.add_get("/api/alerts/{id}/audio", server.handle_alert_audio)
    app.router.add_post("/api/ask", server.handle_ask)
    app.router.add_get("/api/tasks/pending", server.handle_tasks_pending)
    app.router.add_get("/api/posts/today", server.handle_posts_today)
    app.router.add_post("/api/backfill", server.handle_backfill)
    app.router.add_get("/api/voices", server.handle_voices)
    app.router.add_post("/api/tts", server.handle_tts)
    app.router.add_get("/", lambda r: web.FileResponse(WEB_DIR / "index.html"))
    app.router.add_static("/", WEB_DIR)  # 放在最后：/api/* 已被上面精确路由接管
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="VoxEMW 峰哥反指提示器服务")
    parser.add_argument(
        "--config",
        default=os.environ.get("VOXEMW_CONFIG", str(REPO_ROOT / "configs/alerter.yaml")),
        help="YAML 配置路径（默认 %(default)s，可用 VOXEMW_CONFIG 覆盖）",
    )
    parser.add_argument("--no-tts", action="store_true", help="跳过 TTS 模型加载（本地调试）")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    load_dotenv(REPO_ROOT / ".env.local")
    config = load_config(Path(args.config))

    server = AlerterServer(config, no_tts=args.no_tts)
    app = build_app(server)

    host = config["server"].get("host", "0.0.0.0")
    port = int(config["server"].get("port", 8000))
    logger.info("反指提示器服务就绪: http://%s:%s", host, port)
    web.run_app(app, host=host, port=port, print=None)


if __name__ == "__main__":
    main()
