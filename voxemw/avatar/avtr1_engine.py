"""AVTR-1 数字人引擎后端（TensorRT，avaturn-live/avtr-1）。

接口与 voxemw.avatar.service.AvatarEngine（FlashHead）完全一致：
feed_audio/reset/set_image/set_speech_active/set_idle_mode/close +
run_inference_loop(on_frames)/warmup(on_frames)，帧统一输出 960×540 RGB uint8（16:9 原生），
ws 协议层（service.py）无感知。

流式语义对齐官方 streamer（avaturn_live_streamer/worklets/rendering.py +
speech/speech_scheduler.py，2026-08-04 逐条对账）：
- chunk 步进 0.2s（3200 采样，官方 present=5 帧）产 5 帧，输入窗口 6480 采样
  （当前 3280 + 前瞻 3200+80，官方 future=5 帧+80 采样；renderer/models.py）。
  模型需 0.205s 前瞻 → 稳态供帧天然落后音频 ~0.2s（+生成 ~0.08s），
  前端 AVATAR_AUDIO_DELAY 用 0.35s（orchestrator 下发 avatar_backend 选择）。
- 运动上下文（AVTR1State）整个会话连续透传——官方 state 只在 session 开始为
  None，interrupt（打断）不重置（DiscardAvatarSpeechBuffer 只清音频队列）。
  故本引擎 reset() 只清音频缓冲、保留运动上下文（打断后静音 chunk 自然衰减
  闭嘴，无跳变）；仅 set_image 换肖像时冷启动（state 是 avatar 相关的）。
- 句中欠载（speech_active=true 而缓冲不足一个窗口）：停帧等待，不补零
  （官方 midsegment block 等 TTS 新音频；补零会插入假帧使口型滞后）。
- 句尾（speech_active 转 false 即段结束）仍有未消费真音频：立即右补零排空
  （官方 SegmentEnded → pad_right，无等待），帧标 speech，嘴型自然闭合。
- 无活动语音段：静音 chunk 持续渲染（官方 pad_right），按 0.2s 实时节奏节流，
  帧标 idle；listen 轨（用户麦克风）在 idle_mode=="listening" 时生效
  （active listening：点头/注视等倾听反应），thinking/calm 为纯静音。
- 参考帧须为 16:9 胸像（官方 18 帧全部 1920×1080、脸宽占图宽 ~0.20、头顶留白
  ~18%）：loader 会把输入非等比 resize 到 1280×720，非 16:9 输入脸型必失真
  （2026-08-04 实测：方图特写 → 窄长脸 + AR 状态漂移；合规构图下状态永续不漂移）。

运行环境：pixi env（/root/autodl-tmp/avtr-1/.pixi/envs/renderer），
不要 pixi run（会按 lock 重同步覆盖 pip 降级）——启动脚本见
/root/autodl-tmp/restart_avatar_avtr1.sh（env python 直调 + LD_LIBRARY_PATH）。
权重/引擎目录由 AVTR1_LOCAL_STORAGE 指定（含 TRT 引擎，sm89 专用）。
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
FPS = 25
FRAMES_PER_CHUNK = 5
CHUNK_STEP = 3200                                # 0.2s（5 帧 × 640）
CHUNK_WINDOW = (5 + 5) * 640 + 80                # 6480 = 当前 3280 + 前瞻 3200
CHUNK_SECONDS = CHUNK_STEP / SAMPLE_RATE         # 0.2
OUT_H, OUT_W = 720, 1280
FRAME_W = 1280          # 下行统一 1280×720（16:9 官方原生，协议层不变）
FRAME_H = 720

WARMUP_CHUNKS = 2
LISTEN_CAP = SAMPLE_RATE * 8   # listen 环形缓冲上限（最近 8s 用户语音）
# 句尾淡出：段结束时对未消费真音频末尾做余弦渐变（仅模型输入，用户听到的音频
# 不变）。硬切静音会让模型 ~0.3s 内急回中性位（用户感知「说完立马摆正」，实测
# 帧间运动峰值 1.1-1.7）；0.5s 淡出把峰值减半、回落摊到 ~1s（胶片条实测）。
TAIL_FADE_SECONDS = 0.5


class AVTR1Engine:
    """AVTR-1 推理引擎：所有 pipeline 调用序列化在 inference 线程。"""

    def __init__(self, image_path: str, storage: str | None = None,
                 bg_id: str = "plain_white", idle_motion: bool = True,
                 cfg_self_audio: float = 2.0):
        import numpy as _np  # noqa: F401

        if storage:
            os.environ.setdefault("AVTR1_LOCAL_STORAGE", storage)
        if not os.environ.get("AVTR1_LOCAL_STORAGE"):
            raise RuntimeError("AVTR1_LOCAL_STORAGE 未设置（权重/TRT 引擎根目录）")

        from avtr1_renderer.avtr1_artifact_manager import get_artifact_manager
        from avtr1_renderer.avatar_loader import AvatarLoader
        from avtr1_renderer.pipeline import Pipeline
        from avtr1_renderer.types import RenderOptions

        logger.info("加载 AVTR-1 pipeline（TRT 引擎）: storage=%s",
                    os.environ["AVTR1_LOCAL_STORAGE"])
        # avatar_ids=None：注册表留空，肖像由下面的自持 loader 动态加载（set_image）
        self.pipeline, _ = Pipeline.from_artifacts(avatar_ids=None, download_workers=1)

        mgr = get_artifact_manager()
        mask_path = (
            mgr.get_artifact_path("pasteback_mask")
            if "pasteback_mask" in mgr._artifacts
            else None
        )
        self._loader = AvatarLoader(
            engine_files={
                "insightface_det": mgr.get_artifact_path("insightface_det"),
                "landmark106": mgr.get_artifact_path("landmark106"),
                "landmark203": mgr.get_artifact_path("landmark203"),
                "appearance_extractor": mgr.get_artifact_path("appearance_extractor"),
                "motion_extractor": mgr.get_artifact_path("motion_extractor"),
            },
            mask_template_path=mask_path,
            out_h=OUT_H,
            out_w=OUT_W,
            max_dim=max(OUT_H, OUT_W),
        )
        self._options = RenderOptions(
            pixel_format="yuv_i420", bg_id=bg_id, stream_frames=False,
            # speech 轨 CFG 权重（官方默认 2.0）。实测 4.0 口型开合 +3.5%、无伪影
            # （2026-08-04 扫参）；口型幅度的主因是参考图素材（闭嘴+胡须），见审计报告
            cfg_self_audio=cfg_self_audio,
        )
        self.idle_motion = idle_motion

        self._avatar = None
        self._load_avatar(image_path)
        self._state = None  # AVTR1State，跨 chunk 运动上下文（None=冷启动）

        # 音频流账本：_buf = 未裁剪的采样流（含句尾补零），_pos = 已消费到的帧边界，
        # _real_len = 其中真实音频长度（补零不算）。chunk i 取 buf[pos:pos+WINDOW]，
        # 帧对齐音频 [pos, pos+STEP)，与前端 帧序号↔音频时钟 的消费方式一致。
        self._buf = _np.empty(0, dtype=_np.float32)
        self._pos = 0
        self._real_len = 0
        self._tail_faded = False  # 本段句尾淡出是否已施加（防多轮补零重复淡出）

        self._cond = threading.Condition()
        self._closed = False
        self._pending_image = None
        self._speech_active = False
        self._idle_mode = "calm"      # listening 时 listen 轨生效，其余静音
        # listen 环形缓冲（用户麦克风音频，16kHz f32）：实时到达，持续保留最近
        # 一段；chunk 取末尾一窗（不足左补零）。只在 idle_mode=="listening" 时使用。
        self._listen = _np.empty(0, dtype=_np.float32)
        self.on_frames = None

    # ── 内部 ──

    def _load_avatar(self, image_path: str) -> None:
        logger.info("加载 AVTR-1 肖像: %s", image_path)
        self._avatar = self._loader.load(Path(image_path), avatar_id=str(image_path))
        self._avatar_path = image_path

    # ── 生产侧（ws 线程调用）──

    def feed_audio(self, pcm_f32) -> None:
        with self._cond:
            import numpy as np

            # 句尾右补零是投机性的：新一段真音频到达时丢弃未消费的补零段再追加，
            # 否则补零占了帧序号，后续真实语音的口型会整体滞后补零时长（漂移）。
            # buf 布局恒为 [真实音频(real_len)][补零]，补零只在末尾。
            self._buf = np.concatenate([self._buf[: self._real_len], pcm_f32])
            self._real_len += len(pcm_f32)
            self._tail_faded = False  # 新音频到达：上一段的淡出标记作废
            self._cond.notify()

    def reset(self) -> None:
        """打断：丢弃未消费音频（官方 DiscardAvatarSpeechBuffer → scheduler.interrupt
        语义：只清音频队列，**运动上下文保留**——静音 chunk 让姿态自然衰减归位，
        避免参考姿态跳变；state 仅在会话开始/换肖像时冷启动）。"""
        with self._cond:
            import numpy as np

            self._buf = np.empty(0, dtype=np.float32)
            self._pos = 0
            self._real_len = 0
            self._tail_faded = False
            self._cond.notify()

    def set_image(self, image_path: str) -> None:
        with self._cond:
            self._pending_image = image_path
            self._cond.notify()

    def set_speech_active(self, on: bool) -> None:
        with self._cond:
            self._speech_active = on
            self._cond.notify()

    def feed_listen(self, pcm_f32) -> None:
        """用户麦克风音频（listen 轨）。环形保留最近 LISTEN_CAP 采样。"""
        with self._cond:
            import numpy as np

            self._listen = np.concatenate([self._listen, pcm_f32])[-LISTEN_CAP:]
            self._cond.notify()

    def set_idle_mode(self, mode: str) -> None:
        with self._cond:
            self._idle_mode = mode  # listening → listen 轨生效；thinking/calm → 静音
            self._cond.notify()

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify()

    # ── 消费侧（inference 线程）──

    @staticmethod
    def _to_display(frame) -> "object":
        """Frame(yuv_i420 720×1280) → 1280×720 RGB uint8（16:9 官方原生比例）。"""
        import cv2

        rgb = cv2.cvtColor(frame.data, cv2.COLOR_YUV2RGB_I420)
        return cv2.resize(rgb, (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA)

    def _listen_window(self):
        """当前 chunk 的 listen 轨：idle_mode=="listening" 时取环形缓冲末尾一窗
        （不足左补零），否则纯静音。对齐官方 user_speech_scheduler 语义
        （用户音频常开喂入、缺席时 scheduler 补静音）。"""
        import numpy as np

        if self._idle_mode != "listening" or len(self._listen) == 0:
            return np.zeros(CHUNK_WINDOW, dtype=np.float32)
        tail = self._listen[-CHUNK_WINDOW:]
        if len(tail) < CHUNK_WINDOW:
            tail = np.pad(tail, (CHUNK_WINDOW - len(tail), 0))
        return tail

    def _fade_tail(self) -> None:
        """对未消费真音频的末尾做余弦淡出（只改模型输入，用户听到的音频不变）。
        在句尾补零前调用一次，见 TAIL_FADE_SECONDS。"""
        import numpy as np

        n = min(int(TAIL_FADE_SECONDS * SAMPLE_RATE), self._real_len - self._pos)
        if n <= 0:
            return
        ramp = (np.cos(np.linspace(0, np.pi / 2, n)) ** 1.5).astype(np.float32)
        self._buf[self._real_len - n : self._real_len] *= ramp

    def run_inference_loop(self, on_frames) -> None:
        """阻塞循环（官方 scheduler 语义）：
        - 真实音频攒满前瞻窗口（6480）即生成（标 speech）；
        - 段结束（speech_active=false）仍有真音频尾巴：立即右补零排空（官方
          SegmentEnded → pad_right，无超时等待），嘴型自然闭合；
        - 句中欠载（speech_active=true 缓冲不足）：停帧等新音频，不补零不产帧
          （官方 midsegment block——补零假帧会让口型滞后漂移）；
        - 无活动段（缓冲全空且 speech_active=false）：静音 idle chunk 按 0.2s
          实时节流（标 idle）。
        on_frames(frames_uint8: np.ndarray (5,960,540,3), is_idle: bool) 每 chunk 一次。"""
        import time as _time

        import numpy as np
        from avtr1_renderer.types import Chunk

        last_idle_at = 0.0
        while True:
            with self._cond:
                while not self._closed:
                    unconsumed = self._real_len - self._pos      # 真音频余量
                    buffered = len(self._buf) - self._pos        # 含补零
                    if buffered >= CHUNK_WINDOW:
                        break  # 可生成（真实够窗，或句尾补零已补齐）
                    if 0 < unconsumed < CHUNK_WINDOW and not self._speech_active:
                        # 段结束尾巴：先对真音频末尾淡出（防模型急回中性位），
                        # 再立即右补零（下轮循环 break 生成）
                        if not self._tail_faded:
                            self._fade_tail()
                            self._tail_faded = True
                        pad = CHUNK_WINDOW - buffered
                        self._buf = np.concatenate(
                            [self._buf, np.zeros(pad, dtype=np.float32)])
                        continue
                    # 缓冲全空：idle 静音微动（说话期间禁止，防卡帧）
                    if unconsumed == 0 and self.idle_motion and not self._speech_active:
                        wait = last_idle_at + CHUNK_SECONDS - _time.monotonic()
                        if wait > 0:
                            self._cond.wait(timeout=wait)
                            continue  # 唤醒/超时后重查（真音频优先）
                        break
                    # 句中欠载 / 说话期间禁 idle：等状态变化（0.5s 兜底自醒）
                    self._cond.wait(timeout=0.5)
                if self._closed:
                    return
                if self._pending_image:
                    self._load_avatar(self._pending_image)
                    self._state = None  # state 是 avatar 相关的，换图必须冷启动
                    self._pending_image = None
                is_idle = (self._real_len - self._pos) == 0
                if is_idle:
                    audio = np.zeros(CHUNK_WINDOW, dtype=np.float32)
                    last_idle_at = _time.monotonic()
                else:
                    audio = self._buf[self._pos : self._pos + CHUNK_WINDOW]
                    self._pos += CHUNK_STEP
                    if self._pos > 0:  # 裁剪已消费前缀，buf 只留未消费段
                        self._buf = self._buf[self._pos :]
                        # 句尾补零段的 chunk 可能消费越过真实末尾，负余量钳到 0
                        self._real_len = max(0, self._real_len - self._pos)
                        self._pos = 0
            chunk = Chunk(audio_speech=audio,
                          audio_listen=self._listen_window())
            self._state, frames_iter = self.pipeline.process_chunk(
                self._avatar, chunk, self._state, self._options
            )
            frames = np.stack([self._to_display(f) for f in frames_iter])
            on_frames(frames, is_idle)

    def warmup(self, on_frames) -> None:
        """静音跑 2+ chunk：TRT 首个 chunk 初始化 + 运动上下文预填。"""
        import numpy as np

        logger.info("AVTR-1 预热（TRT 首个 chunk 较慢）...")
        self.feed_audio(np.zeros(CHUNK_WINDOW + WARMUP_CHUNKS * CHUNK_STEP,
                                 dtype=np.float32))
        while True:
            with self._cond:
                done = self._real_len - self._pos <= 0
            if done:
                break
            threading.Event().wait(0.1)
        self.reset()  # 清账本；运动上下文保留（静音预热态≈参考姿态，官方会话亦冷启一次后永续）
        logger.info("AVTR-1 预热完成")
