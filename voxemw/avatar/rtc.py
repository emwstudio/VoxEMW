"""WebRTC 音画轨（aiortc）：把 AVSyncScheduler 的音画流以 RTP 轨推给浏览器。

对齐模型对标 AVTR-1 官方 demo：音频轨 20ms 一拍、视频轨 40ms 一拍，
两轨共用同一条单调时钟（track 内 pts 游标各自连续推进、打断不重置），
浏览器按 RTP/RTCP 时间戳原生同步音画——前端零补偿参数。

仅 orchestrator 进程使用；aiortc/av 为重依赖，单测不 import 本模块
（同步逻辑见 voxemw/avatar/avsync.py，纯逻辑独立测试）。
"""

from __future__ import annotations

import asyncio
import fractions
import io
import logging
import time

import numpy as np

logger = logging.getLogger(__name__)

AUDIO_RATE = 48000               # Opus 要求 48k
AUDIO_TICK_48K = 960             # 20ms
VIDEO_CLOCK = 90000              # RTP 视频时钟
VIDEO_PTS_STEP = VIDEO_CLOCK // 25


def _patch_video_bitrate(bitrate: int) -> None:
    """aiortc 视频编码器默认码率不够 720p 数字人（VP8 默认 500k/上限 1.5M，
    H264 默认 1M/上限 3M）。编码器实例由 RTCRtpSender 内部创建，只能在
    import 层改常量；VP8 顺带把 720p 的编码线程 2→4（recv→编码串行，
    2 线程撑不满 25fps 节拍）。"""
    from aiortc.codecs import h264, vpx

    vpx.DEFAULT_BITRATE = bitrate
    vpx.MAX_BITRATE = max(vpx.MAX_BITRATE, bitrate)
    h264.DEFAULT_BITRATE = bitrate
    h264.MAX_BITRATE = max(h264.MAX_BITRATE, bitrate)

    orig_threads = vpx.number_of_threads

    def _more_threads(pixels: int, cpus: int) -> int:
        if pixels >= 1280 * 720 and cpus >= 4:
            return 4
        return orig_threads(pixels, cpus)

    vpx.number_of_threads = _more_threads
    logger.info("视频轨目标码率: %d kbps（H264/VP8）", bitrate // 1000)


class RTCAudioTrack:  # 组合而非继承：构造在 aiortc 导入前不触发重依赖
    """20ms 一拍的音频轨：调度器取 16k PCM → 重采样 48k → Opus（aiortc 编码）。
    无 TTS 音频时发静音，保持 pts 时钟连续（音画对齐的基准）。"""

    def __init__(self, sched):
        from aiortc.mediastreams import AudioStreamTrack

        class _Track(AudioStreamTrack):
            def __init__(self_):
                super().__init__()
                import av

                self_._resampler = av.AudioResampler(format="s16", layout="mono", rate=AUDIO_RATE)
                self_._pts = 0
                self_._start = None

            async def recv(self_):
                import av

                if self_._start is None:
                    self_._start = time.monotonic()
                # 20ms 实时节奏（超前就睡到点；落后直接发，靠 pts 保对齐）
                delay = self_._start + self_._pts / AUDIO_RATE - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
                pcm16 = sched.next_audio_tick()
                samples = np.frombuffer(pcm16, dtype=np.int16).reshape(1, -1)
                frame = av.AudioFrame.from_ndarray(samples, format="s16", layout="mono")
                frame.sample_rate = 16000
                out = self_._resampler.resample(frame)
                if out:
                    of = out[0]
                else:  # 重采样器首拍缓冲：补静音帧
                    of = av.AudioFrame(format="s16", layout="mono", samples=AUDIO_TICK_48K)
                    of.sample_rate = AUDIO_RATE
                    for p in of.planes:
                        p.update(b"\x00" * p.buffer_size)
                of.pts = self_._pts
                of.sample_rate = AUDIO_RATE
                of.time_base = fractions.Fraction(1, AUDIO_RATE)
                self_._pts += AUDIO_TICK_48K
                return of

        self.track = _Track()


class RTCVideoTrack:
    """25fps 视频轨：生产者任务预取帧+打戳（排队≤2 吸收抖动），recv 只取货。
    背景：aiortc 发送链是 recv→编码 串行，recv 里做节拍/解码会把编码
    （720p ≈33ms）顶出 40ms 预算——预取后发送链只剩编码，30fps 可达。
    pts 与音频轨共用同一起点单调推进——帧在哪一拍显示即对应音频同一时间。"""

    def __init__(self, sched, frame_w: int = 1280, frame_h: int = 720):
        from aiortc.mediastreams import VideoStreamTrack

        raw_size = frame_w * frame_h * 3

        class _Track(VideoStreamTrack):
            def __init__(self_):
                super().__init__()
                self_._q: asyncio.Queue = asyncio.Queue(maxsize=2)
                self_._producer = None

            def _start_producer(self_):
                async def produce():
                    import av

                    sched.drop_stale_frames()  # 建连期积压的 idle 帧不播，从最新帧起步
                    pts = 0
                    start = time.monotonic()
                    stat_n, stat_t = 0, start
                    while True:
                        delay = start + (pts // VIDEO_PTS_STEP) / 25 - time.monotonic()
                        if delay > 0:
                            await asyncio.sleep(delay)
                        data = await sched.next_frame_tick()
                        if data is None:
                            await self_._q.put(None)  # 调度器关闭 → 通知 recv 结束
                            return
                        if len(data) == raw_size:
                            # 裸 RGB 帧：零拷贝（RTC 链路默认路径）
                            rgb = np.frombuffer(data, dtype=np.uint8).reshape(frame_h, frame_w, 3)
                        else:
                            # JPEG（WS 兜底/协商失败时）：解码重，上线程池
                            def _decode(d=data):
                                from PIL import Image

                                return np.asarray(Image.open(io.BytesIO(d)).convert("RGB"))

                            rgb = await asyncio.get_running_loop().run_in_executor(None, _decode)
                        frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
                        frame.pts = pts
                        frame.time_base = fractions.Fraction(1, VIDEO_CLOCK)
                        await self_._q.put(frame)  # 满则阻塞 = 背压，不堆内存
                        pts += VIDEO_PTS_STEP
                        stat_n += 1
                        if stat_n % 250 == 0:
                            now = time.monotonic()
                            logger.info("RTC 视频轨产出: %.1f fps（%d 帧）",
                                        250 / max(now - stat_t, 1e-6), stat_n)
                            stat_t = now

                self_._producer = asyncio.create_task(produce())

            async def recv(self_):
                from aiortc.mediastreams import MediaStreamError

                if self_._producer is None:
                    self_._start_producer()
                item = await self_._q.get()
                if item is None:
                    raise MediaStreamError
                return item

            def stop(self_):
                super().stop()
                if self_._producer is not None:
                    self_._producer.cancel()

        self.track = _Track()


class RTCManager:
    """RTCPeerConnection 生命周期：offer → answer，轨绑到会话的 AVSyncScheduler。"""

    def __init__(self, cfg: dict):
        self._turn_url = cfg.get("turn_url", "")        # 服务端侧 TURN（UDP loopback）
        self._turn_user = cfg.get("turn_username", "")
        self._turn_pass = cfg.get("turn_credential", "")
        self.browser_ice_servers = cfg.get("browser_ice_servers", [])  # 下发给前端
        _patch_video_bitrate(int(cfg.get("video_bitrate", 2_000_000)))
        self._pcs: set = set()

    async def _server_ice(self):
        """aiortc 侧 ICE：本地 coturn（浏览器同走 coturn，两侧选择强制一致——
        混配公网 TURN 必败：CF 拒向 127.0.0.1 做 CHANNEL_BIND，SSRF 防护）。"""
        from aiortc import RTCIceServer

        if self._turn_url:
            return [RTCIceServer(
                urls=[self._turn_url], username=self._turn_user, credential=self._turn_pass)]
        return []

    async def handle_offer(self, offer: dict, sched) -> dict:
        from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription

        # 码率可按 offer 要求临时调整（前端 ?vbr=kbps，调参用；编码器常量是全局的，
        # 单用户产品无所谓，下次 offer 会再覆盖）
        vbr = int(offer.get("vbr") or 0)
        if vbr >= 250:
            _patch_video_bitrate(vbr * 1000)

        ice = await self._server_ice()
        pc = RTCPeerConnection(RTCConfiguration(iceServers=ice))
        self._pcs.add(pc)
        pc.addTrack(RTCAudioTrack(sched).track)
        pc.addTrack(RTCVideoTrack(sched).track)


        @pc.on("connectionstatechange")
        async def _on_state():
            if pc.connectionState in ("failed", "closed"):
                logger.info("RTC 连接 %s，释放", pc.connectionState)
                self._pcs.discard(pc)
                await pc.close()

        await pc.setRemoteDescription(
            RTCSessionDescription(sdp=offer["sdp"], type=offer["type"]))
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)  # aiortc 不支持 trickle：此调用完成后候选已收齐
        logger.info("RTC answer 已发（音频 48k Opus + 视频 720p25 VP8）")
        return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

    async def close_all(self) -> None:
        for pc in list(self._pcs):
            self._pcs.discard(pc)
            try:
                await pc.close()
            except Exception:
                pass
