"""WebRTC 音频轨（aiortc）：把 AudioPacer 的音频流以 RTP 轨推给浏览器。

音频轨 20ms 一拍、pts 游标连续推进（打断不重置），浏览器按 RTP/RTCP
时间戳播放。仅 orchestrator 进程使用；aiortc/av 为重依赖，单测不 import
本模块（节拍逻辑见 voxemw/gateway/audio_pacer.py，纯逻辑独立测试）。
"""

from __future__ import annotations

import asyncio
import fractions
import logging
import time

import numpy as np

logger = logging.getLogger(__name__)

AUDIO_RATE = 48000               # Opus 要求 48k
AUDIO_TICK_48K = 960             # 20ms


def _patch_opus_quality() -> None:
    """aiortc 的 Opus 编码器硬编 application=voip：SILK 偏向的语音优化会
    抹平瞬态——回复开头第一个音发闷/发糊（用户实测可闻）。切 audio 模式
    （全频 MDCT，瞬态保真）。96kbps 码率不变。编码器由 RTCRtpSender 内部
    创建，只能在类层面包一层。"""
    from aiortc.codecs import opus

    orig_init = opus.OpusEncoder.__init__

    def _init_audio(self):
        orig_init(self)
        self.codec.options = {"application": "audio"}

    opus.OpusEncoder.__init__ = _init_audio
    logger.info("Opus 编码器: application=audio（瞬态保真）")


class RTCAudioTrack:  # 组合而非继承：构造在 aiortc 导入前不触发重依赖
    """20ms 一拍的音频轨：调度器取 16k PCM → 重采样 48k → Opus（aiortc 编码）。
    无 TTS 音频时发静音，保持 pts 时钟连续。"""

    def __init__(self, pacer):
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
                pcm16 = pacer.next_audio_tick()
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


class RTCManager:
    """RTCPeerConnection 生命周期：offer → answer，轨绑到会话的 AudioPacer。"""

    def __init__(self, cfg: dict):
        self._turn_url = cfg.get("turn_url", "")        # 服务端侧 TURN（UDP loopback）
        self._turn_user = cfg.get("turn_username", "")
        self._turn_pass = cfg.get("turn_credential", "")
        self.browser_ice_servers = cfg.get("browser_ice_servers", [])  # 下发给前端
        _patch_opus_quality()
        self._pcs: set = set()

    async def _server_ice(self):
        """aiortc 侧 ICE：本地 coturn（浏览器同走 coturn，两侧选择强制一致——
        混配公网 TURN 必败：CF 拒向 127.0.0.1 做 CHANNEL_BIND，SSRF 防护）。"""
        from aiortc import RTCIceServer

        if self._turn_url:
            return [RTCIceServer(
                urls=[self._turn_url], username=self._turn_user, credential=self._turn_pass)]
        return []

    async def handle_offer(self, offer: dict, pacer) -> dict:
        from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription

        ice = await self._server_ice()
        pc = RTCPeerConnection(RTCConfiguration(iceServers=ice))
        self._pcs.add(pc)
        pc.addTrack(RTCAudioTrack(pacer).track)

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
        logger.info("RTC answer 已发（音频 48k Opus）")
        return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

    async def close_all(self) -> None:
        for pc in list(self._pcs):
            self._pcs.discard(pc)
            try:
                await pc.close()
            except Exception:
                pass
