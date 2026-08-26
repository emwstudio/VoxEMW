"""服务端音素（A/E/I/O/U）分析：逐行复刻 wLipSync C 实现（mrxz/wLipSync src/）。

为什么在服务端做：Chrome 里 RTC 音轨的音频进不了 WebAudio（实测静音），
浏览器端 wLipSync 拿不到数据（2026-08-25 实测：vol 恒 0，永远走兜底）。
orchestrator 本来就拿得到每块 PCM（lvl 响度就是这么算的），音素权重走
同一条已验证的 WS 通道随音频事件下发，浏览器/iOS 全免疫。

管线（与 wLipSync main.c 完全一致）：
  1024 样本@16k → pre-emphasis 0.97 → Hamming → 时域峰值归一
  → FFT 幅度谱 → 30 通道 mel 滤波（Slaney 面积归一，1127·ln 公式）
  → 10·log10 → DCT-II 取第 1..12 系数（丢弃 c0 能量项）
  → 与 profile 每音素标定均值算余弦相似度 → max(0) → ^100 锐化 → 归一
profile: web/vendor/wlipsync/lip-sync-profile.json（wLipSync 官方标定，
对良子中文语音在浏览器端验证有效，本文件是它的服务端逐行复刻）
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_PROFILE_PATH = Path(__file__).resolve().parents[2] / "web/vendor/wlipsync/lip-sync-profile.json"

SAMPLE_RATE = 16000
FRAME = 1024          # profile.sampleCount
HOP = 512             # 分析步长（32ms），一块 PCM 产出多个权重帧
N_MEL = 30            # profile.melFilterBankChannels
N_MFCC = 12           # 取 DCT 第 1..12（c0 丢弃）
_RMS_GATE = 0.005     # 帧静音门限（float 域）

_cache: dict = {}


def _templates() -> list[tuple[str, np.ndarray]]:
    """12 个音素模板（profile 内同名多条，如两组 A/I/U/E/O）。"""
    if "templates" not in _cache:
        raw = json.loads(_PROFILE_PATH.read_text())
        out = []
        for m in raw["mfccs"]:
            bank = np.array([x["array"] for x in m["mfccCalibrationDataList"]],
                            dtype=np.float64)
            out.append((m["name"], bank.mean(axis=0)))  # wLipSync: 取标定均值
        _cache["templates"] = out
    return _cache["templates"]


def _filterbank() -> np.ndarray:
    """Slaney 风格 mel 滤波器组（逐行对齐 mel_filter_bank.c）。"""
    if "fb" not in _cache:
        f_max = SAMPLE_RATE / 2
        n_max = FRAME // 2
        df = f_max / n_max
        to_mel = lambda hz: 1127.0 * np.log(hz / 700.0 + 1.0)
        to_hz = lambda mel: 700.0 * (np.exp(mel / 1127.0) - 1.0)
        d_mel = to_mel(f_max) / (N_MEL + 1)
        fb = np.zeros((N_MEL, n_max + 1), dtype=np.float64)
        for n in range(N_MEL):
            f_begin, f_center, f_end = (to_hz(d_mel * (n + k)) for k in (0, 1, 2))
            i_begin = int(np.ceil(f_begin / df))
            i_center = int(round(f_center / df))
            i_end = int(np.floor(f_end / df))
            for i in range(i_begin + 1, i_end + 1):
                if i > n_max:
                    break
                f = df * i
                a = ((f - f_begin) / (f_center - f_begin)) if i < i_center \
                    else ((f_end - f) / (f_end - f_center))
                a /= (f_end - f_begin) * 0.5
                fb[n, i] = a
        _cache["fb"] = fb
    return _cache["fb"]


def _dct() -> np.ndarray:
    if "dct" not in _cache:
        j = np.arange(N_MEL)
        i = np.arange(1, N_MFCC + 1).reshape(-1, 1)   # 取 1..12，丢 c0
        _cache["dct"] = np.cos((j + 0.5) * i * np.pi / N_MEL)
    return _cache["dct"]


_hamming = np.hamming(FRAME)


def _mfcc(frame: np.ndarray) -> np.ndarray:
    """1024 float 样本 → 12 维 MFCC（对齐 main.c 的处理顺序）。"""
    x = frame.astype(np.float64).copy()
    x[1:] -= 0.97 * x[:-1]          # pre_emphasis
    x *= _hamming                   # hamming_window
    peak = np.max(np.abs(x))        # normalize(data, 1.0)
    if peak > 1e-9:
        x /= peak
    spec = np.abs(np.fft.rfft(x))   # fft: 幅度谱
    mel = _filterbank() @ spec      # mel_filter_bank
    db = 10.0 * np.log10(np.maximum(mel, 1e-30))  # power_to_db
    return _dct() @ db              # dct → [1..12]


def _scores(cep: np.ndarray) -> dict:
    """余弦相似度 → clamp ≥0 → ^100 → 同名取 max → 归一（对齐 score/*.c）。"""
    raw: dict[str, float] = {}
    n_cep = np.linalg.norm(cep)
    if n_cep < 1e-9:
        return {"A": 0, "E": 0, "I": 0, "O": 0, "U": 0}
    for name, mean in _templates():
        sim = float(np.dot(cep, mean) / (n_cep * np.linalg.norm(mean) + 1e-30))
        score = max(sim, 0.0) ** 100
        raw[name] = max(score, raw.get(name, 0.0))
    raw["I"] = max(raw.get("I", 0.0), raw.pop("S", 0.0))  # S（摩擦/无声）并入 I
    z = sum(raw.values())
    return {k: (v / z if z > 0 else 0.0) for k, v in
            ((k2, raw.get(k2, 0.0)) for k2 in ("A", "E", "I", "O", "U"))}


def analyze(pcm: bytes, carry: bytes = b"") -> tuple[list[dict], bytes]:
    """一块 PCM(int16 16k mono) → (权重帧列表, 尾部留存)。

    按 32ms 步长滑窗分析，每 64ms 窗产出一个权重帧；不足一窗的尾部
    留给下一块拼接。静音帧产出全零（前端嘴闭上）。
    """
    buf = carry + pcm
    n_samples = len(buf) // 2
    out = []
    pos = 0
    while pos + FRAME <= n_samples:
        frame = np.frombuffer(buf, dtype=np.int16,
                              count=FRAME, offset=pos * 2).astype(np.float64) / 32768.0
        rms = float(np.sqrt(np.mean(frame * frame)))
        out.append(_scores(_mfcc(frame)) if rms >= _RMS_GATE
                   else {"A": 0, "E": 0, "I": 0, "O": 0, "U": 0})
        pos += HOP
    return out, buf[pos * 2:]
