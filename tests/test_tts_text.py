"""tts 纯逻辑单测：clean_for_tts / to_wav_bytes（不 import TTS 类/torch）。"""

import io
import wave

import numpy as np

from voxemw.tts import clean_for_tts, to_wav_bytes


class TestCleanForTts:
    def test_strips_parentheses(self):
        assert clean_for_tts("味真足（竖起大拇指）！") == "味真足！"
        assert clean_for_tts("买一送一(laugh)快来") == "买一送一快来"

    def test_strips_markdown(self):
        assert clean_for_tts("**重磅**_特惠_ #今日#") == "重磅特惠 今日"

    def test_strips_emoji(self):
        assert clean_for_tts("好吃到哭😭🔥") == "好吃到哭"

    def test_collapses_whitespace(self):
        assert clean_for_tts("第一句。\n\n第二句。\t完") == "第一句。 第二句。 完"

    def test_empty_and_none(self):
        assert clean_for_tts("") == ""
        assert clean_for_tts(None) == ""
        assert clean_for_tts("（全是括注）") == ""


class TestToWavBytes:
    def test_roundtrip(self):
        audio = np.linspace(-0.5, 0.5, 4800, dtype=np.float32)
        data = to_wav_bytes(audio, 48000)
        with wave.open(io.BytesIO(data), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 48000
            frames = wf.readframes(4800)
        pcm = np.frombuffer(frames, dtype=np.int16)
        assert len(pcm) == 4800
        assert pcm.max() > 16000 and pcm.min() < -16000

    def test_clipping(self):
        audio = np.array([-2.0, 2.0], dtype=np.float32)
        data = to_wav_bytes(audio, 48000)
        with wave.open(io.BytesIO(data), "rb") as wf:
            pcm = np.frombuffer(wf.readframes(2), dtype=np.int16)
        assert pcm.tolist() == [-32767, 32767]
