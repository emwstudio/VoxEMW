"""视觉模块单测：VisionService 流程（HTTP 全 mock，不碰真相机/服务）。"""

import asyncio
import sys
import types

from voxemw.gateway.vision import VisionService


class _FakeResp:
    status_code = 200

    def raise_for_status(self):
        pass

    def json(self):
        return {"choices": [{"message": {"content": "一碗热气腾腾的烩面"}}]}


class _FakeClient:
    def __init__(self, *a, **k):
        pass

    async def get(self, path):
        return _FakeResp()

    async def post(self, path, json=None):
        return _FakeResp()


def _fake_httpx(monkeypatch, client_cls):
    # .venv 无 httpx：往 sys.modules 塞假模块（vision.py 在 _http() 里惰性 import）
    fake = types.ModuleType("httpx")
    fake.AsyncClient = client_cls
    monkeypatch.setitem(sys.modules, "httpx", fake)


def test_describe_and_available(tmp_path, monkeypatch):
    _fake_httpx(monkeypatch, _FakeClient)
    img = tmp_path / "a.jpg"
    img.write_bytes(b"\xff\xd8\xff\xd9")
    vs = VisionService()
    assert asyncio.run(vs.available()) is True
    assert asyncio.run(vs.describe(str(img))) == "一碗热气腾腾的烩面"


def test_describe_failure_returns_none(tmp_path, monkeypatch):
    class _BadClient(_FakeClient):
        async def post(self, path, json=None):
            raise RuntimeError("vlm down")

    _fake_httpx(monkeypatch, _BadClient)
    img = tmp_path / "a.jpg"
    img.write_bytes(b"\xff\xd8\xff\xd9")
    vs = VisionService()
    assert asyncio.run(vs.describe(str(img))) is None
