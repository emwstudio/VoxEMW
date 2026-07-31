"""vision 纯逻辑单测：请求构造 / s2s 注入事件 / 配置开关（不 import aiohttp，不走网络）。"""

import base64

from voxemw import vision


def test_frame_type_distinct_from_avatar():
    # 0x01 是下行数字人帧，用户截帧必须是别的值
    assert vision.FRAME_TYPE_USER_JPEG == 0x02


def test_build_describe_request():
    cfg = {"model_name": "kimi-k3", "max_tokens": 200}
    req = vision.build_describe_request(cfg, b"\xff\xd8fake-jpeg")
    assert req["model"] == "kimi-k3"
    assert req["max_tokens"] == 200
    system, user = req["messages"]
    assert system["role"] == "system"
    assert "不评价" in system["content"]
    parts = user["content"]
    img = next(p for p in parts if p["type"] == "image_url")
    data_url = img["image_url"]["url"]
    assert data_url.startswith("data:image/jpeg;base64,")
    # base64 能解码回原图字节
    assert base64.b64decode(data_url.split(",", 1)[1]) == b"\xff\xd8fake-jpeg"


def test_build_stall_messages():
    item_create, response_create = vision.build_stall_messages()
    assert item_create["type"] == "conversation.item.create"
    item = item_create["item"]
    assert item["type"] == "message" and item["role"] == "user"
    text = item["content"][0]["text"]
    assert "垫个场" in text
    assert "别打分" in text
    assert response_create == {"type": "response.create"}


def test_build_inject_messages():
    item_create, response_create = vision.build_inject_messages("三十岁上下，短发，黑衣，直视镜头")
    assert item_create["type"] == "conversation.item.create"
    item = item_create["item"]
    assert item["type"] == "message" and item["role"] == "user"
    text = item["content"][0]["text"]
    assert item["content"][0]["type"] == "input_text"
    assert "（镜头画面：三十岁上下，短发，黑衣，直视镜头）" in text
    assert "描绘" in text  # 锐评要求先描绘外表细节
    assert "打分" in text
    assert response_create == {"type": "response.create"}


def test_vision_config_disabled_without_section():
    assert vision.vision_config({}) is None
    assert vision.vision_config({"vision": {"enabled": False}}) is None


def test_vision_config_missing_key(monkeypatch):
    monkeypatch.delenv("KIMI_API_KEY", raising=False)
    cfg = {"vision": {"api_key_env": "KIMI_API_KEY"}}
    assert vision.vision_config(cfg) is None


def test_vision_config_merges_key(monkeypatch):
    monkeypatch.setenv("KIMI_API_KEY", "sk-test")
    cfg = {"vision": {"api_key_env": "KIMI_API_KEY", "trigger": "让我好好看看你"}}
    merged = vision.vision_config(cfg)
    assert merged["api_key"] == "sk-test"
    assert merged["trigger"] == "让我好好看看你"
