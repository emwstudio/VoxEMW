"""MiniCPM-V-4.6 视觉边车：transformers 原生 + FastAPI，OpenAI 兼容 /v1/chat/completions。

运行环境：独立 conda env `vlm`（transformers==5.7.0）——
1) 不用 vLLM：vllm 0.28 的 MiniCPM-V-4.6 权重映射有 bug（分体 k_proj
   对不上融合 qkv_proj，加载即 ValueError）；
2) 不用 py312 的 transformers 5.16.1：其窗口合并向量化在非均匀切片下
   reshape 必炸（5.16 回归），36 切片细节档（OCR）只有 5.7.0 能跑。
模型 1.3B（bf16 ~2.7GB），视觉触发低频（说「看看」才调），transformers
直跑足够，还省得为一个边车动主环境的 vllm/torch 版本。

自测：python -m voxemw.gateway.vlm_server --selftest /path/to.jpg
服务：python -m voxemw.gateway.vlm_server --port 18099
"""

from __future__ import annotations

import argparse
import base64
import io
import logging
import time

from fastapi import FastAPI, Request  # 模块级：__future__ annotations 下局部 import 的类型串解析不到
from fastapi.responses import JSONResponse

logger = logging.getLogger("vlm_server")

MODEL_ID = "openbmb/MiniCPM-V-4.6"
_model = None
_processor = None


def _load():
    global _model, _processor
    if _model is not None:
        return
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    t0 = time.perf_counter()
    _processor = AutoProcessor.from_pretrained(MODEL_ID)
    _model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda")
    _model.eval()
    logger.info("MiniCPM-V-4.6 加载完成 %.1fs", time.perf_counter() - t0)


def describe_image(pil_image, prompt: str, max_tokens: int = 120) -> str:
    """一张图 + 一句问 → 中文描述（同步，低频调用不在乎）。"""
    import torch

    _load()
    # 模型卡定式：tokenize+图像处理全在 apply_chat_template 里做，
    # downsample_mode 生成时还要再传一次（16x 是速度档，4x 细节档）。
    # max_slice_nums=36 细节档（OCR 需要）：transformers 5.16.1 的窗口合并
    # 向量化在非均匀切片下 reshape 必炸（5.16 回归，5.7.0 正常）——
    # 所以本服务跑在独立 vlm env（transformers 5.7.0），不用 py312。
    downsample_mode = "16x"
    messages = [{"role": "user", "content": [
        {"type": "image", "image": pil_image},
        {"type": "text", "text": prompt},
    ]}]
    inputs = _processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
        downsample_mode=downsample_mode, max_slice_nums=36,
    ).to("cuda")
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = _model.generate(**inputs, downsample_mode=downsample_mode,
                              max_new_tokens=max_tokens, do_sample=False)
    gen = out[:, inputs["input_ids"].shape[1]:]
    result = _processor.batch_decode(gen, skip_special_tokens=True)[0].strip()
    logger.info("描述耗时 %.2fs: %r", time.perf_counter() - t0, result[:60])
    return result


def create_app():
    app = FastAPI()

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        from PIL import Image

        body = await request.json()
        image = None
        prompt = ""
        for msg in body.get("messages", []):
            for part in msg.get("content", []):
                if part.get("type") == "image_url":
                    url = part["image_url"]["url"]
                    b64 = url.split(",", 1)[1] if "," in url else url
                    image = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
                elif part.get("type") == "text":
                    prompt = part["text"]
        if image is None:
            return JSONResponse({"error": "no image"}, status_code=400)
        t0 = time.perf_counter()
        text = describe_image(image, prompt, int(body.get("max_tokens", 120)))
        return {
            "id": f"vlm-{int(t0)}",
            "object": "chat.completion",
            "created": int(t0),
            "model": MODEL_ID,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="MiniCPM-V-4.6 视觉边车")
    parser.add_argument("--port", type=int, default=18099)
    parser.add_argument("--selftest", default="", help="给一张图路径，加载模型描述后退出")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    if args.selftest:
        from PIL import Image

        img = Image.open(args.selftest).convert("RGB")
        print(describe_image(img, "用三句以内的中文口语描述这张图片的主要内容"
                                "（人物、物体、场景），只描述看到的，不评价"))
        return

    _load()  # 先加载再开服务，health 过了就是真的能用了
    import uvicorn

    uvicorn.run(create_app(), host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
