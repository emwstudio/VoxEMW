#!/usr/bin/env python
"""视频超分后处理（Real-ESRGAN anime6B，插画风数字人专用）。

用法：
    .venv/bin/python scripts/upscale_video.py 输入.mp4 输出.mp4 [--target-w 720 --target-h 1280]

流程：逐帧 x4 超分（tile=256 防爆显存）→ ffmpeg 缩到目标分辨率 + 混回原音轨。
模型：RealESRGAN_x4plus_anime_6B（/root/autodl-tmp/models/realesrgan/）
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import tempfile
import time

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("upscale")

MODEL_PATH = "/root/autodl-tmp/models/realesrgan/RealESRGAN_x4plus_anime_6B.pth"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--short-side", type=int, default=720,
                    help="目标短边像素（长边按比例，偶数对齐）；默认 720p 档")
    args = ap.parse_args()

    import cv2
    import torch
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_block=6, num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(scale=4, model_path=MODEL_PATH, model=model,
                             tile=256, tile_pad=10, pre_pad=0,
                             half=torch.cuda.is_available(),
                             device="cuda" if torch.cuda.is_available() else "cpu")

    cap = cv2.VideoCapture(args.src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    tmp = tempfile.NamedTemporaryFile(prefix="up_", suffix=".mp4", delete=False).name
    writer = None
    n = 0
    t0 = time.monotonic()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out, _ = upsampler.enhance(frame, outscale=4)
        if writer is None:
            h, w = out.shape[:2]
            writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        writer.write(out)
        n += 1
        if n % 50 == 0:
            logger.info("已超分 %d 帧（%.0fs）", n, time.monotonic() - t0)
    cap.release()
    if writer is not None:
        writer.release()
    logger.info("超分完成 %d 帧（%.0fs），缩放混流…", n, time.monotonic() - t0)

    subprocess.run([
        "ffmpeg", "-y", "-v", "error",
        "-i", tmp, "-i", args.src,
        "-map", "0:v", "-map", "1:a?",
        # 短边对齐 --short-side：横屏 → 高=720，竖屏 → 宽=720（-2 自动按比例取偶）
        "-vf", (f"scale='if(gt(iw,ih),-2,{args.short_side})':"
                f"'if(gt(iw,ih),{args.short_side},-2)':flags=lanczos"),
        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", args.dst,
    ], check=True)
    logger.info("完成: %s（总耗时 %.0fs）", args.dst, time.monotonic() - t0)


if __name__ == "__main__":
    main()
