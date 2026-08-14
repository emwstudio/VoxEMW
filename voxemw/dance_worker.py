"""跳舞素材生成工人（Wan-Animate-2 蒸馏版，DiffSynth-Studio 管线）。

用法：
    python -m voxemw.dance_worker --ref-image 全身照.png --driving-video 跳舞.mp4 \
        --name 科目三 --mode move --out data/dance/科目三.mp4

- 管线：DiffSynth-Studio WanVideoPipeline（官方低显存方案，disk offload + vram_limit）
  曾经走过 diffusers PR #14412（未合并）路线：sequential offload 报 meta tensor、
  KV cache KeyError、group offload 顶爆实例 cgroup 内存上限（64G）三连坑，弃用。
- 权重：原版 checkpoint（非 diffusers 格式），见 MODELS_DIR，ModelScope 下载
- 长视频分段（81 帧/段），且**每段一个独立子进程**：
  AutoDL 平台会 SIGKILL 长跑大内存进程（cgroup oom_kill 计数为 0，是宿主动的手），
  单进程跑 5 段必死；拆成每段一个进程后单个进程只活几分钟，且被杀只重跑当前段。
- mode：move=动画模式（照片按驱动视频动起来）；mix=替换模式（官方未开源，降级 move）
- 输出无声，用 ffmpeg 把驱动视频音轨混回成片
- 生成期间会占满显存——调用方（orchestrator 队列）负责先停 avatar/pipeline 服务
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger(__name__)

MODELS_DIR = Path("/root/autodl-tmp/models/wan_animate2")

DEFAULT_PROMPT = (
    "人物外观描述：一位成年男性，短发微翘，蓄有山羊胡，神态放松。"
    "身穿卡其色宽松短袖T恤，下着黑色束脚运动裤，脚穿白色休闲板鞋，双手插在裤兜里。"
    "背景描述：纯白色背景，无其他杂物，光线均匀明亮。"
)
DEFAULT_PROMPT_REF = "视频中的人在做动作，背景静止"

CLIP = 81   # 官方分段长度；单段超 81 画质会崩
STEP = CLIP - 1  # 段间重叠 1 帧衔接


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VoxEMW 跳舞素材生成（Wan-Animate-2 蒸馏版）")
    parser.add_argument("--ref-image", required=True)
    parser.add_argument("--driving-video", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--mode", choices=["move", "mix"], default="move")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--out", required=True)
    parser.add_argument("--height", type=int, default=0,
                        help="默认 0=按驱动视频比例自适应（短边 480，长边对齐 16）")
    parser.add_argument("--width", type=int, default=0,
                        help="同上")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0,
                        help="噪声种子；-1=随机抽卡（父进程解析后把具体值传给各段，保证全片一致）")
    parser.add_argument("--max-frames", type=int, default=1081,
                        help="取驱动视频前 N 帧（4n+1 对齐），默认 1081≈45s；>81 帧自动分段拼接")
    parser.add_argument("--seg-worker", type=int, default=-1,
                        help="内部参数：只生成第 N 段（父进程调度用）")
    return parser.parse_args()


def _load_pipe():
    import torch
    from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline

    # 内存平衡术（AutoDL 宿主压力大时会 SIGKILL 最胖的进程）：
    # - transformer（31G）：offload=disk——mmap 回读走页缓存，宿主缺内存时
    #   内核回收干净页缓存即可，不用杀我们；速度仍是内存级（缓存命中）
    # - T5/CLIP/VAE（共 16G）：驻留 CPU 内存，换块零磁盘
    # 容器占用 ≈ 20G anon + 31G 可回收缓存 ≈ 51G，低于 60G 预警线
    disk_cfg = {
        "offload_dtype": "disk",
        "offload_device": "disk",
        "onload_dtype": torch.bfloat16,
        "onload_device": "cpu",
        "preparing_dtype": torch.bfloat16,
        "preparing_device": "cuda",
        "computation_dtype": torch.bfloat16,
        "computation_device": "cuda",
    }
    cpu_cfg = {**disk_cfg, "offload_dtype": torch.bfloat16, "offload_device": "cpu"}
    t0 = time.monotonic()
    logger.info("加载 Wan-Animate-2 蒸馏版（DiffSynth 混合 offload）…")
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(path=str(MODELS_DIR / "wan_animate_2_bf16_distillation.safetensors"),
                        **disk_cfg),
            ModelConfig(path=str(MODELS_DIR / "models_t5_umt5-xxl-enc-bf16.pth"),
                        **cpu_cfg),
            ModelConfig(path=str(MODELS_DIR / "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
                        **cpu_cfg),
            ModelConfig(path=str(MODELS_DIR / "Wan2.1_VAE.pth"), **cpu_cfg),
        ],
        tokenizer_config=ModelConfig(path=str(MODELS_DIR / "umt5-xxl")),
        # 留 ~4G 给激活值：顶满总显存会在 DiT forward 里 OOM
        vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 4,
    )
    logger.info("模型就绪（%.0fs）", time.monotonic() - t0)
    return pipe


def _plan_segments(num_frames: int) -> int:
    if num_frames <= CLIP:
        return 1
    return (num_frames - CLIP + STEP - 1) // STEP + 1


def _zigzag_pad(seg: list, target: int) -> list:
    """尾段不足：之字形回绕补齐（官方 zigzag padding）。"""
    if len(seg) >= target:
        return seg
    src, idx, flip = list(seg), 0, False
    seg = list(seg)
    while len(seg) < target:
        seg.append(src[idx])
        idx += -1 if flip else 1
        if idx == 0 or idx == len(src) - 1:
            flip = not flip
    return seg


def _gen_one_segment(args, seg_idx: int, n_segs: int, num_frames: int) -> None:
    """子进程模式：只生成第 seg_idx 段，存成 mp4。"""
    from PIL import Image
    from diffsynth.utils.data import VideoData, save_video

    frames = VideoData(args.driving_video).raw_data()[:num_frames]
    seg = frames[seg_idx * STEP: seg_idx * STEP + CLIP]
    if n_segs > 1:  # 多段时尾段之字形补齐；单段短视频按实际帧数生成
        seg = _zigzag_pad(seg, CLIP)

    prev_tail = None
    if seg_idx > 0:
        prev_file = _seg_file(args, seg_idx - 1)
        prev_tail = [VideoData(str(prev_file)).raw_data()[-1]]

    pipe = _load_pipe()
    logger.info("分段 %d/%d 生成中…", seg_idx + 1, n_segs)
    t1 = time.monotonic()
    out = pipe(
        prompt=args.prompt,
        animate2_prompt_ref=DEFAULT_PROMPT_REF,
        animate2_reference_image=Image.open(args.ref_image).convert("RGB"),
        animate2_reference_video=seg,
        animate2_refert_images=prev_tail,
        animate2_offload_kv=True,
        animate2_log_scale=-1.3,  # 蒸馏版官方建议
        num_frames=len(seg),
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        cfg_scale=1.0,  # 蒸馏版无 CFG
        seed=args.seed,
        tiled=True,
    )
    logger.info("分段 %d/%d 完成（%.0fs）", seg_idx + 1, n_segs, time.monotonic() - t1)
    save_video(out, str(_seg_file(args, seg_idx)), fps=24, quality=5)


def _seg_dir(args) -> Path:
    d = Path(args.out).parent / ".segments" / args.name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _seg_file(args, seg_idx: int) -> Path:
    return _seg_dir(args) / f"seg_{seg_idx:03d}.mp4"


def main() -> None:
    args = _parse_args()
    if not args.prompt.strip():
        args.prompt = DEFAULT_PROMPT
    if args.mode == "mix":
        logger.warning("mix（替换）模式官方未开源，按 move 生成")

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    from diffsynth.utils.data import VideoData, save_video

    all_frames = VideoData(args.driving_video).raw_data()
    num_frames = min(len(all_frames), args.max_frames)
    num_frames = (num_frames - 1) // 4 * 4 + 1  # Wan 要求 4n+1
    if not args.height or not args.width:
        # 按驱动视频比例自适应，总像素对齐 640x480 的算力预算（≈307k），
        # 竖屏 9:16 → 416x736：比例正确且速度与横屏版持平（约 6min/段）
        vw, vh = all_frames[0].size
        r = vw / vh
        budget = 640 * 480
        if vw >= vh:
            h = (budget / r) ** 0.5
            w = h * r
        else:
            w = (budget * r) ** 0.5
            h = w / r
        args.width = max(16, round(w / 16) * 16)
        args.height = max(16, round(h / 16) * 16)
        logger.info("自适应分辨率: %dx%d（驱动视频 %dx%d）", args.width, args.height, vw, vh)
    seed_file = Path(args.out).with_suffix(".seed")
    if args.seed < 0:
        if seed_file.is_file():
            args.seed = int(seed_file.read_text().strip())  # 断点续跑/重试沿用同一张卡
        else:
            import random
            args.seed = random.SystemRandom().randrange(2**31)
    if args.seg_worker < 0:
        seed_file.write_text(str(args.seed))  # 固化，供复现（页面展示）
    logger.info("本次 seed=%d", args.seed)
    n_segs = _plan_segments(num_frames)

    if args.seg_worker >= 0:
        _gen_one_segment(args, args.seg_worker, n_segs, num_frames)
        return

    t0 = time.monotonic()
    logger.info("开始生成: %s (%s)，驱动视频 %d 帧，分 %d 段", args.name, args.mode,
                num_frames, n_segs)

    # 父进程：逐段拉起独立子进程（防平台 SIGKILL 团灭；已有分段文件自动跳过=断点续跑）
    for i in range(n_segs):
        seg_file = _seg_file(args, i)
        if seg_file.is_file():
            logger.info("分段 %d/%d 已存在，跳过", i + 1, n_segs)
            continue
        subprocess.run([
            sys.executable, "-m", "voxemw.dance_worker",
            "--ref-image", args.ref_image, "--driving-video", args.driving_video,
            "--name", args.name, "--mode", args.mode, "--prompt", args.prompt,
            "--out", args.out, "--height", str(args.height), "--width", str(args.width),
            "--steps", str(args.steps), "--max-frames", str(args.max_frames),
            "--seed", str(args.seed),
            "--seg-worker", str(i),
        ], cwd=str(REPO_ROOT), check=True, timeout=2400)

    # 拼接：段 0 全取，后续段去首帧（重叠帧），裁到目标帧数
    logger.info("全部分段就绪，拼接导出…")
    video = []
    for i in range(n_segs):
        seg_frames = VideoData(str(_seg_file(args, i))).raw_data()
        video.extend(seg_frames if i == 0 else seg_frames[1:])
    video = video[:num_frames]

    silent = tempfile.NamedTemporaryFile(prefix="dance_", suffix=".mp4", delete=False).name
    save_video(video, silent, fps=24, quality=5)

    # 混回驱动视频音轨（Wan 输出无声）
    subprocess.run([
        "ffmpeg", "-y", "-v", "error",
        "-i", silent, "-i", args.driving_video,
        "-map", "0:v", "-map", "1:a?", "-c:v", "copy", "-c:a", "aac",
        "-shortest", args.out,
    ], check=True)

    # 清理分段残留
    for i in range(n_segs):
        _seg_file(args, i).unlink(missing_ok=True)
    logger.info("完成: %s（总耗时 %.0fs）", args.out, time.monotonic() - t0)


if __name__ == "__main__":
    main()
