"""VoxEMW 语音管线启动器：YAML 积木配置 → speech-to-speech realtime 服务。

用法：
    python -m voxemw.pipeline.launch [--config configs/assistant.yaml] [--dry-run]

- 上游 2026-08 重构（main @5a0c79f）：工厂函数废弃，改 BackendSpec 注册表。
  我们先 register_custom_backends() 把 sensevoice/voxcpm 插进注册表，
  之后 --stt sensevoice --tts voxcpm 就是合法 CLI 参数，走标准 parse/serve 流程。
- persona 人设不进管线进程：realtime 模式下 instructions 由客户端
  （voxemw.avatar.orchestrator）经 session.update 注入。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from voxemw.config import load_config, load_dotenv  # noqa: E402
from voxemw.pipeline.args import render_s2s_argv  # noqa: E402

DEFAULT_CONFIG = "configs/assistant.yaml"


def _patch_torch_flex_attention_compat() -> None:
    """transformers 5.13 的 flex_attention 集成模块顶层 import torch 2.9 才有的
    AuxRequest，torch 2.8 没有 → ImportError 炸在 import 链上（我们的模型根本不用
    flex attention，占位即可）。"""
    try:
        from torch.nn.attention.flex_attention import AuxRequest  # noqa: F401
    except ImportError:
        import torch.nn.attention.flex_attention as _fa

        class AuxRequest:  # 占位：torch 2.9 才有真身；flex attention 不会被调用
            pass

        _fa.AuxRequest = AuxRequest


def _patch_torch_hub_offline_fallback() -> None:
    """silero VAD 走 torch.hub.load("snakers4/silero-vad"):每次启动都向
    github.com 发校验请求,而本机 GitHub 时通时断,断则启动失败。
    加离线兜底:网络失败且本地缓存存在时,source="local" 从缓存加载。"""
    from pathlib import Path

    import torch

    orig_load = torch.hub.load

    def _load_with_local_fallback(repo_or_dir, model, *args, **kwargs):
        try:
            return orig_load(repo_or_dir, model, *args, **kwargs)
        except Exception:
            if repo_or_dir == "snakers4/silero-vad" and kwargs.get("source", "github") == "github":
                local = Path(torch.hub.get_dir()) / "snakers4_silero-vad_master"
                if local.is_dir():
                    kwargs.pop("force_reload", None)
                    return orig_load(str(local), model, *args, source="local", **kwargs)
            raise

    torch.hub.load = _load_with_local_fallback


def _patch_smart_turn_gpu() -> None:
    """上游 SmartTurnAnalyzer 硬编 CPUExecutionProvider。smart_turn_model_path 指到
    *-gpu.onnx 且 CUDA 可用时换成 GPU 优先（复核 ~80ms → ~10ms）。
    做法：包一层 __init__，建完 CPU 会话后原地重建成 CUDA（模型 20MB，双载无感）。
    ⚠️ 2026-08-17 实测：与 AVTR-1 TRT 渲染在同卡上初始化互斥（pipeline 必炸
    illegal memory access），当前配置用 CPU 版模型，本补丁不触发。保留备用——
    换双卡或 TRT 冲突解决后可重新启用。"""
    import logging

    import onnxruntime as ort
    from speech_to_speech.VAD import smart_turn as st_mod

    if "CUDAExecutionProvider" not in ort.get_available_providers():
        return  # 环境没装 onnxruntime-gpu，不动

    logger = logging.getLogger(__name__)
    orig_init = st_mod.SmartTurnAnalyzer.__init__

    def _init_gpu(self, **kw):
        orig_init(self, **kw)
        mp = str(kw.get("model_path") or "")
        if mp.endswith("-gpu.onnx"):
            self.session = ort.InferenceSession(
                mp, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            logger.info("SmartTurn 走 GPU: %s", mp)

    st_mod.SmartTurnAnalyzer.__init__ = _init_gpu


def _patch_speechable_keep_cjk_punct() -> None:
    """两件套:
    1) 上游 remove_unspeechable 的白名单只含 ASCII 标点,中文标点(。、?!)被剥;
       CJK 标点加回白名单(显示有标点,TTS 也能按标点停顿)。
    2) LLM 偶尔输出（笑）（拍大腿）等括号动作(人设禁止但没强制力),
       在 remove_unspeechable 前整段删除——转写和 TTS 都不再出现。
       注意 handler 是 from-import 绑定,必须 patch 其模块命名空间里的引用。"""
    import re

    from speech_to_speech.LLM import utils as llm_utils

    llm_utils.SPEECHABLE_PATTERN = re.compile(
        r"[^\w\s.,!?;:'\"\-()\/\\@#%&*+=$€£¥₹₽¢\[\]{}<>~`^|…—–\n\r\t"
        r"，。、；：？！“”‘’《》【】（）「」·〜～]",
        flags=re.UNICODE,
    )

    orig_remove = llm_utils.remove_unspeechable
    paren_action = re.compile(r"[（(][^（）()]{1,20}[)）]")

    def remove_unspeechable_no_actions(text: str) -> str:
        return orig_remove(paren_action.sub(" ", text))

    for mod_name in (
        "speech_to_speech.LLM.base_openai_compatible_language_model",
        "speech_to_speech.LLM.language_model",
    ):
        import importlib

        mod = importlib.import_module(mod_name)
        if getattr(mod, "remove_unspeechable", None) is orig_remove:
            mod.remove_unspeechable = remove_unspeechable_no_actions


def main() -> None:
    parser = argparse.ArgumentParser(description="VoxEMW 数字人语音管线启动器")
    parser.add_argument(
        "--config",
        default=os.environ.get("VOXEMW_CONFIG", DEFAULT_CONFIG),
        help="YAML 配置路径（默认 %(default)s，可用 VOXEMW_CONFIG 覆盖）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印渲染出的 s2s argv，不启动管线（本机无 GPU/无依赖也可跑）",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    if not config_path.is_file():
        sys.exit(f"ERROR: 配置不存在: {config_path}")

    load_dotenv(REPO_ROOT / ".env.local")
    config = load_config(config_path)
    argv = render_s2s_argv(config, env=os.environ)

    if args.dry_run:
        pairs = [f"{argv[i]} {argv[i + 1]}" for i in range(0, len(argv) - 1, 2)]
        print("speech-to-speech \\")
        print("  " + " \\\n  ".join(pairs))
        print(f"\n# stt={config['stt']['backend']} / tts={config['tts']['backend']}（运行时注册，不在 CLI 里）")
        return

    # ── 以下需要 speech_to_speech 依赖与 GPU，仅在服务器上执行 ──
    # 注意顺序：ModuleArguments 的 stt/tts choices 在 import 时固化，
    # 必须先注册自定义积木再 import s2s_pipeline，否则 --stt sensevoice 报 invalid choice
    from voxemw.pipeline.backends import register_custom_backends

    register_custom_backends(config)

    import speech_to_speech.s2s_pipeline as s2s

    _patch_torch_flex_attention_compat()
    _patch_torch_hub_offline_fallback()
    _patch_speechable_keep_cjk_punct()
    _patch_smart_turn_gpu()

    # 新上游标准 serve 流程（s2s_pipeline.run_pipeline_command 复刻）
    parsed = s2s.parse_arguments(argv, command="serve")
    s2s.setup_logger(parsed.module_kwargs.log_level)
    s2s.prepare_all_args(parsed)

    from threading import Event

    stop_event = Event()
    pipeline_manager = s2s.build_pipeline(parsed, stop_event)

    import signal

    def _shutdown(_sig, _frame):
        pipeline_manager.stop()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)
    pipeline_manager.start()
    pipeline_manager.wait()


if __name__ == "__main__":
    main()
