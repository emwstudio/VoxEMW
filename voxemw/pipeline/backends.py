"""自定义积木注册：qwen3asr STT → 上游 backend_registry。

上游 2026-08 重构（main @5a0c79f）：get_stt_handler/get_tts_handler 工厂函数废弃，
改为 BackendSpec 注册表 + HandlerContext 工厂。往 STT_BACKENDS 插入条目后，
--stt qwen3asr 成为一等公民（argparse choices 由注册表动态生成）。
handler 类零改动：工厂把 HandlerContext 翻译成它的老构造参数
（stop_event + queue_in/out + setup_args/kwargs）。

TTS 用上上游内置的 qwen3 后端（CLI 直传参数），无需在此注册。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Qwen3ASRArgs:
    """STT 积木的 CLI 参数空壳（注册条目必须独立的参数类，
    否则 HfArgumentParser 会因重复选项串冲突）。参数走 VoxEMW YAML。"""

    qwen3asr_placeholder: str = field(default="", metadata={"help": "内部占位，勿传"})


@dataclass
class VoxCPM2TTSArgs:
    """TTS 积木的 CLI 参数空壳（同理必须独立于 STT 的参数类）。"""

    voxcpm2_placeholder: str = field(default="", metadata={"help": "内部占位，勿传"})


def register_custom_backends(config: dict) -> None:
    """把 qwen3asr（STT）与 voxcpm2（TTS，4090 满血版）注册进上游后端注册表（幂等）。"""
    from speech_to_speech.backend_registry import (
        STT_BACKENDS,
        TTS_BACKENDS,
        BackendSpec,
    )

    from voxemw.pipeline.args import stt_setup_kwargs, tts_setup_kwargs

    def _create_stt_qwen3asr(ctx, _cfg):
        from voxemw.pipeline.stt_qwen3asr import Qwen3ASRSTTHandler

        handler = Qwen3ASRSTTHandler(
            ctx.stop_event,
            queue_in=ctx.queue_in,
            queue_out=ctx.queue_out,
            setup_kwargs=stt_setup_kwargs(config),
        )
        # 老工厂在构造后补挂；新世界直接从 HandlerContext 拿
        handler.speculative_turns = ctx.speculative_turns
        return handler

    STT_BACKENDS.setdefault("qwen3asr", BackendSpec(
        name="qwen3asr", kind="stt",
        config_type=Qwen3ASRArgs, create_handler=_create_stt_qwen3asr,
    ))

    tts_cfg = (config.get("tts") or {})
    if tts_cfg.get("backend") == "voxcpm2":
        def _create_tts_voxcpm2(ctx, _cfg):
            from voxemw.pipeline.tts_voxcpm2 import VoxCPM2TTSHandler

            handler = VoxCPM2TTSHandler(
                ctx.stop_event,
                queue_in=ctx.queue_in,
                queue_out=ctx.queue_out,
                setup_kwargs=tts_setup_kwargs(config),
            )
            handler.cancel_scope = ctx.cancel_scope
            handler.speculative_turns = ctx.speculative_turns
            return handler

        TTS_BACKENDS.setdefault("voxcpm2", BackendSpec(
            name="voxcpm2", kind="tts",
            config_type=VoxCPM2TTSArgs,  # 参数走 YAML，独立空壳类防选项串冲突
            create_handler=_create_tts_voxcpm2,
        ))
