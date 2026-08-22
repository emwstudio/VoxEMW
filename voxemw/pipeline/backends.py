"""自定义积木注册：sensevoice STT / voxcpm TTS → 上游 backend_registry。

上游 2026-08 重构（main @5a0c79f）：get_stt_handler/get_tts_handler 工厂函数废弃，
改为 BackendSpec 注册表 + HandlerContext 工厂。往 STT_BACKENDS/TTS_BACKENDS
插入条目后，--stt sensevoice / --tts voxcpm 成为一等公民（argparse choices 由
注册表动态生成）。我们的 handler 类零改动：工厂把 HandlerContext 翻译成
它们的老构造参数（stop_event + queue_in/out + setup_args/kwargs）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SenseVoiceArgs:
    """STT 积木的 CLI 参数空壳（每个注册条目必须独立的参数类，
    否则 HfArgumentParser 会因重复选项串冲突）。参数走 VoxEMW YAML。"""

    sensevoice_placeholder: str = field(default="", metadata={"help": "内部占位，勿传"})


@dataclass
class Qwen3ASRArgs:
    """Qwen3-ASR STT 积木的 CLI 参数空壳。"""

    qwen3asr_placeholder: str = field(default="", metadata={"help": "内部占位，勿传"})


@dataclass
class VoxCPMArgs:
    """TTS 积木的 CLI 参数空壳。"""

    voxcpm_placeholder: str = field(default="", metadata={"help": "内部占位，勿传"})


def register_custom_backends(config: dict) -> None:
    """把 sensevoice/voxcpm 注册进上游后端注册表（幂等）。"""
    from speech_to_speech.backend_registry import (
        STT_BACKENDS,
        TTS_BACKENDS,
        BackendSpec,
    )

    from voxemw.pipeline.args import stt_setup_kwargs, tts_setup_kwargs

    def _create_stt(ctx, _cfg):
        from voxemw.pipeline.stt_sensevoice import SenseVoiceSTTHandler

        handler = SenseVoiceSTTHandler(
            ctx.stop_event,
            queue_in=ctx.queue_in,
            queue_out=ctx.queue_out,
            setup_kwargs=stt_setup_kwargs(config),
        )
        # 老工厂在构造后补挂；新世界直接从 HandlerContext 拿
        handler.speculative_turns = ctx.speculative_turns
        return handler

    def _create_stt_qwen3asr(ctx, _cfg):
        from voxemw.pipeline.stt_qwen3asr import Qwen3ASRSTTHandler

        handler = Qwen3ASRSTTHandler(
            ctx.stop_event,
            queue_in=ctx.queue_in,
            queue_out=ctx.queue_out,
            setup_kwargs=stt_setup_kwargs(config),
        )
        handler.speculative_turns = ctx.speculative_turns
        return handler

    def _create_tts(ctx, _cfg):
        from voxemw.pipeline.tts_voxcpm import VoxCPMTTSHandler

        setup = tts_setup_kwargs(config)
        # 老世界靠扒上游 kwargs dataclass 注入；新世界 HandlerContext 自带
        setup["cancel_scope"] = ctx.cancel_scope
        setup["speculative_turns"] = ctx.speculative_turns
        return VoxCPMTTSHandler(
            ctx.stop_event,
            queue_in=ctx.queue_in,
            queue_out=ctx.queue_out,
            setup_args=(ctx.should_listen,),
            setup_kwargs=setup,
        )

    STT_BACKENDS.setdefault("sensevoice", BackendSpec(
        name="sensevoice", kind="stt",
        config_type=SenseVoiceArgs, create_handler=_create_stt,
    ))
    STT_BACKENDS.setdefault("qwen3asr", BackendSpec(
        name="qwen3asr", kind="stt",
        config_type=Qwen3ASRArgs, create_handler=_create_stt_qwen3asr,
    ))
    TTS_BACKENDS.setdefault("voxcpm", BackendSpec(
        name="voxcpm", kind="tts",
        config_type=VoxCPMArgs, create_handler=_create_tts,
    ))
