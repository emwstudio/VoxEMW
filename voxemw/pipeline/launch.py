"""VoxEMW 语音管线启动器：YAML 积木配置 → speech-to-speech realtime 服务。

用法：
    python -m voxemw.pipeline.launch [--config configs/assistant.yaml] [--dry-run]

- 不 vendor、不打源码补丁：pip 包 speech-to-speech 作依赖，启动时在进程内
  monkeypatch 两个工厂函数（get_stt_handler/get_tts_handler）注册自定义积木
  （qwen3asr STT、voxcpm TTS），再复刻上游 main() 的启动流程。
- stt/tts 不在上游 CLI 的 Literal 选项里：先让 parse_arguments() 用默认值通过
  校验，之后直接改写 module_kwargs.stt/tts（Literal 只是类型注解，运行期不查）。
- DeepSeek 关 thinking：上游 disable_thinking 发的是 vLLM 风格
  chat_template_kwargs（DeepSeek 忽略），这里把 _build_extra_body 换成
  DeepSeek 认的 thinking.type=disabled。
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
from voxemw.pipeline.args import render_s2s_argv, stt_setup_kwargs, tts_setup_kwargs  # noqa: E402

DEFAULT_CONFIG = "configs/assistant.yaml"


def _install_custom_handlers(config: dict) -> None:
    """monkeypatch s2s_pipeline 的 STT/TTS 工厂，注册自定义积木。"""
    import speech_to_speech.s2s_pipeline as s2s

    orig_get_stt = s2s.get_stt_handler
    orig_get_tts = s2s.get_tts_handler

    def get_stt_handler(module_kwargs, stop_event, queue_in, queue_out, speculative_turns, *kwargs):
        if module_kwargs.stt != "qwen3asr":
            return orig_get_stt(module_kwargs, stop_event, queue_in, queue_out, speculative_turns, *kwargs)
        from voxemw.pipeline.stt_qwen3asr import Qwen3ASRSTTHandler

        handler = Qwen3ASRSTTHandler(
            stop_event,
            queue_in=queue_in,
            queue_out=queue_out,
            setup_kwargs=stt_setup_kwargs(config),
        )
        if speculative_turns is not None:
            handler.speculative_turns = speculative_turns
        return handler

    def get_tts_handler(module_kwargs, stop_event, queue_in, queue_out, should_listen, *kwargs):
        if module_kwargs.tts != "voxcpm":
            return orig_get_tts(module_kwargs, stop_event, queue_in, queue_out, should_listen, *kwargs)
        from voxemw.pipeline.tts_voxcpm import VoxCPMTTSHandler

        setup = tts_setup_kwargs(config)
        # realtime 模式在 _build_realtime_pipeline_unit 里把 cancel_scope /
        # speculative_turns 注入上游各 TTS kwargs dataclass，这里顺手取出来
        for kw in kwargs:
            injected = vars(kw)
            for key in ("cancel_scope", "speculative_turns"):
                if key in injected:
                    setup.setdefault(key, injected[key])
        return VoxCPMTTSHandler(
            stop_event,
            queue_in=queue_in,
            queue_out=queue_out,
            setup_args=(should_listen,),
            setup_kwargs=setup,
        )

    s2s.get_stt_handler = get_stt_handler
    s2s.get_tts_handler = get_tts_handler


def _patch_deepseek_thinking() -> None:
    """DeepSeek 的关 thinking 协议是 extra_body {"thinking": {"type": "disabled"}}；
    上游只懂 vLLM 风格 chat_template_kwargs / reasoning_effort。只在 base_url
    指向 DeepSeek 时替换。"""
    from speech_to_speech.LLM.base_openai_compatible_language_model import (
        BaseOpenAICompatibleHandler,
    )

    orig = BaseOpenAICompatibleHandler._build_extra_body

    @classmethod
    def _build_extra_body(cls, base_url, disable_thinking, reasoning_effort):
        if base_url and "deepseek" in base_url and disable_thinking and not reasoning_effort:
            return {"thinking": {"type": "disabled"}}
        return orig(base_url, disable_thinking, reasoning_effort)

    BaseOpenAICompatibleHandler._build_extra_body = _build_extra_body


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
    import speech_to_speech.s2s_pipeline as s2s

    sys.argv = ["speech-to-speech", *argv]
    parsed = s2s.parse_arguments()

    # 自定义积木：parse 用默认值过 Literal 校验，这里改写成我们的 backend 名
    parsed.module_kwargs.stt = config["stt"]["backend"]
    parsed.module_kwargs.tts = config["tts"]["backend"]

    _install_custom_handlers(config)
    _patch_deepseek_thinking()
    _patch_torch_hub_offline_fallback()
    _patch_speechable_keep_cjk_punct()

    # 复刻上游 main() 的启动流程（s2s_pipeline.py:984）
    s2s.setup_logger(parsed.module_kwargs.log_level)
    s2s.prepare_all_args(
        parsed.module_kwargs,
        parsed.whisper_stt_handler_kwargs,
        parsed.paraformer_stt_handler_kwargs,
        parsed.faster_whisper_stt_handler_kwargs,
        parsed.mlx_audio_whisper_stt_handler_kwargs,
        parsed.parakeet_tdt_stt_handler_kwargs,
        parsed.language_model_handler_kwargs,
        parsed.responses_api_language_model_handler_kwargs,
        parsed.chat_tts_handler_kwargs,
        parsed.facebook_mms_tts_handler_kwargs,
        parsed.pocket_tts_handler_kwargs,
        parsed.kokoro_tts_handler_kwargs,
        parsed.qwen3_tts_handler_kwargs,
    )
    queues_and_events = s2s.initialize_queues_and_events()
    pipeline_manager = s2s.build_pipeline(
        parsed.module_kwargs,
        parsed.socket_receiver_kwargs,
        parsed.socket_sender_kwargs,
        parsed.websocket_streamer_kwargs,
        parsed.vad_handler_kwargs,
        parsed.whisper_stt_handler_kwargs,
        parsed.paraformer_stt_handler_kwargs,
        parsed.faster_whisper_stt_handler_kwargs,
        parsed.mlx_audio_whisper_stt_handler_kwargs,
        parsed.parakeet_tdt_stt_handler_kwargs,
        parsed.language_model_handler_kwargs,
        parsed.responses_api_language_model_handler_kwargs,
        parsed.chat_tts_handler_kwargs,
        parsed.facebook_mms_tts_handler_kwargs,
        parsed.pocket_tts_handler_kwargs,
        parsed.kokoro_tts_handler_kwargs,
        parsed.qwen3_tts_handler_kwargs,
        queues_and_events,
    )

    import signal

    def _shutdown(_sig, _frame):
        pipeline_manager.stop()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)
    pipeline_manager.start()
    pipeline_manager.wait()


if __name__ == "__main__":
    main()
