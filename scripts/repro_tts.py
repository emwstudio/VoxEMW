"""临时排障：实例上单独加载 VoxCPM2，完整 traceback 复现 stream_chunks 空异常。

用法：python scripts/repro_tts.py [voice]
"""
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from doudizhu.persona import load_persona


def main():
    voice = sys.argv[1] if len(sys.argv) > 1 else "liangzi"
    root = Path(__file__).resolve().parent.parent
    persona = load_persona(root / "personas" / f"{voice}.md")
    ref_wav = str(root / persona.ref_wav)
    ref_text = (root / persona.ref_text).read_text(encoding="utf-8").strip()

    from doudizhu.voice import TTS

    tts = TTS("openbmb/VoxCPM2", device="cuda", optimize=True)
    tts.add_voice(voice, ref_wav, ref_text)

    t0 = time.time()
    n_chunks = 0
    n_samples = 0

    def run():
        nonlocal n_chunks, n_samples
        for chunk in tts.stream_chunks("老铁们，今天斗地主，味真足！", voice):
            n_chunks += 1
            n_samples += len(chunk)
            if n_chunks == 1:
                print(f"first chunk at {time.time()-t0:.2f}s, len={len(chunk)}", flush=True)

    try:
        if "--thread" in sys.argv:  # 模拟 server.py 的 run_in_executor 工作线程
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(run).result()
        else:
            run()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
    print(f"OK chunks={n_chunks} samples={n_samples} ({n_samples/16000:.2f}s audio) "
          f"in {time.time()-t0:.2f}s")


main()
