"""
Simple Voice Conversion app.

One job, one screen:

    1. Upload SOURCE audio (the speech you want to keep).
    2. Upload REFERENCE voice sample (the voice you want it to sound like).
    3. Click Convert.
    4. Get audio that says the same thing in the reference speaker's voice.

Engines (best free open-source, no online TTS, no translation):
    - Chatterbox VC  (default — fastest, ~6 GB VRAM, no internet at runtime)
    - OpenVoice v2   (tone-color transfer, robust on long files)
    - Seed-VC        (strongest identity transfer — used if installed)

Run:
    venv/bin/python3 simple_vc.py --port 7870
"""

from __future__ import annotations

import argparse
import gc
import sys
import tempfile
import time
import traceback
from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))


def _free_vram() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _resolve_audio(component_value) -> str | None:
    """Gradio Audio(type='filepath') returns a path string; tolerate other shapes too."""
    if component_value is None:
        return None
    if isinstance(component_value, str):
        return component_value
    if isinstance(component_value, dict):
        return component_value.get("path") or component_value.get("name")
    if isinstance(component_value, (tuple, list)) and len(component_value) >= 2:
        # (sr, np.ndarray) — write to temp wav
        sr, arr = component_value[0], component_value[1]
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, arr, int(sr))
            return tmp.name
    return None


def _engine_status() -> dict[str, str]:
    """Return short availability blurb per engine."""
    out: dict[str, str] = {}
    try:
        import chatterbox_engine

        out["Chatterbox VC"] = "ready" if chatterbox_engine.is_vc_available() else "missing (pip install chatterbox-tts)"
    except Exception as e:
        out["Chatterbox VC"] = f"missing ({e.__class__.__name__})"
    try:
        import openvoice_engine

        out["OpenVoice"] = "ready" if openvoice_engine.is_available() else "missing (pip install openvoice-cli)"
    except Exception as e:
        out["OpenVoice"] = f"missing ({e.__class__.__name__})"
    try:
        import seed_vc_engine

        out["Seed-VC"] = "ready" if seed_vc_engine.is_available() else "missing (pip install seed-vc)"
    except Exception as e:
        out["Seed-VC"] = f"missing ({e.__class__.__name__})"
    return out


def _enabled_engines() -> list[str]:
    return [name for name, status in _engine_status().items() if status == "ready"]


def _unload_other_engines(keep: str) -> None:
    """Free VRAM held by engines we are not about to use."""
    mapping = {
        "Chatterbox VC": "chatterbox_engine",
        "OpenVoice": "openvoice_engine",
        "Seed-VC": "seed_vc_engine",
    }
    keep_mod = mapping.get(keep, "")
    for label, mod_name in mapping.items():
        if mod_name == keep_mod:
            continue
        mod = sys.modules.get(mod_name)
        if mod is None:
            continue
        for fn in ("unload_vc", "unload"):
            f = getattr(mod, fn, None)
            if callable(f):
                try:
                    f()
                except Exception as e:
                    print(f"[{label}] unload {fn} failed: {e}", flush=True)
    _free_vram()


def _convert_chatterbox(
    source: str,
    reference: str,
    progress_cb,
    chunk_seconds: float = 6.0,
    overlap_seconds: float = 0.4,
    single_shot_threshold_seconds: float = 8.0,
) -> tuple[np.ndarray, int]:
    """
    Chatterbox VC's s3gen attention is O(seq²) and tries to allocate the full
    matrix in one shot. On a 12 GB GPU that OOMs at ~10 s of source audio.
    Workaround: split the source into ~chunk_seconds segments with a small
    crossfade overlap, run the model per-segment, free VRAM in between, and
    crossfade-stitch the outputs in the model's native sample rate.
    """
    import chatterbox_engine

    src_arr, src_sr = sf.read(source, dtype="float32", always_2d=False)
    if src_arr.ndim == 2:
        src_arr = src_arr.mean(axis=1)
    duration = len(src_arr) / float(src_sr)

    if duration <= single_shot_threshold_seconds:
        progress_cb(f"Chatterbox VC single-shot ({duration:.1f}s)...", 0.4)
        _, arr, sr = chatterbox_engine.convert_voice(input_path=source, ref_path=reference)
        return arr, int(sr)

    chunk_n = max(1, int(chunk_seconds * src_sr))
    overlap_n_in = max(0, int(overlap_seconds * src_sr))
    step_n = max(1, chunk_n - overlap_n_in)
    n_chunks = max(1, (len(src_arr) - overlap_n_in + step_n - 1) // step_n)

    progress_cb(
        f"Chatterbox VC: source is {duration:.1f}s — chunking into ~{chunk_seconds:.1f}s "
        f"pieces ({n_chunks} chunks, {overlap_seconds:.2f}s overlap)...",
        0.2,
    )

    pieces: list[np.ndarray] = []
    out_sr: int | None = None
    pos = 0
    idx = 0
    while pos < len(src_arr):
        seg = src_arr[pos:pos + chunk_n]
        if len(seg) < int(0.5 * src_sr):
            break
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, seg, src_sr)
            seg_path = tmp.name
        try:
            progress_cb(
                f"Chatterbox VC chunk {idx + 1}/{n_chunks} ({len(seg) / src_sr:.1f}s)...",
                0.2 + 0.7 * (idx / max(1, n_chunks)),
            )
            _, conv, sr = chatterbox_engine.convert_voice(
                input_path=seg_path, ref_path=reference
            )
            out_sr = int(sr)
            pieces.append(np.asarray(conv, dtype=np.float32))
        finally:
            Path(seg_path).unlink(missing_ok=True)
            _free_vram()
        pos += step_n
        idx += 1

    if not pieces or out_sr is None:
        raise RuntimeError("Chatterbox VC produced no audio (chunk loop yielded 0 pieces).")

    overlap_n_out = max(0, int(overlap_seconds * out_sr))
    result = pieces[0]
    for nxt in pieces[1:]:
        if overlap_n_out > 0 and len(result) >= overlap_n_out and len(nxt) >= overlap_n_out:
            tail = result[-overlap_n_out:]
            head = nxt[:overlap_n_out]
            fade = np.linspace(0.0, 1.0, overlap_n_out, dtype=np.float32)
            mixed = tail * (1.0 - fade) + head * fade
            result = np.concatenate([result[:-overlap_n_out], mixed, nxt[overlap_n_out:]])
        else:
            result = np.concatenate([result, nxt])

    return result.astype(np.float32), int(out_sr)


def _convert_openvoice(source: str, reference: str, progress_cb) -> tuple[np.ndarray, int]:
    import openvoice_engine

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        out_path = tmp.name
    openvoice_engine.convert(
        input_path=source,
        ref_path=reference,
        output_path=out_path,
        progress_cb=lambda msg, p=None: progress_cb(msg),
        use_cpu_fallback_on_oom=True,
    )
    arr, sr = sf.read(out_path, dtype="float32", always_2d=False)
    Path(out_path).unlink(missing_ok=True)
    if arr.ndim == 2:
        arr = arr.mean(axis=1)
    return arr.astype(np.float32), int(sr)


def _convert_seedvc(source: str, reference: str) -> tuple[np.ndarray, int]:
    import seed_vc_engine

    _, arr, sr = seed_vc_engine.convert_voice(input_path=source, ref_path=reference)
    if arr.ndim == 2:
        arr = arr.mean(axis=1)
    return arr.astype(np.float32), int(sr)


def run_conversion(source_audio, reference_audio, engine: str, progress=gr.Progress()):
    src = _resolve_audio(source_audio)
    ref = _resolve_audio(reference_audio)
    if not src or not Path(src).exists():
        raise gr.Error("Upload the SOURCE audio (the speech to convert).")
    if not ref or not Path(ref).exists():
        raise gr.Error("Upload the REFERENCE voice sample (10–30 s of clean speech of the target speaker).")
    if engine not in _enabled_engines():
        raise gr.Error(f"Engine '{engine}' is not available. Status: {_engine_status()}")

    started = time.time()

    def step(msg: str, p: float | None = None):
        elapsed = time.time() - started
        line = f"[{elapsed:5.1f}s] {msg}"
        print(line, flush=True)
        progress(p if p is not None else None, desc=msg)

    try:
        step(f"Freeing VRAM from other engines (keeping {engine})...", 0.05)
        _unload_other_engines(keep=engine)

        step(f"Loading {engine} model (first run will download weights)...", 0.15)

        if engine == "Chatterbox VC":
            arr, sr = _convert_chatterbox(src, ref, step)
        elif engine == "OpenVoice":
            arr, sr = _convert_openvoice(src, ref, step)
        elif engine == "Seed-VC":
            arr, sr = _convert_seedvc(src, ref)
        else:
            raise gr.Error(f"Unknown engine: {engine}")

        step(f"Done in {time.time() - started:.1f}s. Output: {arr.shape[0] / sr:.1f}s @ {sr} Hz.", 1.0)
        return (sr, arr)
    except gr.Error:
        raise
    except RuntimeError as e:
        msg = str(e)
        if "out of memory" in msg.lower() or "CUDA" in msg:
            traceback.print_exc()
            raise gr.Error(
                f"{engine} ran out of GPU memory or hit a CUDA error.\n"
                "Try: close other GPU apps (`nvidia-smi`), pick OpenVoice "
                "(works on CPU as fallback), or restart the app.\n\n"
                f"Underlying error: {e}"
            ) from e
        traceback.print_exc()
        raise gr.Error(f"{engine} failed: {e}") from e
    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f"{engine} failed: {e.__class__.__name__}: {e}") from e


def build_ui():
    statuses = _engine_status()
    enabled = _enabled_engines()
    preferred_order = ["OpenVoice", "Seed-VC", "Chatterbox VC"]
    default_engine = next(
        (name for name in preferred_order if name in enabled),
        enabled[0] if enabled else "OpenVoice",
    )

    with gr.Blocks(title="Simple Voice Conversion") as demo:
        gr.Markdown(
            "# Simple Voice Conversion\n\n"
            "Goal: take **source audio** and make it sound as if the speaker from the "
            "**reference voice sample** was saying it. No translation, no TTS, no editing — "
            "just speaker identity transfer.\n\n"
            "### Which engine for which language?\n"
            "- **Polish, German, French, and other non-English/non-Chinese sources → use "
            "OpenVoice** (default). It is a *tone-color converter*: keeps your source "
            "audio's phonemes intact and only swaps the timbre. Language stays exactly "
            "as in the source.\n"
            "- **English or Chinese source → Seed-VC** gives the strongest identity "
            "transfer. On other languages it tends to inject English-like phonemes "
            "because its content encoder is biased toward EN/ZH training data.\n"
            "- **Chatterbox VC**: same caveat as Seed-VC (EN-biased), additionally "
            "auto-chunked to fit 12 GB GPU."
        )

        gr.Markdown(
            "**Engine status:** "
            + ", ".join(f"{name} — *{status}*" for name, status in statuses.items())
        )

        with gr.Row():
            with gr.Column():
                src = gr.Audio(label="1. SOURCE audio (speech to convert)", type="filepath")
                ref = gr.Audio(
                    label="2. REFERENCE voice sample (10–30 s of the target voice, clean speech)",
                    type="filepath",
                )
                engine = gr.Radio(
                    choices=list(statuses.keys()),
                    value=default_engine,
                    label="Engine",
                    info=(
                        "OpenVoice (default): tone-color transfer, KEEPS source language "
                        "(Polish stays Polish). Use this for non-EN/ZH sources. "
                        "Seed-VC: strongest identity transfer for EN/ZH; on Polish it "
                        "injects English phonemes (model bias). "
                        "Chatterbox VC: same EN bias as Seed-VC, plus auto-chunked."
                    ),
                )
                btn = gr.Button("Convert", variant="primary")
            with gr.Column():
                out = gr.Audio(label="Converted audio", type="numpy")
                gr.Markdown(
                    "### Tips\n"
                    "- Reference clip: **clean**, single speaker, no music, **10–30 s**.\n"
                    "- **OpenVoice is the default** because it is the only engine in this "
                    "app that preserves the source language exactly (it does not "
                    "re-synthesize from features — it only swaps the speaker timbre). "
                    "Falls back to CPU on OOM.\n"
                    "- **If output sounds English-accented on a Polish source** → you "
                    "are using Seed-VC or Chatterbox VC. Switch to OpenVoice. Those "
                    "two are trained mostly on EN/ZH and their content encoder leaks "
                    "English phonemes on Polish input. There is no fix on our side — "
                    "it is the model itself.\n"
                    "- Chatterbox VC is auto-chunked into ~6 s windows (otherwise its "
                    "attention OOMs beyond ~10 s on 12 GB GPUs).\n"
                    "- This app is intentionally minimal — no segments, no translation, "
                    "no TTS. If you need those, use the original `app.py`."
                )

        btn.click(fn=run_conversion, inputs=[src, ref, engine], outputs=[out])

    return demo


def main() -> int:
    parser = argparse.ArgumentParser(description="Simple Voice Conversion")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7870)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--cpu", action="store_true", help="Disable CUDA for this process")
    args = parser.parse_args()

    if args.cpu:
        import os

        os.environ.setdefault("VOICE_CHANGER_FORCE_DEVICE", "cpu")
        print("CPU mode forced (VOICE_CHANGER_FORCE_DEVICE=cpu).")

    demo = build_ui()
    demo.queue().launch(server_name=args.host, server_port=args.port, share=args.share, inbrowser=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
