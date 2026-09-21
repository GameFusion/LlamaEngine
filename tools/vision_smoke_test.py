#!/usr/bin/env python3
import ctypes
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROMPT_PATH = ROOT / "docs/smoke-tests/echollama-vision-reference-prompt.txt"
OUTPUT_PATH = ROOT / "docs/smoke-tests/echollama-vision-reference-output.txt"
IMAGE_PATH = Path("/Users/andreascarlen/Downloads/WhatsApp Image 2026-07-08 at 23.11.14.jpeg")
MODEL_PATH = Path.home() / ".cache/EchoLlama/models/gemma-3-12b-it-q4_0.gguf"
CLIP_PATH = Path.home() / ".cache/EchoLlama/models/mmproj-google_gemma-3-12b-it-f16.gguf"
ENGINE_PATH = ROOT / "build/Qt_6_10_2_for_macOS-Debug/bin/Metal/libLlamaEngine.1.0.0.dylib"


PARAM_FLOAT = 0
PARAM_INT = 1


class ModelParameter(ctypes.Structure):
    _fields_ = [
        ("key", ctypes.c_char_p),
        ("type", ctypes.c_int),
        ("value", ctypes.c_void_p),
    ]


LoadLogCallback = ctypes.CFUNCTYPE(None, ctypes.c_char_p)
RuntimeCallback = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_void_p)


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(str(path))


def load_engine() -> ctypes.CDLL:
    for dep in (
        "/opt/local/lib/libggml-base.dylib",
        "/opt/local/lib/libggml-cpu.dylib",
        "/opt/local/lib/libggml-metal.dylib",
        "/opt/local/lib/libggml.dylib",
        "/opt/local/lib/libllama.dylib",
    ):
        if Path(dep).exists():
            ctypes.CDLL(dep, mode=ctypes.RTLD_GLOBAL)
    return ctypes.CDLL(str(ENGINE_PATH), mode=ctypes.RTLD_GLOBAL)


def main() -> int:
    for path in (PROMPT_PATH, IMAGE_PATH, MODEL_PATH, CLIP_PATH, ENGINE_PATH):
        require_file(path)

    prompt = PROMPT_PATH.read_text(encoding="utf-8").strip()
    engine = load_engine()

    engine.loadModel.argtypes = [
        ctypes.c_char_p,
        ctypes.POINTER(ModelParameter),
        ctypes.c_size_t,
        LoadLogCallback,
    ]
    engine.loadModel.restype = ctypes.c_bool
    engine.loadClipModel.argtypes = [ctypes.c_char_p, RuntimeCallback, ctypes.c_void_p]
    engine.loadClipModel.restype = ctypes.c_bool
    engine.generateResponseWithImageFile.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_char_p,
        RuntimeCallback,
        RuntimeCallback,
        ctypes.c_void_p,
    ]
    engine.generateResponseWithImageFile.restype = ctypes.c_bool

    temperature = ctypes.c_float(0.2)
    context_size = ctypes.c_int(8192)
    top_k = ctypes.c_float(40.0)
    top_p = ctypes.c_float(0.9)
    repetition_penalty = ctypes.c_float(1.1)

    params = (ModelParameter * 5)(
        ModelParameter(b"temperature", PARAM_FLOAT, ctypes.cast(ctypes.pointer(temperature), ctypes.c_void_p)),
        ModelParameter(b"context_size", PARAM_INT, ctypes.cast(ctypes.pointer(context_size), ctypes.c_void_p)),
        ModelParameter(b"top_k", PARAM_FLOAT, ctypes.cast(ctypes.pointer(top_k), ctypes.c_void_p)),
        ModelParameter(b"top_P", PARAM_FLOAT, ctypes.cast(ctypes.pointer(top_p), ctypes.c_void_p)),
        ModelParameter(b"repetition_penalty", PARAM_FLOAT, ctypes.cast(ctypes.pointer(repetition_penalty), ctypes.c_void_p)),
    )

    log_lines = []
    chunks = []
    final_messages = []

    def on_load_log(message: bytes) -> None:
        text = message.decode("utf-8", errors="replace") if message else ""
        log_lines.append(text.rstrip())
        print(text, flush=True)

    def on_runtime(message: bytes, _user_data: int) -> None:
        text = message.decode("utf-8", errors="replace") if message else ""
        chunks.append(text)
        print(text, end="", flush=True)

    def on_clip_log(message: bytes, _user_data: int) -> None:
        text = message.decode("utf-8", errors="replace") if message else ""
        log_lines.append(text.rstrip())
        print(text, flush=True)

    def on_final(message: bytes, _user_data: int) -> None:
        text = message.decode("utf-8", errors="replace") if message else ""
        final_messages.append(text)

    load_cb = LoadLogCallback(on_load_log)
    runtime_cb = RuntimeCallback(on_runtime)
    clip_log_cb = RuntimeCallback(on_clip_log)
    final_cb = RuntimeCallback(on_final)

    if not engine.loadModel(str(MODEL_PATH).encode(), params, len(params), load_cb):
        raise RuntimeError("loadModel failed")
    if not engine.loadClipModel(str(CLIP_PATH).encode(), clip_log_cb, None):
        raise RuntimeError("loadClipModel failed")
    if not engine.generateResponseWithImageFile(
        0,
        prompt.encode(),
        str(IMAGE_PATH).encode(),
        runtime_cb,
        final_cb,
        None,
    ):
        raise RuntimeError("generateResponseWithImageFile failed")

    response = "".join(chunks).strip()
    OUTPUT_PATH.write_text(
        "Prompt:\n"
        f"{prompt}\n\n"
        "Image:\n"
        f"{IMAGE_PATH}\n\n"
        "Model:\n"
        f"{MODEL_PATH}\n\n"
        "Output:\n"
        f"{response}\n",
        encoding="utf-8",
    )
    print(f"\n\nSaved smoke output to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"smoke test failed: {exc}", file=sys.stderr)
        raise
