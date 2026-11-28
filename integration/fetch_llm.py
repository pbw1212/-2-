"""Download Qwen2.5-0.5B-Instruct GGUF (Q4_K_M) for offline local C/D fallback & NLG."""

from __future__ import annotations

import shutil
from pathlib import Path
from urllib.request import Request, urlopen

MODELS_DIR = Path(__file__).resolve().parent / "models"
DEFAULT_LLM_PATH = MODELS_DIR / "qwen2.5-0.5b-instruct-q4_k_m.gguf"

CANDIDATE_URLS = (
    "https://modelscope.cn/models/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/master/qwen2.5-0.5b-instruct-q4_k_m.gguf",
    "https://hf-mirror.com/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf",
    "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf",
)
MIN_EXPECTED_BYTES = 250 * 1024 * 1024


def fetch_llm(dest: Path = DEFAULT_LLM_PATH) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size >= MIN_EXPECTED_BYTES:
        print(f"离线小模型已就绪：{dest} ({dest.stat().st_size / 1024 / 1024:.1f} MB)")
        return dest

    tmp = dest.with_suffix(".gguf.part")
    errors = []
    for url in CANDIDATE_URLS:
        try:
            print(f"尝试下载离线小模型：{url}", flush=True)
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req, timeout=180) as resp, tmp.open("wb") as out:
                shutil.copyfileobj(resp, out)
            size = tmp.stat().st_size
            if size < MIN_EXPECTED_BYTES:
                raise ValueError(f"Downloaded file too small ({size} bytes)")
            tmp.replace(dest)
            print(f"下载完成：{dest} ({size / 1024 / 1024:.1f} MB)")
            return dest
        except Exception as exc:
            errors.append(f"{url}: {exc}")
            if tmp.exists():
                tmp.unlink()

    raise RuntimeError("未能下载 Qwen2.5-0.5B GGUF 模型:\n" + "\n".join(errors))


if __name__ == "__main__":
    fetch_llm()
