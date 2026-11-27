"""下载并校验模块 A 使用的 Zipformer 和 CAM++ 权重。"""

import hashlib
import shutil
import tarfile
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen


MODELS = Path(__file__).resolve().parent / "models"
ASR_MODEL = "sherpa-onnx-x-asr-160ms-streaming-zipformer-transducer-zh-en-int8-2026-06-05"
ASR_ARCHIVE = f"{ASR_MODEL}.tar.bz2"
ASR_WEIGHTS = {
    "encoder.int8.onnx": "aad87c04ae3a6332f14560f602b830bafa669e6d8f0db9879debc7c5071765c8",
    "decoder.onnx": "fd2ac10c5936a1411f7a68e669e8cfec882b17791dd2ca44ab9baf69bb8ab855",
    "joiner.int8.onnx": "82e88dad29301e29144565cd3b7347fa163f07ddb30ea60346b7c4c536adf4ba",
}
CAM_MODEL = "3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx"
CAM_SHA256 = "e6af4d4c444f3dab01d991bb9b7d5af98eb979e329f6c8880a19d5553561ed25"
RELEASES = "https://github.com/k2-fsa/sherpa-onnx/releases/download"


def matches(path: Path, expected: str) -> bool:
    if not path.is_file():
        return False
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest() == expected


def download(url: str, destination: Path) -> None:
    print(f"下载 {destination.name} …", flush=True)
    with urlopen(url, timeout=90) as response, destination.open("wb") as file:
        shutil.copyfileobj(response, file)


def main() -> None:
    asr_dir = MODELS / "zipformer"
    asr_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=MODELS) as temporary:
        staging = Path(temporary)
        if not all(matches(asr_dir / name, digest) for name, digest in ASR_WEIGHTS.items()):
            archive_path = staging / ASR_ARCHIVE
            download(f"{RELEASES}/asr-models/{ASR_ARCHIVE}", archive_path)
            with tarfile.open(archive_path, "r:bz2") as archive:
                for name, digest in ASR_WEIGHTS.items():
                    member = archive.getmember(f"{ASR_MODEL}/{name}")
                    if not member.isfile():
                        raise ValueError(f"模型包中的文件格式有误：{name}")
                    with archive.extractfile(member) as source, (staging / name).open("wb") as target:
                        shutil.copyfileobj(source, target)
                    if not matches(staging / name, digest):
                        raise ValueError(f"权重校验失败：{name}")
            for name in ASR_WEIGHTS:
                (staging / name).replace(asr_dir / name)
        if not matches(MODELS / CAM_MODEL, CAM_SHA256):
            staged_model = staging / CAM_MODEL
            download(f"{RELEASES}/speaker-recongition-models/{CAM_MODEL}", staged_model)
            if not matches(staged_model, CAM_SHA256):
                raise ValueError(f"权重校验失败：{CAM_MODEL}")
            staged_model.replace(MODELS / CAM_MODEL)
    print("Zipformer 和 CAM++ 权重已就绪。")


if __name__ == "__main__":
    main()
