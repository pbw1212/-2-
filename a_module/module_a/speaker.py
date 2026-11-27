from __future__ import annotations

import hashlib
import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .asr import MODULE_DIR
from .audio import DEFAULT_VAD_DBFS, SAMPLE_RATE, activity, duration_ms, frames, speaker_fbank

DEFAULT_MODEL = MODULE_DIR / "models" / "3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx"
DEFAULT_PROFILES = MODULE_DIR / "profiles.sqlite3"


def unit_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("Invalid speaker embedding")
    return (vector / norm).astype("<f4")


class SpeakerMatcher:
    """Match a local demo profile with a 3D-Speaker CAM++ embedding."""

    def __init__(
        self,
        model_path: Path,
        profiles: Path,
        *,
        threshold: float = 0.55,
        min_voiced_ms: float = 1200,
        vad_dbfs: float = DEFAULT_VAD_DBFS,
    ):
        if not -1 <= threshold <= 1 or not 0 < min_voiced_ms:
            raise ValueError("Invalid speaker threshold or minimum voiced duration")
        digest = hashlib.sha256()
        with model_path.open("rb") as model_file:
            for block in iter(lambda: model_file.read(1024 * 1024), b""):
                digest.update(block)
        self.model_id = digest.hexdigest()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(model_path), options, providers=["CPUExecutionProvider"]
        )
        metadata = self.session.get_modelmeta().custom_metadata_map
        if (
            metadata.get("framework") != "3d-speaker"
            or metadata.get("sample_rate") != str(SAMPLE_RATE)
            or metadata.get("feature_normalize_type") != "global-mean"
            or metadata.get("normalize_samples") != "1"
        ):
            raise ValueError("Expected the 16 kHz 3D-Speaker CAM++ ONNX export")
        self.embedding_size = int(metadata["output_dim"])
        self.input_name = self.session.get_inputs()[0].name
        self.profiles = profiles
        self.threshold = threshold
        self.min_voiced_ms = min_voiced_ms
        self.vad_dbfs = vad_dbfs

    def _embedding(self, pcm: bytes) -> tuple[np.ndarray | None, float]:
        audio_frames = list(frames(pcm))
        voiced = [
            index for index, frame in enumerate(audio_frames)
            if activity(frame, self.vad_dbfs)[1]
        ]
        voiced_ms = sum(duration_ms(audio_frames[index]) for index in voiced)
        if voiced_ms < self.min_voiced_ms:
            return None, voiced_ms
        trimmed = b"".join(audio_frames[voiced[0] : voiced[-1] + 1])
        features = speaker_fbank(trimmed).numpy()
        features -= features.mean(axis=0)
        (vector,) = self.session.run(None, {self.input_name: features[None, :, :]})
        return unit_vector(vector.reshape(self.embedding_size)), voiced_ms

    def _database(self):
        self.profiles.parent.mkdir(parents=True, exist_ok=True)
        database = sqlite3.connect(self.profiles)
        try:
            database.execute(
                """CREATE TABLE IF NOT EXISTS templates (
                       model_id TEXT NOT NULL,
                       profile_id TEXT NOT NULL,
                       vector BLOB NOT NULL,
                       PRIMARY KEY (model_id, profile_id)
                   )"""
            )
        except sqlite3.Error:
            database.close()
            raise
        return database

    def enroll(self, profile_id: str, recordings: list[bytes]) -> dict:
        if not profile_id.strip() or not recordings:
            raise ValueError("A profile ID and at least one recording are required")
        vectors = []
        for pcm in recordings:
            vector, voiced_ms = self._embedding(pcm)
            if vector is None:
                raise ValueError(f"Enrollment audio has only {voiced_ms:.0f} ms above the activity threshold")
            vectors.append(vector)
        template = unit_vector(np.mean(vectors, axis=0))
        with closing(self._database()) as database, database:
            database.execute(
                """INSERT INTO templates (model_id, profile_id, vector)
                   VALUES (?, ?, ?)
                   ON CONFLICT (model_id, profile_id) DO UPDATE SET vector=excluded.vector""",
                (self.model_id, profile_id, template.tobytes()),
            )
        return {"status": "enrolled", "profile_id": profile_id, "recordings": len(vectors)}

    def match(self, pcm: bytes) -> dict:
        vector, voiced_ms = self._embedding(pcm)
        if vector is None:
            return {"status": "too_short", "profile_id": None, "voiced_ms": voiced_ms}
        with closing(self._database()) as database:
            rows = database.execute(
                "SELECT profile_id, vector FROM templates WHERE model_id = ? ORDER BY profile_id",
                (self.model_id,),
            ).fetchall()
        if not rows:
            return {"status": "no_templates", "profile_id": None, "voiced_ms": voiced_ms}
        scores = [
            (float(np.clip(np.dot(vector, np.frombuffer(blob, dtype="<f4")), -1, 1)), profile_id)
            for profile_id, blob in rows
        ]
        score, profile_id = max(scores, key=lambda item: item[0])
        return {
            "status": "candidate" if score >= self.threshold else "unknown",
            "profile_id": profile_id if score >= self.threshold else None,
            "score": score,
            "threshold": self.threshold,
            "voiced_ms": voiced_ms,
        }
