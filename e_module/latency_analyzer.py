"""JSONL latency and timing analyzer for E-module and full-system traces."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np


def _percentile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    return round(float(np.percentile(np.asarray(values, dtype=np.float64), q)), 2)


def analyze_trace_file(trace_path: Path) -> Dict[str, Any]:
    """Parse an E-module or end-to-end JSONL trace and compute turn-level and aggregate latencies."""
    by_turn: Dict[tuple, Dict[str, Any]] = {}

    with trace_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            # Support both raw protocol JSONL lines and service log lines with {"kind": ..., "envelope": ...}
            if "event" in rec:
                env = rec
                sid = env["session_id"]
                tid = env["turn_id"]
                key = (sid, tid)
                bucket = by_turn.setdefault(key, {"session_id": sid, "turn_id": tid})
                ev_name = env["event"]
                bucket[f"{ev_name}_mono_ms"] = env["emitted_mono_ms"]
                if ev_name == "PLAYBACK_RESULT":
                    bucket["playback_status"] = env["payload"].get("status")
                    timing = env["payload"].get("timing") or {}
                    bucket.update(timing)
            elif "kind" in rec:
                kind = rec["kind"]
                if "envelope" in rec:
                    env = rec["envelope"]
                    sid = env["session_id"]
                    tid = env["turn_id"]
                    key = (sid, tid)
                    bucket = by_turn.setdefault(key, {"session_id": sid, "turn_id": tid})
                    bucket[f"{env['event']}_mono_ms"] = env["emitted_mono_ms"]
                    if env["event"] == "PLAYBACK_RESULT":
                        bucket["playback_status"] = env["payload"].get("status")
                        timing = env["payload"].get("timing") or {}
                        bucket.update(timing)
                elif "session_id" in rec and "turn_id" in rec:
                    key = (rec["session_id"], rec["turn_id"])
                    bucket = by_turn.setdefault(key, {"session_id": rec["session_id"], "turn_id": rec["turn_id"]})
                    if kind == "TTS_READY":
                        bucket["tts_ready_mono_ms"] = rec.get("mono_ms")
                        bucket["tts_total_ms"] = rec.get("tts_ms")
                        bucket["audio_duration_s"] = rec.get("audio_duration_s")
                        bucket["rtf"] = rec.get("rtf")
                        bucket["cache_hit"] = rec.get("cache_hit")
                    elif kind == "PLAYBACK_START":
                        bucket["playback_start_mono_ms"] = rec.get("mono_ms")
                    elif kind == "PLAYBACK_END":
                        bucket["playback_end_mono_ms"] = rec.get("mono_ms")

    turn_rows: List[Dict[str, Any]] = []
    tts_latencies: List[float] = []
    e_to_play_start: List[float] = []
    e2e_ttfa: List[float] = []
    rtfs: List[float] = []

    for (_, _), row in sorted(by_turn.items(), key=lambda x: (x[0][0], x[0][1])):
        req_ms = row.get("reply_request_mono_ms") or row.get("REPLY_REQUEST_mono_ms")
        tts_ready_ms = row.get("tts_ready_mono_ms")
        play_start_ms = row.get("playback_start_mono_ms")
        play_end_ms = row.get("playback_end_mono_ms")
        turn_stop_ms = row.get("TURN_STOP_mono_ms")

        tts_ms = row.get("tts_total_ms")
        if tts_ms is None and req_ms is not None and tts_ready_ms is not None:
            tts_ms = float(tts_ready_ms - req_ms)

        e_wait_ms = float(play_start_ms - req_ms) if (req_ms is not None and play_start_ms is not None) else None
        play_dur_ms = float(play_end_ms - play_start_ms) if (play_start_ms is not None and play_end_ms is not None) else None
        ttfa_ms = float(play_start_ms - turn_stop_ms) if (turn_stop_ms is not None and play_start_ms is not None) else None

        if row.get("playback_status") == "completed":
            if tts_ms is not None:
                tts_latencies.append(float(tts_ms))
            if e_wait_ms is not None:
                e_to_play_start.append(e_wait_ms)
            if ttfa_ms is not None:
                e2e_ttfa.append(ttfa_ms)
            if row.get("rtf") is not None and not row.get("cache_hit"):
                rtfs.append(float(row["rtf"]))

        turn_rows.append(
            {
                "session_id": row["session_id"],
                "turn_id": row["turn_id"],
                "playback_status": row.get("playback_status"),
                "cache_hit": row.get("cache_hit"),
                "tts_total_ms": round(tts_ms, 2) if tts_ms is not None else None,
                "e_request_to_play_start_ms": round(e_wait_ms, 2) if e_wait_ms is not None else None,
                "playback_wall_ms": round(play_dur_ms, 2) if play_dur_ms is not None else None,
                "audio_duration_s": row.get("audio_duration_s"),
                "rtf": row.get("rtf"),
                "turn_stop_to_play_start_ms": round(ttfa_ms, 2) if ttfa_ms is not None else None,
                "listen_start_mono_ms": row.get("LISTEN_START_mono_ms"),
            }
        )

    return {
        "trace_path": str(trace_path),
        "turn_count": len(turn_rows),
        "summary": {
            "tts_total_ms_median": _percentile(tts_latencies, 50),
            "tts_total_ms_p90": _percentile(tts_latencies, 90),
            "e_request_to_play_start_ms_median": _percentile(e_to_play_start, 50),
            "e_request_to_play_start_ms_p90": _percentile(e_to_play_start, 90),
            "turn_stop_to_play_start_ms_median": _percentile(e2e_ttfa, 50),
            "turn_stop_to_play_start_ms_p90": _percentile(e2e_ttfa, 90),
            "uncached_rtf_median": _percentile(rtfs, 50),
            "uncached_rtf_p90": _percentile(rtfs, 90),
        },
        "turns": turn_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze E-module JSONL event timing logs.")
    parser.add_argument("trace_file", type=Path, help="Path to JSONL event or service trace")
    args = parser.parse_args()
    report = analyze_trace_file(args.trace_file)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
