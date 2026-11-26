"""Cross-platform audio playback abstraction for Mac local tests and Raspberry Pi ALSA."""

from __future__ import annotations

import platform
import shutil
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from scipy.io import wavfile

from e_module.events import monotonic_ms


@dataclass
class PlaybackOutcome:
    success: bool
    playback_start_mono_ms: Optional[int]
    playback_end_mono_ms: Optional[int]
    error_code: Optional[str] = None
    backend: str = "unknown"
    proxy_timing_note: str = "process_invocation_proxy"


class AudioPlayer(ABC):
    """Abstract player ensuring single-active playback and clean stop on failure."""

    @abstractmethod
    def play(self, wav_path: Path) -> PlaybackOutcome:
        """Play a WAV file synchronously and return timestamps or failure code."""

    @abstractmethod
    def stop(self) -> None:
        """Force-stop any active playback process and verify device release."""

    @property
    @abstractmethod
    def is_playing(self) -> bool:
        """Whether playback is currently active."""


class SubprocessAudioPlayer(AudioPlayer):
    """Subprocess-backed audio player wrapping afplay (macOS) or aplay (Linux/Pi)."""

    def __init__(self, command_prefix: list[str], backend_name: str) -> None:
        self._cmd_prefix = list(command_prefix)
        self._backend_name = backend_name
        self._proc: Optional[subprocess.Popen] = None

    @property
    def is_playing(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def play(self, wav_path: Path) -> PlaybackOutcome:
        if not wav_path.exists() or not wav_path.is_file():
            return PlaybackOutcome(
                success=False,
                playback_start_mono_ms=None,
                playback_end_mono_ms=None,
                error_code="file_missing",
                backend=self._backend_name,
            )

        self.stop()
        start_ms = monotonic_ms()
        try:
            self._proc = subprocess.Popen(
                self._cmd_prefix + [str(wav_path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            _, stderr = self._proc.communicate()
            end_ms = monotonic_ms()
            rc = self._proc.returncode
            self._proc = None
            if rc != 0:
                return PlaybackOutcome(
                    success=False,
                    playback_start_mono_ms=start_ms,
                    playback_end_mono_ms=end_ms,
                    error_code=f"device_error:rc={rc}",
                    backend=self._backend_name,
                )
            return PlaybackOutcome(
                success=True,
                playback_start_mono_ms=start_ms,
                playback_end_mono_ms=end_ms,
                error_code=None,
                backend=self._backend_name,
            )
        except OSError:
            self.stop()
            return PlaybackOutcome(
                success=False,
                playback_start_mono_ms=start_ms,
                playback_end_mono_ms=monotonic_ms(),
                error_code="device_error:os_error",
                backend=self._backend_name,
            )

    def stop(self) -> None:
        if self._proc is not None:
            if self._proc.poll() is None:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
                    self._proc.wait(timeout=1.0)
            self._proc = None


class MacAudioPlayer(SubprocessAudioPlayer):
    """macOS audio playback using afplay."""

    def __init__(self) -> None:
        super().__init__(["afplay"], "mac_afplay")


class AlsaAudioPlayer(SubprocessAudioPlayer):
    """Raspberry Pi Linux ALSA playback using aplay."""

    def __init__(self, device: Optional[str] = None) -> None:
        cmd = ["aplay", "-q"]
        if device:
            cmd.extend(["-D", device])
        super().__init__(cmd, f"alsa_aplay({device or 'default'})")


class MockAudioPlayer(AudioPlayer):
    """Deterministic in-memory mock player for automated integration & failure testing."""

    def __init__(self, speed_factor: float = 0.0, fail_next_count: int = 0) -> None:
        self.speed_factor = speed_factor
        self.fail_next_count = fail_next_count
        self._playing = False
        self.played_paths: list[str] = []
        self.stop_calls: int = 0

    @property
    def is_playing(self) -> bool:
        return self._playing

    def play(self, wav_path: Path) -> PlaybackOutcome:
        if not wav_path.exists() or not wav_path.is_file():
            return PlaybackOutcome(
                success=False,
                playback_start_mono_ms=None,
                playback_end_mono_ms=None,
                error_code="file_missing",
                backend="mock_player",
                proxy_timing_note="mock_playback",
            )

        sr, data = wavfile.read(str(wav_path))
        duration_s = float(len(data) / max(sr, 1))

        self._playing = True
        start_ms = monotonic_ms()
        if self.fail_next_count > 0:
            self.fail_next_count -= 1
            self.stop()
            return PlaybackOutcome(
                success=False,
                playback_start_mono_ms=start_ms,
                playback_end_mono_ms=monotonic_ms(),
                error_code="device_error:simulated_failure",
                backend="mock_player",
                proxy_timing_note="mock_playback",
            )

        if self.speed_factor > 0:
            time.sleep(duration_s * self.speed_factor)
        end_ms = monotonic_ms()
        self._playing = False
        self.played_paths.append(str(wav_path))
        return PlaybackOutcome(
            success=True,
            playback_start_mono_ms=start_ms,
            playback_end_mono_ms=end_ms,
            error_code=None,
            backend="mock_player",
            proxy_timing_note="mock_playback",
        )

    def stop(self) -> None:
        self._playing = False
        self.stop_calls += 1


def create_default_player(device: Optional[str] = None, mock: bool = False) -> AudioPlayer:
    """Create platform-appropriate AudioPlayer (Mac afplay, Pi aplay, or Mock)."""
    if mock:
        return MockAudioPlayer()
    system = platform.system().lower()
    if system == "darwin" and shutil.which("afplay"):
        return MacAudioPlayer()
    if system == "linux" and shutil.which("aplay"):
        return AlsaAudioPlayer(device=device)
    return MockAudioPlayer()
