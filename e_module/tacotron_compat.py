"""Compatibility wrapper around the unmodified course Tacotron2 baseline.

Provides clean model loading, modern PyTorch/Librosa compatibility shims,
cached window_sumsquare STFT inversion, and Fast Griffin-Lim reconstruction
without touching any file in 语音信息处理综合实践网盘/.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from librosa.util import pad_center, tiny
from scipy.signal import get_window

from e_module.config import (
    AUDIO_CONFIG,
    BASELINE_TACOTRON2_DIR,
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_FAST_GL_MOMENTUM,
    MODEL_CONFIG,
)


def ensure_baseline_on_path() -> None:
    """Append the course Tacotron2 directory to sys.path once."""
    baseline_str = str(BASELINE_TACOTRON2_DIR)
    if baseline_str not in sys.path:
        sys.path.append(baseline_str)


ensure_baseline_on_path()

from common.audio_processing import (  # noqa: E402
    dynamic_range_compression,
    dynamic_range_decompression,
)
from common.utils import de_emphasize  # noqa: E402
from tacotron2.model import Tacotron2  # noqa: E402
from tacotron2.text import text_to_sequence  # noqa: E402


def window_sumsquare_compat(
    window: str,
    n_frames: int,
    hop_length: int = 256,
    win_length: int = 1024,
    n_fft: int = 1024,
    dtype=np.float32,
    norm=None,
) -> np.ndarray:
    """Compute window sum-square envelope compatible with librosa 0.10+ pad_center."""
    if win_length is None:
        win_length = n_fft
    n = n_fft + hop_length * (n_frames - 1)
    x = np.zeros(n, dtype=dtype)
    win_sq = get_window(window, win_length, fftbins=True)
    win_sq = librosa.util.normalize(win_sq, norm=norm) ** 2
    win_sq = pad_center(win_sq, size=n_fft)
    for i in range(n_frames):
        sample = i * hop_length
        x[sample : min(n, sample + n_fft)] += win_sq[: max(0, min(n_fft, n - sample))]
    return x


def _torch_load_compat(filepath: Path) -> Dict:
    """Load a PyTorch checkpoint safely across PyTorch 1.x and 2.6+."""
    sig = inspect.signature(torch.load)
    kwargs = {"map_location": "cpu"}
    if "weights_only" in sig.parameters:
        kwargs["weights_only"] = False
    return torch.load(str(filepath), **kwargs)


def _unwrap_distributed_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remove DDP 'module.' prefixes if present."""
    if any("module." in k for k in state_dict):
        return {k.replace("module.", ""): v for k, v in state_dict.items()}
    return state_dict


def load_tacotron2_model(
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    max_decoder_steps: Optional[int] = None,
) -> Tacotron2:
    """Instantiate Tacotron2 with fixed training hyperparameters and load checkpoint."""
    cfg = dict(MODEL_CONFIG)
    if max_decoder_steps is not None:
        cfg["max_decoder_steps"] = max_decoder_steps

    model = Tacotron2(**cfg)
    ckpt = _torch_load_compat(Path(checkpoint_path))
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    state_dict = _unwrap_distributed_state_dict(state_dict)
    model.load_state_dict(state_dict)
    if isinstance(ckpt, dict) and "epoch" in ckpt:
        model.elapsed_epochs = ckpt["epoch"]
    model.eval()
    return model.cpu()


def prepare_input_sequence(
    texts: List[str], speaker_id: int = 0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode cleaned text strings to padded symbol ID tensors sorted by descending length."""
    sequences = [text_to_sequence(text, speaker_id, ["basic_cleaners"])[:] for text in texts]
    lengths = torch.tensor([len(x) for x in sequences], dtype=torch.int64)
    text_lengths, ids_sorted_decreasing = torch.sort(lengths, dim=0, descending=True)
    max_text_len = int(text_lengths[0].item())

    padded = []
    for idx in ids_sorted_decreasing.tolist():
        seq = sequences[idx]
        padded.append(np.pad(seq, [0, max_text_len - len(seq)], mode="constant"))

    texts_tensor = torch.from_numpy(np.stack(padded)).long()
    return texts_tensor, text_lengths.int(), ids_sorted_decreasing


class CompatSTFT(torch.nn.Module):
    """STFT/ISTFT compatible with modern librosa and optional window_sum caching."""

    def __init__(
        self,
        filter_length: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        window: str = "hann",
        cache_window_sum: bool = True,
    ) -> None:
        super().__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.cache_window_sum = cache_window_sum
        self._window_sum_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

        scale = self.filter_length / self.hop_length
        fourier_basis = np.fft.fft(np.eye(self.filter_length))
        cutoff = int(self.filter_length / 2 + 1)
        fourier_basis = np.vstack(
            [np.real(fourier_basis[:cutoff, :]), np.imag(fourier_basis[:cutoff, :])]
        )

        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
        inverse_basis = torch.FloatTensor(np.linalg.pinv(scale * fourier_basis).T[:, None, :])

        if window is not None:
            assert filter_length >= win_length
            fft_window = get_window(window, win_length, fftbins=True)
            fft_window = pad_center(fft_window, size=filter_length)
            fft_window = torch.from_numpy(fft_window).float()
            forward_basis *= fft_window
            inverse_basis *= fft_window

        self.register_buffer("forward_basis", forward_basis.float())
        self.register_buffer("inverse_basis", inverse_basis.float())

    def _get_window_sum_and_indices(self, n_frames: int, is_cuda: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.cache_window_sum and n_frames in self._window_sum_cache:
            return self._window_sum_cache[n_frames]

        window_sum = window_sumsquare_compat(
            self.window,
            n_frames,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_fft=self.filter_length,
            dtype=np.float32,
        )
        nonzero_idx = torch.from_numpy(np.where(window_sum > tiny(window_sum))[0])
        window_sum_tensor = torch.from_numpy(window_sum)
        if is_cuda:
            window_sum_tensor = window_sum_tensor.cuda()
            nonzero_idx = nonzero_idx.cuda()

        if self.cache_window_sum:
            self._window_sum_cache[n_frames] = (window_sum_tensor, nonzero_idx)
        return window_sum_tensor, nonzero_idx

    def transform(self, input_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        num_batches = input_data.size(0)
        num_samples = input_data.size(1)

        input_data = input_data.view(num_batches, 1, num_samples)
        input_data = F.pad(
            input_data.unsqueeze(1),
            (int(self.filter_length / 2), int(self.filter_length / 2), 0, 0),
            mode="reflect",
        ).squeeze(1)

        forward_transform = F.conv1d(
            input_data,
            self.forward_basis,
            stride=self.hop_length,
            padding=0,
        )

        cutoff = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]

        magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)
        phase = torch.atan2(imag_part.data, real_part.data)
        return magnitude, phase

    def transform_complex(self, input_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return real and imaginary STFT parts directly for Fast Griffin-Lim."""
        num_batches = input_data.size(0)
        num_samples = input_data.size(1)

        input_data = input_data.view(num_batches, 1, num_samples)
        input_data = F.pad(
            input_data.unsqueeze(1),
            (int(self.filter_length / 2), int(self.filter_length / 2), 0, 0),
            mode="reflect",
        ).squeeze(1)

        forward_transform = F.conv1d(
            input_data,
            self.forward_basis,
            stride=self.hop_length,
            padding=0,
        )
        cutoff = int((self.filter_length / 2) + 1)
        return forward_transform[:, :cutoff, :], forward_transform[:, cutoff:, :]

    def inverse(self, magnitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        recombine = torch.cat(
            [magnitude * torch.cos(phase), magnitude * torch.sin(phase)], dim=1
        )
        return self._inverse_from_recombined(recombine, magnitude.size(-1), magnitude.is_cuda)

    def inverse_complex(self, real_part: torch.Tensor, imag_part: torch.Tensor) -> torch.Tensor:
        recombine = torch.cat([real_part, imag_part], dim=1)
        return self._inverse_from_recombined(recombine, real_part.size(-1), real_part.is_cuda)

    def _inverse_from_recombined(
        self, recombine: torch.Tensor, n_frames: int, is_cuda: bool
    ) -> torch.Tensor:
        inverse_transform = F.conv_transpose1d(
            recombine,
            self.inverse_basis,
            stride=self.hop_length,
            padding=0,
        )
        if self.window is not None:
            window_sum, nonzero_idx = self._get_window_sum_and_indices(n_frames, is_cuda)
            inverse_transform[:, :, nonzero_idx] /= window_sum[nonzero_idx]
            inverse_transform *= float(self.filter_length) / self.hop_length

        half = int(self.filter_length / 2)
        return inverse_transform[:, :, half:-half]


class CompatTacotronSTFT(torch.nn.Module):
    """TacotronSTFT with modern librosa kwargs and cached pseudo-inverse mel basis."""

    def __init__(
        self,
        filter_length: int = AUDIO_CONFIG["filter_length"],
        hop_length: int = AUDIO_CONFIG["hop_length"],
        win_length: int = AUDIO_CONFIG["win_length"],
        n_mel_channels: int = AUDIO_CONFIG["n_mel_channels"],
        sampling_rate: int = AUDIO_CONFIG["sampling_rate"],
        mel_fmin: float = AUDIO_CONFIG["mel_fmin"],
        mel_fmax: float = AUDIO_CONFIG["mel_fmax"],
        cache_window_sum: bool = True,
    ) -> None:
        super().__init__()
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.stft_fn = CompatSTFT(
            filter_length=filter_length,
            hop_length=hop_length,
            win_length=win_length,
            cache_window_sum=cache_window_sum,
        )
        mel_basis = librosa.filters.mel(
            sr=sampling_rate,
            n_fft=filter_length,
            n_mels=n_mel_channels,
            fmin=mel_fmin,
            fmax=mel_fmax,
        )
        inv_mel_basis = np.linalg.pinv(mel_basis)
        self.register_buffer("mel_basis", torch.from_numpy(mel_basis).float())
        self.register_buffer("inv_mel_basis", torch.from_numpy(inv_mel_basis).float())

    def spectral_normalize(self, magnitudes: torch.Tensor) -> torch.Tensor:
        return dynamic_range_compression(magnitudes)

    def spectral_de_normalize(self, magnitudes: torch.Tensor) -> torch.Tensor:
        return dynamic_range_decompression(magnitudes)

    def inv_mel_spectrogram(self, mel: torch.Tensor) -> torch.Tensor:
        mel = self.spectral_de_normalize(mel.float())
        magnitudes = torch.matmul(self.inv_mel_basis, mel.data)
        magnitudes = torch.clamp(magnitudes, min=1e-10)
        return magnitudes.data


def _ensure_3d_mag(magnitudes: torch.Tensor, power: float) -> torch.Tensor:
    if magnitudes.dim() == 2:
        return magnitudes.unsqueeze(0) ** power
    if magnitudes.dim() == 3:
        return magnitudes ** power
    raise ValueError(f"Expected 2D or 3D magnitudes, got shape {tuple(magnitudes.shape)}")


def griffin_lim_compat(
    magnitudes: torch.Tensor,
    stft_fn: CompatSTFT,
    n_iters: int = 50,
    power: float = 1.5,
) -> np.ndarray:
    """Standard Griffin-Lim waveform reconstruction matching course baseline."""
    with torch.no_grad():
        mag = _ensure_3d_mag(magnitudes, power)
        angles = np.angle(np.exp(2j * np.pi * np.random.rand(*mag.size())))
        angles_t = torch.from_numpy(angles.astype(np.float32))
        signal = stft_fn.inverse(mag, angles_t).squeeze(1)

        for _ in range(n_iters):
            _, angles_t = stft_fn.transform(signal)
            signal = stft_fn.inverse(mag, angles_t).squeeze(1)
        return de_emphasize(signal.squeeze().cpu().numpy())


def fast_griffin_lim(
    magnitudes: torch.Tensor,
    stft_fn: CompatSTFT,
    n_iters: int = 25,
    power: float = 1.5,
    momentum: float = DEFAULT_FAST_GL_MOMENTUM,
) -> np.ndarray:
    """Fast Griffin-Lim (Perraudin et al. 2013) using Nesterov-style momentum."""
    with torch.no_grad():
        mag = _ensure_3d_mag(magnitudes, power)
        angles = 2 * np.pi * np.random.rand(*mag.size())
        angles_t = torch.from_numpy(angles.astype(np.float32))

        prev_real = mag * torch.cos(angles_t)
        prev_imag = mag * torch.sin(angles_t)
        signal = stft_fn.inverse_complex(prev_real, prev_imag).squeeze(1)

        for _ in range(n_iters):
            cur_real, cur_imag = stft_fn.transform_complex(signal)
            norm = torch.sqrt(cur_real ** 2 + cur_imag ** 2).clamp(min=1e-8)
            proj_real = mag * (cur_real / norm)
            proj_imag = mag * (cur_imag / norm)

            acc_real = proj_real + momentum * (proj_real - prev_real)
            acc_imag = proj_imag + momentum * (proj_imag - prev_imag)

            prev_real = proj_real
            prev_imag = proj_imag
            signal = stft_fn.inverse_complex(acc_real, acc_imag).squeeze(1)

        return de_emphasize(signal.squeeze().cpu().numpy())
