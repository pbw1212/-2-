"""Data-driven Mandarin Prosodic Structure Prediction (Syllable BiGRU-CRF) + EC-Viterbi.

Trained on the 10,000-utterance Biaobei corpus (#0: none, #1: prosodic word, #3: prosodic phrase)
with zero train/inference feature skew, plus Entity-Constrained Viterbi decoding for clinic slots.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from e_module.config import BASELINE_TACOTRON2_DIR, E_MODULE_DIR, OUTPUTS_DIR

BIAOBEI_METADATA_PATH = BASELINE_TACOTRON2_DIR / "data_mandarin" / "biaobei" / "metadata.csv"
MODELS_DIR = E_MODULE_DIR / "models"
DEFAULT_PSP_CHECKPOINT = MODELS_DIR / "syllable_psp_bigru_crf.pt"

LABEL_NAMES = ["#0", "#1", "#3"]
LABEL_TO_ID = {"#0": 0, "#1": 1, "#3": 2}
PUNCT_TO_ID = {"": 0, ",": 1, ".": 2, "?": 3, "!": 4, ";": 1, ":": 1}
_SYLLABLE_RE = re.compile(r"^([a-z]+)([1-5])$")


@dataclass
class ProsodyUtterance:
    utt_id: str
    bases: List[str]
    tones: List[int]
    punct_after: List[int]
    labels: List[int]


def parse_biaobei_metadata(
    metadata_path: Path = BIAOBEI_METADATA_PATH,
) -> List[ProsodyUtterance]:
    """Parse 10,000 Biaobei sentences into syllable sequences and #0/#1/#3 boundary labels."""
    utterances: List[ProsodyUtterance] = []
    with metadata_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) < 2:
                continue
            utt_id = parts[0].strip()
            tokens = parts[1].strip().split()
            bases: List[str] = []
            tones: List[int] = []
            puncts: List[int] = []
            labels: List[int] = []

            for tok in tokens:
                m = _SYLLABLE_RE.match(tok.lower())
                if m:
                    bases.append(m.group(1))
                    tones.append(int(m.group(2)))
                    puncts.append(0)
                    labels.append(0)
                elif tok in ("#1", "#2", "#3", "#4"):
                    if labels:
                        labels[-1] = 2 if tok in ("#3", "#4") else 1
                elif tok in PUNCT_TO_ID:
                    if puncts:
                        puncts[-1] = PUNCT_TO_ID[tok]

            if bases:
                utterances.append(
                    ProsodyUtterance(
                        utt_id=utt_id,
                        bases=bases,
                        tones=tones,
                        punct_after=puncts,
                        labels=labels,
                    )
                )
    return utterances


def compute_clause_position_features(punct_after: Sequence[int]) -> np.ndarray:
    """Compute 4-dim clause-relative positional features matching online inference."""
    n = len(punct_after)
    feats = np.zeros((n, 4), dtype=np.float32)
    if n == 0:
        return feats

    # Partition sequence into clauses ended by any punctuation token (>0) or end of utterance
    start = 0
    for i, p in enumerate(punct_after):
        if p > 0 or i == n - 1:
            end = i + 1
            cl_len = max(end - start, 1)
            for j in range(start, end):
                idx_in_cl = j - start
                dist_end = (end - 1) - j
                feats[j, 0] = idx_in_cl / float(max(cl_len - 1, 1))
                feats[j, 1] = min(dist_end, 20) / 20.0
                feats[j, 2] = min(cl_len, 25) / 25.0
                feats[j, 3] = 1.0 if j == n - 1 else 0.0
            start = end
    return feats


class ProsodyDataset(Dataset):
    def __init__(self, utterances: List[ProsodyUtterance], base_vocab: Dict[str, int]) -> None:
        self.samples: List[Tuple[torch.Tensor, ...]] = []
        unk_id = base_vocab["<UNK>"]
        for utt in utterances:
            base_ids = torch.tensor([base_vocab.get(b, unk_id) for b in utt.bases], dtype=torch.long)
            tone_ids = torch.tensor(utt.tones, dtype=torch.long)
            punct_ids = torch.tensor(utt.punct_after, dtype=torch.long)
            pos = torch.from_numpy(compute_clause_position_features(utt.punct_after))
            labels = torch.tensor(utt.labels, dtype=torch.long)
            self.samples.append((base_ids, tone_ids, punct_ids, pos, labels))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        return self.samples[idx]


def collate_prosody_batch(
    batch: List[Tuple[torch.Tensor, ...]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    max_len = max(item[0].size(0) for item in batch)
    bsz = len(batch)
    base_ids = torch.zeros((bsz, max_len), dtype=torch.long)
    tone_ids = torch.zeros((bsz, max_len), dtype=torch.long)
    punct_ids = torch.zeros((bsz, max_len), dtype=torch.long)
    pos_feats = torch.zeros((bsz, max_len, 4), dtype=torch.float32)
    labels = torch.zeros((bsz, max_len), dtype=torch.long)
    mask = torch.zeros((bsz, max_len), dtype=torch.bool)

    for i, (b, t, p, pos, y) in enumerate(batch):
        length = b.size(0)
        base_ids[i, :length] = b
        tone_ids[i, :length] = t
        punct_ids[i, :length] = p
        pos_feats[i, :length] = pos
        labels[i, :length] = y
        mask[i, :length] = True

    return base_ids, tone_ids, punct_ids, pos_feats, labels, mask


class LinearChainCRF(nn.Module):
    """Batched exact Linear-Chain CRF with constrained Viterbi decoding."""

    def __init__(self, num_tags: int = 3) -> None:
        super().__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)

    def forward(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute mean negative log-likelihood loss over a batch."""
        gold_score = self._compute_score(emissions, tags, mask)
        partition = self._compute_normalizer(emissions, mask)
        nll = partition - gold_score
        return nll.mean()

    def _compute_score(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = emissions.shape
        score = self.start_transitions[tags[:, 0]] + emissions[:, 0].gather(1, tags[:, 0:1]).squeeze(1)
        for t in range(1, seq_len):
            m_t = mask[:, t].float()
            trans_score = self.transitions[tags[:, t - 1], tags[:, t]]
            emit_score = emissions[:, t].gather(1, tags[:, t : t + 1]).squeeze(1)
            score = score + (trans_score + emit_score) * m_t
        last_idx = mask.long().sum(dim=1) - 1
        last_tags = tags.gather(1, last_idx.unsqueeze(1)).squeeze(1)
        return score + self.end_transitions[last_tags]

    def _compute_normalizer(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = emissions.shape
        score = self.start_transitions + emissions[:, 0]  # (B, C)
        for t in range(1, seq_len):
            broadcast_score = score.unsqueeze(2)  # (B, C, 1)
            broadcast_emit = emissions[:, t].unsqueeze(1)  # (B, 1, C)
            next_score = broadcast_score + self.transitions + broadcast_emit  # (B, C, C)
            next_score = torch.logsumexp(next_score, dim=1)  # (B, C)
            m_t = mask[:, t].unsqueeze(1)
            score = torch.where(m_t, next_score, score)
        score = score + self.end_transitions
        return torch.logsumexp(score, dim=1)

    def viterbi_decode(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor,
        entity_spans_batch: Optional[List[List[Tuple[int, int]]]] = None,
        entity_end_boost: float = 4.0,
    ) -> List[List[int]]:
        """Decode optimal tag paths with optional Entity-Constrained Viterbi (EC-Viterbi)."""
        adj_emissions = emissions.clone()
        bsz, seq_len, _ = adj_emissions.shape
        lengths = mask.long().sum(dim=1).tolist()

        if entity_spans_batch is not None:
            for b, spans in enumerate(entity_spans_batch):
                utt_len = lengths[b]
                for s, e in spans:
                    s_c = max(0, s)
                    e_c = min(utt_len, e)
                    if e_c - s_c >= 2:
                        # Strictly forbid #3 phrase breaks inside medical entity slots
                        adj_emissions[b, s_c : e_c - 1, 2] -= 1e4
                    if e_c - 1 >= 0 and e_c - 1 < utt_len - 1:
                        # Encourage a phrase boundary right after a completed medical entity
                        adj_emissions[b, e_c - 1, 2] += entity_end_boost

        results: List[List[int]] = []
        trans = self.transitions.detach()
        start_t = self.start_transitions.detach()
        end_t = self.end_transitions.detach()

        for b in range(bsz):
            L = lengths[b]
            emit_b = adj_emissions[b, :L].detach()  # (L, C)
            score = start_t + emit_b[0]
            history: List[torch.Tensor] = []
            for t in range(1, L):
                next_scores = score.unsqueeze(1) + trans  # (C_prev, C_curr)
                best_prev_scores, best_prev_idx = next_scores.max(dim=0)
                score = best_prev_scores + emit_b[t]
                history.append(best_prev_idx)
            score = score + end_t
            best_last = int(score.argmax(dim=0).item())
            path = [best_last]
            for back_ptr in reversed(history):
                best_last = int(back_ptr[best_last].item())
                path.append(best_last)
            path.reverse()
            results.append(path)
        return results


class SyllableBiGRUCRF(nn.Module):
    """Lightweight Mandarin prosody predictor using syllable, tone, punct, and clause-position features."""

    def __init__(
        self,
        vocab_size: int,
        base_emb_dim: int = 64,
        tone_emb_dim: int = 16,
        punct_emb_dim: int = 16,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.base_emb = nn.Embedding(vocab_size, base_emb_dim, padding_idx=0)
        self.tone_emb = nn.Embedding(6, tone_emb_dim, padding_idx=0)
        self.punct_emb = nn.Embedding(5, punct_emb_dim)

        in_dim = base_emb_dim + tone_emb_dim + punct_emb_dim + 4
        self.in_drop = nn.Dropout(dropout)
        self.bigru = nn.GRU(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out_drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, 3)
        self.crf = LinearChainCRF(num_tags=3)

    def _encode_emissions(
        self,
        base_ids: torch.Tensor,
        tone_ids: torch.Tensor,
        punct_ids: torch.Tensor,
        pos_feats: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat(
            [self.base_emb(base_ids), self.tone_emb(tone_ids), self.punct_emb(punct_ids), pos_feats],
            dim=-1,
        )
        x = self.in_drop(x)
        lengths = mask.long().sum(dim=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        enc_packed, _ = self.bigru(packed)
        enc, _ = nn.utils.rnn.pad_packed_sequence(enc_packed, batch_first=True, total_length=x.size(1))
        enc = self.out_drop(enc)
        return self.classifier(enc)

    def loss(
        self,
        base_ids: torch.Tensor,
        tone_ids: torch.Tensor,
        punct_ids: torch.Tensor,
        pos_feats: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        emissions = self._encode_emissions(base_ids, tone_ids, punct_ids, pos_feats, mask)
        return self.crf(emissions, labels, mask)

    def predict(
        self,
        base_ids: torch.Tensor,
        tone_ids: torch.Tensor,
        punct_ids: torch.Tensor,
        pos_feats: torch.Tensor,
        mask: torch.Tensor,
        entity_spans_batch: Optional[List[List[Tuple[int, int]]]] = None,
        entity_end_boost: float = 4.0,
    ) -> List[List[int]]:
        with torch.no_grad():
            emissions = self._encode_emissions(base_ids, tone_ids, punct_ids, pos_feats, mask)
            return self.crf.viterbi_decode(
                emissions, mask, entity_spans_batch=entity_spans_batch, entity_end_boost=entity_end_boost
            )


def compute_f1_metrics(
    y_true_seqs: List[List[int]],
    y_pred_seqs: List[List[int]],
    punct_seqs: List[List[int]],
) -> Dict[str, Any]:
    """Compute per-class precision/recall/F1, Macro-F1, and non-punctuation internal boundary metrics."""
    tp = [0, 0, 0]
    fp = [0, 0, 0]
    fn = [0, 0, 0]
    # Also track non-punctuation positions (excludes sentence/clause punctuation where #3 is trivial)
    np_tp = [0, 0, 0]
    np_fp = [0, 0, 0]
    np_fn = [0, 0, 0]

    for yt_seq, yp_seq, p_seq in zip(y_true_seqs, y_pred_seqs, punct_seqs):
        for yt, yp, pu in zip(yt_seq, yp_seq, p_seq):
            if yt == yp:
                tp[yt] += 1
            else:
                fp[yp] += 1
                fn[yt] += 1
            if pu == 0:
                if yt == yp:
                    np_tp[yt] += 1
                else:
                    np_fp[yp] += 1
                    np_fn[yt] += 1

    def _prf(c_tp: int, c_fp: int, c_fn: int) -> Dict[str, float]:
        prec = c_tp / max(c_tp + c_fp, 1)
        rec = c_tp / max(c_tp + c_fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        return {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}

    per_class = {LABEL_NAMES[i]: _prf(tp[i], fp[i], fn[i]) for i in range(3)}
    non_punct = {LABEL_NAMES[i]: _prf(np_tp[i], np_fp[i], np_fn[i]) for i in range(3)}
    macro_f1 = round(sum(per_class[name]["f1"] for name in LABEL_NAMES) / 3.0, 4)
    total_correct = sum(tp)
    total_count = sum(tp) + sum(fp)
    return {
        "accuracy": round(total_correct / max(total_count, 1), 4),
        "macro_f1": macro_f1,
        "per_class": per_class,
        "non_punct_per_class": non_punct,
    }


def _eval_rule_baselines(test_utts: List[ProsodyUtterance]) -> Dict[str, Dict[str, Any]]:
    y_true = [u.labels for u in test_utts]
    puncts = [u.punct_after for u in test_utts]

    # 1. no_prosody: all #0
    pred_none = [[0] * len(u.labels) for u in test_utts]

    # 2. punct_only: #3 at punctuation, #0 elsewhere
    pred_punct = [[2 if p > 0 else 0 for p in u.punct_after] for u in test_utts]

    # 3. fixed_stride_2: #1 every 2 syllables inside clause + #3 at punctuation
    pred_stride2: List[List[int]] = []
    for u in test_utts:
        seq: List[int] = []
        step = 0
        for idx, p in enumerate(u.punct_after):
            if p > 0 or idx == len(u.punct_after) - 1:
                seq.append(2 if p > 0 else 0)
                step = 0
            else:
                step += 1
                if step % 2 == 0:
                    seq.append(1)
                else:
                    seq.append(0)
        pred_stride2.append(seq)

    return {
        "no_prosody": compute_f1_metrics(y_true, pred_none, puncts),
        "punct_only": compute_f1_metrics(y_true, pred_punct, puncts),
        "fixed_stride_2": compute_f1_metrics(y_true, pred_stride2, puncts),
    }


class ProsodyPredictor:
    """Inference wrapper for trained SyllableBiGRUCRF supporting EC-Viterbi."""

    def __init__(self, checkpoint_path: Path = DEFAULT_PSP_CHECKPOINT) -> None:
        ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        self.base_vocab: Dict[str, int] = ckpt["base_vocab"]
        cfg = ckpt["model_config"]
        self.model = SyllableBiGRUCRF(**cfg)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.checkpoint_path = Path(checkpoint_path)

    def predict_syllables(
        self,
        syllables: List[str],
        punct_after: List[int],
        entity_spans: Optional[List[Tuple[int, int]]] = None,
        entity_end_boost: float = 4.0,
    ) -> List[int]:
        """Predict [#0=0, #1=1, #3=2] boundary tags after each syllable."""
        if not syllables:
            return []
        unk_id = self.base_vocab["<UNK>"]
        b_ids: List[int] = []
        t_ids: List[int] = []
        for syl in syllables:
            m = _SYLLABLE_RE.match(syl.lower())
            if m:
                b_ids.append(self.base_vocab.get(m.group(1), unk_id))
                t_ids.append(int(m.group(2)))
            else:
                b_ids.append(unk_id)
                t_ids.append(0)

        base_t = torch.tensor([b_ids], dtype=torch.long)
        tone_t = torch.tensor([t_ids], dtype=torch.long)
        punct_t = torch.tensor([punct_after], dtype=torch.long)
        pos_t = torch.from_numpy(compute_clause_position_features(punct_after)).unsqueeze(0)
        mask_t = torch.ones((1, len(syllables)), dtype=torch.bool)
        spans_batch = [entity_spans] if entity_spans is not None else None
        preds = self.model.predict(
            base_t, tone_t, punct_t, pos_t, mask_t, entity_spans_batch=spans_batch, entity_end_boost=entity_end_boost
        )
        return preds[0]


def _evaluate_clinic_entity_integrity(predictor: ProsodyPredictor) -> Dict[str, Any]:
    """Compare unconstrained BiGRU-CRF vs EC-Viterbi on 12 clinic utterances with medical entities."""
    from e_module.tts_frontend import chinese_clause_to_pinyin, extract_clinic_entity_spans, normalize_text

    clinic_cases = [
        "康复科张医生，12月5日 14:30，对吗？",
        "已帮您把日期改到2026-12-04下午，康复科张医生，对吗？",
        "门诊取号时间是上午08:05到11:30，下午13:30到16:30，请在3号诊室门口候诊。",
        "您预约的是心内科李医生，2026年12月8日上午九点半。",
        "复诊请提前十五分钟到一楼大厅自助机或1号窗口取号。",
        "骨科王医生本周五下午两点半出诊，请去二楼3号诊室。",
        "请问您想挂康复科还是心内科？",
        "好的，已为您改到十二月十日上午十点整。",
        "神经内科刘大夫明天上午08:30有号。",
        "请确认您预约的科室是中医科，时间是12月12日 15:00。",
        "如改约康复科张医生，最早可约十二月六日上午九点十五分。",
        "请在102室门口候诊，上午十一点前完成登记。",
    ]

    unconstrained_splits = 0
    ec_splits = 0
    total_entities = 0
    latencies_ms: List[float] = []

    for raw in clinic_cases:
        norm, _, _ = normalize_text(raw)
        # Convert whole normalized text into flat syllables + punct_after + syllable entity spans
        py_flat = chinese_clause_to_pinyin(norm, insert_prosody=False)
        tokens = py_flat.split()
        syllables: List[str] = []
        puncts: List[int] = []
        for tok in tokens:
            if _SYLLABLE_RE.match(tok):
                syllables.append(tok)
                puncts.append(0)
            elif tok in PUNCT_TO_ID and puncts:
                puncts[-1] = PUNCT_TO_ID[tok]

        # Extract character-level entities on normalized text and map to syllable spans
        char_entities = extract_clinic_entity_spans(norm)
        syl_spans: List[Tuple[int, int]] = []
        for cs, ce, _ in char_entities:
            prefix_syls = [
                t for t in chinese_clause_to_pinyin(norm[:cs], insert_prosody=False).split() if _SYLLABLE_RE.match(t)
            ]
            span_syls = [
                t for t in chinese_clause_to_pinyin(norm[cs:ce], insert_prosody=False).split() if _SYLLABLE_RE.match(t)
            ]
            if span_syls:
                s_idx = len(prefix_syls)
                e_idx = min(len(syllables), s_idx + len(span_syls))
                if e_idx - s_idx >= 2:
                    syl_spans.append((s_idx, e_idx))

        total_entities += len(syl_spans)

        tags_plain = predictor.predict_syllables(syllables, puncts, entity_spans=None)
        t0 = time.perf_counter()
        tags_ec = predictor.predict_syllables(syllables, puncts, entity_spans=syl_spans)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        for s_idx, e_idx in syl_spans:
            if any(tags_plain[k] == 2 for k in range(s_idx, e_idx - 1)):
                unconstrained_splits += 1
            if any(tags_ec[k] == 2 for k in range(s_idx, e_idx - 1)):
                ec_splits += 1

    return {
        "clinic_utterances": len(clinic_cases),
        "total_multi_syllable_entities": total_entities,
        "unconstrained_internal_#3_splits": unconstrained_splits,
        "ec_viterbi_internal_#3_splits": ec_splits,
        "ec_viterbi_entity_integrity_rate": round(1.0 - ec_splits / max(total_entities, 1), 4),
        "single_utterance_infer_ms_median": round(float(np.median(latencies_ms)), 3),
        "single_utterance_infer_ms_p90": round(float(np.percentile(latencies_ms, 90)), 3),
    }


def train_and_evaluate(
    metadata_path: Path = BIAOBEI_METADATA_PATH,
    checkpoint_out: Path = DEFAULT_PSP_CHECKPOINT,
    report_out: Path = OUTPUTS_DIR / "exp1_prosody_eval.json",
    epochs: int = 24,
    batch_size: int = 64,
    hidden_dim: int = 128,
    lr: float = 2e-3,
    seed: int = 42,
) -> Dict[str, Any]:
    """Train SyllableBiGRUCRF on 8500 Biaobei utterances, validate on 500, test on 1000."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    utterances = parse_biaobei_metadata(metadata_path)
    indices = list(range(len(utterances)))
    random.Random(seed).shuffle(indices)

    train_utts = [utterances[i] for i in indices[:8500]]
    val_utts = [utterances[i] for i in indices[8500:9000]]
    test_utts = [utterances[i] for i in indices[9000:]]

    # Build base syllable vocabulary from training split only (strict holdout hygiene)
    base_vocab: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
    for u in train_utts:
        for b in u.bases:
            if b not in base_vocab:
                base_vocab[b] = len(base_vocab)

    train_ds = ProsodyDataset(train_utts, base_vocab)
    val_ds = ProsodyDataset(val_utts, base_vocab)
    test_ds = ProsodyDataset(test_utts, base_vocab)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_prosody_batch)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_prosody_batch)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_prosody_batch)

    model_cfg = {
        "vocab_size": len(base_vocab),
        "base_emb_dim": 64,
        "tone_emb_dim": 16,
        "punct_emb_dim": 16,
        "hidden_dim": hidden_dim,
        "num_layers": 2,
        "dropout": 0.2,
    }
    model = SyllableBiGRUCRF(**model_cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)

    def _run_eval(loader: DataLoader, utts: List[ProsodyUtterance]) -> Dict[str, Any]:
        model.eval()
        preds: List[List[int]] = []
        for b_id, t_id, p_id, pos_f, _, m_t in loader:
            preds.extend(model.predict(b_id, t_id, p_id, pos_f, m_t))
        return compute_f1_metrics([u.labels for u in utts], preds, [u.punct_after for u in utts])

    best_val_f1 = -1.0
    best_epoch = 0
    history: List[Dict[str, Any]] = []
    t_train_0 = time.perf_counter()

    checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for b_id, t_id, p_id, pos_f, y_t, m_t in train_loader:
            optimizer.zero_grad()
            loss = model.loss(b_id, t_id, p_id, pos_f, y_t, m_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1
        scheduler.step()

        val_metrics = _run_eval(val_loader, val_utts)
        val_f1 = val_metrics["macro_f1"]
        val_p1_f1 = val_metrics["per_class"]["#1"]["f1"]
        avg_loss = round(total_loss / max(n_batches, 1), 4)
        history.append({"epoch": ep, "train_nll": avg_loss, "val_macro_f1": val_f1, "val_#1_f1": val_p1_f1})
        print(f"Epoch {ep:02d}/{epochs} | loss={avg_loss:.4f} | val_macro_f1={val_f1:.4f} | val_#1_f1={val_p1_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = ep
            torch.save(
                {
                    "base_vocab": base_vocab,
                    "model_config": model_cfg,
                    "state_dict": model.state_dict(),
                    "best_epoch": best_epoch,
                    "best_val_macro_f1": best_val_f1,
                },
                str(checkpoint_out),
            )

    train_elapsed_s = round(time.perf_counter() - t_train_0, 2)
    ckpt = torch.load(str(checkpoint_out), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])

    bigru_test_metrics = _run_eval(test_loader, test_utts)
    baseline_metrics = _eval_rule_baselines(test_utts)

    predictor = ProsodyPredictor(checkpoint_out)
    clinic_integrity = _evaluate_clinic_entity_integrity(predictor)
    ckpt_size_mb = round(checkpoint_out.stat().st_size / (1024 * 1024), 3)

    report = {
        "dataset": {
            "total_utterances": len(utterances),
            "train_count": len(train_utts),
            "val_count": len(val_utts),
            "test_count": len(test_utts),
            "vocab_size": len(base_vocab),
        },
        "training": {
            "epochs": epochs,
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_val_f1,
            "train_elapsed_s": train_elapsed_s,
            "checkpoint_path": str(checkpoint_out),
            "checkpoint_size_mb": ckpt_size_mb,
            "param_count": sum(p.numel() for p in model.parameters()),
            "history": history,
        },
        "test_set_comparison": {
            **baseline_metrics,
            "syllable_bigru_crf": bigru_test_metrics,
        },
        "clinic_entity_integrity": clinic_integrity,
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate Syllable BiGRU-CRF prosody predictor.")
    parser.add_argument("--train", action="store_true", help="Train model on 10k Biaobei corpus")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    args = parser.parse_args()

    if args.train:
        report = train_and_evaluate(epochs=args.epochs, hidden_dim=args.hidden_dim, lr=args.lr)
        print("=== TEST SET COMPARISON (1000 Holdout Biaobei) ===")
        for name, m in report["test_set_comparison"].items():
            print(
                f"{name:20s} | acc={m['accuracy']:.4f} | macro_f1={m['macro_f1']:.4f} | "
                f"#1_f1={m['per_class']['#1']['f1']:.4f} | #3_f1={m['per_class']['#3']['f1']:.4f} | "
                f"nonpunct_#3_f1={m['non_punct_per_class']['#3']['f1']:.4f}"
            )
        print("=== CLINIC ENTITY INTEGRITY ===")
        print(json.dumps(report["clinic_entity_integrity"], ensure_ascii=False, indent=2))
    else:
        print("Use --train to train and evaluate SyllableBiGRUCRF.")


if __name__ == "__main__":
    main()
