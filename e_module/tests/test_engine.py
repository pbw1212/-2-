"""Unit tests for engine wrappers including LSTMCellAsLSTM equivalence."""

import torch

from e_module.tts_engine import LSTMCellAsLSTM


def test_lstmcell_as_lstm_numerical_equivalence():
    torch.manual_seed(42)
    cell = torch.nn.LSTMCell(input_size=16, hidden_size=32)
    cell.eval()
    wrapped = LSTMCellAsLSTM(cell)
    wrapped.eval()

    x = torch.randn(2, 16)
    h = torch.randn(2, 32)
    c = torch.randn(2, 32)

    with torch.no_grad():
        h_ref, c_ref = cell(x, (h, c))
        h_new, c_new = wrapped(x, (h, c))

    max_h_diff = float((h_ref - h_new).abs().max().item())
    max_c_diff = float((c_ref - c_new).abs().max().item())
    assert max_h_diff < 1e-5
    assert max_c_diff < 1e-5
