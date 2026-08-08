"""rembg adapter tests: mask quality heuristic (no model download needed)."""

import numpy as np
import pytest

from adapters.background.rembg_adapter import _mask_quality


def test_empty_mask_zero_quality():
    alpha = np.zeros((100, 100), dtype=np.uint8)
    q, fg, cov = _mask_quality(alpha)
    assert q == 0.0 and fg == 0.0 and cov == 0.0


def test_compact_centered_mask_high_quality():
    alpha = np.zeros((100, 100), dtype=np.uint8)
    alpha[30:70, 30:70] = 255
    q, fg, cov = _mask_quality(alpha)
    assert fg == pytest.approx(0.16, abs=0.01)
    assert cov == pytest.approx(1.0, abs=0.01)
    assert q > 0.8


def test_tiny_mask_low_quality():
    alpha = np.zeros((100, 100), dtype=np.uint8)
    alpha[49:51, 49:51] = 255
    q, _, _ = _mask_quality(alpha)
    assert q < 0.5


def test_scattered_mask_penalized():
    alpha = np.zeros((100, 100), dtype=np.uint8)
    alpha[20:30, 20:30] = 255
    alpha[70:80, 70:80] = 255  # two islands in a big bbox
    q, _, cov = _mask_quality(alpha)
    assert cov < 0.5
    assert q < 0.5
