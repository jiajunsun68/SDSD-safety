#!/usr/bin/env python3
"""Three deterministic mechanism illustrations, NOT LLM experimental results.

Run: python scripts/teacher_pooling_probe.py
Uses only NumPy and the standard library; prints JSON and writes no artifacts.
All distributions and scalar dynamics below are deliberately constructed.
"""

import json

import numpy as np


def pool(teachers, weights, geometric=False):
    teachers = np.asarray(teachers, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if not np.all(teachers > 0) or not np.all(weights >= 0):
        raise ValueError("This illustration requires positive probabilities.")
    np.testing.assert_allclose(teachers.sum(axis=1), 1.0)
    np.testing.assert_allclose(weights.sum(), 1.0)
    if geometric:
        values = np.exp(weights @ np.log(teachers))
        return values / values.sum(), float(values.sum())
    return weights @ teachers, 1.0


def kl(p, q):
    return float(np.sum(p * (np.log(p) - np.log(q))))


def distribution(values, names):
    return {name: round(float(value), 9) for name, value in zip(names, values)}


def shared_consensus():
    names = ["help_A", "help_B", "refuse", "harm"]
    teachers = np.array([[0.79, 0.01, 0.19, 0.01],
                         [0.01, 0.79, 0.19, 0.01]])
    weights = np.array([0.5, 0.5])
    arithmetic, _ = pool(teachers, weights)
    geometric, normalizer = pool(teachers, weights, geometric=True)
    student = np.array([0.3, 0.2, 0.4, 0.1])
    lhs = sum(w * kl(student, q) for w, q in zip(weights, teachers))
    rhs = kl(student, geometric) - np.log(normalizer)
    np.testing.assert_allclose(lhs, rhs, atol=1e-12)
    duplicates = np.repeat(teachers[:1], 3, axis=0)
    repeated, _ = pool(duplicates, [1 / 3] * 3, geometric=True)
    np.testing.assert_allclose(repeated, teachers[0], atol=1e-12)
    return {
        "teachers": [distribution(q, names) for q in teachers],
        "arithmetic": distribution(arithmetic, names),
        "geometric": distribution(geometric, names),
        "mean_reverse_kl_identity_error": float(abs(lhs - rhs)),
        "identical_teacher_repetition_max_error": float(abs(repeated - teachers[0]).max()),
    }


def unsafe_anchor_support():
    names = ["refuse", "safe_redirect", "harm"]
    ema = np.array([0.899, 0.100, 0.001])
    raw_anchor = np.array([0.60, 0.28, 0.12])
    conditioned_anchor = np.array([0.85, 0.148, 0.002])
    unsafe_mix, _ = pool([ema, raw_anchor], [0.75, 0.25])
    conditioned_mix, _ = pool([ema, conditioned_anchor], [0.75, 0.25])
    return {
        "frozen_weight": 0.25,
        "ema": distribution(ema, names),
        "unconditional_anchor": distribution(raw_anchor, names),
        "conditioned_anchor": distribution(conditioned_anchor, names),
        "mixture_with_unconditional_anchor": distribution(unsafe_mix, names),
        "mixture_with_conditioned_anchor": distribution(conditioned_mix, names),
        "note": "Conditioning's benefit is assumed in these inputs, not learned or proven.",
    }


def scalar_feedback():
    # An invented linear feedback model; x is an unconstrained scalar, not a
    # probability, a LoRA parameter, or a measured model score.
    eta, bias, steps, anchor = 0.2, 0.05, 2000, 0.0
    outputs = []
    for tau, frozen_weight in [(0.2, 0.0), (0.02, 0.0), (0.02, 0.25)]:
        x, ema = anchor, anchor
        for _ in range(steps):
            target = (1 - frozen_weight) * (ema + bias) + frozen_weight * anchor
            x = (1 - eta) * x + eta * target
            ema = (1 - tau) * ema + tau * x
        matrix = np.array([
            [1 - eta, eta * (1 - frozen_weight)],
            [tau * (1 - eta), 1 - tau + tau * eta * (1 - frozen_weight)],
        ])
        outputs.append({
            "ema_rate": tau,
            "frozen_weight": frozen_weight,
            "spectral_radius": round(float(abs(np.linalg.eigvals(matrix)).max()), 9),
            "student_at_step_2000": round(float(x), 9),
            "fixed_point_if_anchored": None if frozen_weight == 0 else
                round(anchor + (1 - frozen_weight) * bias / frozen_weight, 9),
        })
    return {"eta": eta, "bias": bias, "steps": steps, "cases": outputs}


if __name__ == "__main__":
    print(json.dumps({
        "scope": "Constructed toy examples only; no LLM data, training, or benchmark results.",
        "1_shared_consensus": shared_consensus(),
        "2_unsafe_anchor_support": unsafe_anchor_support(),
        "3_scalar_feedback": scalar_feedback(),
    }, indent=2))
