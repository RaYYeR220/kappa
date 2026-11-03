"""Rearing from SAM 3.1 silhouettes: a small classifier trained on the tuning videos.

Muse Wonder watching zoomed clips could not tell a rear from a walk in this
footage (frame-level kappa 0.10 to 0.15 against raters at 0.75 to 0.78 on the tuning
videos). The silhouettes SAM 3.1 draws on every frame carry the signal instead: a
gradient-boosted classifier over per-frame silhouette descriptors and their context
gives a rearing probability per frame, which becomes bouts by hysteresis.

The model is trained only on the pre-registered tuning videos, on the frames where
at least two of the three raters agree, and ships with the list of what it was
trained on.
"""

from __future__ import annotations

import hashlib
import io
import json
import pickle
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import median_filter

from kappa.model import Span
from kappa.shape import frame_features, silhouettes

MODEL_DIR = Path(__file__).parent / "models"
FEATURE_VERSION = "silhouette/1"


@dataclass(frozen=True)
class BoutRule:
    """Probability → bouts: smooth, enter above ``high``, stay while above ``low``."""

    smooth_s: float = 0.28
    high: float = 0.5
    low: float = 0.35
    min_s: float = 0.24
    merge_gap_s: float = 0.2


@dataclass
class RearingModel:
    classifier: Any
    rule: BoutRule = field(default_factory=BoutRule)
    trained_on: tuple[str, ...] = ()
    feature_version: str = FEATURE_VERSION
    notes: str = ""

    def probabilities(self, features: np.ndarray) -> np.ndarray:
        x = np.where(np.isfinite(features), features, np.nan)
        return self.classifier.predict_proba(x)[:, 1]

    def bouts(self, probability: np.ndarray, present: np.ndarray, fps: float) -> list[tuple[Span, float]]:
        r = self.rule
        p = median_filter(np.where(present, probability, 0.0), size=max(1, int(r.smooth_s * fps)) | 1)
        out: list[list[int]] = []
        inside = False
        for i, v in enumerate(p):
            if not inside and v >= r.high:
                inside = True
                start = i
            elif inside and v < r.low:
                inside = False
                out.append([start, i])
        if inside:
            out.append([start, len(p)])
        merged: list[list[int]] = []
        gap = round(r.merge_gap_s * fps)
        for s, e in out:
            if merged and s - merged[-1][1] <= gap:
                merged[-1][1] = e
            else:
                merged.append([s, e])
        shortest = round(r.min_s * fps)
        return [(Span(s, e), float(p[s:e].mean())) for s, e in merged if e - s >= shortest]

    def save(self, path: Path) -> str:
        buf = io.BytesIO()
        pickle.dump(self, buf)
        data = buf.getvalue()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        digest = hashlib.sha256(data).hexdigest()
        path.with_suffix(".json").write_text(
            json.dumps(
                {"sha256": digest, "trained_on": list(self.trained_on), "feature_version": self.feature_version, "rule": self.rule.__dict__, "notes": self.notes},
                indent=2,
            ),
            encoding="utf-8",
        )
        return digest


def load(path: Path | None = None) -> RearingModel:
    """Load the shipped model, refusing a file that doesn't match its recorded sha256.

    The classifier is a scikit-learn object, which only pickles; the file is Kappa's
    own artifact from ``kappa/models`` and is checked against the hash written next
    to it at training time before it is unpickled. Never point this at a file from
    elsewhere.
    """
    path = path or MODEL_DIR / "rearing_topdown.pkl"
    data = path.read_bytes()
    recorded = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))["sha256"]
    if hashlib.sha256(data).hexdigest() != recorded:
        raise ValueError(f"{path.name} does not match its recorded sha256; refusing to load it")
    return pickle.loads(data)


def features_for(raw: str, track, arena_cm: tuple[float, float], by_source: bool = False) -> np.ndarray:
    shape = silhouettes(raw, track.n_frames, source=track.source if by_source else None)
    return frame_features(shape, track.x_cm, track.y_cm, arena_cm, track.fps)


def train(rows: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]], seed: int = 0):
    """Fit on (features, label, weight) per video; frames the raters split on carry less weight."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    x = np.vstack([r[0] for r in rows])
    y = np.concatenate([r[1] for r in rows])
    w = np.concatenate([r[2] for r in rows])
    clf = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0, random_state=seed
    )
    clf.fit(x, y, sample_weight=w)
    return clf
