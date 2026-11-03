"""Train Kappa's rearing model on the pre-registered tuning videos, and only on them.

Labels are the frames where at least two of the three raters marked a rear
(supported or unsupported); frames the raters split on count half. Only frames
inside the raters' common scoring window and with the mouse in view are used. The
videos are tracked at the evaluation's frame rate, so the model sees the same
sampling it will be applied to.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from kappa.behaviour_model import MODEL_DIR, BoutRule, RearingModel, train
from kappa.eval.datasets import fetch
from kappa.eval.harness import TUNING, with_union
from kappa.eval.stats import bouts_to_frames
from kappa.eval.sturman import load_sturman
from kappa.runner import Runner
from kappa.shape import frame_features, silhouettes

RATERS = ("Furkan", "Jin", "Oliver")


def video_rows(runner: Runner, video_id: str, fps: float | None, labels) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    state = runner.load(f"sturman-{video_id}")
    analysis = runner.analysis(state)
    track = analysis.tracks[1]
    step = max(1, round(track.fps / fps)) if fps else 1
    shape = silhouettes(runner.store.get_text(state.sam_main_sha256), track.n_frames)[::step]
    rate = track.fps / step
    x, y, present = track.x_cm[::step], track.y_cm[::step], track.present[::step]
    arena = (analysis.protocol.arena.width_cm, analysis.protocol.arena.height_cm)
    features = frame_features(shape, x, y, arena, rate)
    n = len(shape)
    votes = np.sum([bouts_to_frames(labels.bouts[video_id][r]["rear"], rate, n) for r in RATERS], axis=0)
    lo, hi = labels.common_window(video_id)
    keep = present.copy()
    keep[: int(lo * rate)] = False
    keep[min(n, int(hi * rate)):] = False
    weight = np.where((votes == 0) | (votes == len(RATERS)), 1.0, 0.5)
    return features[keep], (votes >= 2)[keep], weight[keep]


def train_and_save(runner: Runner, fps: float | None, rule: BoutRule, out: Path | None = None) -> str:
    labels = with_union(load_sturman(fetch("sturman_oft_labels")))
    rows = [video_rows(runner, v, fps, labels) for v in TUNING]
    model = RearingModel(
        classifier=train(rows),
        rule=rule,
        trained_on=tuple(TUNING),
        notes=(
            "Gradient-boosted decision trees over SAM 3.1 silhouette descriptors, trained on the five "
            f"pre-registered tuning videos of Sturman et al. (2020) at {fps or 'native'} fps; labels = frames "
            "where at least two of three raters marked a rear."
        ),
    )
    return model.save(out or MODEL_DIR / "rearing_topdown.pkl")
