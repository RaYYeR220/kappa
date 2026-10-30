import json
import math
import os
from pathlib import Path

import pytest

from kappa.eval import datasets
from kappa.eval.sturman import (
    compare_video,
    human_agreement,
    load_sturman,
    normalise_behaviour,
)

# Synthetic rows in the published format (semicolon-separated, quoted; times in seconds).
OFT_LIKE = """\
"file";"Experimenter";"from";"to";"type";"ID";"DLCFile"
"V1_A.csv";"Ann";2;3;"Start/End";"V1";"x.csv"
"V1_A.csv";"Ann";4;6;"Supported";"V1";"x.csv"
"V1_A2.json";"Ann";10;11;"Unsupported";"V1";"x.csv"
"V1_A2.json";"Ann";12;13;"_DEFAULT";"V1";"x.csv"
"V1_A.csv";"Ann";20;25;"Grooming";"V1";"x.csv"
"V1_A.csv";"Ann";598;599;"Start/End";"V1";"x.csv"
"V1_B.csv";"Bob";1.5;2;"StartEnd";"V1";"x.csv"
"V1_B.csv";"Bob";4.1;6.1;"Supported";"V1";"x.csv"
"V1_B.csv";"Bob";30;NA;"NA";"V1";"x.csv"
"V1_B.csv";"Bob";40;40.5;"Jumping";"V1";"x.csv"
"V1_B.csv";"Bob";600;600.5;"StartEnd";"V1";"x.csv"
"V2_A.csv";"Ann";1;2;"StartEnd";"V2";"y.csv"
"V2_A.csv";"Ann";5;7;"Supported";"V2";"y.csv"
"V2_A.csv";"Ann";10;11;"Supported";"V2";"y.csv"
"V2_A.csv";"Ann";500;501;"StartEnd";"V2";"y.csv"
"V2_B.csv";"Bob";3;4;"StartEnd";"V2";"y.csv"
"V2_B.csv";"Bob";5;7;"Supported";"V2";"y.csv"
"V2_B.csv";"Bob";10.2;11;"Supported";"V2";"y.csv"
"""

EPM_LIKE = """\
file;Experimenter;from;to;type;ID;DLCFile
E1_Jin.json;Jin;0;1;Start_End;E1;e.csv
E1_Jin.json;Jin;2.5;3;Head Dip;E1;e.csv
E1_Jin.json;Jin;4;6;Protected Stretch;E1;e.csv
E1_Jin.json;Jin;299;300;Start/End;E1;e.csv
"""


@pytest.fixture
def oft_csv(tmp_path):
    p = tmp_path / "labels.csv"
    p.write_text(OFT_LIKE, encoding="utf-8")
    return p


def test_behaviour_names_are_normalised():
    assert normalise_behaviour("Supported") == "supported_rear"
    assert normalise_behaviour("Unsupported") == "unsupported_rear"
    assert normalise_behaviour("Grooming") == "grooming"
    assert normalise_behaviour("Jumping") == "jumping"
    assert normalise_behaviour("Head Dip") == "head_dip"
    for marker_or_placeholder in ("Start/End", "StartEnd", "Start_End", "_DEFAULT", "NA", ""):
        assert normalise_behaviour(marker_or_placeholder) is None


def test_loader_groups_bouts_by_video_rater_behaviour(oft_csv):
    labels = load_sturman(oft_csv)
    assert labels.videos == ("V1", "V2")
    assert labels.raters == ("Ann", "Bob")
    ann = labels.bouts["V1"]["Ann"]
    assert ann["supported_rear"] == [(4.0, 6.0)]
    assert ann["unsupported_rear"] == [(10.0, 11.0)]  # second annotation file, same rater
    assert ann["grooming"] == [(20.0, 25.0)]
    assert ann["jumping"] == []  # scored by someone in the file, so present (empty) for everyone
    assert labels.bouts["V1"]["Bob"]["jumping"] == [(40.0, 40.5)]
    assert labels.n_skipped_rows == 2  # the "_DEFAULT" placeholder and the NA row


def test_loader_reads_each_raters_window(oft_csv):
    labels = load_sturman(oft_csv)
    assert labels.windows["V1"] == {"Ann": (2.0, 599.0), "Bob": (1.5, 600.5)}
    assert labels.windows["V2"]["Bob"] is None  # only one marker: the end is unknown
    assert labels.common_window("V1") == (2.0, 599.0)
    assert labels.common_window("V2") == (1.0, 501.0)


def test_loader_handles_the_unquoted_epm_file(tmp_path):
    p = tmp_path / "epm.csv"
    p.write_text(EPM_LIKE, encoding="utf-8")
    labels = load_sturman(p)
    assert labels.bouts["E1"]["Jin"] == {
        "head_dip": [(2.5, 3.0)],
        "protected_stretch": [(4.0, 6.0)],
    }
    assert labels.windows["E1"]["Jin"] == (0.0, 300.0)


def test_human_agreement_on_synthetic_labels(oft_csv):
    agreement = human_agreement(
        load_sturman(oft_csv), fps=25.0, behaviours=("supported_rear", "grooming")
    )
    (pair,) = agreement.pairs["supported_rear"]
    assert (pair.a, pair.b, pair.n_videos) == ("Ann", "Bob", 2)
    assert 0.8 < pair.kappa < 1.0
    assert pair.count_r == pytest.approx(1.0)  # both score 1 bout on V1 and 2 on V2
    assert agreement.kappa_range("supported_rear") == (pair.kappa, pair.kappa)
    (groom,) = agreement.pairs["grooming"]
    assert groom.kappa == 0.0  # Ann grooms, Bob never does
    json.dumps(agreement.to_dict(), allow_nan=False)


def test_compare_video_uses_the_common_window(oft_csv):
    labels = load_sturman(oft_csv)
    rep = compare_video(labels, "V1", "supported_rear", [(4.0, 6.0)], fps=25.0)
    assert rep.offset_s == 2.0
    assert rep.n_frames == round((599.0 - 2.0) * 25)
    assert {p.b: p.kappa for p in rep.vs_raters}["Ann"] == 1.0
    assert rep.within_human_range is True


# ---------------------------------------------------------------- the real Sturman et al. 2020 labels


def _real_labels() -> Path | None:
    env = os.environ.get("KAPPA_STURMAN_OFT_LABELS")
    if env:
        return Path(env) if Path(env).is_file() else None
    cached = datasets.default_cache_dir() / datasets.SOURCES["sturman_oft_labels"].filename
    return cached if cached.is_file() else None


REAL = _real_labels()


@pytest.mark.skipif(
    REAL is None,
    reason="Sturman OFT labels not available (set KAPPA_STURMAN_OFT_LABELS or fetch them)",
)
def test_reproduces_published_human_agreement_on_real_labels():
    labels = load_sturman(REAL)
    assert len(labels.videos) == 20 and labels.raters == ("Furkan", "Jin", "Oliver")
    agreement = human_agreement(labels, fps=25.0)

    def rounded(rng):
        return tuple(round(v, 2) for v in rng)

    assert rounded(agreement.kappa_range("supported_rear")) == (0.85, 0.89)
    assert rounded(agreement.kappa_range("unsupported_rear")) == (0.68, 0.75)
    assert rounded(agreement.kappa_range("grooming")) == (0.67, 0.73)
    assert rounded(agreement.count_r_range("supported_rear")) == (0.93, 0.95)
    assert rounded(agreement.count_r_range("unsupported_rear")) == (0.87, 0.94)
    assert all(not math.isnan(p.bout_f1) for pairs in agreement.pairs.values() for p in pairs)
