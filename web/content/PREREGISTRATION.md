# Pre-registered evaluation

This file is committed before any test-set video is analysed. The commit that adds it is the timestamp: everything below was fixed before the results existed.

## Question

Does Kappa score rearing in the open-field test as well as a trained human does?

## Data

Sturman et al. (2020), *Deep learning-based behavioral analysis reaches human accuracy and is capable of outperforming commercial solutions*, Neuropsychopharmacology 45, 1942–1952.

- Videos: Zenodo record 3608658 (CC BY 4.0), 20 open-field tests, 10 min each, 928×576 px, 25 fps, one mouse, top view, 45 × 45 cm arena.
- Labels: `ETHZ-INS/DLCAnalyzer`, `data/OFT/Labels/AllLabDataOFT_final.csv` (GPL-3.0; downloaded at run time, never redistributed here). Three raters scored every bout of supported rearing, unsupported rearing, grooming and jumping, with onset and offset in seconds.

## Split

Videos are ordered by `sha256("kappa-2026:" + video_id)`.

- **Tuning set (5):** OFT_12, OFT_52, OFT_54, OFT_15, OFT_41. The candidate-detector thresholds, the adjudication prompt and the confidence cut-off may be adjusted on these only.
- **Test set (15):** OFT_16, OFT_14, OFT_6, OFT_50, OFT_43, OFT_5, OFT_51, OFT_23, OFT_49, OFT_38, OFT_58, OFT_11, OFT_44, OFT_24, OFT_39. They are analysed once, with the configuration frozen in a commit made before the first test-set run.

## Primary endpoint

Frame-level Cohen's κ for **rearing** (supported or unsupported) at 25 fps over each video's common rater window, pooled across the 15 test videos, between Kappa and each of the three raters.

**Success criterion:** the mean of Kappa's three κ values is at least the lowest of the three rater-vs-rater κ values on the same 15 videos.

For reference, the rater-vs-rater κ for rearing on all 20 videos is 0.801, 0.815 and 0.825.

## Secondary endpoints (reported, no success threshold)

- Per-video rear count: Pearson r and ICC(2,1) against the mean of the raters, next to the same figures for each rater against the mean of the other two.
- Bout-level F1 (one-to-one matching, temporal IoU ≥ 0.3).
- Supported vs unsupported rearing, split by the body centre's distance to the nearest wall at bout midpoint (12.5 cm, the convention of the original TSE setup), each against the raters' own labels.
- Distance travelled and time in the centre, compared with the values the paper reports for the commercial systems (there are no human labels for these).
- Integrity outcome: how many values were verified, locked or refused, and why.

## Negative controls

1. Kappa's rearing bouts shifted circularly by 30 s must score κ near 0 against every rater.
2. An all-false prediction must score κ = 0 and F1 = 0.
3. A two-animal clip with an injected identity swap must lock every number that depends on the swapped animals until a person resolves it.
4. A question asking for something not observable from a top-down video (for example heart rate) must be refused rather than approximated.

## Not claimed

Grooming and jumping are scored by the raters but not by Kappa in this version, so nothing is claimed about them.

## Models

Segmentation and tracking: `sam-3.1`. Candidate adjudication and protocol drafting: `muse-spark-1.3`. Run review: `muse-wonder-1.0`. All on Meta Model API.
