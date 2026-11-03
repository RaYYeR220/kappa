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

## Amendment 1 (2026-11-03, before any test-set video was analysed)

Everything below was decided on the five tuning videos only. No test-set video had been tracked, scored or looked at when this amendment was committed; the commit that adds it is the configuration freeze.

**What the tuning set showed.** Rearing candidates proposed from mask area and reviewed by Muse Wonder on zoomed clips reached a frame-level κ of 0.145 against the raters on OFT_12; scanning the whole video in 4-second windows with Muse Wonder reached 0.10. The raters agree with each other at 0.75 to 0.78 on that video. Seen from above in this grayscale, 928×576 footage, the model could not tell a rear from a walk.

**What replaces it.** Rearing is now detected frame by frame from the SAM 3.1 masks: silhouette descriptors (area, perimeter, solidity, extent, ellipse axes, radius, three Hu moments, each normalised by the animal's own rolling typical value), the animal's speed and distance to the wall, and their context over 0.2, 0.6 and 1.2 seconds go into a small gradient-boosted decision-tree classifier. Its probabilities become bouts by hysteresis (enter at 0.4, leave below 0.3, 0.28 s smoothing, bouts of at least 0.24 s, gaps under 0.2 s merged). The classifier is trained on the five tuning videos only, on frames where at least two of the three raters marked a rear; frames where the raters split count half. The trained model is committed with its sha256 (`engine/src/kappa/models/rearing_topdown.json`) and the code that trains it (`kappa train-rearing`).

**Tuning-set estimate.** Leave-one-video-out over the five tuning videos at 12.5 fps: κ 0.566, 0.384, 0.500, 0.430 and 0.694 (mean 0.515) against a rater-vs-rater mean of 0.812 on the same videos. On this evidence the primary success criterion is not expected to be met. It is kept unchanged and will be reported as it comes out.

**Frame rate.** Videos are tracked and scored at 12.5 fps (every other frame). This halves the GPU time of the self-hosted segmenter, and a rear still spans several frames. Agreement is still computed at 25 fps from bout times in seconds, as pre-registered.

**Models.** Segmentation runs on SAM 3.1 open weights (`sam3.1_multiplex.pt`, sha256 `0567debe…` as published by Meta) on Kappa's own GPU endpoint, which returns the Model API output format. Arena corners are located by Muse Spark Open (perception grounding). Run review uses Muse Wonder. Behaviour adjudication by a language model is no longer part of the rearing detector.

**Added secondary endpoint.** Agreement on rear counts and total rearing time per video (Pearson r and ICC(2,1) against the mean rater), beside the same figures for each rater against the others.
