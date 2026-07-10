# 3D Multi-Object Detection & Tracking on nuScenes — Project Plan
**Native Apple Silicon · From-scratch KF/tracker/fusion · nuScenes tracking leaderboard target**

## Context

A learning project whose end state is a complete, benchmarked, leaderboard-submitted, tri-modal (lidar + radar + camera) 3D multi-object tracking system — built so the owner can explain and defend every core autonomy component (Kalman filter, data association, sensor fusion) in interviews. The 3D detector is deliberately taken from a library (understood, not reimplemented); the KF, association logic, and fusion are hand-written with AI tools as tutor only. Everything runs natively on Apple Silicon macOS; the simulator project on Windows comes later and is out of scope.

**Calibration (from user):** ~15 hrs/week sustained · M-series Max/Ultra, 48GB+ RAM · zero cloud spend planned — pretrained weights and published artifacts only (cloud GPU rental remains the *noted fallback* if training ever became unavoidable; the plan is designed so it never does) · tracker-before-fusion ordering approved · math/ML foundations still developing, so **understanding, not typing, is the rate limiter** — learning steps are budgeted as first-class work, and estimates assume a first-timer's debugging velocity, not an expert's.

---

## Research-verified facts this plan is built on (checked July 2026)

1. **The nuScenes tracking eval server (EvalAI challenge 476) is open year-round** and still active — but allows **1 account and only 3 test submissions per user per year**, which must be genuinely different methods. Test submissions are a scarce resource; all iteration happens locally on val.
2. **Official val evaluation needs metadata only** (`v1.0-trainval_meta.tgz`, 0.46 GB) plus your results JSON — no sensor blobs. Ground truth is read from annotation tables.
3. **Published detection files exist for train/val/test**: CenterPoint detections (MIT SharePoint — the modern standard every 2022–2025 tracking paper builds on) and MEGVII/PointPillars/Mapillary detections (hosted on nuScenes.org — durable fallback). The tracking leaderboard scores the *tracker*; building on public detections is standard, disclosed practice.
4. **Sparse-conv detectors can never run natively on a Mac** (spconv is CUDA/Windows-only; mmcv has no MPS ops beyond BBoxOverlaps), and **training anything real on MPS is impractical** (CPU fallbacks, no sparse conv, documented silent-correctness bugs in optimizer paths). But **pillar-based models run inference CPU-only**: a known-good zero-compile pin exists (Python 3.11 + torch 2.1.2 + prebuilt mmcv 2.1.0 arm64 wheel + mmdet3d 1.4.0), and **CenterPoint-pillar02** (val mAP 48.7 / NDS 59.6, 23 MB checkpoint) is the best Mac-runnable detector in MMDetection3D. Caveat: MMDetection3D is frozen upstream (last release Jan 2024) — treat it as a vendored artifact.
5. **Disk is a non-issue on the chosen path**: mini 4.2 GB + trainval metadata 0.46 GB + detection JSONs + **trainval radar 2.2 GB** + test metadata 71 MB + test radar 0.4 GB ≈ **under 10 GB for the entire core project including radar fusion and the leaderboard submission**. Only running your own detector on full val would need the 135 GB lidar blobs (optional, off the critical path).
6. **The devkit pip package excludes the tracking eval code** — you must `git clone nuscenes-devkit` and put `python-sdk` on `PYTHONPATH` for AMOTA evaluation. The devkit also pins `numpy<2.0` (use a dedicated env).
7. **Realistic AMOTA expectations on val with CenterPoint detections** (numbers are only comparable on the same detections): the original AB3DMOT IoU-association baseline scored ~0.18 (MEGVII dets; reported numbers vary by config — the durable lesson is that IoU gating starves at nuScenes' 2 Hz). A naive greedy-L2 constant-velocity tracker lands ~0.55–0.60. Chiu et al.'s published Mahalanobis-KF result is **0.561 val — on weaker MEGVII detections**; on CenterPoint detections the research band for a clean hand-written KF tracker is **~0.55–0.64 first-pass**, and **0.63–0.66 is an extrapolated target after tuning + SimpleTrack-style upgrades** (two-stage association, coasted output), not a verified given. CenterPoint's own tracker = 0.637; SimpleTrack = 0.687; 2025 SOTA ≈ 0.73. Reproducing 0.60–0.66 from scratch is unambiguous success.
8. **AMOTA silently punishes format mistakes**: the tracker must emit confidence-ranked results *including coasted (unmatched, motion-predicted) tracks at low score* (SimpleTrack: 0.01 × previous confidence), track-level scores, a constant class per track, tracks reset at scene boundaries, ≤500 boxes/sample. Getting this wrong is the #1 reason correct-looking from-scratch trackers score far below where they should.
9. **Radar fusion has a clean linear path**: nuScenes radar provides ego-motion-compensated velocities (`vx_comp`/`vy_comp`), so radar can update the KF's velocity states with a plain linear measurement update — **no EKF required**. Radar has no elevation (z always 0), is sparse, comes from **5 separate sensors** that must be transformed and merged, and needs quality filtering (`pdh0`, `ambig_state`, `invalid_state`).
10. **Camera fusion without images**: nuScenes.org hosts Mapillary *camera-based* detection JSONs for train/val/test — enabling camera-modality fusion on full val and test with zero image downloads. **Honest caveat:** Mapillary detections are weak (29.8 mAP vs CenterPoint's 48.7), so expect a ~zero AMOTA delta from the camera leg on clean data; its value is the fusion design story and the robustness experiments.
11. For robustness, ready-made references exist: **Occluded nuScenes (Oct 2025)** ships parameterized lidar/radar/camera degradation scripts (radar Gaussian noise there is on point *coordinates*, σ 0.1–2.0 m); MultiCorrupt defines the standard beam-reduction scheme; ego-pose Gaussian noise sweeps are the standard ad-hoc GPS-denial protocol. Report an mRR-style "% of clean performance retained" column.

---

## Strategy: three load-bearing decisions

**1. Dual detection strategy.** Run a library detector natively for the *learning* goal (P2, on mini); feed the tracker with *published CenterPoint detections* for all full-split work and the leaderboard (disclosed via the submission's meta flags — standard practice). This decouples the riskiest toolchain item from the critical path, and tracker AMOTA then reflects tracking skill, not Mac-constrained detector quality.

**2. Metadata-first data plan.** The core loop (tracker dev → official val AMOTA) costs < 1 GB. Sensor data is pulled only where a milestone demands it: mini (4.2 GB) for development, trainval radar (2.2 GB) for fusion, test metadata + radar (< 0.5 GB) for submission. All published detection files and the trainval metadata are archived in week 1 — the CenterPoint SharePoint links can rot, and P4 needs the train-split files.

**3. Spend test submissions like gold.** 3 per year. Slot 1: lidar-only from-scratch tracker (LiDAR track). Slot 2: the fused tracker (Open track — a genuinely different method; radar fusion alone qualifies, camera delta is a bonus not a gate). Slot 3: reserve. Every submission JSON is validated locally with the devkit first.

---

## Milestones

Each lists prerequisite learning (hours included in the estimate), tasks, a testable **Done** gate (objective criteria gate; self-assessed items are marked as stretch), and a conservative range. **The from-scratch rule (owner-written code, AI as tutor only) applies to P1 transforms, P3, P4, and P7** — P1 is included because hand-writing the transforms is the vaccine for this plan's most insidious risk.

### P0 — Environment, data, and artifact archive — **15–25 h (cum. wk 1–1.5)**
- **Learning prereq (2–3 h):** nuScenes schema — scene / sample / sample_data / sample_annotation / calibrated_sensor / ego_pose tables and how they link.
- Miniforge; **tracker env** (Python 3.12): latest PyTorch (≥2.9; MPS verified with a tensor op), numpy 1.26.x (devkit pins `<2.0`), scipy, matplotlib, pyquaternion; `pip install nuscenes-devkit` **plus `git clone nuscenes-devkit`** with `python-sdk` on `PYTHONPATH` (pip package lacks the tracking eval).
- Downloads (resumable — aria2c or `aws s3 cp --no-sign-request s3://motional-nuscenes/...`; Tokyo origin can be slow; verify checksums): **nuScenes-mini (4.2 GB)** and **v1.0-trainval_meta.tgz (0.46 GB — P4's covariance statistics need it, so it comes home now)**. Work through the devkit tutorial notebook.
- **Archive the fallback artifacts (budget 2–3 h, not 30 min):** MEGVII detection zip (nuscenes.org), CenterPoint train/val/test detection JSONs (MIT SharePoint — browser-driven download, GB-scale train file), Mapillary camera detections. Keep a backup copy. **Load-test every file** (`json.load` + sample-token count) **and confirm each zip actually contains train/val/test splits as expected** — a silently missing train file wouldn't otherwise surface until P4.
- **Done (gate):** BEV lidar render of a mini sample with GT boxes + camera image with projected 3D GT boxes; MPS op confirmed; all detection files + trainval metadata archived, load-tested, split-verified.

### P1 — Learning block: 3D geometry & coordinate frames — **25–40 h (cum. wk 2.5–4.5)**
- **Learning (10–15 h):** rotation matrices, quaternions, homogeneous transforms; the frame chain **sensor → ego(t) → global**; timestamps and ego motion between sweeps.
- **Exercises (owner-written, devkit's `map_pointcloud_to_image` as the checking oracle):**
  1. Project lidar points into all 6 cameras (colored by depth).
  2. Transform GT boxes global↔ego↔sensor; verify round-trips with unit tests.
  3. Accumulate 10 lidar sweeps into one frame with ego-motion compensation — exactly the input detectors consume.
- **Done (gate):** round-trip transform unit tests assert error < 1e-6; static-scene multi-sweep alignment check passes at a stated cm-level threshold (visual "no smearing" as the sanity view, not the gate). One-page frame-conventions note exists (tracking in the global frame, yaw wrapped to [-π, π]) — the project's single source of truth.

### P2 — Library detector running + understood — **30–45 h (cum. wk 4.5–7.5)**
- **Learning (8–12 h):** voxel/pillar encoding; PointPillars paper; CenterPoint paper (center heads, velocity regression); NMS; detection scores. Output: a one-page "how my detector works" note (interview artifact).
- **Detector env** (separate, Python 3.11 — the only version with a prebuilt arm64 mmcv wheel): torch 2.1.2 + mmcv 2.1.0 (prebuilt universal2 wheel from the OpenMMLab index — zero compilation) + mmengine 0.10.x + mmdet 3.2.0 + mmdet3d 1.4.0 from source (`pip install -v -e .`, pure Python). **Timebox install fights to ~10 h.** MMDetection3D is frozen upstream — expect to self-patch stray `.cuda()` calls; never wait for fixes.
- Model: **CenterPoint-pillar02 + circle-NMS** (23 MB checkpoint, val mAP 48.7 / NDS 59.6 — the best detector that runs natively on Apple Silicon, CPU-only inference; sparse-conv "voxel" variants can never run on Mac). PointPillars-FPN as simpler fallback. **Benchmark per-frame CPU latency day 1** — no published M-series numbers exist (expect ~0.5–3 s/frame); also record it for P9.
- Run inference on mini; visualize pred vs GT (reuse P0/P1 viz). Build the **canonical detection interface** (detections per sample token, global frame) with two producers: own detector output; loader for the archived detection JSONs.
- **Fallback ladder if the timebox blows:** ① run a small **pure-PyTorch PointPillars** with a pretrained checkpoint on mini instead, producing the *same* Done artifacts (the "run + understand a real detector" goal survives); ② ONNX inference via onnxruntime if a pre-exported model is available (the mmdeploy export step itself needs Linux). Either way, full-split work continues on archived detections — zero critical-path loss.
- **Done (gate):** side-by-side pred-vs-GT visualization on mini; own detections serialized in nuScenes results-JSON format; archived detections load through the same interface; latency benchmark recorded; "how it works" note written. **← Morale checkpoint: a 3D detector running natively on Apple Silicon is a shareable artifact.**

### P3 — Learning block + build: Kalman filter from scratch — **40–60 h (cum. wk 7.5–11.5)**
- **Learning (20–30 h, scoped to prevent completionism):** Gaussians, covariance, conditional/marginal, Bayes; KF predict/update at intuition level (Kalman gain as an uncertainty-weighted average). Text: Labbe, *Kalman and Bayesian Filters in Python* (free, notebook-based) — **through the multivariate KF chapters only; skip smoothers/UKF/particle filters.** For a profile with developing probability foundations, genuinely absorbing this is the milestone — budget accordingly.
- Implement a KF class from scratch in numpy (float64, CPU — MPS has no float64 and the KF is microseconds anyway): constant-velocity model, configurable dims, predict/update.
- **Synthetic validation before nuScenes:** 1D position+velocity from noisy positions; 2D maneuvering target; `filterpy` as *test oracle only*; NIS (normalized innovation squared) consistency check (timeboxed — it's a check, not a required derivation); deliberately mis-tune Q/R to watch divergence and build tuning intuition.
- **Done (gate):** KF matches the filterpy oracle within stated tolerance on all synthetic scenarios; NIS falls in the chi-square 95% band when well-tuned. *(Stretch, interview prep: write and explain predict/update from memory.)*

### P4 — Tracker from scratch (lidar-only) — **60–90 h total (cum. wk 11.5–17.5)**
Split into two sub-milestones — the base tracker is the critical path; the comparison study is not.

**P4a — Base tracker, format-correct, evaluated on mini_val — 45–65 h**
- **Learning (8–12 h):** tracking-by-detection; AB3DMOT paper (and *why* IoU association starves at 2 Hz); Chiu et al. 2020 (the design to copy); CenterPoint's greedy tracker; SimpleTrack's ID-switch analysis (~95% of IDS are premature terminations).
- The Chiu et al. design, every piece hand-written:
  - 11-D state [x, y, z, yaw, l, w, h, vx, vy, vz, vyaw]; 7-D measurement; H = [I₇ | 0].
  - **Statistics-driven covariances — a named sub-task (10–15 h inside P4a):** R = Var(detection − matched GT), Q = Var(GT acceleration), per class, computed from the archived train detections + trainval metadata (downloaded in P0). Chiu's `covariance.py` hard-coded values are the sanity oracle and the emergency shortcut.
  - Association: Mahalanobis distance with chi-square gating, **greedy matching** (beats Hungarian in Chiu's ablation).
  - Yaw handling (top bug source): wrap to [-π, π] after every predict/update; 180° flip correction when detection-vs-predicted yaw differs by (90°, 270°).
  - Lifecycle: min-hits = 3 / max-age = 2 (Chiu operating point).
  - **AMOTA-correct output from day one:** coasted tracks at score 0.01 × previous confidence; track-level (averaged) scores; constant class per track; tracks reset at scene boundaries (online rule); ≤500 boxes/sample.
- Build the BEV track-animation debug view (IDs + trails) — the highest-value association-debugging tool. **← Morale checkpoint: the track animation is a shareable video.**
- **Done (gate):** stable IDs visually on mini scenes; association edge cases unit-tested (no dets, no tracks, all-gated-out); **official devkit TrackingEval runs end-to-end on mini_val** (archived val detections filtered to those scenes) — proving the output format before scaling; design note written.

**P4b — Association comparison + SimpleTrack upgrades — 15–25 h (may interleave with P5 tuning)**
- Implement the CenterPoint-style velocity-based greedy L2 matcher; compare Mahalanobis-greedy vs velocity-L2 vs Hungarian (`scipy.linear_sum_assignment` as the solver baseline — the *solver* is a primitive; the association logic is yours). This comparison table is the learning goal made concrete and an interview artifact.
- SimpleTrack upgrades: NMS pre-processing on detections (BEV IoU ~0.1); **two-stage association** (rescue unmatched tracks with low-score detections — the single cheapest large IDS reduction).
- **Done (gate):** comparison table on mini_val/val; two-stage association measurably reduces IDS.

### P5 — Close the loop: official val metrics + tuning — **25–40 h (cum. wk 13–20)**
- **Learning (4–6 h):** AMOTA/AMOTP exactly (MOTAR averaged over 40 recall points, 2 m center-distance matching), IDS/FRAG, why track scores drive everything.
- Run the tracker over all 150 val scenes on archived CenterPoint val detections; official `TrackingEval`. **Record the wall-clock time of one full val run+eval — P8's matrix is costed from this number.**
- **Expectation management is part of the milestone:** the first full-val run is the *simple greedy-L2 tracker*, whose expected ladder position is ~0.55 — a low first number is a ladder rung, not a failure. mini_val (2 scenes) proves format only; per-class failures (motorcycle, bicycle, trailer) surface only at full-val scale. Debug with a pre-written triage checklist ordered by likelihood: coasted-track output → track-level scores → class constancy → scene reset → box cap.
- Iterate (per-class gates, max-age, score handling, two-stage thresholds) with a tuning log.
- **Target ladder (CenterPoint detections; if ever forced onto the MEGVII fallback, rebase everything to Chiu's 0.561 band):** ≥ 0.55 first pass = on track; **≥ 0.60 after upgrades = the P5 gate**; 0.63–0.66 = headline goal (matching CenterPoint's own 0.637 tracker); ~0.68 = stretch (SimpleTrack territory).
- Create the EvalAI account and read current submission rules **now** (not in P6).
- **Done (gate):** one reproducible script prints the full official val metrics table; AMOTA ≥ 0.60; per-class results inspected and worst classes understood; single-run wall-clock recorded; tuning log kept. **← Morale checkpoint: first full-val AMOTA is a celebrated number, whatever it is.**

### P6 — Leaderboard submission #1: lidar-only baseline — **15–25 h (cum. wk 14–22)** ← **SHIP POINT 1**
- Learning prereq: none.
- Test metadata (71 MB) + archived CenterPoint test detections; run tracker; produce submission JSON; **validate locally with devkit checks (zero errors) before upload**; write the required method description (5+ sentences, team info, measured FPS); submit to the **LiDAR track**. Spends slot 1 of 3 — deliberately.
- **Done (gate):** entry visible on the public leaderboard; test AMOTA within ~0.05 absolute of val AMOTA; screenshot taken. **Emergency-fallback line reached (~3.5–5 months in) — but note: this alone falls short of the project's tri-modal spec.**

### P7 — Sensor fusion from scratch (radar + camera) — **55–85 h (cum. wk 17.5–27.5)**
Fusion is **required** for the complete project — it's what makes the pipeline tri-modal. Three sub-parts:
- **Learning (10–15 h):** radar measurement model — sparse point targets, Doppler → `vx_comp`/`vy_comp` (ego-compensated, Cartesian, hence a *linear* measurement), no elevation, RCS, quality flags; 13 Hz radar vs 2 Hz keyframes (time-alignment by `time_lag` before any update); early/late/deep fusion taxonomy (discuss the space, defend the late-fusion choice).
- **P7a — Radar → KF (the substantive from-scratch fusion math), ~30–45 h:** download trainval radar (2.2 GB). Named tasks, each a real chunk: transform + merge the **5 radar sensors** into the global frame; quality-filter (`pdh0`/`ambig_state`/`invalid_state`); associate radar points to tracks (BEV position gate + velocity-consistency gate — a second association problem with its own bug class); build a **second measurement model (new H, new R)** updating velocity states from `vx_comp`/`vy_comp` — ignore radar z, down-weight/skip radar position. Linear update; no EKF needed (and you can explain why).
  **AMOTA-independent validation (critical for a first-timer):** per-track velocity RMSE vs GT before/after radar updates — so "is my fusion buggy?" is answerable without waiting on a small AMOTA delta.
- **P7b — Camera fusion (detection-level by design — the 177 GB image blobs stay untouched), ~10–15 h:** score/class refinement of tracks using the archived **Mapillary camera detection JSONs** (val + test) — confirmation boosts confidence, absence attenuates. Expect ~zero clean-data AMOTA delta (Mapillary is 19 mAP weaker than CenterPoint); the value is the design story + P8. **Named deliverable (not optional): a 2D-projection confirmation-gating demo on mini images**, so raw camera data demonstrably enters the pipeline at least once.
- **P7c — Open-track submission, ~8–15 h (its own process, same rigor as P6):** best fused config on test (test radar = 0.4 GB); radar fusion alone qualifies as a genuinely different method; local validation; method description; slot 2 of 3.
- **Done (gate):** before/after table on **full val**: AMOTA/AMOTP/IDS/FRAG for lidar-only vs +radar vs +radar+camera; radar velocity-RMSE improvement demonstrated; fusion design note; Open-track entry live.

### P8 — Robustness layer + degradation table — **30–45 h focused + overnight batch runs (cum. wk 19.5–30.5)** ← **SHIP POINT 2 (complete project)**
- **Learning (2–4 h):** corruption-benchmark conventions — MultiCorrupt / Occluded nuScenes protocols, mRR ("% of clean retained") reporting.
- Build an **injection layer** wrapping the canonical inputs (tracker code untouched). Two injection surfaces, stated honestly:
  - *Detection/pose/radar-level (full val — the headline experiments):*
    1. **Sensor dropout:** lidar-detection frames dropped at p ∈ {0.1, 0.3, 0.5}; **radar dropout; camera-detection dropout; single-survivor cases (lidar-only / radar+camera-only)** — each an explicit matrix row.
    2. **Noise:** Gaussian on detection centers/sizes/yaw (σ sweep); radar point-coordinate noise σ ∈ {0.1, 0.5, 2.0 m} (matching the Occluded nuScenes protocol); optionally radar velocity noise σ ∈ {0.1, 0.5, 1 m/s} as an own-designed extra (justified against the sensor's ~0.03 m/s accuracy).
    3. **GPS/localization denial:** ego-pose corruption — translation noise σ ∈ {0.1, 0.5, 2 m} + yaw noise, and a bias/random-walk drift variant. Tracking lives in the global frame; this stresses exactly what GPS denial stresses.
  - *Point-cloud-level (own detector on mini — smaller demo):* point dropout and MultiCorrupt-style beam reduction (32→16→8 by ring index). References: Occluded nuScenes scripts, thu-ml 3D_Corruptions_AD.
- **Cost the matrix before running it:** ~27–30 cells × (val-run wall-clock recorded in P5, ~10–30 min each) ≈ 5–13 h of compute — so build a **scripted batch runner with cached, named per-cell results** and run overnight. **Seed policy:** severity sweeps at 1 seed; the stochastic cells (dropout, noise) at 3 seeds, reported as mean.
- **Done (gate):** the degradation table — rows = {clean, each degradation × severity}, configs = {lidar-only, fused}, columns = AMOTA / AMOTP / IDS / FRAG + **% of clean AMOTA retained (mRR-style)** — plus a 1-page analysis earning (or honestly failing to earn) the claim *"the fused tracker degrades more gracefully."* Final README (architecture, results, reproduction steps). **Complete: tri-modal, benchmarked, submitted, robustness-quantified.**

### P9 — Optimization pass (stretch — explicitly optional) — **15–30 h (beyond SHIP 2)**
- Learning prereq: none.
- **Primary (achievable, on-theme):** end-to-end profiling — time voxelization / backbone / NMS / association / KF separately; vectorize the tracker's hot loops (sub-ms/frame in numpy is reachable); per-stage latency table.
- **Secondary (stretch):** hand-rolled MPS pipeline — pillar voxelization in ~50 lines of vectorized PyTorch + dense 2D backbone on MPS + torchvision's MPS NMS; CPU-vs-MPS latency comparison. (Stays on the Mac; is itself a hand-written component.)
- **Tertiary (timebox ~10 h):** Core ML export of the dense 2D backbone via coremltools 9 (`torch.jit.trace` path — `torch.export` is still beta); CPU/GPU/ANE backbone latency. Known limits: voxelization/NMS/tracker stay outside Core ML; full-model ANE lidar pipelines don't exist. Any success is bonus. (mmdeploy ONNX export requires Linux — excluded by the Mac-only constraint.)
- **Done (gate):** latency table + short writeup framing the story honestly (tracker is µs–ms; the detector dominates).

---

## Critical path & dependencies

```
P0 ──▶ P1 ──▶ [detections: archived files (P0) or own detector (P2)] ──▶ P4a ──▶ P5 ──▶ P6
                        │                                                 ▲        ▲
                        └── P2 install pain? fallback ladder —            │        │
                            zero critical-path delay                      │   P4b interleaves
P3 (KF) ──────────────────────────────────────────────────────────────────┘
P7 (fusion: radar 2.2 GB + Mapillary JSONs) ← needs P4/P5 · required for SHIP 2 · gates slot-2 submission
P8 (robustness) ← needs P5's eval harness + P7's fused config
P9 ← needs P2; independent of everything else
```

- **Critical path:** P0 → P1 → detections → P3 → P4a → P5 → P6 → P7 → P8. Detections are available from week 1 (archived), so **P2 is off the critical path by construction**.
- Overlap P3 with P2 **only during unattended waits** (downloads, overnight inference runs). If the install fight is active, finish or timebox it first — first-contact Bayesian filtering cannot be learned in fragments between debugging sessions. Keep whole sessions single-topic.
- EvalAI account + rules reading happens in P5; every upload is preceded by local devkit validation (3/year budget).

## Timeline at 15 h/week (conservative)

| Milestone | Hours | Cumulative weeks (hrs ÷ 15) |
|---|---|---|
| P0 Environment + archive | 15–25 | 1–1.5 |
| P1 Geometry & frames | 25–40 | 2.5–4.5 |
| P2 Detector 🏁 | 30–45 | 4.5–7.5 |
| P3 Kalman filter | 40–60 | 7.5–11.5 |
| P4 Tracker (P4a+P4b) 🏁 | 60–90 | 11.5–17.5 |
| P5 Val eval + tuning 🏁 | 25–40 | 13–20 |
| **P6 Leaderboard #1 — SHIP 1 (emergency-fallback line)** | 15–25 | **14–22 (~3.5–5 mo)** |
| P7 Fusion + Open-track submission | 55–85 | 17.5–27.5 |
| **P8 Robustness — SHIP 2 (complete project)** | 30–45 | **19.5–30.5 (~4.5–7 mo)** |
| P9 Optimization (optional) | 15–30 | +1–2 wks |

🏁 = named morale checkpoints (shareable artifacts) — deliberately placed in the weeks-6–14 trough where heavy math meets low visible progress and abandonment risk peaks.

**Total core (P0–P8): ~295–455 h.** At a *realized* 15 h/wk that is **~4.5–7 months; plan on the upper half** — sustained side-project hours typically realize at ~80% of intention across holidays, crunches, and life. These ranges price in first-timer debugging velocity and Apple-Silicon friction. If you run ahead, enjoy it; don't plan on it.

## Risk register (prioritized)

| # | Risk | Likelihood / Impact | Mitigation |
|---|---|---|---|
| 1 | **AMOTA output-format traps** — missing coasted-track output, per-frame instead of track-level scores, class switches, no scene-boundary reset | Near-certain if unaware / High (silently tanks score) | Format rules encoded in P4a's Done; official eval on mini_val *before* full val; P5 triage checklist; devkit validation before every submission. |
| 2 | **Coordinate-frame & calibration bugs** — wrong frame, inverted transform, timestamp mismatch; silently wrong | Near-certain / High (insidious) | P1 is the vaccine: owner-written transforms, round-trip unit tests, static-scene alignment gate, frame-conventions note; *visualize before trusting*; all tracking in the global frame. |
| 3 | **Learning pace on math-heavy steps** (KF, geometry, metrics) | Likely / Medium | Learning hours are budgeted line items, scoped reading lists, raised estimates (P3 = 40–60 h). If a concept runs over, cut from the flex list (risk 5) — never cut the understanding; it's the point. |
| 4 | **KF divergence / yaw bugs** — unwrapped angles, 180° flips, mis-scaled Q/R, association errors poisoning the state | Likely / Medium | Synthetic validation + NIS checks first; statistics-driven Q/R with Chiu's published values as oracle; yaw wrap + flip correction as explicit tasks. |
| 5 | **Scope creep / schedule collapse** | High / High | Two ship points; overbuild-trap list; flex list for cuts = P4b breadth, the 2D-projection demo, P9 — **not** radar/camera fusion (tri-modal is the spec) and not learning time. If the schedule truly collapses, SHIP 1 + a lidar-only robustness slice is the explicit emergency fallback, acknowledged as falling short of the tri-modal spec. |
| 6 | **Training on MPS is impractical** — key ops fall back to CPU, no sparse conv, documented silent-correctness bugs in optimizer paths; a full detector train would take days-to-weeks | Certain (documented) / None on chosen path | The plan requires **zero training by design** (pretrained checkpoint + published detections). If fine-tuning ever became genuinely unavoidable: rent a cloud GPU for a few hours and bring weights back to the Mac (the noted fallback — currently outside budget by user decision), else drop the feature. |
| 7 | **Test-submission scarcity** — 3/year, 1 account/year | Certain (rule) / High if squandered | Slot plan (baseline → fused → reserve); local devkit validation before every upload; all tuning on val. |
| 8 | **Detection-file link rot** — CenterPoint files live on MIT SharePoint | Possible / High if late | Archive in week 1 (P0) with split-verification + backup. If forced onto the MEGVII fallback: **rebase all AMOTA targets to Chiu's 0.561 band** — the ladder's absolute numbers hold only on CenterPoint detections. |
| 9 | **MMDetection3D on Apple Silicon** — frozen upstream, version-pin hell, stray `.cuda()` calls | Medium (exact pin known) / Medium (off critical path) | Zero-compile pin (Py 3.11 / torch 2.1.2 / mmcv 2.1.0 wheel / mmdet3d 1.4.0); pillar models only; 10 h timebox; fallback ladder with substitute Done. |
| 10 | **Env/tooling papercuts** — pip devkit lacks tracking eval; `numpy<2.0` pin; Python-version wheel traps | Certain (documented) / Low (now known) | Clone devkit for eval; two dedicated envs (tracker: Py 3.12; detector: Py 3.11); pins recorded in P0/P2. |
| 11 | **Slow downloads / slow CPU inference** — Tokyo-hosted blobs; ~0.5–3 s/frame detector | Possible / Low (chosen path avoids both) | Metadata-first plan (<10 GB); resumable downloads + checksums; own detector only on mini or overnight batches; detections always cached to JSON. |
| 12 | **ID switches / AMOTA plateau** | Possible / Medium | Two-stage association (the researched highest-leverage fix — ~95% of IDS are premature terminations); per-class gates; error analysis on worst scenes with the BEV animation. |
| 13 | **Fusion deltas small on clean data** (radar modest, camera ~zero — morale + debuggability risk) | Likely / Low-Med | Expected and documented in advance; **AMOTA-independent validation** (velocity RMSE vs GT) separates "buggy" from "modest"; the payoff is the P8 graceful-degradation story — the better interview story anyway. |

## Where to stop and ship

**The complete project = SHIP POINT 2:** tri-modal from-scratch tracker (lidar detections + radar-in-KF fusion + camera detection-level fusion) · official val metrics · two leaderboard entries (LiDAR + Open track) · the quantitative degradation table with dropout (all three modalities), noise, and GPS-denial rows · README. That is the resume claim the plan is built to earn.

**Emergency fallback = SHIP POINT 1 + a lidar-only robustness slice** (dropout + pose noise + detection noise on the lidar-only config). Explicitly labeled: this falls short of the tri-modal spec — use only if the schedule genuinely collapses. It is still a leaderboard-submitted, robustness-tested project.

**Overbuild traps — explicitly out of scope:**
- Implementing the Hungarian algorithm's internals (scipy for the baseline; greedy — which you do hand-write — beats it here anyway).
- Training or fine-tuning any detector (zero training by design; cloud GPU is the noted never-expected fallback).
- Chasing leaderboard *rank* — a defensible, disclosed entry is the goal (SOTA ≈ 0.73; you are not competing with it).
- Learned appearance/ReID features (Chiu 2021-style) — a separate ML project.
- Fancy interactive visualizers, UKF/IMM/CTRA motion models, BEVFusion-class detectors, simulator work.

## Verification

Each milestone's **Done gate** is the verification: objective, testable criteria (unit-tested transforms with stated tolerances, oracle-matched KF, official devkit metrics on mini_val → val → leaderboard with a stated val-to-test tolerance, velocity-RMSE fusion check, the degradation table). Self-assessed items are marked stretch. The project-level acceptance test is SHIP POINT 2: tri-modal, benchmarked, submitted, robustness-quantified, reproducible from the README.

## Key resources (verified July 2026)

- Devkit + tracking rules/eval: `github.com/nutonomy/nuscenes-devkit` (tracking README = submission schema, rules, tracks, detection links)
- Leaderboard: EvalAI challenge 476 · data: `s3://motional-nuscenes` (public, `--no-sign-request`)
- Detections: CenterPoint (tianweiy/CenterPoint → MIT SharePoint links); MEGVII/PointPillars/Mapillary (nuscenes.org/data/...)
- Tracker designs: Chiu et al. arXiv:2001.05673 + `eddyhkchiu/mahalanobis_3d_multi_object_tracking` (covariance.py, get_nuscenes_stats.py); CenterPoint arXiv:2006.11275; SimpleTrack arXiv:2111.09621; AB3DMOT arXiv:1907.03961
- KF learning: Labbe, *Kalman and Bayesian Filters in Python* (free)
- Robustness: Occluded nuScenes arXiv:2510.18552 (scripts); MultiCorrupt (`ika-rwth-aachen/MultiCorrupt`); `thu-ml/3D_Corruptions_AD`
- Detector on Mac: mmdet3d v1.4.0 + mmcv 2.1.0 cp311 universal2 wheel (`download.openmmlab.com/mmcv/dist/cpu/torch2.1/`); config `centerpoint_pillar02_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d`
