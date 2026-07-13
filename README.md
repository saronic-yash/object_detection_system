# 3D Multi-Object Detection & Tracking on nuScenes

A 3D multi-object detection and tracking system built on the [nuScenes](https://www.nuscenes.org) autonomous-driving dataset, targeting the public nuScenes tracking leaderboard — running natively on Apple Silicon macOS (PyTorch/MPS, no CUDA).

The core autonomy components are **written from scratch** as a learning exercise: the Kalman filter, the data-association / track-lifecycle logic, and the camera–lidar–radar sensor-fusion strategy. Only the 3D object detector comes from a library (MMDetection3D, CenterPoint-pillar), plus published public detection files for full-split evaluation. A robustness layer injects sensor dropout, localization (GPS) denial, and noise, and quantifies graceful degradation before/after fusion.

**Full project plan:** [PROJECT_PLAN.md](PROJECT_PLAN.md) — milestones, critical path, risk register, and timeline.

## Status

- ✅ **P0 — Environment + data checkpoint:** Apple Silicon env (Python 3.12, PyTorch w/ MPS), nuScenes-mini + trainval metadata loaded, official tracking eval verified, all published detection files (CenterPoint / MEGVII / PointPillars / Mapillary) archived and split-verified.
- 🔨 **P1 — 3D geometry & coordinate frames** (in progress): hand-written sensor→ego→global transforms, multi-sweep lidar accumulation. Spec: `src/geometry.py` + `src/nuscenes_frames.py` stubs, gated by `tests/`.

## Data

The `data/` directory is intentionally not committed — nuScenes terms don't permit redistribution. To reproduce:

1. Register (free, non-commercial) at [nuscenes.org](https://www.nuscenes.org/nuscenes#download).
2. Download `v1.0-mini.tgz` and `v1.0-trainval_meta.tgz` (public S3 mirror: `s3://motional-nuscenes`, `--no-sign-request`) and extract both into `data/nuscenes/`.
3. Detection files: MEGVII / PointPillars / Mapillary from nuscenes.org (`https://www.nuscenes.org/data/detection-<name>.zip`) into `data/detections/<name>/`; CenterPoint detections from the [CenterPoint repo](https://github.com/tianweiy/CenterPoint)'s links into `data/detections/centerpoint/`.

## Environment

```bash
mamba create -n nuscenes -c conda-forge python=3.12 "numpy<2" scipy matplotlib jupyterlab tqdm pandas pytorch
mamba run -n nuscenes pip install nuscenes-devkit motmetrics filterpy
```

(PyTorch from conda-forge, not pip, to avoid a duplicate-OpenMP crash with conda's numpy on Apple Silicon.)

Sanity check: `python scripts/p0_checkpoint.py` renders a BEV lidar view and camera view with ground-truth boxes from nuScenes-mini.

## Acknowledgments

- [nuScenes devkit](https://github.com/nutonomy/nuscenes-devkit) (Apache 2.0) — dataset API and official evaluation; `notebooks/nuscenes_tutorial.ipynb` is a lightly-patched copy of its tutorial.
- Public detection files: [CenterPoint](https://github.com/tianweiy/CenterPoint) (Yin et al.), MEGVII (Zhu et al.), and the nuScenes baseline releases.
