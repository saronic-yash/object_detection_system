"""P1 Exercise 3: visualize your 10-sweep ego-motion-compensated accumulation.

Uses YOUR accumulate_sweeps (src/nuscenes_frames.py). Plots a single sweep
next to the 10-sweep accumulation. What to look for:

  - Static world (walls, poles, parked cars): razor sharp. If it smears,
    your ego-motion compensation is wrong.
  - Moving objects: smeared into short trails. That is physics, not a bug —
    detectors exploit exactly this to regress velocity.

Run:  python scripts/p1_sweep_accumulation.py [scene_idx] [sample_idx]
Quantitative gate: pytest tests/test_nuscenes_chain.py -k accumulate
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud

from nuscenes_frames import accumulate_sweeps

DATAROOT = os.path.join(os.path.dirname(__file__), "..", "data", "nuscenes")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "p1")


def bev(ax, pts, title):
    ax.scatter(pts[:, 0], pts[:, 1], s=0.05, c=pts[:, 2], cmap="viridis",
               vmin=-2, vmax=6)
    ax.set_xlim(-50, 50); ax.set_ylim(-50, 50)
    ax.set_aspect("equal"); ax.set_title(title)


def main() -> None:
    scene_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    sample_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 15

    os.makedirs(OUT_DIR, exist_ok=True)
    nusc = NuScenes(version="v1.0-mini", dataroot=DATAROOT, verbose=False)
    sample = nusc.get("sample", nusc.scene[scene_idx]["first_sample_token"])
    for _ in range(sample_idx):
        if not sample["next"]:
            break
        sample = nusc.get("sample", sample["next"])

    lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    single = LidarPointCloud.from_file(
        os.path.join(nusc.dataroot, lidar_sd["filename"])).points[:3].T

    accumulated = accumulate_sweeps(nusc, sample, nsweeps=10)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    bev(ax1, single, f"single sweep ({len(single):,} pts)")
    bev(ax2, accumulated, f"10 sweeps, ego-motion compensated "
                          f"({len(accumulated):,} pts)")
    fig.tight_layout()
    out = os.path.join(OUT_DIR, f"sweep_accumulation_s{scene_idx}.png")
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
