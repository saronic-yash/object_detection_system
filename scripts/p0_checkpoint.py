"""P0 done-gate checkpoint: prove the environment + data are correctly set up.

Renders, for one nuScenes-mini sample:
  1. BEV lidar point cloud with ground-truth boxes
  2. Front-camera image with projected 3D ground-truth boxes
  3. Lidar points projected into the front camera (calibration sanity view)

Run:  python scripts/p0_checkpoint.py
"""
import os

import matplotlib
matplotlib.use("Agg")

from nuscenes.nuscenes import NuScenes

DATAROOT = "data/nuscenes"
OUT_DIR = "outputs/p0_checkpoint"


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    nusc = NuScenes(version="v1.0-mini", dataroot=DATAROOT, verbose=False)

    scene = nusc.scene[0]
    sample = nusc.get("sample", scene["first_sample_token"])
    print(f"scene: {scene['name']} — {scene['description']}")
    print(f"sample token: {sample['token']}")
    print(f"annotations in sample: {len(sample['anns'])}")

    nusc.render_sample_data(
        sample["data"]["LIDAR_TOP"],
        with_anns=True,
        underlay_map=False,
        out_path=f"{OUT_DIR}/bev_lidar_gt.png",
        verbose=False,
    )
    print(f"wrote {OUT_DIR}/bev_lidar_gt.png")

    nusc.render_sample_data(
        sample["data"]["CAM_FRONT"],
        with_anns=True,
        out_path=f"{OUT_DIR}/cam_front_gt.png",
        verbose=False,
    )
    print(f"wrote {OUT_DIR}/cam_front_gt.png")

    nusc.render_pointcloud_in_image(
        sample["token"],
        pointsensor_channel="LIDAR_TOP",
        camera_channel="CAM_FRONT",
        out_path=f"{OUT_DIR}/lidar_in_cam_front.png",
        verbose=False,
    )
    print(f"wrote {OUT_DIR}/lidar_in_cam_front.png")


if __name__ == "__main__":
    main()
