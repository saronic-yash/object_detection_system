"""P1 Exercise 1: project the lidar point cloud into all 6 cameras.

The data loading, figure layout, and oracle rendering are provided (plumbing).
The geometry — the actual exercise — is the TODO block, built from YOUR
functions in src/geometry.py and src/nuscenes_frames.py.

Run:  python scripts/p1_lidar_to_cameras.py
Then compare outputs/p1/lidar_in_all_cams.png (yours, CAM_FRONT panel)
against outputs/p1/oracle_cam_front.png (devkit). They should look identical.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from PIL import Image

from geometry import apply_transform, invert_transform, project_points
from nuscenes_frames import get_sensor_to_global

DATAROOT = os.path.join(os.path.dirname(__file__), "..", "data", "nuscenes")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "p1")

CAMERAS = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
           "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    nusc = NuScenes(version="v1.0-mini", dataroot=DATAROOT, verbose=False)
    sample = nusc.get("sample", nusc.scene[0]["first_sample_token"])

    lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    pc = LidarPointCloud.from_file(os.path.join(nusc.dataroot, lidar_sd["filename"]))
    points_lidar = pc.points[:3].T  # (N, 3) in the LIDAR sensor frame

    fig, axes = plt.subplots(2, 3, figsize=(24, 9))
    for ax, cam in zip(axes.flat, CAMERAS):
        cam_sd = nusc.get("sample_data", sample["data"][cam])
        img = Image.open(os.path.join(nusc.dataroot, cam_sd["filename"]))
        K = np.array(nusc.get("calibrated_sensor",
                              cam_sd["calibrated_sensor_token"])["camera_intrinsic"])

        # ------------------------------------------------------------------
        # TODO(you): put the lidar points into this camera's image.
        #
        # The lidar and the camera fired at (slightly) different times, so
        # they have DIFFERENT ego poses. The correct chain is:
        #
        #   lidar frame -> ego(t_lidar) -> global -> ego(t_cam) -> camera frame
        #
        # which in transform language is:
        #   T_cam_lidar = inv(T_global_cam) @ T_global_lidar
        #
        # Steps:
        #   1. T_global_lidar from get_sensor_to_global(nusc, lidar sd token)
        #   2. T_global_cam   from get_sensor_to_global(nusc, camera sd token)
        #   3. compose into T_cam_lidar, apply_transform to points_lidar
        #   4. project_points with K -> (uv, depth)
        #   5. mask: depth > 1.0, and uv inside the image (img.size = (W, H))
        #
        # Produce: uv_vis (M, 2) and depth_vis (M,) for the scatter below.
        # ------------------------------------------------------------------
        raise NotImplementedError("P1 exercise: write the projection chain")

        ax.imshow(img)
        ax.scatter(uv_vis[:, 0], uv_vis[:, 1], c=depth_vis, s=1.5, cmap="viridis")
        ax.set_title(cam)
        ax.axis("off")

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "lidar_in_all_cams.png")
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")

    # Oracle for comparison (devkit does the same chain internally)
    nusc.render_pointcloud_in_image(
        sample["token"], pointsensor_channel="LIDAR_TOP",
        camera_channel="CAM_FRONT",
        out_path=os.path.join(OUT_DIR, "oracle_cam_front.png"), verbose=False)
    print(f"wrote {os.path.join(OUT_DIR, 'oracle_cam_front.png')}")


if __name__ == "__main__":
    main()
