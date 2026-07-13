"""Integration tests: the hand-written frame chain against real nuScenes-mini
records, with the devkit as oracle. These are the P1 done-gate tests.
"""
import numpy as np
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion
from scipy.spatial import cKDTree

from geometry import invert_pose, transform_box
from nuscenes_frames import (accumulate_sweeps, get_sensor_pose,
                             get_sensor_to_global)


def _oracle_sensor_to_global(nusc, sd_token):
    sd = nusc.get("sample_data", sd_token)
    cal = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    ego = nusc.get("ego_pose", sd["ego_pose_token"])
    return (transform_matrix(ego["translation"], Quaternion(ego["rotation"]))
            @ transform_matrix(cal["translation"], Quaternion(cal["rotation"])))


def test_sensor_to_global_matches_devkit(nusc):
    sample = nusc.get("sample", nusc.scene[0]["first_sample_token"])
    for chan in ("LIDAR_TOP", "CAM_FRONT", "RADAR_FRONT"):
        sd_token = sample["data"][chan]
        np.testing.assert_allclose(get_sensor_to_global(nusc, sd_token),
                                   _oracle_sensor_to_global(nusc, sd_token),
                                   atol=1e-9)
        q, t = get_sensor_pose(nusc, sd_token)
        from geometry import make_transform
        np.testing.assert_allclose(make_transform(q, t),
                                   _oracle_sensor_to_global(nusc, sd_token),
                                   atol=1e-9)


def test_gt_box_round_trip_global_to_sensor_and_back(nusc):
    """Done-gate: every GT box of a real sample survives global -> lidar-frame
    -> global with < 1e-6 error."""
    sample = nusc.get("sample", nusc.scene[0]["first_sample_token"])
    q_gs, t_gs = get_sensor_pose(nusc, sample["data"]["LIDAR_TOP"])
    q_sg, t_sg = invert_pose(q_gs, t_gs)
    assert len(sample["anns"]) > 0
    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        center = np.array(ann["translation"])
        orientation = np.array(ann["rotation"])
        c_sensor, o_sensor = transform_box(center, orientation, q_sg, t_sg)
        c_back, o_back = transform_box(c_sensor, o_sensor, q_gs, t_gs)
        assert np.abs(c_back - center).max() < 1e-6
        assert np.isclose(abs(np.dot(o_back, orientation)), 1.0, atol=1e-9)


def test_accumulate_sweeps_matches_devkit(nusc):
    """Done-gate: 10-sweep accumulation within 1 cm of the devkit oracle."""
    # a sample mid-scene so 10 previous sweeps exist
    sample = nusc.get("sample", nusc.scene[0]["first_sample_token"])
    for _ in range(15):
        sample = nusc.get("sample", sample["next"])

    mine = accumulate_sweeps(nusc, sample, nsweeps=10, min_distance=1.0)

    oracle_pc, _ = LidarPointCloud.from_file_multisweep(
        nusc, sample, chan="LIDAR_TOP", ref_chan="LIDAR_TOP",
        nsweeps=10, min_distance=1.0)
    oracle = oracle_pc.points[:3].T

    assert mine.shape == oracle.shape, (
        f"point count mismatch: mine {mine.shape} vs oracle {oracle.shape}")
    d_mine_to_oracle, _ = cKDTree(oracle).query(mine, k=1)
    d_oracle_to_mine, _ = cKDTree(mine).query(oracle, k=1)
    assert d_mine_to_oracle.max() < 0.01, "points >1 cm from oracle cloud"
    assert d_oracle_to_mine.max() < 0.01, "oracle points >1 cm from your cloud"
