"""Oracle + property tests for the hand-written geometry (P1).

pyquaternion and the devkit are used here as ORACLES only — the code under
test may not import them.
"""
import numpy as np
import pytest
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion

from geometry import (apply_transform, compose_poses, invert_pose,
                      invert_transform, make_transform, project_points,
                      quat_inverse, quat_multiply, quat_to_rot, transform_box)

RNG = np.random.default_rng(42)


def random_unit_quat(n=1):
    q = RNG.normal(size=(n, 4))
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def random_pose():
    q = random_unit_quat(1)[0]
    t = RNG.uniform(-50, 50, size=3)
    return q, t


# ---------------------------------------------------------------- quaternions

def test_quat_to_rot_matches_pyquaternion():
    for q in random_unit_quat(200):
        np.testing.assert_allclose(quat_to_rot(q), Quaternion(q).rotation_matrix,
                                   atol=1e-9)


def test_quat_to_rot_is_a_rotation():
    for q in random_unit_quat(50):
        R = quat_to_rot(q)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-9)


def test_quat_to_rot_known_answers():
    np.testing.assert_allclose(quat_to_rot(np.array([1.0, 0, 0, 0])), np.eye(3),
                               atol=1e-12)
    # 90 degrees about +z maps x-axis onto y-axis
    s = np.sqrt(0.5)
    R = quat_to_rot(np.array([s, 0, 0, s]))
    np.testing.assert_allclose(R @ np.array([1.0, 0, 0]), np.array([0, 1.0, 0]),
                               atol=1e-9)


def test_quat_multiply_matches_pyquaternion():
    for _ in range(100):
        a, b = random_unit_quat(2)
        expected = (Quaternion(a) * Quaternion(b)).elements
        np.testing.assert_allclose(quat_multiply(a, b), expected, atol=1e-9)


def test_quat_inverse_undoes_rotation():
    for q in random_unit_quat(50):
        prod = quat_multiply(q, quat_inverse(q))
        # identity quaternion up to sign
        assert np.isclose(abs(prod[0]), 1.0, atol=1e-9)
        np.testing.assert_allclose(prod[1:], 0, atol=1e-9)


# ------------------------------------------------------------- 4x4 transforms

def test_make_transform_layout():
    q, t = random_pose()
    T = make_transform(q, t)
    assert T.shape == (4, 4)
    np.testing.assert_allclose(T[:3, :3], quat_to_rot(q), atol=1e-12)
    np.testing.assert_allclose(T[:3, 3], t, atol=1e-12)
    np.testing.assert_allclose(T[3], [0, 0, 0, 1], atol=1e-12)


def test_invert_transform_matches_linalg_and_composes_to_identity():
    for _ in range(20):
        q, t = random_pose()
        T = make_transform(q, t)
        Ti = invert_transform(T)
        np.testing.assert_allclose(Ti, np.linalg.inv(T), atol=1e-9)
        np.testing.assert_allclose(T @ Ti, np.eye(4), atol=1e-9)


def test_apply_transform_round_trip():
    """Done-gate: transform points out and back, error < 1e-6."""
    for _ in range(20):
        q, t = random_pose()
        T = make_transform(q, t)
        pts = RNG.uniform(-100, 100, size=(1000, 3))
        back = apply_transform(invert_transform(T), apply_transform(T, pts))
        assert np.abs(back - pts).max() < 1e-6


# ------------------------------------------------------------- (q, t) poses

def test_compose_poses_matches_matrix_composition():
    for _ in range(50):
        qa, ta = random_pose()
        qb, tb = random_pose()
        q_ac, t_ac = compose_poses(qa, ta, qb, tb)
        T_expected = make_transform(qa, ta) @ make_transform(qb, tb)
        np.testing.assert_allclose(make_transform(q_ac, t_ac), T_expected,
                                   atol=1e-9)


def test_invert_pose_matches_invert_transform():
    for _ in range(50):
        q, t = random_pose()
        qi, ti = invert_pose(q, t)
        np.testing.assert_allclose(make_transform(qi, ti),
                                   invert_transform(make_transform(q, t)),
                                   atol=1e-9)


def test_transform_box_round_trip():
    for _ in range(50):
        q, t = random_pose()
        center = RNG.uniform(-100, 100, size=3)
        orientation = random_unit_quat(1)[0]
        c2, o2 = transform_box(center, orientation, q, t)
        qi, ti = invert_pose(q, t)
        c3, o3 = transform_box(c2, o2, qi, ti)
        assert np.abs(c3 - center).max() < 1e-6
        # quaternions are equal up to global sign
        assert np.isclose(abs(np.dot(o3, orientation)), 1.0, atol=1e-9)


# ------------------------------------------------------------------- cameras

def test_project_points_matches_devkit_view_points():
    K = np.array([[1266.42, 0.0, 816.27],
                  [0.0, 1266.42, 491.51],
                  [0.0, 0.0, 1.0]])
    pts = RNG.uniform(-20, 20, size=(500, 3))
    pts[:, 2] = RNG.uniform(1.0, 80.0, size=500)  # in front of the camera
    uv, depth = project_points(pts, K)
    oracle = view_points(pts.T, K, normalize=True)  # 3xN, rows = u, v, 1
    np.testing.assert_allclose(uv, oracle[:2].T, atol=1e-6)
    np.testing.assert_allclose(depth, pts[:, 2], atol=1e-12)
