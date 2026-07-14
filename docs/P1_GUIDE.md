# P1 Working Guide — 3D Geometry & Coordinate Frames

Your reference for the current milestone. Everything you need without scrolling
through the chat. Budget: ~25–40 h. The done-gate is at the bottom.

**The rule for this milestone:** every function body in `src/geometry.py` and
`src/nuscenes_frames.py` is written BY YOU. AI explains concepts and decodes
errors; it does not write these bodies. `pyquaternion` / devkit appear only in
tests, as oracles. If you're stuck on the same bug for 45+ minutes, ask.

---

## Running the tests

```bash
conda activate nuscenes
cd ~/Documents/object_detection_system    # repo root (habit — scripts assume it)

pytest -k quat_to_rot -q     # just quat_to_rot (3 tests)
pytest -k quat -q            # all quaternion tests (5)
pytest tests/test_geometry.py -q    # all pure-geometry tests (11)
pytest -q                    # everything incl. dataset integration tests (15)
```

Starting state: everything fails with `NotImplementedError`. You turn them
green stub by stub. The harness itself was validated against a reference
implementation before you started — **if a test fails, it's the code, not the
test.**

Reading failures: the assertion diff is the clue. All-random-quats wrong =
formula bug. Only known-answers wrong = sign/direction bug. Passes
orthonormality but fails oracle = you built the transpose (inverse rotation).

---

## Session plan

| # | Do | Gate |
|---|---|------|
| 1 | Watch 3B1B *Essence of Linear Algebra* ch. 3–4; Ben Eater quaternion series (eater.net/quaternions — do the interactive parts). Implement `quat_to_rot`, `quat_multiply`, `quat_inverse`. | `pytest -k quat -q` → 5 passed |
| 2 | On paper: derive pose inverse + composition. Implement `make_transform`, `invert_transform`, `apply_transform`, `compose_poses`, `invert_pose`, `transform_box`, `project_points`. | `pytest tests/test_geometry.py -q` → 11 passed |
| 3 | Implement `get_sensor_pose`, `get_sensor_to_global` (src/nuscenes_frames.py). | `pytest -k "sensor_to_global or box_round_trip" -q` green |
| 4 | Fill the TODO block in `scripts/p1_lidar_to_cameras.py`. | Your CAM_FRONT panel visually identical to `outputs/p1/oracle_cam_front.png` |
| 5 | Implement `accumulate_sweeps`. Run `scripts/p1_sweep_accumulation.py`, study the output. | `pytest -k accumulate -q` green (1 cm vs oracle) |
| 6 | Write `docs/frame_conventions.md` from memory, in your own words. | Every TODO in it filled |

Commit after each green gate: `git add -A && git commit -m "..."` — this
history is your portfolio evidence.

---

## Concepts (the short version)

### Frames and transforms
A frame is a point of view; the same physical point has different coordinates
in each. A transform converts between them. Naming discipline (non-negotiable,
it prevents the #2 project risk): `T_A_B` takes coordinates in frame B to
frame A, i.e. `p_A = T_A_B @ p_B`. Chains cancel like dominoes:
`T_global_lidar = T_global_ego @ T_ego_lidar`. Frame bugs don't crash — they
silently put everything 2 m to the left. Visualize before trusting.

### Quaternions — [w, x, y, z], always
A unit quaternion encodes axis-angle rotation in 4 numbers:

```
w = cos(θ/2)          (x, y, z) = axis · sin(θ/2)
```

- Identity (no rotation): `[1, 0, 0, 0]`
- 90° about z: `[√0.5, 0, 0, √0.5]` — sends x̂ to ŷ
- **q and −q are the SAME rotation** (double cover). Tests compare
  `|dot| ≈ 1`, not elements. This resurfaces in P4 as the 180° yaw-flip issue.
- Composition = Hamilton product, same right-to-left order as matrices.
  In scalar/vector form: `(w₁,v₁)⊗(w₂,v₂) = (w₁w₂ − v₁·v₂,  w₁v₂ + w₂v₁ + v₁×v₂)`
- Inverse of a unit quaternion = conjugate: `[w, −x, −y, −z]`.
- Storage-order trap: nuScenes/pyquaternion are w-first; scipy is x,y,z,w.
  We are w-first everywhere. Never use scipy Rotation here.

### quat_to_rot
Columns of R = where the rotation sends x̂, ŷ, ẑ (3B1B ch. 3). Standard formula:

```
      ⎡ 1−2(y²+z²)   2(xy−wz)     2(xz+wy)  ⎤
R  =  ⎢ 2(xy+wz)     1−2(x²+z²)   2(yz−wx)  ⎥
      ⎣ 2(xz−wy)     2(yz+wx)     1−2(x²+y²)⎦
```

Structure: diagonal entry for axis i is `1 − 2(sum of squares of the OTHER
two)`. Off-diagonals are mirror pairs — same product term, w-term sign FLIPS
across the diagonal ((0,1) has −wz, (1,0) has +wz). Consistent wrong signs =
transpose = inverse rotation = passes orthonormality but fails known-answers.

Hand-checks before running pytest: identity quat → `np.eye(3)` by inspection;
90°-about-z → column 1 must be (0, 1, 0). Build the matrix as ONE
`np.array([[...],[...],[...]])` literal so the code looks like the formula.

Bugs already survived (keep dodging them): missing `+` before a paren reads as
a function call (`'numpy.float64' object is not callable`); `int()`/`dtype=int`
silently truncates cos/sin values to 0 and ±1; rotation matrices are 3×3.

### Rigid transforms (4×4 homogeneous)
`p' = R p + t`, packed as `[[R, t], [0 0 0 1]]` so composition = matmul.
Inverse: solve `p' = R p + t` for p — do it on paper; the answer gives you
`invert_transform` (rotation part transposes; translation is NOT just −t) and
the same derivation in (q, t) form gives `invert_pose`. `compose_poses`: push
a point through both poses on paper, read off the result — translation is NOT
`t₁ + t₂`. `apply_transform` must be vectorized ((N,3) in, (N,3) out, no
Python loop — hint: `pts @ R.T + t`... derive why the transpose appears).

### Pinhole projection
Camera frame: +z forward. `uvw = K @ p`, then divide by z (depth):
`u = uvw[0]/z, v = uvw[1]/z`. Return all points' (u,v) and depth; the CALLER
masks depth > small-positive and image bounds. Must match devkit
`view_points(..., normalize=True)`.

### The nuScenes chain (the one idea everything sits on)
Every `sample_data` record links to:
- `calibrated_sensor` → pose (ego ← sensor): where the sensor is bolted on
  the car. Fixed per log.
- `ego_pose` → pose (global ← ego) **at that record's timestamp**.

Compose → (global ← sensor). Order: which pose acts on a sensor point first?

**Timestamps are the trap:** every record has its OWN ego_pose because every
sensor fires at a different instant, and the car moves between instants
(~20 cm at highway speed). Lidar→camera is therefore a FOUR-link chain:

```
lidar → ego(t_lidar) → global → ego(t_cam) → camera
T_cam_lidar = inv(T_global_cam) @ T_global_lidar
```

### Sweep accumulation (exercise 3)
Walk `sample_data['prev']` backwards from the keyframe lidar for 10 sweeps.
Each sweep: load points (`LidarPointCloud.from_file` — I/O is allowed),
`pc.remove_close(1.0)` (devkit helper, allowed — its criterion is a |x|,|y|
box, don't hand-roll it), then chain sweep-sensor → global → ref-sensor using
each sweep's own ego pose. Concatenate newest-first. Gate: within 1 cm of
devkit `from_file_multisweep`.

What the picture should show: static world razor-sharp (smearing = your
ego-motion compensation is wrong); moving objects smeared into trails (that's
physics — detectors use exactly this to regress velocity).

---

## Done-gate (P1 complete when ALL of these hold)

- [ ] `pytest -q` → 15/15 green
- [ ] `outputs/p1/lidar_in_all_cams.png` CAM_FRONT panel matches oracle
- [ ] Sweep-accumulation render studied; you can explain sharp-vs-smeared
- [ ] `docs/frame_conventions.md` fully written, from memory, own words
- [ ] Self-test you can pass out loud: why does each sample_data carry its own
      ego_pose, and what breaks if you ignore that?
