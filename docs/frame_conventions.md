# Frame conventions — single source of truth

> P1 deliverable, written by hand in your own words. This is the note you (and
> every later milestone) consult whenever a transform is in doubt. Fill in
> every TODO; keep it to ~one page.

## Frames in this project

TODO: list each frame (global, ego, lidar, camera, radar) — where its origin
is, where its axes point, and which nuScenes table defines it.

## Conventions

- Quaternion storage order: TODO (and why it matters — what does the devkit use?)
- Transform naming: TODO (what does `T_A_B` mean? which way do points flow?)
- Rotation handedness / direction: TODO

## The chain

TODO: write out, in both words and symbols, how a point measured by a sensor
at time t ends up in the global frame — naming the two tables involved and
what each contributes.

## Timestamps — the part that bites

TODO: why does each sample_data record carry its OWN ego_pose? What goes wrong
if you project lidar into a camera using only one ego pose? Where else will
this matter (hint: sweeps, radar at 13 Hz vs keyframes at 2 Hz)?

## Where tracking lives

TODO: which frame does the tracker (P4) operate in, and why that one?

## Yaw

TODO: how do you get a BEV yaw angle out of a box's quaternion? What range
convention will this project use, and where must angles be wrapped?
