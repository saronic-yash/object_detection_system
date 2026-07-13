import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="session")
def nusc():
    from nuscenes.nuscenes import NuScenes
    return NuScenes(version="v1.0-mini", dataroot=str(ROOT / "data" / "nuscenes"),
                    verbose=False)
