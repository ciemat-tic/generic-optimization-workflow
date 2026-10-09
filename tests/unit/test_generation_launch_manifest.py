from __future__ import annotations

import json
from pathlib import Path

from gow.postprocess import write_generation_launch_manifest


class Launch:
    def __init__(self, launch_id: int, state: str, launch_dir: str):
        self.launch_id = launch_id
        self.state = state
        self.launch_dir = launch_dir


class FakeLaunchPad:
    def __init__(self, root: Path):
        self.root = root

    def get_fw_dict_by_id(self, fw_id: int):
        if fw_id == 10:
            return {"state": "COMPLETED", "launches": [100], "archived_launches": []}
        return {"state": "FIZZLED", "launches": [101], "archived_launches": [99]}

    def get_launch_by_id(self, launch_id: int):
        return Launch(launch_id, "COMPLETED" if launch_id != 101 else "FIZZLED", str(self.root / f"launcher-{launch_id}"))


def test_manifest_keeps_initial_and_retry_fireworks(tmp_path: Path) -> None:
    outdir = tmp_path / "results"
    path = write_generation_launch_manifest(
        outdir=outdir, run_id="r1", generation_id=7, fw_ids=[11, 10, 11], lp=FakeLaunchPad(tmp_path)
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["fw_ids"] == [10, 11]
    assert data["generation_id"] == 7
    assert data["all_terminal"] is True
    assert [row["fw_id"] for row in data["fireworks"]] == [10, 11]
    assert data["fireworks"][1]["launches"][0]["launch_dir"].endswith("launcher-101")
    assert data["fireworks"][1]["launches"][1]["archived"] is True
