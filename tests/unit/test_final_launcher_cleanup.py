from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from gow.postprocess import finalize_run_launcher_tree, final_launcher_archive_path


class FakeLaunchPad:
    def get_fw_dict_by_id(self, fw_id: int):
        return {"state": "COMPLETED", "launches": [], "archived_launches": []}


@pytest.mark.skipif(shutil.which("zstd") is None, reason="zstd required")
def test_final_cleanup_archives_remaining_pilot_tree(tmp_path: Path) -> None:
    outdir = tmp_path / "results"
    run_id = "run-a"
    run_dir = outdir / "runs" / run_id
    launchers = run_dir / "launchers"
    pilot = launchers / "block_a" / "launcher-job-1"
    pilot.mkdir(parents=True)
    (pilot / "slurm.out").write_text("done", encoding="utf-8")

    gen = run_dir / "generations" / "g000000.launches.json"
    gen.parent.mkdir(parents=True)
    gen.write_text(json.dumps({"fw_ids": [10], "all_terminal": True, "fireworks": []}), encoding="utf-8")

    archive = finalize_run_launcher_tree(outdir=outdir, run_id=run_id, lp=FakeLaunchPad(), delete_source=True)
    assert archive == final_launcher_archive_path(outdir, run_id)
    assert archive.exists()
    assert not launchers.exists()


def test_final_cleanup_refuses_non_terminal_firework(tmp_path: Path) -> None:
    class RunningLaunchPad:
        def get_fw_dict_by_id(self, fw_id: int):
            return {"state": "RUNNING", "launches": [], "archived_launches": []}

    outdir = tmp_path / "results"
    run_id = "run-a"
    launchers = outdir / "runs" / run_id / "launchers"
    launchers.mkdir(parents=True)
    gen = outdir / "runs" / run_id / "generations" / "g000000.launches.json"
    gen.parent.mkdir(parents=True)
    gen.write_text(json.dumps({"fw_ids": [10]}), encoding="utf-8")

    with pytest.raises(RuntimeError, match="non-terminal"):
        finalize_run_launcher_tree(outdir=outdir, run_id=run_id, lp=RunningLaunchPad(), delete_source=True)
    assert launchers.exists()
