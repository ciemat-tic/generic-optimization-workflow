from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from gow.postprocess import AsyncGenerationArchiver, generation_archive_report_path


@pytest.mark.skipif(shutil.which("zstd") is None, reason="zstd required")
def test_async_archiver_deletes_only_archived_leaves(tmp_path: Path) -> None:
    outdir = tmp_path / "results"
    run_id = "run-a"
    run_dir = outdir / "runs" / run_id
    candidate = run_dir / "candidate-1"
    candidate.mkdir(parents=True)
    (candidate / "result.json").write_text('{"ok": true}', encoding="utf-8")

    pilot = run_dir / "launchers" / "block_a" / "launcher-job"
    leaf = pilot / "launcher-task"
    leaf.mkdir(parents=True)
    (leaf / "FW.json").write_text("{}", encoding="utf-8")

    manifest = run_dir / "generations" / "g000000.launches.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({
        "all_terminal": True,
        "fireworks": [{
            "terminal": True,
            "launches": [{"state": "COMPLETED", "launch_dir": str(leaf)}],
        }],
    }), encoding="utf-8")

    archiver = AsyncGenerationArchiver(
        outdir=outdir,
        run_id=run_id,
        delete_candidate_workdirs=True,
        delete_launcher_dirs=True,
        max_pending=1,
    )
    archiver.submit(generation_id=0, candidate_ids=["candidate-1"], launch_manifest=manifest)
    archiver.close()

    report = json.loads(generation_archive_report_path(outdir, run_id, 0).read_text(encoding="utf-8"))
    candidate_archive = Path(report["candidate_archive"])
    launcher_archive = Path(report["launcher_archive"])
    assert candidate_archive.exists()
    assert launcher_archive.exists()
    assert not candidate.exists()
    assert not leaf.exists()
    assert pilot.exists(), "pilot parent must remain alive for future FireWorks"
    assert subprocess.run(["zstd", "-t", "-q", str(candidate_archive)]).returncode == 0
    assert subprocess.run(["zstd", "-t", "-q", str(launcher_archive)]).returncode == 0
