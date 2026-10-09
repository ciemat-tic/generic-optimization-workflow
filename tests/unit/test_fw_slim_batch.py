from __future__ import annotations

from pathlib import Path

import yaml

from gow.fw.workflow import BatchEvalSpec, SingleEvalSpec, build_batch_evaluate_workflow


def test_batch_workflow_has_one_task_and_shared_fields_once(tmp_path: Path) -> None:
    config = {
        "id": "slim",
        "objective": {"direction": "minimize"},
        "parameters": {"x": {"type": "real", "value": 0.0, "bounds": [-1.0, 1.0]}},
        "evaluator": {"command": ["python", "-c", "print(1)"], "timeout_s": 30},
        "optimizer": {"name": "random_search", "seed": 1, "max_evaluations": 2, "batch_size": 2},
    }
    config_path = tmp_path / "problem.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    outdir = tmp_path / "out"
    items = [
        SingleEvalSpec(config_path, outdir, "r", "c0", {"x": 0.1}, generation_id=0, candidate_index=0),
        SingleEvalSpec(config_path, outdir, "r", "c1", {"x": 0.2}, generation_id=0, candidate_index=1),
    ]
    wf = build_batch_evaluate_workflow(BatchEvalSpec(config_path, outdir, "r", items))
    fw = wf.fws[0]
    assert len(fw.tasks) == 1
    task = fw.tasks[0]
    assert task["problem_config"] == str(config_path.resolve())
    assert task["outdir"] == str(outdir.resolve())
    assert task["run_id"] == "r"
    assert all("problem_config" not in item and "outdir" not in item and "run_id" not in item for item in task["items"])
    assert "candidate_ids" not in fw.spec
