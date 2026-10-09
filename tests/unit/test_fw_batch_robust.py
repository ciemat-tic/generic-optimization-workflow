from __future__ import annotations

import json
from pathlib import Path

import yaml

from gow.candidate_ids import format_candidate_id
from gow.fw.tasks import EvaluateBatchTask, rebuild_problem_results_jsonl, rebuild_run_results_jsonl


def _write_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def test_evaluate_batch_task_continues_after_single_candidate_exception(tmp_path: Path, monkeypatch) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    config = {
        "id": "toy-fw-batch-robust",
        "objective": {"direction": "minimize"},
        "parameters": {
            "x": {"type": "real", "value": 0.0, "bounds": [-1.0, 1.0]},
        },
        "evaluator": {
            "command": ["python", "-c", "print('unused in monkeypatched test')"],
            "timeout_s": 30,
        },
        "optimizer": {
            "name": "random_search",
            "seed": 123,
            "max_evaluations": 2,
            "batch_size": 2,
        },
    }

    config_yaml = project_dir / "problem.yaml"
    _write_yaml(config_yaml, config)

    outdir = tmp_path / "out"
    run_id = "robust-batch-run"
    candidate_ok = format_candidate_id(0, 0, run_id=run_id)
    candidate_fail = format_candidate_id(0, 1, run_id=run_id)

    ok_record = {
        "problem_id": "toy-fw-batch-robust",
        "run_id": run_id,
        "candidate_id": candidate_ok,
        "candidate_local_id": "g000000_c000000",
        "attempt_id": f"{candidate_ok}_a000",
        "generation_id": 0,
        "candidate_index": 0,
        "attempt_index": 0,
        "params": {"x": 0.1},
        "fitness": {
            "status": "ok",
            "metrics": {"objective": 0.01},
            "objective": 0.01,
            "constraints": {},
            "artifacts": {},
            "error": None,
            "failure_kind": None,
        },
        "failure_kind": None,
        "returncode": 0,
        "wall_time_s": 0.001,
        "started_at": "2026-01-01T00:00:00Z",
        "finished_at": "2026-01-01T00:00:00Z",
        "evaluator": {"resolved_command": ["python", "ok.py"]},
        "workdir": str(outdir / "runs" / run_id / candidate_ok),
        "stdout_path": str(outdir / "runs" / run_id / candidate_ok / "stdout.txt"),
        "stderr_path": str(outdir / "runs" / run_id / candidate_ok / "stderr.txt"),
        "input_path": str(outdir / "runs" / run_id / candidate_ok / "input.json"),
        "output_path": str(outdir / "runs" / run_id / candidate_ok / "output.json"),
    }

    def fake_evaluate_one_candidate(**kwargs):
        candidate_id = kwargs["candidate_id"]
        if candidate_id == candidate_fail:
            raise RuntimeError("boom for batch candidate")

        workdir = Path(kwargs["outdir"]) / "runs" / kwargs["run_id"] / candidate_id
        workdir.mkdir(parents=True, exist_ok=True)
        record = dict(ok_record)
        record.update(
            {
                "candidate_id": candidate_id,
                "candidate_local_id": "g000000_c000000",
                "attempt_id": f"{candidate_id}_a000",
                "generation_id": kwargs["generation_id"],
                "candidate_index": kwargs["candidate_index"],
                "params": {"x": kwargs["candidate_params"]["x"]},
                "workdir": str(workdir),
                "stdout_path": str(workdir / "stdout.txt"),
                "stderr_path": str(workdir / "stderr.txt"),
                "input_path": str(workdir / "input.json"),
                "output_path": str(workdir / "output.json"),
            }
        )
        (workdir / "result.json").write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
        return record

    monkeypatch.setattr("gow.fw.tasks._evaluate_one_candidate", fake_evaluate_one_candidate)

    items = [
        {
            "candidate_id": candidate_ok,
            "candidate_params": {"x": 0.1},
            "generation_id": 0,
            "candidate_index": 0,
            "attempt_index": 0,
        },
        {
            "candidate_id": candidate_fail,
            "candidate_params": {"x": 0.2},
            "generation_id": 0,
            "candidate_index": 1,
            "attempt_index": 0,
        },
    ]

    action = EvaluateBatchTask(
        {
            "problem_config": str(config_yaml.resolve()),
            "outdir": str(outdir.resolve()),
            "run_id": run_id,
            "items": items,
        }
    ).run_task({})
    assert action.stored_data == {"evaluated": 2, "failed": 1}
    assert action.update_spec == {}

    ok = json.loads((outdir / "runs" / run_id / candidate_ok / "result.json").read_text(encoding="utf-8"))
    failed_result_path = outdir / "runs" / run_id / candidate_fail / "result.json"
    assert failed_result_path.exists()
    failed_payload = json.loads(failed_result_path.read_text(encoding="utf-8"))
    assert failed_payload["candidate_id"] == candidate_fail
    assert failed_payload["fitness"]["status"] == "failed"
    assert failed_payload["failure_kind"] == "internal_error"

    assert not (outdir / "results.jsonl").exists()
    assert not (outdir / "runs" / run_id / "results.jsonl").exists()
    run_rows = _read_jsonl(rebuild_run_results_jsonl(outdir, run_id))

    assert [r["candidate_id"] for r in run_rows] == [candidate_ok, candidate_fail]
    assert run_rows[1]["fitness"]["status"] == "failed"
    assert run_rows[1]["failure_kind"] == "internal_error"

    problem_rows = _read_jsonl(rebuild_problem_results_jsonl(outdir))
    assert [r["candidate_id"] for r in problem_rows] == [candidate_ok, candidate_fail]
    assert problem_rows[1]["fitness"]["status"] == "failed"
    assert problem_rows[1]["failure_kind"] == "internal_error"
