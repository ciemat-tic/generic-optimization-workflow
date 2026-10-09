from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from gow.candidate_ids import format_candidate_id
from gow.cli import fw_run_cmd, _build_missing_result_record


class _OptCfg:
    name = "random_search"
    seed = 123
    max_evaluations = 2
    batch_size = 2

    def model_dump(self):
        return {
            "name": self.name,
            "seed": self.seed,
            "max_evaluations": self.max_evaluations,
            "batch_size": self.batch_size,
            "settings": {},
        }


class _Problem:
    id = "toy-queue-retry"
    objective = SimpleNamespace(direction="minimize")
    optimizer = _OptCfg()

    def runtime_params(self):
        return {}


class _FakeOptimizer:
    def __init__(self):
        self.tell_calls = []

    def ask(self, problem, n_batch):
        return [{"x": 0.1}, {"x": 0.2}][:n_batch]

    def tell(self, candidates, fitness_dicts):
        self.tell_calls.append((list(candidates), list(fitness_dicts)))


class _FakeMongoAdmin:
    def command(self, name: str):
        assert name == "hello"
        return {
            "maxBsonObjectSize": 16 * 1024 * 1024,
            "maxMessageSizeBytes": 48_000_000,
            "maxWriteBatchSize": 100_000,
        }


class _FakeMongoClient:
    admin = _FakeMongoAdmin()


class _FakeLP:
    def __init__(self):
        self.next_id = 100
        self.connection = _FakeMongoClient()

    def add_wf(self, wf):
        self.next_id += 1
        return {0: self.next_id}

    def get_fw_by_id(self, fw_id: int):
        return SimpleNamespace(state="FIZZLED")


def test_build_missing_result_record_persists_result_json_only(tmp_path: Path) -> None:
    problem = _Problem()
    run_id = "run-a"
    candidate_id = format_candidate_id(0, 0, run_id=run_id)

    record = _build_missing_result_record(
        problem=problem,
        results_dir=tmp_path,
        run_id=run_id,
        candidate_id=candidate_id,
        candidate_params={"x": 0.5},
        generation_id=0,
        candidate_index=0,
        attempt_index=2,
        reason="node lost",
    )

    result_path = tmp_path / "runs" / run_id / candidate_id / "result.json"
    assert result_path.exists()
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["fitness"]["status"] == "failed"
    assert payload["failure_kind"] == "missing_result_after_retries"
    assert record["attempt_index"] == 2
    assert not (tmp_path / "results.jsonl").exists()


def test_fw_run_cmd_retries_missing_queue_candidates_then_rebuilds_global(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = tmp_path / "problem.yaml"
    config.write_text("id: dummy\n", encoding="utf-8")

    fake_problem = _Problem()
    fake_lp = _FakeLP()
    fake_optimizer = _FakeOptimizer()
    submitted_attempt_indexes: list[list[int]] = []

    monkeypatch.setattr("gow.cli.load_problem_config", lambda path: fake_problem)
    monkeypatch.setattr("gow.fw.launchpad.load_launchpad", lambda _: fake_lp)
    monkeypatch.setattr("gow.optimizer.make_optimizer", lambda *args, **kwargs: fake_optimizer)
    monkeypatch.setattr("gow.cli._launch_fireworks", lambda **kwargs: None)
    monkeypatch.setattr(
        "gow.cli._wait_for_batch_results",
        lambda *args, **kwargs: (False, list(kwargs.get("candidate_ids", args[3] if len(args) > 3 else []))),
    )

    class _FakeWorkflow:
        fws = []

        def to_db_dict(self):
            return {"name": "fake-workflow"}

    def fake_build_batch(spec):
        submitted_attempt_indexes.append([item.attempt_index for item in spec.items])
        return _FakeWorkflow()

    monkeypatch.setattr("gow.fw.workflow.build_batch_evaluate_workflow", fake_build_batch)

    fw_run_cmd(
        config=config,
        launchpad=None,
        outdir=tmp_path / "out",
        run_id="run-q",
        launch=True,
        launch_dir=None,
        sleep=0,
        nlaunches=0,
        launcher="queue",
        qadapter=None,
        njobs_queue=10,
        reserve=False,
        group_size=2,
        queue_wait=True,
        queue_poll_seconds=1,
        queue_wait_timeout=5,
        max_missing_result_retries=1,
    )

    assert submitted_attempt_indexes == [[0, 0], [1, 1]]
    assert len(fake_optimizer.tell_calls) == 1
    _, fitness_dicts = fake_optimizer.tell_calls[0]
    assert [fit["status"] for fit in fitness_dicts] == ["failed", "failed"]
    assert [fit["failure_kind"] for fit in fitness_dicts] == [
        "missing_result_after_retries",
        "missing_result_after_retries",
    ]

    candidate_ids = [
        format_candidate_id(0, 0, run_id="run-q"),
        format_candidate_id(0, 1, run_id="run-q"),
    ]
    for candidate_id in candidate_ids:
        result_path = tmp_path / "out" / "runs" / "run-q" / candidate_id / "result.json"
        assert result_path.exists()
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        assert payload["attempt_index"] == 2
        assert payload["fitness"]["status"] == "failed"

    run_results = (tmp_path / "out" / "runs" / "run-q" / "results.jsonl").read_text(encoding="utf-8")
    assert candidate_ids[0] in run_results
    assert candidate_ids[1] in run_results

    global_results = (tmp_path / "out" / "results.jsonl").read_text(encoding="utf-8")
    assert candidate_ids[0] in global_results
    assert candidate_ids[1] in global_results
