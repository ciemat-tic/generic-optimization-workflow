from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

from gow.candidate_ids import format_attempt_id, parse_candidate_id
from gow.config import load_problem_config
from gow.evaluation import evaluate_candidate
from gow.layout import candidate_workdir, run_root as run_root_dir
from gow.postprocess import iter_generation_shards
from gow.output.jsonl import append_jsonl_line


def _to_jsonable(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return str(obj)


def _ensure_fireworks_imports():
    try:
        from fireworks import FiretaskBase, FWAction, explicit_serialize  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "FireWorks is not installed. Install with: pip install -e '.[fireworks]'"
        ) from e


def _ensure_filelock_import():
    try:
        from filelock import FileLock  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise RuntimeError("filelock is not installed. Install with: pip install filelock") from e


_ensure_fireworks_imports()
_ensure_filelock_import()

from fireworks import FWAction, FiretaskBase, explicit_serialize  # type: ignore  # noqa: E402
from filelock import FileLock  # type: ignore  # noqa: E402


def _parse_candidate_id(candidate_id: str) -> Tuple[Optional[int], Optional[int]]:
    parts = parse_candidate_id(candidate_id)
    if parts is None:
        return None, None
    return parts.generation_id, parts.candidate_index


def _fill_generation_metadata(
    *,
    candidate_id: str,
    generation_id: Optional[int],
    candidate_index: Optional[int],
) -> Tuple[Optional[int], Optional[int]]:
    """
    Fill missing generation_id / candidate_index.
    Priority:
      1) explicitly provided values
      2) parse from candidate_id if missing
    """
    g = generation_id
    c = candidate_index
    if g is not None and c is not None:
        return g, c

    pg, pc = _parse_candidate_id(candidate_id)
    if g is None:
        g = pg
    if c is None:
        c = pc
    return g, c


def _unique_key(record: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Idempotency key for results.jsonl lines.
    Prefer (run_id, attempt_id). Fall back to candidate_id for legacy records.
    """
    rid = record.get("run_id")
    aid = record.get("attempt_id")
    cid = record.get("candidate_id")
    rid = str(rid) if rid is not None else None
    aid = str(aid) if aid is not None else None
    cid = str(cid) if cid is not None else None
    return rid, aid, cid


def _evaluate_one_candidate(
    *,
    problem_config: Path,
    outdir: Path,
    run_id: str,
    candidate_id: str,
    candidate_params: Dict[str, Any],
    context_override: Optional[Dict[str, Any]] = None,
    generation_id: Optional[int] = None,
    candidate_index: Optional[int] = None,
    attempt_index: int = 0,
) -> Dict[str, Any]:
    generation_id, candidate_index = _fill_generation_metadata(
        candidate_id=candidate_id,
        generation_id=generation_id,
        candidate_index=candidate_index,
    )

    candidate_parts = parse_candidate_id(candidate_id)
    candidate_local_id = candidate_parts.candidate_local_id if candidate_parts is not None else None
    attempt_id = format_attempt_id(candidate_id, attempt_index)

    problem = load_problem_config(problem_config)

    workdir = candidate_workdir(outdir, run_id, candidate_id)
    workdir.mkdir(parents=True, exist_ok=True)

    res = evaluate_candidate(
        problem,
        run_id=run_id,
        candidate_id=candidate_id,
        candidate_local_id=candidate_local_id,
        attempt_id=attempt_id,
        candidate_params=candidate_params,
        workdir=workdir,
        context_override=context_override,
    )

    stored: Dict[str, Any] = {
        "problem_id": problem.id,
        "run_id": run_id,
        "candidate_id": candidate_id,
        "candidate_local_id": candidate_local_id,
        "attempt_id": attempt_id,
        "generation_id": generation_id,
        "candidate_index": candidate_index,
        "attempt_index": attempt_index,
        "params": _to_jsonable({**problem.runtime_params(), **candidate_params}),
        "fitness": _to_jsonable(res.fitness.model_dump()),
        "failure_kind": res.fitness.failure_kind,
        "returncode": res.returncode,
        "wall_time_s": res.wall_time_s,
        "started_at": res.started_at,
        "finished_at": res.finished_at,
        "evaluator": _to_jsonable(res.evaluator),
        "workdir": str(workdir),
        "stdout_path": str(res.stdout_path),
        "stderr_path": str(res.stderr_path),
        "input_path": str(res.input_path),
        "output_path": str(res.output_path),
    }

    (workdir / "result.json").write_text(
        json.dumps(stored, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return stored


# ---------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------

@explicit_serialize
class EvaluateCandidateTask(FiretaskBase):
    """
    Evaluate one candidate and write:
      <outdir>/runs/<run_id>/<candidate_id>/{input.json,output.json,stdout.txt,stderr.txt,result.json}

    `outdir` is the flattened problem root (same as CLI --outdir).
    """
    required_params = ["problem_config", "run_id", "candidate_id", "candidate_params", "outdir"]
    optional_params = ["context_override", "generation_id", "candidate_index", "attempt_index"]

    def run_task(self, fw_spec: Dict[str, Any]) -> FWAction:
        stored = _evaluate_one_candidate(
            problem_config=Path(self["problem_config"]).expanduser().resolve(),
            outdir=Path(self["outdir"]).expanduser().resolve(),
            run_id=str(self["run_id"]),
            candidate_id=str(self["candidate_id"]),
            candidate_params=dict(self["candidate_params"]),
            context_override=self.get("context_override"),
            generation_id=self.get("generation_id"),
            candidate_index=self.get("candidate_index"),
            attempt_index=int(self.get("attempt_index", 0)),
        )
        return FWAction(stored_data=stored, update_spec={"last_result": stored})


@explicit_serialize
class EvaluateBatchTask(FiretaskBase):
    """Evaluate multiple candidates sequentially inside one slim FireWork.

    Shared paths/configuration live once on the task. Candidate-specific items only
    contain candidate_id, params and metadata. Every result remains persisted in the
    candidate workdir; MongoDB only receives small aggregate counters.
    """
    required_params = ["problem_config", "outdir", "run_id", "items"]

    def _failed_record_from_exception(
        self,
        item: Dict[str, Any],
        exc: Exception,
        *,
        problem_config: Path,
        outdir: Path,
        run_id: str,
    ) -> Dict[str, Any]:
        candidate_id = str(item["candidate_id"])
        candidate_params = dict(item.get("candidate_params", {}))
        generation_id = item.get("generation_id")
        candidate_index = item.get("candidate_index")
        attempt_index = int(item.get("attempt_index", 0))

        generation_id, candidate_index = _fill_generation_metadata(
            candidate_id=candidate_id,
            generation_id=generation_id,
            candidate_index=candidate_index,
        )

        candidate_parts = parse_candidate_id(candidate_id)
        candidate_local_id = candidate_parts.candidate_local_id if candidate_parts is not None else None
        attempt_id = format_attempt_id(candidate_id, attempt_index)

        workdir = candidate_workdir(outdir, run_id, candidate_id)
        workdir.mkdir(parents=True, exist_ok=True)

        problem_id: Optional[str] = None
        runtime_params: Dict[str, Any] = {}
        try:
            problem = load_problem_config(problem_config)
            problem_id = problem.id
            runtime_params = problem.runtime_params()
        except Exception:
            pass

        fitness = {
            "status": "failed",
            "metrics": {},
            "objective": None,
            "constraints": {},
            "artifacts": {},
            "error": f"Unhandled batch exception: {exc}",
            "failure_kind": "internal_error",
        }
        record: Dict[str, Any] = {
            "problem_id": problem_id,
            "run_id": run_id,
            "candidate_id": candidate_id,
            "candidate_local_id": candidate_local_id,
            "attempt_id": attempt_id,
            "generation_id": generation_id,
            "candidate_index": candidate_index,
            "attempt_index": attempt_index,
            "params": _to_jsonable({**runtime_params, **candidate_params}),
            "fitness": _to_jsonable(fitness),
            "failure_kind": "internal_error",
            "returncode": None,
            "wall_time_s": None,
            "started_at": None,
            "finished_at": None,
            "evaluator": None,
            "workdir": str(workdir),
            "stdout_path": str(workdir / "stdout.txt"),
            "stderr_path": str(workdir / "stderr.txt"),
            "input_path": str(workdir / "input.json"),
            "output_path": str(workdir / "output.json"),
        }
        (workdir / "result.json").write_text(
            json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
        )
        return record

    def run_task(self, fw_spec: Dict[str, Any]) -> FWAction:
        problem_config = Path(self["problem_config"]).expanduser().resolve()
        outdir = Path(self["outdir"]).expanduser().resolve()
        run_id = str(self["run_id"])
        evaluated = 0
        failed = 0

        for item in self["items"]:
            try:
                record = _evaluate_one_candidate(
                    problem_config=problem_config,
                    outdir=outdir,
                    run_id=run_id,
                    candidate_id=str(item["candidate_id"]),
                    candidate_params=dict(item["candidate_params"]),
                    context_override=item.get("context_override"),
                    generation_id=item.get("generation_id"),
                    candidate_index=item.get("candidate_index"),
                    attempt_index=int(item.get("attempt_index", 0)),
                )
            except Exception as exc:
                record = self._failed_record_from_exception(
                    item, exc, problem_config=problem_config, outdir=outdir, run_id=run_id
                )
            evaluated += 1
            if record.get("failure_kind") is not None or (record.get("fitness") or {}).get("status") != "ok":
                failed += 1

        # Deliberately do not update fw_spec with per-candidate results. Each record
        # already exists as result.json and is consolidated by the generation pipeline.
        return FWAction(stored_data={"evaluated": evaluated, "failed": failed})


def _iter_jsonl_records(path: Path) -> Iterable[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    yield obj
    except FileNotFoundError:
        return


def _result_json_path(outdir: Path | str, run_id: str, candidate_id: str) -> Path:
    return candidate_workdir(Path(outdir).expanduser().resolve(), run_id, candidate_id) / "result.json"


def persist_result_record(
    outdir: Path | str,
    run_id: str,
    record: Dict[str, Any],
) -> Path:
    outdir = Path(outdir).expanduser().resolve()
    candidate_id = str(record["candidate_id"])
    result_path = _result_json_path(outdir, run_id, candidate_id)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    return result_path


def append_run_result_record(
    outdir: Path | str,
    run_id: str,
    record: Dict[str, Any],
    *,
    results_filename: str = "results.jsonl",
    lock_filename: str | None = None,
    skip_if_exists: bool = True,
) -> bool:
    """Backward-compatible wrapper.

    Persist the candidate result in its own workdir only. The run-level JSONL is rebuilt
    later by the coordinator from result.json files, so workers never contend on a shared
    run results file.
    """
    persist_result_record(outdir, run_id, record)
    return True


def _iter_run_result_json_records(outdir: Path | str, run_id: str) -> Iterable[Dict[str, Any]]:
    outdir = Path(outdir).expanduser().resolve()
    run_dir = run_root_dir(outdir, run_id)
    if not run_dir.exists():
        return
    for candidate_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        result_path = candidate_dir / "result.json"
        if not result_path.exists():
            continue
        try:
            obj = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(obj, dict):
            yield obj


def rebuild_run_results_jsonl(
    outdir: Path | str,
    run_id: str,
    *,
    results_filename: str = "results.jsonl",
) -> Path:
    outdir = Path(outdir).expanduser().resolve()
    run_dir = run_root_dir(outdir, run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / results_filename

    records: list[Dict[str, Any]] = []
    seen: set[tuple[str | None, str | None, str | None]] = set()

    generation_shards = list(iter_generation_shards(outdir, run_id))
    if generation_shards:
        for shard_path in generation_shards:
            for obj in _iter_jsonl_records(shard_path):
                key = _unique_key(obj)
                if key in seen:
                    continue
                seen.add(key)
                records.append(obj)
    else:
        for obj in _iter_run_result_json_records(outdir, run_id):
            key = _unique_key(obj)
            if key in seen:
                continue
            seen.add(key)
            records.append(obj)

    records.sort(key=lambda obj: (
        obj.get("generation_id") if isinstance(obj.get("generation_id"), int) else 10**12,
        obj.get("candidate_index") if isinstance(obj.get("candidate_index"), int) else 10**12,
        str(obj.get("candidate_id", "")),
        obj.get("attempt_index") if isinstance(obj.get("attempt_index"), int) else 10**12,
    ))

    with results_path.open("w", encoding="utf-8") as f:
        for obj in records:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    return results_path


def verify_run_results_complete(
    outdir: Path | str,
    run_id: str,
    expected_count: int,
    *,
    results_filename: str = "results.jsonl",
) -> tuple[bool, int]:
    outdir = Path(outdir).expanduser().resolve()
    seen: set[tuple[str | None, str | None, str | None]] = set()
    count = 0
    generation_shards = list(iter_generation_shards(outdir, run_id))
    if generation_shards:
        for shard_path in generation_shards:
            for obj in _iter_jsonl_records(shard_path):
                key = _unique_key(obj)
                if key in seen:
                    continue
                seen.add(key)
                count += 1
    else:
        for obj in _iter_run_result_json_records(outdir, run_id):
            key = _unique_key(obj)
            if key in seen:
                continue
            seen.add(key)
            count += 1
    return count >= expected_count, count


def rebuild_problem_results_jsonl(
    outdir: Path | str,
    *,
    results_filename: str = "results.jsonl",
    lock_filename: str | None = None,
) -> Path:
    outdir = Path(outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    results_path = outdir / results_filename
    lock_path = outdir / (lock_filename or f"{results_filename}.lock")
    runs_root = outdir / "runs"

    records: list[Dict[str, Any]] = []
    seen: set[tuple[str | None, str | None, str | None]] = set()
    if runs_root.exists():
        for run_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
            run_results_path = run_dir / results_filename
            for obj in _iter_jsonl_records(run_results_path):
                key = _unique_key(obj)
                if key in seen:
                    continue
                seen.add(key)
                records.append(obj)

    lock = FileLock(str(lock_path))
    with lock:
        with results_path.open("w", encoding="utf-8") as f:
            for obj in records:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    return results_path


@explicit_serialize
class AppendResultJsonlTask(FiretaskBase):
    """
    Append <workdir>/result.json as one line to:
      1) <outdir>/runs/<run_id>/results.jsonl

    The global <outdir>/results.jsonl is rebuilt only after a run completes.
    Uses file locks to prevent corruption under parallel execution.
    Idempotent by (run_id, attempt_id), falling back to (run_id, candidate_id)
    for legacy records without attempt metadata.

    HARD SAFETY:
      - will refuse to append if result.json's candidate_id != task candidate_id
      - will refuse to append if result.json's run_id != task run_id
    """
    required_params = ["outdir", "problem_id", "run_id", "candidate_id"]
    optional_params = [
        "workdir",
        "result_filename",
        "results_filename",
        "lock_filename",
        "skip_if_exists",
        "append_run_level",
        "generation_id",
        "candidate_index",
        "attempt_index",
    ]

    def _already_appended(
        self,
        results_path: Path,
        *,
        run_id: str,
        candidate_id: str,
        attempt_id: str | None,
    ) -> bool:
        try:
            with results_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    rid, aid, cid = _unique_key(obj)
                    if rid != run_id:
                        continue
                    if attempt_id is not None:
                        if aid == attempt_id:
                            return True
                        continue
                    if cid == candidate_id:
                        return True
        except FileNotFoundError:
            return False
        return False

    def _append_one(
        self,
        results_path: Path,
        lock_path: Path,
        record: Dict[str, Any],
        *,
        run_id: str,
        candidate_id: str,
        attempt_id: str | None,
        skip_if_exists: bool,
    ) -> bool:
        lock = FileLock(str(lock_path))
        with lock:
            if skip_if_exists and self._already_appended(
                results_path,
                run_id=run_id,
                candidate_id=candidate_id,
                attempt_id=attempt_id,
            ):
                return False
            append_jsonl_line(results_path, record)
        return True

    def run_task(self, fw_spec: Dict[str, Any]) -> FWAction:
        outdir = Path(self["outdir"]).expanduser().resolve()
        problem_id: str = self["problem_id"]
        run_id: str = self["run_id"]
        candidate_id: str = self["candidate_id"]
        attempt_index = self["attempt_index"] if isinstance(self.get("attempt_index"), int) else 0

        result_filename = str(self.get("result_filename", "result.json"))
        results_filename = str(self.get("results_filename", "results.jsonl"))
        lock_filename = str(self.get("lock_filename", f"{results_filename}.lock"))
        skip_if_exists = bool(self.get("skip_if_exists", True))
        append_run_level = bool(self.get("append_run_level", True))

        if self.get("workdir"):
            workdir = Path(str(self["workdir"])).expanduser().resolve()
        else:
            workdir = candidate_workdir(outdir, run_id, candidate_id)

        result_path = workdir / result_filename
        if not result_path.exists():
            raise FileNotFoundError(f"Expected result file not found: {result_path}")

        record = json.loads(result_path.read_text(encoding="utf-8"))
        attempt_id = str(record.get("attempt_id")) if record.get("attempt_id") is not None else format_attempt_id(candidate_id, attempt_index)
        record.setdefault("attempt_id", attempt_id)
        record.setdefault("attempt_index", attempt_index)
        parts = parse_candidate_id(candidate_id)
        if record.get("candidate_local_id") is None and parts is not None:
            record["candidate_local_id"] = parts.candidate_local_id

        rec_pid = record.get("problem_id")
        if rec_pid and rec_pid != problem_id:
            raise RuntimeError(
                f"Record problem_id={rec_pid!r} does not match task problem_id={problem_id!r}"
            )

        rec_rid = record.get("run_id")
        if rec_rid is not None and str(rec_rid) != str(run_id):
            raise RuntimeError(
                f"Record run_id={rec_rid!r} does not match task run_id={run_id!r} "
                f"(workdir={workdir})"
            )

        rec_cid = record.get("candidate_id")
        if rec_cid is not None and str(rec_cid) != str(candidate_id):
            raise RuntimeError(
                f"Record candidate_id={rec_cid!r} does not match task candidate_id={candidate_id!r} "
                f"(workdir={workdir})"
            )
        rec_aid = record.get("attempt_id")
        if rec_aid is not None and str(rec_aid) != attempt_id:
            raise RuntimeError(
                f"Record attempt_id={rec_aid!r} does not match task attempt_id={attempt_id!r} "
                f"(workdir={workdir})"
            )

        task_gen = self.get("generation_id")
        task_idx = self.get("candidate_index")
        gen_filled, idx_filled = _fill_generation_metadata(
            candidate_id=candidate_id,
            generation_id=task_gen if isinstance(task_gen, int) else record.get("generation_id"),
            candidate_index=task_idx if isinstance(task_idx, int) else record.get("candidate_index"),
        )
        if record.get("generation_id") is None:
            record["generation_id"] = gen_filled
        if record.get("candidate_index") is None:
            record["candidate_index"] = idx_filled

        outdir.mkdir(parents=True, exist_ok=True)
        run_dir = run_root_dir(outdir, run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        run_results_path = run_dir / results_filename

        return FWAction(
            stored_data={
                "candidate_id": candidate_id,
                "attempt_id": attempt_id,
                "problem_id": problem_id,
                "run_results": str(run_results_path),
                "appended_run": None,
                "deferred_run_rebuild": True,
            }
        )
