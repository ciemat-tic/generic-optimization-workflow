# src/gow/cli.py
from __future__ import annotations

import json
import time
import os
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

import typer

from gow.candidate_ids import format_attempt_id, format_candidate_id, parse_candidate_id
from gow.config import load_problem_config
from gow.layout import candidate_workdir, run_launchers_dir, run_root
from gow.run import run_local_optimization
from gow.postprocess import AsyncGenerationArchiver, archive_generation_workdirs, finalize_generation, finalize_run_launcher_tree, merge_runs, write_generation_launch_manifest

app = typer.Typer(help="Generic Optimization Workflow (gow)")
commands = typer.Typer(help="Commands")
fw_app = typer.Typer(help="FireWorks backend (optional)")

app.add_typer(commands)
app.add_typer(fw_app, name="fw")

ENV_OUTDIR = "GOW_OUTDIR"


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _default_run_id() -> str:
    return str(uuid.uuid4())


def _resolve_results_dir(config: Path, outdir: Path | None) -> Path:
    """
    Resolution order:
      1) explicit --outdir
      2) env var GOW_OUTDIR
      3) <config_dir>/results

    IMPORTANT: this is the flattened results directory (problem root).
    Outputs go directly under:
      <outdir>/
        results.jsonl
        summary.json
        runs/<run_id>/...
    """
    if outdir is not None:
        return outdir.expanduser().resolve()

    env = os.environ.get(ENV_OUTDIR)
    if env:
        return Path(env).expanduser().resolve()

    return config.expanduser().resolve().parent / "results"


def _parse_kv_params(items: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise typer.BadParameter(f"Invalid --param '{item}'. Use NAME=VALUE.")
        k, v = item.split("=", 1)
        k = k.strip()
        v = v.strip()

        if v.lower() in {"true", "false"}:
            out[k] = (v.lower() == "true")
            continue

        try:
            out[k] = int(v)
            continue
        except ValueError:
            pass

        try:
            out[k] = float(v)
            continue
        except ValueError:
            pass

        out[k] = v
    return out


def _resolve_manual_candidate_id(
    *,
    candidate_id: str | None,
    run_id: str,
    generation_id: int | None,
    candidate_index: int | None,
) -> str:
    """Resolve the candidate id for manual single-candidate evaluation."""
    if candidate_id is not None:
        return candidate_id
    if generation_id is not None and candidate_index is not None:
        return format_candidate_id(
            generation_id=generation_id,
            candidate_index=candidate_index,
            run_id=run_id,
        )
    return "manual"


def _coerce_bool_option(value: Any, default: bool = False) -> bool:
    if type(value).__name__ == "OptionInfo":
        return default
    return bool(value)


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _pick_best(records: Iterable[dict[str, Any]], *, direction: str = "minimize") -> list[dict[str, Any]]:
    if direction not in {"minimize", "maximize"}:
        raise ValueError("direction must be 'minimize' or 'maximize'")

    good: list[dict[str, Any]] = []
    for r in records:
        fit = (r or {}).get("fitness", {}) or {}
        if fit.get("status") != "ok":
            continue
        obj = fit.get("objective", None)
        if obj is None:
            continue
        try:
            obj_val = float(obj)
        except (TypeError, ValueError):
            continue
        rr = dict(r)
        rr["_objective"] = obj_val
        good.append(rr)

    reverse = direction == "maximize"
    good.sort(key=lambda x: x["_objective"], reverse=reverse)
    return good


def _optimizer_kwargs(opt_cfg: Any) -> dict[str, Any]:
    """
    Extract optimizer-specific kwargs from opt_cfg.

    In GOW, optimizer hyperparameters are stored in opt_cfg.settings (dict).
    This function flattens that dict and returns it as kwargs for make_optimizer().

    Clean mode: requires Pydantic v2 models (model_dump()).
    """
    if not hasattr(opt_cfg, "model_dump"):
        raise TypeError("optimizer config must be a Pydantic v2 model (missing model_dump())")

    data = opt_cfg.model_dump()
    settings = data.get("settings") or {}
    if not isinstance(settings, dict):
        raise ValueError(f"optimizer.settings must be a dict, got {type(settings)}")

    # remove the standard fields (not forwarded)
    for k in ("name", "seed", "max_evaluations", "batch_size", "settings"):
        data.pop(k, None)

    out = {k: v for k, v in data.items() if not str(k).startswith("_")}
    out.update(settings)
    out = {k: v for k, v in out.items() if not str(k).startswith("_")}
    return out


@contextmanager
def _pushd(path: Path):
    """
    Temporarily chdir. Ensures FireWorks launcher_* directories are created under `path`.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    prev = Path.cwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(str(prev))


def _coerce_objective(fitness: dict[str, Any]) -> float | None:
    if not isinstance(fitness, dict):
        return None
    if fitness.get("status") != "ok":
        return None
    obj = fitness.get("objective")
    if obj is None:
        return None
    try:
        return float(obj)
    except (TypeError, ValueError):
        return None


def _is_better(a: float, b: float, direction: str) -> bool:
    if direction == "minimize":
        return a < b
    if direction == "maximize":
        return a > b
    raise ValueError("direction must be 'minimize' or 'maximize'")


def _parse_group_size_option(value: Any) -> tuple[bool, int]:
    """Return ``(auto_mode, fixed_size)`` for --group-size."""
    if type(value).__name__ == "OptionInfo":
        value = "0"
    text = str(value).strip().lower()
    if text == "auto":
        return True, 0
    try:
        size = int(text)
    except (TypeError, ValueError) as exc:
        raise typer.BadParameter("--group-size must be a non-negative integer or 'auto'") from exc
    if size < 0:
        raise typer.BadParameter("--group-size must be >= 0 or 'auto'")
    return False, size


def _representative_candidate(problem: Any) -> dict[str, Any]:
    """Build a BSON-size-oriented representative candidate without running an optimizer."""
    candidate: dict[str, Any] = {}
    for name, param in problem.optimizable_parameters().items():
        ptype = str(getattr(param, "type", ""))
        if ptype == "categorical":
            choices = list(getattr(param, "choices", None) or [])
            if choices:
                candidate[name] = max(choices, key=lambda x: len(str(x).encode("utf-8")))
            else:
                candidate[name] = getattr(param, "value", "")
        elif ptype == "int":
            bounds = list(getattr(param, "bounds", None) or [])
            if bounds:
                candidate[name] = max(bounds, key=lambda x: abs(int(x)))
            else:
                candidate[name] = int(getattr(param, "value", 0))
        elif ptype == "real":
            bounds = list(getattr(param, "bounds", None) or [])
            if bounds:
                candidate[name] = max(bounds, key=lambda x: abs(float(x)))
            else:
                candidate[name] = float(getattr(param, "value", 0.0))
        else:
            candidate[name] = getattr(param, "value", None)
    return candidate


def _launch_fireworks(
    *,
    lp: Any,
    mode: str,
    nlaunches: int,
    sleep: int,
    qadapter_path: Path | None = None,
    njobs_queue: int = 0,
    reserve: bool = False,
) -> None:
    mode_norm = str(mode).strip().lower()
    if mode_norm not in {"local", "queue"}:
        raise typer.BadParameter("--launcher must be 'local' or 'queue'")

    if mode_norm == "local":
        from fireworks.core.rocket_launcher import rapidfire as local_rapidfire

        local_rapidfire(lp, nlaunches=nlaunches, sleep_time=sleep)
        return

    from fireworks.queue.queue_launcher import rapidfire as queue_rapidfire
    from gow.fw.launchpad import load_qadapter

    qadapter = load_qadapter(qadapter_path)
    queue_rapidfire(
        lp,
        fworker=None,
        qadapter=qadapter,
        launch_dir=".",
        nlaunches=nlaunches,
        njobs_queue=njobs_queue,
        sleep_time=sleep,
        reserve=reserve,
    )



def _terminal_fw_state(state: str | None) -> bool:
    return str(state or "").upper() in {"COMPLETED", "FIZZLED", "ARCHIVED"}


def _get_fw_state(lp: Any, fw_id: int | None) -> str | None:
    if fw_id is None:
        return None
    try:
        if hasattr(lp, "get_fw_by_id"):
            fw = lp.get_fw_by_id(fw_id)
            state = getattr(fw, "state", None)
            if state is not None:
                return str(state)
    except Exception:
        pass
    try:
        if hasattr(lp, "get_fw_dict_by_id"):
            data = lp.get_fw_dict_by_id(fw_id)
            if isinstance(data, dict) and data.get("state") is not None:
                return str(data.get("state"))
    except Exception:
        pass
    return None


def _extract_fw_ids(id_map: Any) -> list[int]:
    if isinstance(id_map, dict):
        out: list[int] = []
        for value in id_map.values():
            try:
                out.append(int(value))
            except Exception:
                continue
        return out
    return []


def _wait_for_batch_results(
    lp: Any,
    results_dir: Path,
    run_id: str,
    candidate_ids: list[str],
    fw_ids: list[int],
    *,
    poll_seconds: int,
    wait_timeout: int,
) -> tuple[bool, list[str]]:
    started = time.monotonic()
    missing = list(candidate_ids)
    while True:
        missing = []
        for candidate_id in candidate_ids:
            result_path = candidate_workdir(results_dir, run_id, candidate_id) / "result.json"
            if not result_path.exists():
                missing.append(candidate_id)
        if not missing:
            return True, []

        states = [_get_fw_state(lp, fw_id) for fw_id in fw_ids]
        known_states = [s for s in states if s is not None]
        if known_states and all(_terminal_fw_state(s) for s in known_states):
            return False, missing

        if wait_timeout > 0 and (time.monotonic() - started) >= wait_timeout:
            return False, missing

        time.sleep(max(1, poll_seconds))


def _wait_for_candidate_results(
    lp: Any,
    results_dir: Path,
    run_id: str,
    candidate_ids: list[str],
    fw_ids: list[int],
    *,
    poll_seconds: int,
    timeout_seconds: int,
) -> None:
    ok, missing = _wait_for_batch_results(
        lp=lp,
        results_dir=results_dir,
        run_id=run_id,
        candidate_ids=candidate_ids,
        fw_ids=fw_ids,
        poll_seconds=poll_seconds,
        wait_timeout=timeout_seconds,
    )
    if not ok:
        raise RuntimeError(f"Missing result.json for candidate(s): {', '.join(missing)}")


def _build_missing_result_record(
    *,
    problem: Any,
    results_dir: Path,
    run_id: str,
    candidate_id: str,
    candidate_params: dict[str, Any],
    generation_id: int | None,
    candidate_index: int | None,
    attempt_index: int,
    reason: str,
) -> dict[str, Any]:
    workdir = candidate_workdir(results_dir, run_id, candidate_id)
    workdir.mkdir(parents=True, exist_ok=True)
    parts = parse_candidate_id(candidate_id)
    candidate_local_id = parts.candidate_local_id if parts is not None else None
    attempt_id = format_attempt_id(candidate_id, attempt_index)
    record = {
        "problem_id": problem.id,
        "run_id": run_id,
        "generation_id": generation_id,
        "candidate_index": candidate_index,
        "candidate_id": candidate_id,
        "candidate_local_id": candidate_local_id,
        "attempt_id": attempt_id,
        "attempt_index": attempt_index,
        "params": {**problem.runtime_params(), **candidate_params},
        "fitness": {
            "status": "failed",
            "metrics": {},
            "objective": None,
            "constraints": {},
            "artifacts": {},
            "error": reason,
            "failure_kind": "missing_result_after_retries",
        },
        "failure_kind": "missing_result_after_retries",
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
    (workdir / "result.json").write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    return record

def _write_summary_json(
    *,
    results_dir: Path,
    problem_id: str,
    run_id: str,
    max_evaluations: int,
    direction: str,
    best_candidate: dict[str, Any] | None,
) -> Path:
    results_dir = results_dir.expanduser().resolve()
    run_root_dir = run_root(results_dir, run_id)

    summary: dict[str, Any] = {
        "best": best_candidate,
        "max_evaluations": max_evaluations,
        "objective": {"direction": direction},
        "outdir": str(results_dir),
        "problem_id": problem_id,
        "results_file": str(results_dir / "results.jsonl"),
        "run_id": run_id,
        "run_results_file": str(run_root_dir / "results.jsonl"),
        "run_root": str(run_root_dir),
    }

    path = results_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path


# -----------------------------------------------------------------------------
# Core (no FireWorks)
# -----------------------------------------------------------------------------
@commands.command("run")
def run_cmd(
    config: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to optimization specs (YAML/JSON).",
    ),
    outdir: Path | None = typer.Option(
        None,
        "--outdir",
        "-o",
        help=(
            "Results directory (flattened). Resolution order: "
            "explicit --outdir, else $GOW_OUTDIR, else <config_dir>/results."
        ),
    ),
    run_id: str | None = typer.Option(None, "--run-id", help="Optional run id (defaults to a UUID)."),
    archive_generations: bool = typer.Option(False, "--archive-generations/--no-archive-generations", help="Archive each completed generation into a single tar.gz."),
    delete_archived_workdirs: bool = typer.Option(False, "--delete-archived-workdirs/--keep-archived-workdirs", help="Delete candidate workdirs after successfully archiving a generation."),
):
    """
    Run a local optimization loop (no FireWorks).

    Outputs:
      <outdir>/results.jsonl
      <outdir>/summary.json
      <outdir>/runs/<run_id>/...
    """
    config_abs = config.expanduser().resolve()
    results_dir = _resolve_results_dir(config_abs, outdir)

    archive_generations = _coerce_bool_option(archive_generations, False)
    delete_archived_workdirs = _coerce_bool_option(delete_archived_workdirs, False)
    problem = load_problem_config(config_abs)
    results_path = run_local_optimization(problem, outdir=results_dir, run_id=run_id, archive_generations=archive_generations, delete_archived_workdirs=delete_archived_workdirs)
    typer.echo(f"Results: {results_path}")


@commands.command("info")
def info():
    typer.echo("gow is installed and commands are registered correctly.")


@commands.command("evaluate")
def evaluate_cmd(
    config: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to optimization specs (YAML/JSON).",
    ),
    outdir: Path | None = typer.Option(
        None,
        "--outdir",
        "-o",
        help=(
            "Results directory (flattened). Resolution order: "
            "explicit --outdir, else $GOW_OUTDIR, else <config_dir>/results."
        ),
    ),
    run_id: str = typer.Option("manual", "--run-id", help="Run id used to build the workdir path."),
    candidate_id: str | None = typer.Option(
        None,
        "--candidate-id",
        help=(
            "Candidate id used to build the workdir path. If omitted and both "
            "--generation-id and --candidate-index are provided, GOW generates a "
            "canonical run-aware candidate id; otherwise it falls back to 'manual'."
        ),
    ),
    generation_id: int | None = typer.Option(
        None,
        "--generation-id",
        help="Optional zero-based generation number used for metadata and canonical candidate-id generation.",
    ),
    candidate_index: int | None = typer.Option(
        None,
        "--candidate-index",
        help="Optional zero-based global candidate sequence number within the run.",
    ),
    attempt_index: int = typer.Option(
        0,
        "--attempt-index",
        min=0,
        help="Attempt index for this execution. Use 0 for the first attempt and increment on manual re-execution.",
    ),
    param: list[str] = typer.Option([], "--param", "-p", help="Override parameter as NAME=VALUE (repeatable)."),
    params_file: Path | None = typer.Option(None, "--params-file", help="JSON file with parameter overrides."),
):
    """
    Evaluate a single candidate (useful for debugging external evaluators).
    """
    config_abs = config.expanduser().resolve()
    results_dir = _resolve_results_dir(config_abs, outdir)

    archive_generations = _coerce_bool_option(archive_generations, False)
    delete_archived_workdirs = _coerce_bool_option(delete_archived_workdirs, False)

    problem = load_problem_config(config_abs)

    overrides: dict[str, Any] = {}
    if params_file is not None:
        data = json.loads(params_file.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise typer.BadParameter("--params-file must contain a JSON object (dict).")
        overrides.update(data)

    overrides.update(_parse_kv_params(param))

    candidate_id_value = _resolve_manual_candidate_id(
        candidate_id=candidate_id,
        run_id=run_id,
        generation_id=generation_id,
        candidate_index=candidate_index,
    )

    workdir = candidate_workdir(results_dir, run_id, candidate_id_value)
    candidate_parts = parse_candidate_id(candidate_id_value)
    candidate_local_id = candidate_parts.candidate_local_id if candidate_parts is not None else None
    attempt_id = format_attempt_id(candidate_id_value, attempt_index)

    from gow.evaluation import evaluate_candidate  # local import

    res = evaluate_candidate(
        problem,
        run_id=run_id,
        candidate_id=candidate_id_value,
        candidate_local_id=candidate_local_id,
        attempt_id=attempt_id,
        candidate_params=overrides,
        workdir=workdir,
    )

    typer.echo(f"Workdir: {workdir}")
    typer.echo(f"Status:  {res.fitness.status}")
    typer.echo(f"Objective: {res.fitness.objective}")
    typer.echo(f"Metrics: {res.fitness.metrics}")
    if res.fitness.error:
        typer.echo(f"Error: {res.fitness.error}")
    typer.echo(f"Return code: {res.returncode}  Wall time (s): {res.wall_time_s:.3f}")


@commands.command("best")
def best_cmd(
    results_dir: Path = typer.Argument(..., exists=True, file_okay=False, readable=True, help="Results dir (<outdir>)."),
    top: int = typer.Option(1, "--top", "-n", min=1, help="Show top N candidates."),
    config: Path | None = typer.Option(None, "--config", help="Optimization specs (YAML/JSON) to read objective direction."),
    direction: str | None = typer.Option(None, "--direction", help="Override objective direction: minimize or maximize."),
    run_id: str | None = typer.Option(None, "--run-id", help="If set, read runs/<run_id>/results.jsonl instead of <outdir>/results.jsonl."),
):
    """
    Show the best candidate(s).

    Reads:
      - <outdir>/results.jsonl (default), OR
      - <outdir>/runs/<run_id>/results.jsonl if --run-id is provided.
    """
    results_dir = results_dir.expanduser().resolve()
    results_path = results_dir / "runs" / run_id / "results.jsonl" if run_id else results_dir / "results.jsonl"

    if not results_path.exists():
        raise typer.BadParameter(f"Could not find results.jsonl at {results_path}")

    chosen_direction = "minimize"
    if config is not None:
        problem = load_problem_config(config.expanduser().resolve())
        chosen_direction = problem.objective.direction

    if direction is not None:
        direction_norm = direction.strip().lower()
        if direction_norm not in {"minimize", "maximize"}:
            raise typer.BadParameter("--direction must be 'minimize' or 'maximize'")
        chosen_direction = direction_norm

    ranked = _pick_best(_iter_jsonl(results_path), direction=chosen_direction)
    if not ranked:
        typer.echo("No successful candidates with an objective found.")
        raise typer.Exit(code=1)

    show = ranked[:top]
    for i, r in enumerate(show, start=1):
        fit = r["fitness"]
        typer.echo(f"#{i}")
        typer.echo(f"  candidate_id: {r.get('candidate_id')}")
        typer.echo(f"  objective:    {fit.get('objective')}")
        typer.echo(f"  metrics:      {fit.get('metrics')}")
        typer.echo(f"  workdir:      {r.get('workdir')}")
        if i == 1:
            typer.echo(f"  direction:    {chosen_direction}")
        if top > 1 and i != top:
            typer.echo("")




@commands.command("merge-runs")
def merge_runs_cmd(
    results_dir: Path = typer.Argument(..., exists=True, file_okay=False, readable=True, help="Results dir (<outdir>)."),
    target_run_id: str = typer.Option(..., "--target-run-id", help="Run id for the merged output."),
    source_run_id: list[str] = typer.Option(..., "--source-run-id", help="Source run id to merge (repeatable)."),
):
    path = merge_runs(outdir=results_dir, target_run_id=target_run_id, source_run_ids=source_run_id)
    typer.echo(f"Merged run results: {path}")


@commands.command("archive-generation")
def archive_generation_cmd(
    results_dir: Path = typer.Argument(..., exists=True, file_okay=False, readable=True, help="Results dir (<outdir>)."),
    run_id: str = typer.Option(..., "--run-id", help="Run id."),
    generation_id: int = typer.Option(..., "--generation-id", min=0, help="Generation id to archive."),
    delete_source: bool = typer.Option(False, "--delete-source/--keep-source", help="Delete candidate workdirs after archiving."),
):
    run_dir = run_root(results_dir, run_id)
    candidate_ids: list[str] = []
    if run_dir.exists():
        for candidate_dir in sorted(p for p in run_dir.iterdir() if p.is_dir() and p.name not in {"launchers", "generations", "archives"}):
            parts = parse_candidate_id(candidate_dir.name)
            if parts is not None and parts.generation_id == generation_id:
                candidate_ids.append(candidate_dir.name)
    archive_path = archive_generation_workdirs(
        outdir=results_dir,
        run_id=run_id,
        generation_id=generation_id,
        candidate_ids=candidate_ids,
        delete_source=delete_source,
    )
    typer.echo(f"Archive: {archive_path}")


# -----------------------------------------------------------------------------
# FireWorks backend
# -----------------------------------------------------------------------------
@fw_app.command("evaluate")
def fw_evaluate_cmd(
    config: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True, help="Path to optimization specs (YAML/JSON)."),
    launchpad: Path | None = typer.Option(None, "--launchpad", help="Path to my_launchpad.yaml (FireWorks LaunchPad config)."),
    outdir: Path | None = typer.Option(
        None,
        "--outdir",
        "-o",
        help=(
            "Results directory (flattened). Resolution order: "
            "explicit --outdir, else $GOW_OUTDIR, else <config_dir>/results."
        ),
    ),
    run_id: str = typer.Option("fw-manual", "--run-id", help="Run id used to build the workdir path."),
    candidate_id: str | None = typer.Option(
        None,
        "--candidate-id",
        help=(
            "Candidate id used to build the workdir path. If omitted and both "
            "--generation-id and --candidate-index are provided, GOW generates a "
            "canonical run-aware candidate id; otherwise it falls back to 'manual'."
        ),
    ),
    generation_id: int | None = typer.Option(
        None,
        "--generation-id",
        help="Optional zero-based generation number used for metadata and canonical candidate-id generation.",
    ),
    candidate_index: int | None = typer.Option(
        None,
        "--candidate-index",
        help="Optional zero-based global candidate sequence number within the run.",
    ),
    attempt_index: int = typer.Option(
        0,
        "--attempt-index",
        min=0,
        help="Attempt index for this execution. Use 0 for the first attempt and increment on manual re-execution.",
    ),
    param: list[str] = typer.Option([], "--param", "-p", help="Override parameter as NAME=VALUE (repeatable)."),
    params_file: Path | None = typer.Option(None, "--params-file", help="JSON file with parameter overrides."),
    launch: bool = typer.Option(False, "--launch/--no-launch", help="Launch immediately (rapidfire) after submitting."),
    launch_dir: Path | None = typer.Option(
        None,
        "--launch-dir",
        help="Directory for FireWorks launcher_* dirs (default: <outdir>/runs/<run_id>/launchers).",
    ),
    sleep: int = typer.Option(0, "--sleep", help="Seconds to sleep between rocket launches (rapidfire)."),
    nlaunches: int = typer.Option(0, "--nlaunches", help="Max launches for rapidfire (0 means until queue empty)."),
    launcher: str = typer.Option("local", "--launcher", help="Launcher backend: 'local' or 'queue'."),
    qadapter: Path | None = typer.Option(
        None,
        "--qadapter",
        help="Path to my_qadapter.yaml, or a directory containing it. Used when --launcher queue.",
    ),
    njobs_queue: int = typer.Option(0, "--njobs-queue", min=0, help="Max queued jobs for FireWorks queue rapidfire."),
    reserve: bool = typer.Option(False, "--reserve/--no-reserve", help="Reserve jobs before launching when using queue rapidfire."),
):
    """
    Submit a single-candidate evaluation workflow to FireWorks (optionally launch).
    """
    overrides: dict[str, Any] = {}
    if params_file is not None:
        data = json.loads(params_file.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise typer.BadParameter("--params-file must contain a JSON object (dict).")
        overrides.update(data)
    overrides.update(_parse_kv_params(param))

    try:
        from gow.fw.launchpad import load_launchpad
        from gow.fw.workflow import SingleEvalSpec, build_single_evaluate_workflow
    except Exception as e:
        raise typer.BadParameter(str(e)) from e

    config_abs = config.expanduser().resolve()
    results_dir = _resolve_results_dir(config_abs, outdir)

    _ = load_problem_config(config_abs)  # validate config early
    lp = load_launchpad(launchpad)

    launchers_dir = launch_dir.expanduser().resolve() if launch_dir else run_launchers_dir(results_dir, run_id)
    candidate_id_value = _resolve_manual_candidate_id(
        candidate_id=candidate_id,
        run_id=run_id,
        generation_id=generation_id,
        candidate_index=candidate_index,
    )

    spec = SingleEvalSpec(
        problem_config=config_abs,
        outdir=results_dir,
        run_id=run_id,
        candidate_id=candidate_id_value,
        candidate_params=overrides,
        generation_id=generation_id,
        candidate_index=candidate_index,
        attempt_index=attempt_index,
    )
    wf = build_single_evaluate_workflow(spec)

    id_map = lp.add_wf(wf)
    fw_id = next(iter(id_map.values()), None) if isinstance(id_map, dict) else None

    typer.echo(f"Submitted workflow. id_map={id_map}  fw_id={fw_id}")
    typer.echo(f"Results dir: {results_dir}")
    typer.echo(f"Launchers dir: {launchers_dir}")
    typer.echo(f"Launcher: {launcher}")
    if str(launcher).strip().lower() == "queue":
        typer.echo(f"QAdapter: {qadapter or 'auto'}")
        typer.echo(f"njobs_queue: {njobs_queue}")

    if launch:
        with _pushd(launchers_dir):
            _launch_fireworks(
                lp=lp,
                mode=launcher,
                nlaunches=nlaunches,
                sleep=sleep,
                qadapter_path=qadapter,
                njobs_queue=njobs_queue,
                reserve=reserve,
            )
        typer.echo(f"Launch complete for current queue using launcher={launcher}.")


@fw_app.command("group-size-preflight")
def fw_group_size_preflight_cmd(
    config: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True, help="Path to optimization specs (YAML/JSON)."),
    launchpad: Path | None = typer.Option(None, "--launchpad", help="Path to my_launchpad.yaml (FireWorks LaunchPad config)."),
    outdir: Path | None = typer.Option(None, "--outdir", "-o", help="Results directory used to build the real FireWork payload."),
    group_size: str = typer.Option("auto", "--group-size", help="Candidate group size to test, or 'auto'."),
    njobs_queue: int = typer.Option(10, "--njobs-queue", min=1, help="Queue jobs to keep occupied when --group-size auto is used."),
    bson_safety_fraction: float = typer.Option(0.75, "--bson-safety-fraction", min=0.05, max=1.0, help="Fraction of the MongoDB-reported BSON limit considered safe."),
):
    """Measure a representative batch FireWork against the connected MongoDB limits.

    No Workflow is inserted. MongoDB capabilities are read from the actual FireWorks
    LaunchPad connection using the ``hello`` command.
    """
    try:
        from gow.fw.bson_preflight import (
            htc_group_size_target,
            largest_safe_group_size,
            measure_workflow_bson,
            query_mongo_server_limits,
        )
        from gow.fw.launchpad import load_launchpad
        from gow.fw.workflow import BatchEvalSpec, SingleEvalSpec, build_batch_evaluate_workflow
    except Exception as e:
        raise typer.BadParameter(str(e)) from e

    config_abs = config.expanduser().resolve()
    results_dir = _resolve_results_dir(config_abs, outdir)
    problem = load_problem_config(config_abs)
    lp = load_launchpad(launchpad)
    limits = query_mongo_server_limits(lp)
    safe_bytes = limits.safe_document_limit(bson_safety_fraction)

    auto_mode, fixed_size = _parse_group_size_option(group_size)
    generation_size = min(problem.optimizer.batch_size, problem.optimizer.max_evaluations)
    htc_target = htc_group_size_target(generation_size, njobs_queue)
    requested_max = htc_target if auto_mode else (fixed_size or generation_size)
    representative = _representative_candidate(problem)
    run_id_val = "bson-preflight"
    last_index = max(0, problem.optimizer.max_evaluations - 1)

    def build_for_size(size: int):
        items: list[SingleEvalSpec] = []
        start_index = max(0, last_index - size + 1)
        for offset in range(size):
            idx = start_index + offset
            cid = format_candidate_id(generation_id=0, candidate_index=idx, run_id=run_id_val)
            items.append(
                SingleEvalSpec(
                    problem_config=config_abs,
                    outdir=results_dir,
                    run_id=run_id_val,
                    candidate_id=cid,
                    candidate_params=dict(representative),
                    generation_id=0,
                    candidate_index=idx,
                    attempt_index=0,
                )
            )
        return build_batch_evaluate_workflow(
            BatchEvalSpec(
                problem_config=config_abs,
                outdir=results_dir,
                run_id=run_id_val,
                items=items,
            )
        )

    safe_size, safe_measurement = largest_safe_group_size(
        requested_max,
        build_workflow_for_size=build_for_size,
        safe_document_bytes=safe_bytes,
    )

    requested_measurement = measure_workflow_bson(build_for_size(requested_max)) if requested_max == safe_size else None

    typer.echo("MongoDB-aware FireWork BSON preflight")
    typer.echo(f"maxBsonObjectSize:    {limits.max_bson_object_size / (1024 * 1024):.2f} MiB")
    typer.echo(f"maxMessageSizeBytes:  {limits.max_message_size_bytes / (1024 * 1024):.2f} MiB")
    typer.echo(f"maxWriteBatchSize:    {limits.max_write_batch_size}")
    typer.echo(f"safety fraction:      {bson_safety_fraction:.3f}")
    typer.echo(f"safe document limit:  {safe_bytes / (1024 * 1024):.2f} MiB")
    typer.echo(f"generation candidates:{generation_size:>10}")
    typer.echo(f"njobs_queue:          {njobs_queue:>10}")
    typer.echo(f"HTC group target:     {htc_target:>10}")
    typer.echo(f"requested maximum:    {requested_max:>10}")
    typer.echo(f"BSON-safe maximum:    {safe_size:>10}")

    if safe_measurement is not None:
        typer.echo(f"BSON at safe maximum: {safe_measurement.max_document_mib:.2f} MiB")

    if safe_size <= 0:
        raise typer.BadParameter("Even one candidate exceeds the configured safe MongoDB BSON threshold.")

    if auto_mode:
        selected = min(htc_target, safe_size)
        typer.echo(f"AUTO selected:        {selected:>10}")
        typer.echo(f"FireWorks/generation: {(generation_size + selected - 1) // selected:>10}")
    elif safe_size < requested_max:
        typer.echo(f"UNSAFE: --group-size {requested_max} exceeds the server-aware safety threshold.")
        typer.echo(f"Recommended maximum for this representative payload: {safe_size}")
        raise typer.Exit(code=2)
    else:
        measurement = requested_measurement or safe_measurement
        if measurement is not None:
            typer.echo(f"Requested BSON size:  {measurement.max_document_mib:.2f} MiB")
        typer.echo("Status: OK")


@fw_app.command("run")
def fw_run_cmd(
    config: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True, help="Path to optimization specs (YAML/JSON)."),
    launchpad: Path | None = typer.Option(None, "--launchpad", help="Path to my_launchpad.yaml (FireWorks LaunchPad config)."),
    outdir: Path | None = typer.Option(
        None,
        "--outdir",
        "-o",
        help=(
            "Results directory (flattened). Resolution order: "
            "explicit --outdir, else $GOW_OUTDIR, else <config_dir>/results."
        ),
    ),
    run_id: str | None = typer.Option(None, "--run-id", help="Run id (defaults to UUID)."),
    launch: bool = typer.Option(True, "--launch/--no-launch", help="Launch immediately (rapidfire) after submitting."),
    launch_dir: Path | None = typer.Option(
        None,
        "--launch-dir",
        help="Directory for FireWorks launcher_* dirs (default: <outdir>/runs/<run_id>/launchers).",
    ),
    sleep: int = typer.Option(0, "--sleep", help="Seconds to sleep between rocket launches (rapidfire)."),
    nlaunches: int = typer.Option(0, "--nlaunches", help="Max launches for rapidfire (0 means until queue empty)."),
    launcher: str = typer.Option("local", "--launcher", help="Launcher backend: 'local' or 'queue'."),
    qadapter: Path | None = typer.Option(
        None,
        "--qadapter",
        help="Path to my_qadapter.yaml, or a directory containing it. Used when --launcher queue.",
    ),
    njobs_queue: int = typer.Option(10, "--njobs-queue", min=0, help="Max queued jobs for FireWorks queue rapidfire."),
    reserve: bool = typer.Option(False, "--reserve/--no-reserve", help="Reserve jobs before launching when using queue rapidfire."),
    group_size: str = typer.Option(
        "0",
        "--group-size",
        help="Number of candidates per FireWork, 0 for the whole batch, or 'auto' for MongoDB/HTC-aware sizing.",
    ),
    bson_safety_fraction: float = typer.Option(
        0.75,
        "--bson-safety-fraction",
        min=0.05,
        max=1.0,
        help="Fraction of MongoDB hello.maxBsonObjectSize used as the safe FireWork document threshold.",
    ),
    queue_wait: bool = typer.Option(True, "--queue-wait/--no-queue-wait", help="Wait for queue-launched batches to finish before tell()."),
    queue_poll_seconds: int = typer.Option(10, "--queue-poll-seconds", min=1, help="Polling interval in seconds while waiting for queue batches."),
    queue_wait_timeout: int = typer.Option(0, "--queue-wait-timeout", min=0, help="Timeout in seconds for waiting on a queue batch. 0 means no timeout."),
    max_missing_result_retries: int = typer.Option(2, "--max-missing-result-retries", min=0, help="How many times to resubmit candidates missing result.json after terminal queue states."),
    archive_generations: bool = typer.Option(False, "--archive-generations/--no-archive-generations", help="Archive each completed generation into a single tar.gz."),
    delete_archived_workdirs: bool = typer.Option(False, "--delete-archived-workdirs/--keep-archived-workdirs", help="Delete candidate workdirs after successfully archiving a generation."),
    delete_archived_launchers: bool = typer.Option(True, "--delete-archived-launchers/--keep-archived-launchers", help="Delete terminal leaf launcher directories after they are safely archived."),
    archive_backlog: int = typer.Option(3, "--archive-backlog", min=1, help="Maximum completed generations queued for the single background archiver."),
):
    """
    Submit AND (optionally) launch a full optimization loop using FireWorks.

    NOTE: This is a simple synchronous loop (submit batch -> launch -> read results -> tell).
    """
    try:
        from gow.fw.bson_preflight import (
            htc_group_size_target,
            largest_safe_prefix_size,
            measure_workflow_bson,
            query_mongo_server_limits,
        )
        from gow.fw.launchpad import load_launchpad
        from gow.fw.tasks import append_run_result_record, rebuild_problem_results_jsonl, rebuild_run_results_jsonl, verify_run_results_complete
        from gow.fw.workflow import BatchEvalSpec, SingleEvalSpec, build_batch_evaluate_workflow
        from gow.optimizer import make_optimizer
    except Exception as e:
        raise typer.BadParameter(str(e)) from e

    config_abs = config.expanduser().resolve()
    results_dir = _resolve_results_dir(config_abs, outdir)

    if type(bson_safety_fraction).__name__ == "OptionInfo":
        bson_safety_fraction = 0.75
    archive_generations = _coerce_bool_option(archive_generations, False)
    delete_archived_workdirs = _coerce_bool_option(delete_archived_workdirs, False)

    problem = load_problem_config(config_abs)
    lp = load_launchpad(launchpad)
    mongo_limits = query_mongo_server_limits(lp)
    safe_bson_bytes = mongo_limits.safe_document_limit(bson_safety_fraction)
    auto_group_size, fixed_group_size = _parse_group_size_option(group_size)

    run_id_val = run_id or _default_run_id()
    launchers_dir = launch_dir.expanduser().resolve() if launch_dir else run_launchers_dir(results_dir, run_id_val)

    opt_cfg = problem.optimizer
    opt_kwargs = _optimizer_kwargs(opt_cfg)

    name_norm = str(opt_cfg.name).lower().strip()
    if name_norm in {"differential_evolution", "de"}:
        opt_kwargs.setdefault("population_size", opt_cfg.batch_size)
        if opt_cfg.max_evaluations % opt_cfg.batch_size != 0:
            raise ValueError(
                "Differential Evolution requires max_evaluations to be a multiple of batch_size "
                "(one full population per generation). "
                f"Got max_evaluations={opt_cfg.max_evaluations}, batch_size={opt_cfg.batch_size}."
            )

    optimizer = make_optimizer(opt_cfg.name, seed=opt_cfg.seed, **opt_kwargs)

    direction = str(problem.objective.direction).strip().lower()
    if direction not in {"minimize", "maximize"}:
        direction = "minimize"

    typer.echo(f"Problem: {problem.id}")
    typer.echo(f"run_id:  {run_id_val}")
    typer.echo(f"results_dir: {results_dir}")
    typer.echo(f"launchers_dir: {launchers_dir}")
    typer.echo(f"launcher: {launcher}")
    if str(launcher).strip().lower() == "queue":
        typer.echo(f"qadapter: {qadapter or 'auto'}")
        typer.echo(f"njobs_queue: {njobs_queue}")
    typer.echo(
        "MongoDB BSON limits: "
        f"maxBsonObjectSize={mongo_limits.max_bson_object_size / (1024 * 1024):.2f} MiB, "
        f"maxMessageSizeBytes={mongo_limits.max_message_size_bytes / (1024 * 1024):.2f} MiB, "
        f"safety={bson_safety_fraction:.3f} -> {safe_bson_bytes / (1024 * 1024):.2f} MiB"
    )
    typer.echo(f"group_size: {'auto' if auto_group_size else (fixed_group_size or 'batch_size')}")
    typer.echo(f"max_evaluations={opt_cfg.max_evaluations}  batch_size={opt_cfg.batch_size}")

    def _read_candidate_record(workdir: Path) -> dict[str, Any] | None:
        """
        Read <workdir>/result.json written by EvaluateCandidateTask.

        Returns the full record dict, or None if missing/unreadable.
        """
        result_path = workdir / "result.json"
        if not result_path.exists():
            return None
        try:
            return json.loads(result_path.read_text(encoding="utf-8"))
        except Exception:
            return None

    best_obj: float | None = None
    best_info: dict[str, Any] | None = None
    archiver = (
        AsyncGenerationArchiver(
            outdir=results_dir,
            run_id=run_id_val,
            delete_candidate_workdirs=delete_archived_workdirs,
            delete_launcher_dirs=delete_archived_launchers,
            max_pending=archive_backlog,
        )
        if archive_generations
        else None
    )

    n_done = 0
    while n_done < opt_cfg.max_evaluations:
        n_batch = min(opt_cfg.batch_size, opt_cfg.max_evaluations - n_done)
        generation_id = n_done // opt_cfg.batch_size
        generation_fw_ids: set[int] = set()

        candidates = optimizer.ask(problem, n_batch)

        candidate_ids: list[str] = []
        specs: list[SingleEvalSpec] = []
        for i, cand in enumerate(candidates):
            idx = n_done + i  # global index
            candidate_id = format_candidate_id(
                generation_id=generation_id,
                candidate_index=idx,
                run_id=run_id_val,
            )
            candidate_ids.append(candidate_id)

            specs.append(
                SingleEvalSpec(
                    problem_config=config_abs,
                    outdir=results_dir,
                    run_id=run_id_val,
                    candidate_id=candidate_id,
                    candidate_params=cand,
                    generation_id=generation_id,
                    candidate_index=idx,
                    attempt_index=0,
                )
            )

        specs_by_candidate_id = {spec.candidate_id: spec for spec in specs}
        pending_specs = list(specs)
        retry_counts: dict[str, int] = {spec.candidate_id: 0 for spec in specs}
        # Once BSON constrains a generation, keep the discovered cap for its retries.
        generation_bson_cap: int | None = None

        def _build_batch_wf(items: list[SingleEvalSpec]):
            return build_batch_evaluate_workflow(
                BatchEvalSpec(
                    problem_config=config_abs,
                    outdir=results_dir,
                    run_id=run_id_val,
                    items=items,
                )
            )

        while pending_specs:
            is_queue_launcher = str(launcher).strip().lower() == "queue"
            htc_target = (
                htc_group_size_target(len(pending_specs), njobs_queue)
                if auto_group_size and is_queue_launcher
                else len(pending_specs)
            )
            requested_group_size = (
                min(htc_target, generation_bson_cap)
                if auto_group_size and generation_bson_cap is not None
                else htc_target
                if auto_group_size
                else (fixed_group_size or len(pending_specs))
            )
            requested_group_size = max(1, min(requested_group_size, len(pending_specs)))

            # Plan the complete submission wave before inserting any Workflow.  This
            # prevents a fixed --group-size from partially submitting a generation
            # before a later, larger candidate payload is discovered to be unsafe.
            planned_chunk_sizes: list[int] = []
            planned_max_bson = 0
            cursor = 0
            while cursor < len(pending_specs):
                requested = min(requested_group_size, len(pending_specs) - cursor)
                remaining_specs = pending_specs[cursor:]
                chunk = remaining_specs[:requested]
                wf = _build_batch_wf(chunk)
                measurement = measure_workflow_bson(wf)

                if measurement.max_document_bytes > safe_bson_bytes:
                    safe_size, safe_measurement = largest_safe_prefix_size(
                        remaining_specs,
                        requested,
                        build_workflow=lambda prefix: _build_batch_wf(list(prefix)),
                        safe_document_bytes=safe_bson_bytes,
                    )
                    if safe_size <= 0:
                        raise RuntimeError(
                            "MongoDB BSON preflight failed: even one candidate produces a FireWorks "
                            f"document larger than the safe limit of {safe_bson_bytes / (1024 * 1024):.2f} MiB."
                        )
                    if not auto_group_size:
                        raise typer.BadParameter(
                            f"--group-size {requested} is unsafe for this generation: largest FireWorks "
                            f"document is {measurement.max_document_mib:.2f} MiB, while the server-aware "
                            f"safe limit is {safe_bson_bytes / (1024 * 1024):.2f} MiB. "
                            f"Use --group-size {safe_size} or smaller, or use --group-size auto."
                        )
                    requested = safe_size
                    generation_bson_cap = (
                        safe_size
                        if generation_bson_cap is None
                        else min(generation_bson_cap, safe_size)
                    )
                    measurement = safe_measurement or measure_workflow_bson(
                        _build_batch_wf(remaining_specs[:requested])
                    )
                    requested_group_size = min(requested_group_size, requested)

                planned_chunk_sizes.append(requested)
                planned_max_bson = max(planned_max_bson, measurement.max_document_bytes)
                cursor += requested

            if auto_group_size:
                typer.echo(
                    "AUTO group-size plan: "
                    f"pending={len(pending_specs)}, njobs_queue={njobs_queue if is_queue_launcher else 0}, "
                    f"HTC target={htc_target}, BSON cap={generation_bson_cap or 'not limiting'}, "
                    f"groups={len(planned_chunk_sizes)}, sizes={min(planned_chunk_sizes)}..{max(planned_chunk_sizes)}, "
                    f"max BSON={planned_max_bson / (1024 * 1024):.2f} MiB"
                )
            else:
                typer.echo(
                    "BSON preflight OK: "
                    f"group_size={requested_group_size}, groups={len(planned_chunk_sizes)}, "
                    f"max BSON={planned_max_bson / (1024 * 1024):.2f} MiB"
                )

            submitted_fw_ids: list[int] = []
            submitted_candidate_ids: list[str] = []
            cursor = 0
            for chunk_size in planned_chunk_sizes:
                chunk = pending_specs[cursor : cursor + chunk_size]
                wf = _build_batch_wf(chunk)
                # Re-measure immediately before add_wf().  This is deliberately
                # redundant with planning: no oversized document reaches PyMongo.
                measurement = measure_workflow_bson(wf)
                if measurement.max_document_bytes > safe_bson_bytes:
                    raise RuntimeError(
                        "FireWork BSON payload changed after preflight and now exceeds the "
                        f"server-aware safe limit ({measurement.max_document_mib:.2f} MiB > "
                        f"{safe_bson_bytes / (1024 * 1024):.2f} MiB). Nothing further was submitted."
                    )
                id_map = lp.add_wf(wf)
                new_fw_ids = _extract_fw_ids(id_map)
                submitted_fw_ids.extend(new_fw_ids)
                generation_fw_ids.update(new_fw_ids)
                submitted_candidate_ids.extend([item.candidate_id for item in chunk])
                cursor += chunk_size

            typer.echo(
                f"Submitted batch of {len(submitted_candidate_ids)} candidate(s) in "
                f"{len(planned_chunk_sizes)} FireWork group(s)."
            )

            if launch:
                with _pushd(launchers_dir):
                    _launch_fireworks(
                        lp=lp,
                        mode=launcher,
                        nlaunches=nlaunches,
                        sleep=sleep,
                        qadapter_path=qadapter,
                        njobs_queue=njobs_queue,
                        reserve=reserve,
                    )
                typer.echo(f"Launch complete for current queue using launcher={launcher}.")

            is_queue = str(launcher).strip().lower() == "queue"
            if is_queue and launch and queue_wait:
                complete, missing_ids = _wait_for_batch_results(
                    lp,
                    results_dir,
                    run_id_val,
                    submitted_candidate_ids,
                    submitted_fw_ids,
                    poll_seconds=queue_poll_seconds,
                    wait_timeout=queue_wait_timeout,
                )
            else:
                complete = True
                missing_ids = []

            if missing_ids:
                typer.echo(f"Batch ended with {len(missing_ids)} missing result(s).")

            next_pending_specs: list[SingleEvalSpec] = []
            for missing_id in missing_ids:
                prev = specs_by_candidate_id[missing_id]
                retry_counts[missing_id] += 1
                if retry_counts[missing_id] <= max_missing_result_retries:
                    next_pending_specs.append(
                        SingleEvalSpec(
                            problem_config=prev.problem_config,
                            outdir=prev.outdir,
                            run_id=prev.run_id,
                            candidate_id=prev.candidate_id,
                            candidate_params=prev.candidate_params,
                            generation_id=prev.generation_id,
                            candidate_index=prev.candidate_index,
                            attempt_index=prev.attempt_index + retry_counts[missing_id],
                            context_override=prev.context_override,
                        )
                    )
                else:
                    record = _build_missing_result_record(
                        problem=problem,
                        results_dir=results_dir,
                        run_id=run_id_val,
                        candidate_id=prev.candidate_id,
                        candidate_params=prev.candidate_params,
                        generation_id=prev.generation_id,
                        candidate_index=prev.candidate_index,
                        attempt_index=prev.attempt_index + retry_counts[missing_id],
                        reason=(
                            f"Missing result.json after {max_missing_result_retries} retry attempts "
                            f"for candidate {prev.candidate_id}"
                        ),
                    )
                    append_run_result_record(results_dir, run_id_val, record)

            if next_pending_specs:
                typer.echo(f"Retrying {len(next_pending_specs)} candidate(s) missing results.")
            pending_specs = next_pending_specs

        # IMPORTANT:
        # - tell() uses fitness in the *same order* as `candidates`
        # - best_info MUST be derived from the evaluated record/result to avoid mismatches
        fitness_dicts: list[dict[str, Any]] = []
        for i, candidate_id in enumerate(candidate_ids):
            workdir = candidate_workdir(results_dir, run_id_val, candidate_id)
            rec = _read_candidate_record(workdir)

            if rec is None:
                fit: dict[str, Any] = {"status": "failed", "error": f"Missing/unreadable result.json at {workdir / 'result.json'}"}
            else:
                fit = (rec.get("fitness") or {}) if isinstance(rec.get("fitness"), dict) else {}
                if "status" not in fit:
                    fit["status"] = "failed"

            fitness_dicts.append(fit)

            obj_val = _coerce_objective(fit)
            if obj_val is None:
                continue

            if best_obj is None or _is_better(obj_val, best_obj, direction):
                best_obj = obj_val

                # Prefer evaluated record params (guaranteed consistent with objective).
                # Fall back to the asked candidate only if needed.
                params = None
                if rec is not None and isinstance(rec.get("params"), dict):
                    params = rec["params"]
                else:
                    params = candidates[i] if isinstance(candidates[i], dict) else {"candidate": candidates[i]}

                best_info = {
                    "candidate_id": candidate_id,
                    "generation_id": generation_id,
                    "objective": obj_val,
                    "params": params,
                }

        try:
            optimizer.tell(candidates, fitness_dicts)
        except Exception as e:
            typer.echo(f"Warning: optimizer.tell failed: {e}")

        finalize_generation(
            outdir=results_dir,
            run_id=run_id_val,
            problem_id=problem.id,
            generation_id=generation_id,
            candidate_ids=candidate_ids,
            max_evaluations=opt_cfg.max_evaluations,
            direction=direction,
            completed_generations=(n_done + n_batch + opt_cfg.batch_size - 1) // opt_cfg.batch_size,
            final=(n_done + n_batch) >= opt_cfg.max_evaluations,
        )
        launch_manifest = write_generation_launch_manifest(
            outdir=results_dir,
            run_id=run_id_val,
            generation_id=generation_id,
            fw_ids=sorted(generation_fw_ids),
            lp=lp,
        )
        if archiver is not None:
            archiver.submit(
                generation_id=generation_id,
                candidate_ids=candidate_ids,
                launch_manifest=launch_manifest,
            )

        n_done += n_batch

    if archiver is not None:
        archiver.close()
        if delete_archived_launchers:
            finalize_run_launcher_tree(
                outdir=results_dir,
                run_id=run_id_val,
                lp=lp,
                delete_source=True,
            )

    ok_run, actual_run = verify_run_results_complete(results_dir, run_id_val, opt_cfg.max_evaluations)
    if not ok_run:
        raise RuntimeError(
            f"Run {run_id_val} incomplete before rebuilding run/global results: expected {opt_cfg.max_evaluations}, found {actual_run}"
        )
    rebuild_run_results_jsonl(results_dir, run_id_val)
    rebuild_problem_results_jsonl(results_dir)

    summary_path = _write_summary_json(
        results_dir=results_dir,
        problem_id=problem.id,
        run_id=run_id_val,
        max_evaluations=opt_cfg.max_evaluations,
        direction=direction,
        best_candidate=best_info,
    )

    typer.echo("Done.")
    typer.echo(f"Results dir: {results_dir}")
    typer.echo(f"Results.jsonl: {results_dir / 'results.jsonl'}")
    typer.echo(f"Run results.jsonl: {run_root(results_dir, run_id_val) / 'results.jsonl'}")
    typer.echo(f"Summary: {summary_path}")
    typer.echo(f"Launchers dir: {launchers_dir}")


if __name__ == "__main__":
    app()
