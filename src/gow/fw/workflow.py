from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from gow.config import load_problem_config


def _ensure_fireworks_imports():
    try:
        from fireworks import Firework, Workflow  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "FireWorks is not installed. Install with: pip install -e '.[fireworks]'"
        ) from e


_ensure_fireworks_imports()
from fireworks import Firework, Workflow  # type: ignore  # noqa: E402

from .tasks import (  # noqa: E402
    AppendResultJsonlTask,
    EvaluateBatchTask,
    EvaluateCandidateTask,
    _to_jsonable,
)


def default_run_id() -> str:
    return str(uuid.uuid4())


@dataclass(frozen=True)
class SingleEvalSpec:
    problem_config: Path
    outdir: Path
    run_id: str
    candidate_id: str
    candidate_params: Dict[str, Any]

    generation_id: Optional[int] = None
    candidate_index: Optional[int] = None
    attempt_index: int = 0

    context_override: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class BatchEvalSpec:
    problem_config: Path
    outdir: Path
    run_id: str
    items: list[SingleEvalSpec]


def _batch_item_payload(spec: SingleEvalSpec) -> Dict[str, Any]:
    """Serialize only candidate-specific batch fields.

    problem_config, outdir and run_id are shared by the whole FireWork and are
    intentionally kept out of every item to keep FireWorks/Mongo documents small.
    """
    payload: Dict[str, Any] = {
        "candidate_id": spec.candidate_id,
        "candidate_params": _to_jsonable(spec.candidate_params),
        "generation_id": spec.generation_id,
        "candidate_index": spec.candidate_index,
        "attempt_index": spec.attempt_index,
    }
    if spec.context_override is not None:
        payload["context_override"] = _to_jsonable(spec.context_override)
    return payload


def build_single_evaluate_workflow(spec: SingleEvalSpec) -> Workflow:
    """
    Workflow (single FireWork, two Firetasks):
      1) EvaluateCandidateTask -> writes runs/<run_id>/<candidate_id>/result.json
      2) AppendResultJsonlTask -> appends to runs/<run_id>/results.jsonl

    The problem-level <outdir>/results.jsonl is rebuilt after the run completes.

    This design reduces launches by 2x (previously 2 FireWorks per candidate).
    """
    problem_config_abs = Path(spec.problem_config).expanduser().resolve()
    outdir_abs = Path(spec.outdir).expanduser().resolve()

    problem = load_problem_config(problem_config_abs)

    eval_task_params: Dict[str, Any] = {
        "problem_config": str(problem_config_abs),
        "outdir": str(outdir_abs),
        "run_id": spec.run_id,
        "candidate_id": spec.candidate_id,
        "candidate_params": _to_jsonable(spec.candidate_params),
        "generation_id": spec.generation_id,
        "candidate_index": spec.candidate_index,
        "attempt_index": spec.attempt_index,
    }
    if spec.context_override:
        eval_task_params["context_override"] = _to_jsonable(spec.context_override)

    append_task_params: Dict[str, Any] = {
        "outdir": str(outdir_abs),
        "problem_id": problem.id,
        "run_id": spec.run_id,
        "candidate_id": spec.candidate_id,
        "generation_id": spec.generation_id,
        "candidate_index": spec.candidate_index,
        "attempt_index": spec.attempt_index,
    }

    fw = Firework(
        [
            EvaluateCandidateTask(eval_task_params),
            AppendResultJsonlTask(append_task_params),
        ],
        name=f"evaluate+append:{problem.id}:{spec.run_id}:{spec.candidate_id}",
        spec={
            "problem_id": problem.id,
            "run_id": spec.run_id,
            "candidate_id": spec.candidate_id,
            "generation_id": spec.generation_id,
            "candidate_index": spec.candidate_index,
            "attempt_index": spec.attempt_index,
        },
    )

    wf_name = f"gow-single-eval:{problem.id}:{spec.run_id}:{spec.candidate_id}"
    return Workflow([fw], name=wf_name)


def build_batch_evaluate_workflow(spec: BatchEvalSpec) -> Workflow:
    """Build one slim FireWork for a batch of candidates.

    EvaluateBatchTask persists one result.json per candidate. No result payload is
    copied back into the FireWork spec and there is no append task: generation
    consolidation is performed later by the coordinator from those result files.
    """
    problem_config_abs = Path(spec.problem_config).expanduser().resolve()
    outdir_abs = Path(spec.outdir).expanduser().resolve()

    problem = load_problem_config(problem_config_abs)
    items_payload = [_batch_item_payload(item) for item in spec.items]
    generation_ids = {item.generation_id for item in spec.items}
    generation_id = next(iter(generation_ids)) if len(generation_ids) == 1 else None

    fw = Firework(
        [
            EvaluateBatchTask(
                {
                    "problem_config": str(problem_config_abs),
                    "outdir": str(outdir_abs),
                    "run_id": spec.run_id,
                    "items": items_payload,
                }
            ),
        ],
        name=f"evaluate-batch:{problem.id}:{spec.run_id}:n{len(spec.items)}",
        spec={
            "problem_id": problem.id,
            "run_id": spec.run_id,
            "generation_id": generation_id,
            "batch_size": len(spec.items),
        },
    )

    wf_name = f"gow-batch-eval:{problem.id}:{spec.run_id}:n{len(spec.items)}"
    return Workflow([fw], name=wf_name)
