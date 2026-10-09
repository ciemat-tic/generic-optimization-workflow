from __future__ import annotations

import gzip
import json
import os
import queue
import shutil
import subprocess
import tarfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from gow.candidate_ids import parse_candidate_id
from gow.layout import candidate_workdir, run_launchers_dir, run_root


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def generation_results_dir(outdir: Path | str, run_id: str) -> Path:
    return run_root(outdir, run_id) / "generations"


def generation_results_path(outdir: Path | str, run_id: str, generation_id: int) -> Path:
    return generation_results_dir(outdir, run_id) / f"g{int(generation_id):06d}.jsonl"


def generation_archive_dir(outdir: Path | str, run_id: str) -> Path:
    return run_root(outdir, run_id) / "archives"


def generation_archive_path(outdir: Path | str, run_id: str, generation_id: int) -> Path:
    return generation_archive_dir(outdir, run_id) / f"g{int(generation_id):06d}.tar.gz"


_TERMINAL_FW_STATES = {"COMPLETED", "FIZZLED", "ARCHIVED"}


def generation_launch_manifest_path(outdir: Path | str, run_id: str, generation_id: int) -> Path:
    return generation_results_dir(outdir, run_id) / f"g{int(generation_id):06d}.launches.json"


def _launch_record(lp: Any, launch_ref: Any, *, archived: bool) -> dict[str, Any]:
    launch = launch_ref
    launch_id = None
    if isinstance(launch_ref, (int, str)):
        try:
            launch_id = int(launch_ref)
        except Exception:
            launch_id = launch_ref
        getter = getattr(lp, "get_launch_by_id", None)
        if callable(getter):
            try:
                launch = getter(launch_ref)
            except Exception as exc:
                return {
                    "launch_id": launch_id,
                    "state": None,
                    "launch_dir": None,
                    "archived": bool(archived),
                    "inspection_error": str(exc),
                }
    if isinstance(launch, dict):
        return {
            "launch_id": launch.get("launch_id", launch_id),
            "state": launch.get("state"),
            "launch_dir": launch.get("launch_dir"),
            "archived": bool(archived),
        }
    return {
        "launch_id": getattr(launch, "launch_id", launch_id),
        "state": getattr(launch, "state", None),
        "launch_dir": getattr(launch, "launch_dir", None),
        "archived": bool(archived),
    }


def collect_firework_launch_records(lp: Any, fw_ids: Sequence[int]) -> list[dict[str, Any]]:
    """Collect compact FireWork/Launch metadata without serializing FireWork payloads."""
    rows: list[dict[str, Any]] = []
    for fw_id in sorted({int(v) for v in fw_ids}):
        fw_data: Any = None
        error: str | None = None
        try:
            getter = getattr(lp, "get_fw_dict_by_id", None)
            if callable(getter):
                fw_data = getter(fw_id)
            elif callable(getattr(lp, "get_fw_by_id", None)):
                fw_data = lp.get_fw_by_id(fw_id)
        except Exception as exc:
            error = str(exc)

        if isinstance(fw_data, dict):
            state = fw_data.get("state")
            live_refs = list(fw_data.get("launches") or [])
            archived_refs = list(fw_data.get("archived_launches") or [])
        else:
            state = getattr(fw_data, "state", None) if fw_data is not None else None
            live_refs = list(getattr(fw_data, "launches", []) or []) if fw_data is not None else []
            archived_refs = list(getattr(fw_data, "archived_launches", []) or []) if fw_data is not None else []

        launches = [_launch_record(lp, ref, archived=False) for ref in live_refs]
        launches.extend(_launch_record(lp, ref, archived=True) for ref in archived_refs)
        row = {
            "fw_id": fw_id,
            "state": str(state) if state is not None else None,
            "terminal": str(state or "").upper() in _TERMINAL_FW_STATES,
            "launches": launches,
        }
        if error is not None:
            row["inspection_error"] = error
        rows.append(row)
    return rows


def write_generation_launch_manifest(
    *,
    outdir: Path | str,
    run_id: str,
    generation_id: int,
    fw_ids: Sequence[int],
    lp: Any,
) -> Path:
    """Persist the FireWork ids (including retries) and their launch directories."""
    path = generation_launch_manifest_path(outdir, run_id, generation_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    fireworks = collect_firework_launch_records(lp, fw_ids)
    payload = {
        "created_at": _now_iso(),
        "run_id": str(run_id),
        "generation_id": int(generation_id),
        "fw_ids": sorted({int(v) for v in fw_ids}),
        "all_terminal": bool(fireworks) and all(row.get("terminal") is True for row in fireworks),
        "fireworks": fireworks,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)
    return path


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
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


def _unique_key(record: dict[str, Any]) -> tuple[str | None, str | None, str | None]:
    rid = record.get("run_id")
    aid = record.get("attempt_id")
    cid = record.get("candidate_id")
    return (
        str(rid) if rid is not None else None,
        str(aid) if aid is not None else None,
        str(cid) if cid is not None else None,
    )


def _sort_key(obj: dict[str, Any]) -> tuple[int, int, str, int]:
    return (
        obj.get("generation_id") if isinstance(obj.get("generation_id"), int) else 10**12,
        obj.get("candidate_index") if isinstance(obj.get("candidate_index"), int) else 10**12,
        str(obj.get("candidate_id", "")),
        obj.get("attempt_index") if isinstance(obj.get("attempt_index"), int) else 10**12,
    )


def _best_from_records(records: Iterable[dict[str, Any]], direction: str) -> dict[str, Any] | None:
    best_obj: float | None = None
    best_record: dict[str, Any] | None = None
    for rec in records:
        fit = rec.get("fitness")
        if not isinstance(fit, dict) or fit.get("status") != "ok":
            continue
        obj = fit.get("objective")
        try:
            obj_val = float(obj)
        except Exception:
            continue
        if best_obj is None:
            best_obj = obj_val
            best_record = rec
            continue
        if direction == "maximize" and obj_val > best_obj:
            best_obj = obj_val
            best_record = rec
        elif direction != "maximize" and obj_val < best_obj:
            best_obj = obj_val
            best_record = rec
    if best_record is None:
        return None
    return {
        "candidate_id": best_record.get("candidate_id"),
        "generation_id": best_record.get("generation_id"),
        "objective": best_obj,
        "params": best_record.get("params"),
        "attempt_id": best_record.get("attempt_id"),
    }


def verify_generation_results_complete(
    outdir: Path | str,
    run_id: str,
    generation_id: int,
    candidate_ids: Sequence[str],
) -> tuple[bool, int, list[str]]:
    outdir = Path(outdir).expanduser().resolve()
    missing: list[str] = []
    count = 0
    for candidate_id in candidate_ids:
        result_path = candidate_workdir(outdir, run_id, candidate_id) / "result.json"
        if result_path.exists():
            count += 1
        else:
            missing.append(candidate_id)
    return not missing, count, missing


def rebuild_generation_results_jsonl(
    outdir: Path | str,
    run_id: str,
    generation_id: int,
    candidate_ids: Sequence[str],
) -> Path:
    outdir = Path(outdir).expanduser().resolve()
    gen_path = generation_results_path(outdir, run_id, generation_id)
    gen_path.parent.mkdir(parents=True, exist_ok=True)

    seen: set[tuple[str | None, str | None, str | None]] = set()
    records: list[dict[str, Any]] = []
    for candidate_id in candidate_ids:
        result_path = candidate_workdir(outdir, run_id, candidate_id) / "result.json"
        obj = _load_json(result_path)
        if not isinstance(obj, dict):
            continue
        key = _unique_key(obj)
        if key in seen:
            continue
        seen.add(key)
        if obj.get("generation_id") is None:
            parts = parse_candidate_id(candidate_id)
            if parts is not None:
                obj["generation_id"] = parts.generation_id
                obj.setdefault("candidate_index", parts.candidate_index)
                obj.setdefault("candidate_local_id", parts.candidate_local_id)
        records.append(obj)

    records.sort(key=_sort_key)
    with gen_path.open("w", encoding="utf-8") as f:
        for obj in records:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    return gen_path


def _read_generation_summary_rows(summary_path: Path) -> list[dict[str, Any]]:
    return list(_iter_jsonl(summary_path))


def update_run_summary(
    *,
    outdir: Path | str,
    run_id: str,
    problem_id: str,
    max_evaluations: int,
    direction: str,
    generation_id: int,
    generation_results_file: Path,
    expected_count: int,
    actual_count: int,
    completed_generations: int,
    final: bool,
) -> tuple[Path, Path, Path]:
    outdir = Path(outdir).expanduser().resolve()
    run_dir = run_root(outdir, run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    generation_records = list(_iter_jsonl(generation_results_file))
    generation_best = _best_from_records(generation_records, direction)

    summary_jsonl_path = run_dir / "summary.jsonl"
    rows = _read_generation_summary_rows(summary_jsonl_path) if summary_jsonl_path.exists() else []
    rows = [row for row in rows if int(row.get("generation_id", -1)) != int(generation_id)]
    row = {
        "timestamp": _now_iso(),
        "problem_id": problem_id,
        "run_id": run_id,
        "generation_id": int(generation_id),
        "expected_count": int(expected_count),
        "actual_count": int(actual_count),
        "generation_results_file": str(generation_results_file),
        "generation_best": generation_best,
    }
    rows.append(row)
    rows.sort(key=lambda obj: int(obj.get("generation_id", 10**12)))
    with summary_jsonl_path.open("w", encoding="utf-8") as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    best_so_far = _best_from_records(
        ({
            "candidate_id": r.get("generation_best", {}).get("candidate_id"),
            "generation_id": r.get("generation_best", {}).get("generation_id"),
            "params": r.get("generation_best", {}).get("params"),
            "attempt_id": r.get("generation_best", {}).get("attempt_id"),
            "fitness": {"status": "ok", "objective": r.get("generation_best", {}).get("objective")},
        } for r in rows if isinstance(r.get("generation_best"), dict)),
        direction,
    )

    evaluations_done = sum(int(r.get("actual_count", 0)) for r in rows)
    summary = {
        "problem_id": problem_id,
        "run_id": run_id,
        "max_evaluations": int(max_evaluations),
        "objective": {"direction": direction},
        "best": best_so_far,
        "results_file": str(outdir / "results.jsonl"),
        "run_results_file": str(run_dir / "results.jsonl"),
        "run_root": str(run_dir),
        "outdir": str(outdir),
        "summary_jsonl": str(summary_jsonl_path),
        "completed_generations": int(completed_generations),
        "evaluations_done": evaluations_done,
        "finalized": bool(final),
        "updated_at": _now_iso(),
    }
    run_summary = run_dir / "summary.json"
    problem_summary = outdir / "summary.json"
    run_summary.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    problem_summary.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary_jsonl_path, run_summary, problem_summary


def finalize_generation(
    *,
    outdir: Path | str,
    run_id: str,
    problem_id: str,
    generation_id: int,
    candidate_ids: Sequence[str],
    max_evaluations: int,
    direction: str,
    completed_generations: int,
    final: bool = False,
) -> Path:
    ok, actual_count, missing = verify_generation_results_complete(outdir, run_id, generation_id, candidate_ids)
    if not ok:
        raise RuntimeError(
            f"Generation {generation_id} of run {run_id} incomplete: missing result.json for {len(missing)} candidate(s): {', '.join(missing[:10])}"
        )
    gen_path = rebuild_generation_results_jsonl(outdir, run_id, generation_id, candidate_ids)
    update_run_summary(
        outdir=outdir,
        run_id=run_id,
        problem_id=problem_id,
        max_evaluations=max_evaluations,
        direction=direction,
        generation_id=generation_id,
        generation_results_file=gen_path,
        expected_count=len(candidate_ids),
        actual_count=actual_count,
        completed_generations=completed_generations,
        final=final,
    )
    return gen_path


def iter_generation_shards(outdir: Path | str, run_id: str) -> Iterable[Path]:
    gen_dir = generation_results_dir(outdir, run_id)
    if not gen_dir.exists():
        return
    for path in sorted(gen_dir.glob("g*.jsonl")):
        if path.is_file():
            yield path


def merge_runs(
    *,
    outdir: Path | str,
    target_run_id: str,
    source_run_ids: Sequence[str],
) -> Path:
    outdir = Path(outdir).expanduser().resolve()
    target_run = run_root(outdir, target_run_id)
    target_run.mkdir(parents=True, exist_ok=True)
    target_results = target_run / "results.jsonl"

    seen: set[tuple[str | None, str | None, str | None]] = set()
    rows: list[dict[str, Any]] = []
    for run_id in source_run_ids:
        path = run_root(outdir, run_id) / "results.jsonl"
        for obj in _iter_jsonl(path):
            key = _unique_key(obj)
            if key in seen:
                continue
            seen.add(key)
            rows.append(obj)
    rows.sort(key=_sort_key)
    with target_results.open("w", encoding="utf-8") as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    return target_results


def archive_generation_workdirs(
    *,
    outdir: Path | str,
    run_id: str,
    generation_id: int,
    candidate_ids: Sequence[str],
    delete_source: bool = False,
) -> Path:
    outdir = Path(outdir).expanduser().resolve()
    archive_path = generation_archive_path(outdir, run_id, generation_id)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, mode="w:gz") as tar:
        for candidate_id in candidate_ids:
            workdir = candidate_workdir(outdir, run_id, candidate_id)
            if not workdir.exists():
                continue
            tar.add(workdir, arcname=workdir.relative_to(run_root(outdir, run_id)).as_posix())
    if delete_source:
        import shutil

        for candidate_id in candidate_ids:
            workdir = candidate_workdir(outdir, run_id, candidate_id)
            if workdir.exists():
                shutil.rmtree(workdir)
    return archive_path


_TERMINAL_LAUNCH_STATES = {"COMPLETED", "FIZZLED", "ARCHIVED"}


def candidate_generation_archive_path(outdir: Path | str, run_id: str, generation_id: int) -> Path:
    return generation_archive_dir(outdir, run_id) / "candidates" / f"g{int(generation_id):06d}.tar.zst"


def launcher_generation_archive_path(outdir: Path | str, run_id: str, generation_id: int) -> Path:
    return generation_archive_dir(outdir, run_id) / "launchers" / f"g{int(generation_id):06d}.tar.zst"


def generation_archive_report_path(outdir: Path | str, run_id: str, generation_id: int) -> Path:
    return generation_archive_dir(outdir, run_id) / "manifests" / f"g{int(generation_id):06d}.archive.json"


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _create_tar_zst_transactional(*, archive_path: Path, base_dir: Path, source_dirs: Sequence[Path]) -> Path | None:
    """Create archive.partial -> validate -> atomic rename, without deleting sources."""
    sources = [Path(p).resolve() for p in source_dirs if Path(p).exists()]
    if not sources:
        return None
    zstd = shutil.which("zstd")
    if zstd is None:
        raise RuntimeError("zstd executable is required for .tar.zst generation archives")

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    partial = archive_path.with_suffix(archive_path.suffix + ".partial")
    partial.unlink(missing_ok=True)

    proc = subprocess.Popen([zstd, "-T0", "-1", "-q", "-o", str(partial)], stdin=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        with tarfile.open(fileobj=proc.stdin, mode="w|") as tar:
            for source in sources:
                if not _is_relative_to(source, base_dir):
                    raise RuntimeError(f"Refusing to archive path outside run root: {source}")
                tar.add(source, arcname=source.relative_to(base_dir).as_posix(), recursive=True)
    finally:
        try:
            proc.stdin.close()
        except Exception:
            pass
    rc = proc.wait()
    if rc != 0:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"zstd failed with exit code {rc} while creating {archive_path}")

    check = subprocess.run([zstd, "-t", "-q", str(partial)], check=False)
    if check.returncode != 0:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"zstd validation failed for {partial}")
    os.replace(partial, archive_path)
    return archive_path


def _safe_launcher_leaf_dirs(manifest_path: Path, launchers_root: Path) -> tuple[list[Path], list[dict[str, Any]]]:
    payload = _load_json(manifest_path)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Launch manifest is missing or invalid: {manifest_path}")
    if payload.get("all_terminal") is not True:
        raise RuntimeError(f"Refusing to archive non-terminal generation launches: {manifest_path}")

    safe: list[Path] = []
    skipped: list[dict[str, Any]] = []
    seen: set[Path] = set()
    launchers_root = launchers_root.resolve()
    for fw in payload.get("fireworks") or []:
        if not isinstance(fw, dict) or fw.get("terminal") is not True:
            continue
        for launch in fw.get("launches") or []:
            if not isinstance(launch, dict):
                continue
            raw = launch.get("launch_dir")
            state = str(launch.get("state") or "").upper()
            if not raw or state not in _TERMINAL_LAUNCH_STATES:
                skipped.append({"launch_dir": raw, "reason": f"non-terminal-or-unknown-state:{state or 'NONE'}"})
                continue
            path = Path(str(raw)).expanduser().resolve()
            if not _is_relative_to(path, launchers_root) or path == launchers_root:
                skipped.append({"launch_dir": str(path), "reason": "outside-launchers-root"})
                continue
            if (path / "FW_offline.json").exists():
                skipped.append({"launch_dir": str(path), "reason": "offline-recovery-marker"})
                continue
            if path.exists() and path not in seen:
                safe.append(path)
                seen.add(path)
    return safe, skipped


def archive_generation_payload(
    *,
    outdir: Path | str,
    run_id: str,
    generation_id: int,
    candidate_ids: Sequence[str],
    launch_manifest: Path | str,
    delete_candidate_workdirs: bool,
    delete_launcher_dirs: bool,
) -> dict[str, Any]:
    """Archive one completed generation's candidate and leaf launcher directories."""
    outdir = Path(outdir).expanduser().resolve()
    run_dir = run_root(outdir, run_id).resolve()
    candidate_dirs = [candidate_workdir(outdir, run_id, cid) for cid in candidate_ids]
    launcher_dirs, skipped = _safe_launcher_leaf_dirs(Path(launch_manifest), run_launchers_dir(outdir, run_id))

    candidate_archive = _create_tar_zst_transactional(
        archive_path=candidate_generation_archive_path(outdir, run_id, generation_id),
        base_dir=run_dir,
        source_dirs=candidate_dirs,
    )
    launcher_archive = _create_tar_zst_transactional(
        archive_path=launcher_generation_archive_path(outdir, run_id, generation_id),
        base_dir=run_dir,
        source_dirs=launcher_dirs,
    )

    # Delete only after the corresponding archive has been atomically published.
    if delete_candidate_workdirs and candidate_archive is not None:
        for path in candidate_dirs:
            if path.exists():
                shutil.rmtree(path)
    if delete_launcher_dirs and launcher_archive is not None:
        for path in launcher_dirs:
            if path.exists():
                shutil.rmtree(path)

    report = {
        "created_at": _now_iso(),
        "run_id": str(run_id),
        "generation_id": int(generation_id),
        "candidate_archive": str(candidate_archive) if candidate_archive else None,
        "candidate_dirs": len([p for p in candidate_dirs if p.exists()]) if not delete_candidate_workdirs else len(candidate_dirs),
        "launcher_archive": str(launcher_archive) if launcher_archive else None,
        "launcher_dirs": len(launcher_dirs),
        "skipped_launchers": skipped,
        "delete_candidate_workdirs": bool(delete_candidate_workdirs),
        "delete_launcher_dirs": bool(delete_launcher_dirs),
    }
    report_path = generation_archive_report_path(outdir, run_id, generation_id)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = report_path.with_suffix(report_path.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(report_path)
    return report


class AsyncGenerationArchiver:
    """Single background archive worker with bounded backlog/backpressure."""

    def __init__(
        self,
        *,
        outdir: Path | str,
        run_id: str,
        delete_candidate_workdirs: bool,
        delete_launcher_dirs: bool,
        max_pending: int = 3,
    ) -> None:
        self.outdir = Path(outdir).expanduser().resolve()
        self.run_id = str(run_id)
        self.delete_candidate_workdirs = bool(delete_candidate_workdirs)
        self.delete_launcher_dirs = bool(delete_launcher_dirs)
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=max(1, int(max_pending)))
        self._errors: list[BaseException] = []
        self._thread = threading.Thread(target=self._worker, name=f"gow-archiver-{self.run_id}", daemon=True)
        self._thread.start()

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                generation_id, candidate_ids, manifest_path = item
                archive_generation_payload(
                    outdir=self.outdir,
                    run_id=self.run_id,
                    generation_id=generation_id,
                    candidate_ids=candidate_ids,
                    launch_manifest=manifest_path,
                    delete_candidate_workdirs=self.delete_candidate_workdirs,
                    delete_launcher_dirs=self.delete_launcher_dirs,
                )
            except BaseException as exc:
                self._errors.append(exc)
            finally:
                self._queue.task_done()

    def submit(self, *, generation_id: int, candidate_ids: Sequence[str], launch_manifest: Path | str) -> None:
        if self._errors:
            raise RuntimeError(f"Generation archiver failed: {self._errors[0]}") from self._errors[0]
        self._queue.put((int(generation_id), list(candidate_ids), Path(launch_manifest)))

    def close(self) -> None:
        self._queue.join()
        self._queue.put(None)
        self._thread.join()
        if self._errors:
            raise RuntimeError(f"Generation archiver failed: {self._errors[0]}") from self._errors[0]



def iter_generation_launch_manifests(outdir: Path | str, run_id: str) -> Iterable[Path]:
    gen_dir = generation_results_dir(outdir, run_id)
    if not gen_dir.exists():
        return
    for path in sorted(gen_dir.glob("g*.launches.json")):
        if path.is_file():
            yield path


def run_firework_ids_from_manifests(outdir: Path | str, run_id: str) -> list[int]:
    fw_ids: set[int] = set()
    for path in iter_generation_launch_manifests(outdir, run_id):
        payload = _load_json(path)
        if not isinstance(payload, dict):
            continue
        for value in payload.get("fw_ids") or []:
            try:
                fw_ids.add(int(value))
            except Exception:
                continue
    return sorted(fw_ids)


def final_launcher_archive_path(outdir: Path | str, run_id: str) -> Path:
    return generation_archive_dir(outdir, run_id) / "launchers" / "run-final.tar.zst"


def final_launcher_report_path(outdir: Path | str, run_id: str) -> Path:
    return generation_archive_dir(outdir, run_id) / "launchers" / "run-final.json"


def finalize_run_launcher_tree(
    *,
    outdir: Path | str,
    run_id: str,
    lp: Any,
    delete_source: bool = True,
) -> Path | None:
    """Archive the remaining pilot/block launcher tree after every FireWork is terminal."""
    outdir = Path(outdir).expanduser().resolve()
    run_dir = run_root(outdir, run_id).resolve()
    launchers_root = run_launchers_dir(outdir, run_id).resolve()
    if not launchers_root.exists():
        return None

    fw_ids = run_firework_ids_from_manifests(outdir, run_id)
    if not fw_ids:
        raise RuntimeError(
            f"Refusing final launcher cleanup for run {run_id}: no generation launch manifests found"
        )
    fireworks = collect_firework_launch_records(lp, fw_ids)
    unsafe = [row for row in fireworks if row.get("terminal") is not True or row.get("inspection_error")]
    if unsafe:
        details = ", ".join(f"{row.get('fw_id')}:{row.get('state')}" for row in unsafe[:10])
        raise RuntimeError(
            f"Refusing final launcher cleanup for run {run_id}: non-terminal/uninspectable FireWorks: {details}"
        )

    offline_markers = list(launchers_root.rglob("FW_offline.json"))
    if offline_markers:
        raise RuntimeError(
            f"Refusing final launcher cleanup for run {run_id}: offline recovery markers remain ({len(offline_markers)})"
        )

    archive = _create_tar_zst_transactional(
        archive_path=final_launcher_archive_path(outdir, run_id),
        base_dir=run_dir,
        source_dirs=[launchers_root],
    )
    if archive is None:
        return None
    if delete_source and launchers_root.exists():
        shutil.rmtree(launchers_root)

    report = {
        "created_at": _now_iso(),
        "run_id": str(run_id),
        "fw_ids": fw_ids,
        "fireworks": len(fireworks),
        "archive": str(archive),
        "deleted_source": bool(delete_source),
    }
    report_path = final_launcher_report_path(outdir, run_id)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = report_path.with_suffix(report_path.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(report_path)
    return archive
