from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Sequence

DEFAULT_MAX_BSON_OBJECT_SIZE = 16 * 1024 * 1024
DEFAULT_MAX_MESSAGE_SIZE_BYTES = 48_000_000
DEFAULT_MAX_WRITE_BATCH_SIZE = 100_000
DEFAULT_SAFETY_FRACTION = 0.75


@dataclass(frozen=True)
class MongoServerLimits:
    """MongoDB wire/BSON limits reported by the connected server."""

    max_bson_object_size: int
    max_message_size_bytes: int
    max_write_batch_size: int

    def safe_document_limit(self, safety_fraction: float = DEFAULT_SAFETY_FRACTION) -> int:
        if not (0.0 < float(safety_fraction) <= 1.0):
            raise ValueError("safety_fraction must be > 0 and <= 1")
        # A BSON document must fit both the BSON-object and wire-message limits.
        # maxBsonObjectSize is normally the tighter one, but use the actual minimum.
        return int(min(self.max_bson_object_size, self.max_message_size_bytes) * float(safety_fraction))


@dataclass(frozen=True)
class WorkflowBsonMeasurement:
    max_document_bytes: int
    workflow_bytes: int
    firework_bytes: tuple[int, ...]

    @property
    def max_document_mib(self) -> float:
        return self.max_document_bytes / (1024.0 * 1024.0)


def _mongo_client_from_launchpad(lp: Any) -> Any:
    """Return the PyMongo client used by a FireWorks LaunchPad."""
    client = getattr(lp, "connection", None)
    if client is not None:
        return client

    db = getattr(lp, "db", None)
    client = getattr(db, "client", None) if db is not None else None
    if client is not None:
        return client

    collection = getattr(lp, "fireworks", None)
    database = getattr(collection, "database", None) if collection is not None else None
    client = getattr(database, "client", None) if database is not None else None
    if client is not None:
        return client

    raise RuntimeError("Could not obtain the PyMongo client from the FireWorks LaunchPad")


def query_mongo_server_limits(lp: Any) -> MongoServerLimits:
    """Query the MongoDB actually used by ``lp`` using the ``hello`` command.

    The command is intentionally executed against the live LaunchPad connection.  If
    ``hello`` itself cannot be queried, preflight fails rather than silently assuming
    that a different MongoDB installation has the standard limits.

    The MongoDB protocol defines 16 MiB as the fallback for maxBsonObjectSize when the
    field is absent from the hello response.  Equivalent protocol defaults are used
    for the other advertised limits.
    """
    client = _mongo_client_from_launchpad(lp)
    try:
        hello = client.admin.command("hello")
    except Exception as exc:
        raise RuntimeError(
            "Could not query MongoDB 'hello' through the FireWorks LaunchPad; "
            "cannot perform a server-aware BSON preflight"
        ) from exc

    if not isinstance(hello, dict):
        raise RuntimeError("MongoDB 'hello' returned an unexpected response")

    return MongoServerLimits(
        max_bson_object_size=int(hello.get("maxBsonObjectSize", DEFAULT_MAX_BSON_OBJECT_SIZE)),
        max_message_size_bytes=int(hello.get("maxMessageSizeBytes", DEFAULT_MAX_MESSAGE_SIZE_BYTES)),
        max_write_batch_size=int(hello.get("maxWriteBatchSize", DEFAULT_MAX_WRITE_BATCH_SIZE)),
    )


def _db_dict(obj: Any) -> dict[str, Any]:
    to_db = getattr(obj, "to_db_dict", None)
    if callable(to_db):
        doc = to_db()
    else:
        to_dict = getattr(obj, "to_dict", None)
        if not callable(to_dict):
            raise TypeError(f"Object {type(obj)!r} has neither to_db_dict() nor to_dict()")
        doc = to_dict()
    if not isinstance(doc, dict):
        raise TypeError(f"Serialized {type(obj)!r} is not a dict")
    return doc


def _bson_size(document: dict[str, Any]) -> int:
    try:
        from bson import BSON
    except Exception as exc:  # pragma: no cover - FireWorks installs PyMongo/bson
        raise RuntimeError(
            "PyMongo/bson is required for FireWorks BSON preflight. "
            "Install with: pip install -e '.[fireworks]'"
        ) from exc
    return len(BSON.encode(document))


def _workflow_fireworks(workflow: Any) -> list[Any]:
    fws = getattr(workflow, "fws", None)
    if fws is not None:
        try:
            return list(fws)
        except TypeError:
            pass

    id_fw = getattr(workflow, "id_fw", None)
    if isinstance(id_fw, dict):
        return list(id_fw.values())

    return []


def measure_workflow_bson(workflow: Any) -> WorkflowBsonMeasurement:
    """Measure the BSON documents FireWorks is about to persist for a Workflow.

    FireWorks stores Firework and Workflow documents separately.  Measure both and
    enforce the server limit against the largest document rather than against an
    estimate based on candidate count.
    """
    workflow_bytes = _bson_size(_db_dict(workflow))
    firework_bytes = tuple(_bson_size(_db_dict(fw)) for fw in _workflow_fireworks(workflow))
    max_document_bytes = max((workflow_bytes, *firework_bytes), default=workflow_bytes)
    return WorkflowBsonMeasurement(
        max_document_bytes=max_document_bytes,
        workflow_bytes=workflow_bytes,
        firework_bytes=firework_bytes,
    )


def htc_group_size_target(candidate_count: int, njobs_queue: int) -> int:
    """Group size that creates approximately one FireWork per available queue job."""
    if candidate_count <= 0:
        return 1
    if njobs_queue <= 0:
        return candidate_count
    return max(1, int(math.ceil(candidate_count / float(njobs_queue))))


def largest_safe_group_size(
    max_group_size: int,
    *,
    build_workflow_for_size: Callable[[int], Any],
    safe_document_bytes: int,
) -> tuple[int, WorkflowBsonMeasurement | None]:
    """Find the largest safe size without first materializing ``max_group_size`` items.

    Probe exponentially (1, 2, 4, ...) until the BSON threshold is crossed, then
    finish with a binary search.  This is important for standalone preflight of very
    large generations: a nonsensical group size of millions does not require creating
    millions of candidate specs merely to discover that MongoDB rejects it.
    """
    upper = max(0, int(max_group_size))
    if upper <= 0:
        return 0, None

    best_size = 0
    best_measurement: WorkflowBsonMeasurement | None = None
    first_unsafe: int | None = None

    probe = 1
    while probe <= upper:
        measurement = measure_workflow_bson(build_workflow_for_size(probe))
        if measurement.max_document_bytes <= safe_document_bytes:
            best_size = probe
            best_measurement = measurement
            if probe == upper:
                return best_size, best_measurement
            next_probe = min(upper, probe * 2)
            if next_probe == probe:
                return best_size, best_measurement
            probe = next_probe
        else:
            first_unsafe = probe
            break

    if first_unsafe is None:
        return best_size, best_measurement

    low = best_size + 1
    high = first_unsafe - 1
    while low <= high:
        mid = (low + high) // 2
        measurement = measure_workflow_bson(build_workflow_for_size(mid))
        if measurement.max_document_bytes <= safe_document_bytes:
            best_size = mid
            best_measurement = measurement
            low = mid + 1
        else:
            high = mid - 1

    return best_size, best_measurement


def largest_safe_prefix_size(
    items: Sequence[Any],
    max_group_size: int,
    *,
    build_workflow: Callable[[Sequence[Any]], Any],
    safe_document_bytes: int,
) -> tuple[int, WorkflowBsonMeasurement | None]:
    """Return the largest safe prefix of an already-materialized candidate sequence."""
    upper = min(len(items), max(0, int(max_group_size)))
    return largest_safe_group_size(
        upper,
        build_workflow_for_size=lambda size: build_workflow(items[:size]),
        safe_document_bytes=safe_document_bytes,
    )
