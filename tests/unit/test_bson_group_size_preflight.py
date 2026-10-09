from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_MODULE_PATH = Path(__file__).resolve().parents[2] / "src" / "gow" / "fw" / "bson_preflight.py"
_SPEC = importlib.util.spec_from_file_location("gow_bson_preflight_test_module", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
bp = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = bp
_SPEC.loader.exec_module(bp)


class _FakeAdmin:
    def command(self, name: str):
        assert name == "hello"
        return {
            "maxBsonObjectSize": 20_000_000,
            "maxMessageSizeBytes": 50_000_000,
            "maxWriteBatchSize": 123_456,
        }


class _FakeClient:
    admin = _FakeAdmin()


class _FakeLP:
    connection = _FakeClient()


class _FakeDoc:
    def __init__(self, size: int):
        self.size = size

    def to_db_dict(self):
        return {"_fake_size": self.size}


class _FakeWorkflow(_FakeDoc):
    def __init__(self, fw_size: int, workflow_size: int = 100):
        super().__init__(workflow_size)
        self.fws = [_FakeDoc(fw_size)]


def test_query_limits_uses_live_launchpad_hello() -> None:
    limits = bp.query_mongo_server_limits(_FakeLP())
    assert limits.max_bson_object_size == 20_000_000
    assert limits.max_message_size_bytes == 50_000_000
    assert limits.max_write_batch_size == 123_456
    assert limits.safe_document_limit(0.75) == 15_000_000


def test_htc_target_keeps_queue_occupied() -> None:
    assert bp.htc_group_size_target(200_000, 400) == 500
    assert bp.htc_group_size_target(10_000, 400) == 25
    assert bp.htc_group_size_target(137, 400) == 1


def test_largest_safe_group_uses_exponential_then_binary_search(monkeypatch) -> None:
    monkeypatch.setattr(bp, "_bson_size", lambda doc: int(doc["_fake_size"]))

    def build(size: int):
        return _FakeWorkflow(fw_size=size * 1_000)

    safe_size, measurement = bp.largest_safe_group_size(
        10_000_000,
        build_workflow_for_size=build,
        safe_document_bytes=12_500,
    )
    assert safe_size == 12
    assert measurement is not None
    assert measurement.max_document_bytes == 12_000


def test_measurement_checks_firework_and_workflow_documents(monkeypatch) -> None:
    monkeypatch.setattr(bp, "_bson_size", lambda doc: int(doc["_fake_size"]))
    measurement = bp.measure_workflow_bson(_FakeWorkflow(fw_size=9_000, workflow_size=12_000))
    assert measurement.workflow_bytes == 12_000
    assert measurement.firework_bytes == (9_000,)
    assert measurement.max_document_bytes == 12_000
