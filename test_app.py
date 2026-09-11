"""Tests for the Flask endpoints — stubbed client, never live HRIS.

Run with:
    pytest test_app.py -v
"""

import pytest

import app as app_module
from hris_client import Employee, HrisError


def _emp(emp_id, last, first):
    return Employee(
        id=emp_id, last_name=last, first_name=first, middle_name="",
        suffix="", position="Teacher I", school="Lumbia NHS",
    )


class StubClient:
    """Records every query it is asked for, so de-duplication is observable."""

    def __init__(self):
        self.queries = []

    def search_employees(self, query):
        self.queries.append(query)
        if query.startswith("Santos"):
            return [_emp(1, "SANTOS", "JUANA")]
        if query.startswith("Reyes"):
            return [_emp(2, "REYES", "PEDRO"), _emp(3, "REYES-LIM", "PEDRO")]
        if query.startswith("Boom"):
            raise HrisError("upstream exploded")
        return []


@pytest.fixture
def client():
    stub = StubClient()
    app_module._clients["test-sid"] = stub
    test_client = app_module.app.test_client()
    with test_client.session_transaction() as sess:
        sess["sid"] = "test-sid"
    yield test_client, stub
    app_module._clients.pop("test-sid", None)


def test_match_requires_login():
    anon = app_module.app.test_client()
    assert anon.post("/api/match", json={"names": ["Santos, Juana"]}).status_code == 401


def test_match_rejects_non_list(client):
    test_client, _ = client
    assert test_client.post("/api/match", json={"names": "Santos"}).status_code == 400


def test_match_deduplicates_repeated_names(client):
    """A transmittal repeats a name once per date range; each person should
    only cost one HRIS lookup."""
    test_client, stub = client
    name = "Santos, Juana R."
    resp = test_client.post("/api/match", json={"names": [name, name, name]})
    assert resp.status_code == 200
    assert stub.queries == [name]
    assert len(resp.get_json()["matches"][name]["results"]) == 1


def test_match_reports_each_outcome(client):
    test_client, _ = client
    resp = test_client.post(
        "/api/match",
        json={"names": ["Santos, Juana R.", "Reyes, Pedro", "Nobody, Here", "Boom, Bad"]},
    )
    matches = resp.get_json()["matches"]
    assert len(matches["Santos, Juana R."]["results"]) == 1
    assert len(matches["Reyes, Pedro"]["results"]) == 2
    assert matches["Nobody, Here"]["results"] == []
    # One bad name must not sink the whole batch.
    assert "upstream exploded" in matches["Boom, Bad"]["error"]


def test_match_skips_blank_names(client):
    test_client, stub = client
    test_client.post("/api/match", json={"names": ["", "   ", None, 5]})
    assert stub.queries == []


# ---- /api/add: paid vs without-pay routing ----

class RecordingClient:
    def __init__(self):
        self.calls = []

    def create_leave_credit(self, **kwargs):
        self.calls.append(kwargs)
        return {"ok": True}


@pytest.fixture
def add_client():
    recorder = RecordingClient()
    app_module._clients["add-sid"] = recorder
    test_client = app_module.app.test_client()
    with test_client.session_transaction() as sess:
        sess["sid"] = "add-sid"
    yield test_client, recorder
    app_module._clients.pop("add-sid", None)


ADD_BODY = {
    "employee_id": 1,
    "description": "Sick leave June 22, 2026",
    "date_from": "2026-06-22",
    "date_to": "2026-06-22",
    "used": "1",
}


def test_add_paid_leave_goes_to_used(add_client):
    test_client, recorder = add_client
    test_client.post("/api/add", json=dict(ADD_BODY))
    assert recorder.calls[0]["used"] == 1.0
    assert recorder.calls[0]["wo_pay"] == 0


def test_add_without_pay_goes_to_wo_pay(add_client):
    """Leave without pay consumes no credit, so it must not land in "used" —
    that would deduct it from the employee's balance."""
    test_client, recorder = add_client
    test_client.post("/api/add", json={**ADD_BODY, "without_pay": True})
    assert recorder.calls[0]["used"] == 0
    assert recorder.calls[0]["wo_pay"] == 1.0


def test_add_without_pay_from_form_checkbox(add_client):
    """An HTML checkbox posts the string "on", not a boolean."""
    test_client, recorder = add_client
    test_client.post("/api/add", json={**ADD_BODY, "used": "0.5", "without_pay": "on"})
    assert recorder.calls[0]["used"] == 0
    assert recorder.calls[0]["wo_pay"] == 0.5


def _ledger_row(rid, desc, used, wo_pay, kind="Service Credit"):
    return {
        "id": rid, "description": desc, "used": used, "wo_pay": wo_pay,
        "date_from": "2026-06-22T00:00:00.000Z", "date_to": "2026-06-22T09:00:00.000Z",
        "leave_credit_type": {"leave_credit_type": kind},
    }


class LedgerClient:
    def __init__(self, ledger):
        self.ledger = ledger
        self.updates = []

    def get_leave_credits(self, employee_id):
        return self.ledger.get(employee_id, [])

    def update_leave_credit(self, record_id, **kwargs):
        self.updates.append((record_id, kwargs))
        return {"ok": True}


@pytest.fixture
def wop_client():
    ledger = {
        77: [
            _ledger_row(101, "Sick leave June 22, 2026", 1, 0),
            _ledger_row(102, "Sick leave June 30, 2026", 0, 1),
            _ledger_row(103, "Sick leave June 22, 2026", 1, 0, kind="Vacation Leave"),
        ],
    }
    stub = LedgerClient(ledger)
    app_module._clients["wop-sid"] = stub
    test_client = app_module.app.test_client()
    with test_client.session_transaction() as sess:
        sess["sid"] = "wop-sid"
    yield test_client, stub
    app_module._clients.pop("wop-sid", None)


def test_wop_audit_finds_only_misfiled_service_credit(wop_client):
    """Only a Service Credit row still holding days in "used" needs repair."""
    test_client, _ = wop_client
    resp = test_client.post("/api/wop-audit", json={"rows": [
        {"employee_id": 77, "description": "Sick leave June 22, 2026"},
        {"employee_id": 77, "description": "Sick leave June 30, 2026"},
    ]})
    records = resp.get_json()["records"]
    assert [r["record_id"] for r in records] == [101]


def test_wop_audit_is_read_only(wop_client):
    test_client, stub = wop_client
    test_client.post("/api/wop-audit", json={"rows": [
        {"employee_id": 77, "description": "Sick leave June 22, 2026"},
    ]})
    assert stub.updates == []


def test_wop_audit_deduplicates_records(wop_client):
    """The same ledger entry must never be offered for repair twice."""
    test_client, _ = wop_client
    row = {"employee_id": 77, "description": "Sick leave June 22, 2026"}
    resp = test_client.post("/api/wop-audit", json={"rows": [row, row, row]})
    assert len(resp.get_json()["records"]) == 1


def test_wop_fix_moves_days_to_wo_pay(wop_client):
    test_client, stub = wop_client
    resp = test_client.post("/api/wop-fix", json={"record_id": 101, "days": 1})
    assert resp.status_code == 200
    record_id, kwargs = stub.updates[0]
    assert record_id == 101
    assert kwargs["used"] == 0
    assert kwargs["wo_pay"] == 1.0


def test_wop_fix_rejects_bad_input(wop_client):
    test_client, stub = wop_client
    assert test_client.post("/api/wop-fix", json={"record_id": 101, "days": 0}).status_code == 400
    assert test_client.post("/api/wop-fix", json={"days": 1}).status_code == 400
    assert stub.updates == []


def test_add_still_requires_positive_days(add_client):
    test_client, recorder = add_client
    resp = test_client.post("/api/add", json={**ADD_BODY, "used": "0", "without_pay": True})
    assert resp.status_code == 400
    assert recorder.calls == []
