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
    """Behaves like HRIS: what is created shows up in that employee's ledger."""

    def __init__(self):
        self.calls = []
        self.ledger = {}          # employee_id -> [record, ...] in HRIS's shape
        self.deleted = []
        self.next_id = 1000
        self.fail_ledger = False
        self.ignore_deletes = False

    def _add(self, employee_id, description, date_from, date_to, earned=0, used=0, wo_pay=0):
        self.next_id += 1
        self.ledger.setdefault(employee_id, []).append({
            "id": self.next_id,
            "attributes": {
                "description": description, "date_from": date_from, "date_to": date_to,
                "earned": earned, "used": used, "wo_pay": wo_pay,
                "leave_credit_type": {"leave_credit_type": "Service Credit"},
            },
        })
        return self.next_id

    def get_leave_credits(self, employee_id):
        if self.fail_ledger:
            raise HrisError("HRIS is down")
        return list(self.ledger.get(employee_id, []))

    def create_leave_credit(self, **kwargs):
        self.calls.append(kwargs)
        self._add(kwargs["employee_id"], kwargs["description"], kwargs["date_from"], kwargs["date_to"],
                  kwargs.get("earned", 0), kwargs.get("used", 0), kwargs.get("wo_pay", 0))
        return {"ok": True}

    def delete_leave_credit(self, record_id, dry_run=True):
        self.deleted.append((record_id, dry_run))
        if not dry_run and not self.ignore_deletes:
            for records in self.ledger.values():
                records[:] = [r for r in records if r["id"] != record_id]
        return {}


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


def test_both_upload_tabs_ship_the_progress_ui(client):
    """Each upload's two slow steps (OCR, then one HRIS search per name) show
    a live indicator — a silent page reads as a hung app."""
    test_client, _ = client
    html = test_client.get("/").get_data(as_text=True)
    assert "busyPanel(" in html            # the shared spinner/bar panel
    assert ".spinner {" in html            # and it has a spinner to show
    assert "MATCH_CHUNK" in html           # names go up in groups, not one silent batch
    assert "just-matched" in html          # rows settle visibly as they resolve


def test_leave_and_service_credits_are_separate_tabs(client):
    """Earning credits is its own workflow: its own tab, upload and review."""
    import re
    test_client, _ = client
    html = test_client.get("/").get_data(as_text=True)
    assert 'data-tab="vsc"' in html and 'id="tab-vsc"' in html
    # Every element a workspace looks up exists for both prefixes.
    for suffix in ("File", "Upload", "Status", "Notes", "Kpis", "KpiPeople", "KpiPeopleFoot",
                   "KpiAuto", "KpiAutoFoot", "KpiAttention", "Kpi4", "ReviewCard", "Summary",
                   "Table", "CheckAll", "Empty", "BarTitle", "Progress", "SubmitStatus", "SubmitAll"):
        for prefix in ("form6", "vsc"):
            assert html.count(f'id="{prefix}{suffix}"') == 1, f"{prefix}{suffix}"
    # No element id is defined twice anywhere on the page.
    ids = re.findall(r'\sid="([^"{}]+)"', html)
    dupes = {i for i in ids if ids.count(i) > 1}
    assert not dupes, dupes


def test_the_individual_tab_only_deducts_now(client):
    test_client, _ = client
    html = test_client.get("/").get_data(as_text=True)
    employee_tab = html[html.index('id="tab-employee"'):html.index('id="tab-wop"')]
    assert 'name="earned"' not in employee_tab and 'value="earn"' not in employee_tab


def test_match_handles_a_chunk_of_names(client):
    """The UI now sends names a few at a time; each small request must behave
    exactly like the old single big one."""
    test_client, stub = client
    resp = test_client.post("/api/match", json={"names": ["Santos, Juana", "Reyes, Pedro"]})
    matches = resp.get_json()["matches"]
    assert len(matches["Santos, Juana"]["results"]) == 1
    assert len(matches["Reyes, Pedro"]["results"]) == 2
    assert stub.queries == ["Santos, Juana", "Reyes, Pedro"]


def test_write_paths_confirm_in_a_modal_not_a_native_dialog(client):
    """Native confirm() can be suppressed for a session ("don't ask again"),
    which would leave nothing between a typo and a real HRIS record."""
    test_client, _ = client
    html = test_client.get("/").get_data(as_text=True)
    script = html[html.index("<script>"):]
    # Comments talk about confirm()/alert() on purpose — judge the code only.
    code = "\n".join(
        line for line in script.splitlines() if not line.lstrip().startswith("//")
    )
    assert "confirmModal(" in code
    assert "dialog.modal" in html                 # the styled dialog element
    assert "confirm(" not in code.replace("confirmModal(", "")
    assert "alert(" not in code.replace("alertModal(", "")


# --- Recording service credits earned ---------------------------------------

def _record(**overrides):
    body = {
        "employee_id": 7, "description": "Vacation service credits May 6 - June 2, 2026 (138 hrs)",
        "date_from": "2026-05-06", "date_to": "2026-06-02",
    }
    body.update(overrides)
    return body


def test_credits_earned_go_to_earned_and_deduct_nothing(add_client):
    test_client, recorder = add_client
    resp = test_client.post("/api/add", json=_record(earned="25.944"))
    assert resp.status_code == 200, resp.get_json()
    written = recorder.calls[-1]
    assert written["earned"] == 25.944
    assert float(written["used"]) == 0 and written["wo_pay"] == 0


def test_earned_is_rounded_to_the_ledgers_three_decimals(add_client):
    test_client, recorder = add_client
    test_client.post("/api/add", json=_record(earned=25.943999999))
    assert recorder.calls[-1]["earned"] == 25.944


def test_a_record_cannot_both_add_and_deduct(add_client):
    test_client, recorder = add_client
    resp = test_client.post("/api/add", json=_record(earned="1.5", used="1"))
    assert resp.status_code == 400 and not recorder.calls


def test_without_pay_cannot_ride_along_on_a_grant(add_client):
    """A stray WOP tick on a grant must not route the credits to wo_pay."""
    test_client, recorder = add_client
    test_client.post("/api/add", json=_record(earned="1.5", without_pay=True))
    written = recorder.calls[-1]
    assert written["earned"] == 1.5 and written["wo_pay"] == 0


def test_leave_deductions_are_unchanged(add_client):
    test_client, recorder = add_client
    test_client.post("/api/add", json=_record(used="1", description="Sick leave July 3, 2026"))
    written = recorder.calls[-1]
    assert float(written["used"]) == 1 and written["earned"] == 0 and written["wo_pay"] == 0
    test_client.post("/api/add", json=_record(used="2", without_pay=True, description="Sick leave"))
    written = recorder.calls[-1]
    assert written["wo_pay"] == 2 and float(written["used"]) == 0 and written["earned"] == 0


def test_nothing_to_record_is_refused(add_client):
    test_client, recorder = add_client
    assert test_client.post("/api/add", json=_record()).status_code == 400
    assert not recorder.calls


def test_ocr_passes_unreadable_page_notes_to_the_ui(client, monkeypatch):
    import io
    test_client, _ = client
    monkeypatch.setattr(
        app_module, "extract_form6_with_notes",
        lambda path: ([], ["Page 4: found a table with 11 rows but couldn't read any of them."]),
    )
    resp = test_client.post(
        "/api/ocr", data={"file": (io.BytesIO(b"fake"), "scan.pdf")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 200
    assert resp.get_json()["notes"] == ["Page 4: found a table with 11 rows but couldn't read any of them."]


def _ocr_rows(*kinds):
    import ocr_form6
    rows = []
    for i, kind in enumerate(kinds):
        r = ocr_form6.Form6Row(str(i + 1), "Cruz", "Ana", "", "", "", "", "SL" if kind == "leave" else "VSC", "WP")
        r.kind = kind
        rows.append(r)
    return rows


def _post_scan(test_client, expect):
    import io
    return test_client.post(
        f"/api/ocr?expect={expect}", data={"file": (io.BytesIO(b"fake"), "scan.jpg")},
        content_type="multipart/form-data",
    )


def test_a_grant_uploaded_on_the_form6_tab_is_sent_to_its_own_tab(client, monkeypatch):
    test_client, _ = client
    monkeypatch.setattr(app_module, "extract_form6_with_notes", lambda path: (_ocr_rows("vsc", "vsc"), []))
    resp = _post_scan(test_client, "leave")
    assert resp.status_code == 422 and "Service Credits Earned tab" in resp.get_json()["error"]


def test_a_form6_uploaded_on_the_credits_tab_is_sent_back(client, monkeypatch):
    test_client, _ = client
    monkeypatch.setattr(app_module, "extract_form6_with_notes", lambda path: (_ocr_rows("leave"), []))
    resp = _post_scan(test_client, "vsc")
    assert resp.status_code == 422 and "Form 6 tab" in resp.get_json()["error"]


def test_a_mixed_upload_keeps_only_this_tabs_rows_and_says_so(client, monkeypatch):
    test_client, _ = client
    monkeypatch.setattr(app_module, "extract_form6_with_notes", lambda path: (_ocr_rows("vsc", "leave", "vsc"), []))
    data = _post_scan(test_client, "vsc").get_json()
    assert [r["kind"] for r in data["rows"]] == ["vsc", "vsc"]
    assert any("other kind" in n for n in data["notes"])


def test_the_page_defines_the_submit_switch_every_write_path_reads(client):
    """Removing the old earned mode once cut this declaration out with it —
    the page still parsed, and every submit button on every tab threw."""
    test_client, _ = client
    html = test_client.get("/").get_data(as_text=True)
    script = html[html.index("<script>"):]
    assert script.count("const SUBMIT_ENABLED =") == 1
    assert script.index("const SUBMIT_ENABLED =") < script.index("createReviewWorkspace('form6'")


# --- Refusing to file the same record twice ----------------------------------

GRANT = {
    "employee_id": 7, "description": "Vacation service credits, SO No. 1673 s. 2026, Division Training, June 12-13, 2026 (16 hrs)",
    "date_from": "2026-06-12", "date_to": "2026-06-13", "earned": "3",
}


def test_an_identical_grant_is_refused_the_second_time(add_client):
    """SO 1673 was submitted twice and every teacher got 6.000, not 3.000."""
    test_client, recorder = add_client
    first = test_client.post("/api/add", json=dict(GRANT))
    second = test_client.post("/api/add", json=dict(GRANT))
    assert first.status_code == 200
    assert second.status_code == 409
    body = second.get_json()
    assert body["duplicate"] is True and body["record_id"] == 1001 and "already has this exact record" in body["error"]
    assert len(recorder.calls) == 1


def test_a_second_submit_is_refused_even_after_a_page_reload(add_client):
    """The check reads HRIS, not the page: nothing the browser forgets matters."""
    test_client, recorder = add_client
    recorder._add(7, GRANT["description"], "2026-06-12", "2026-06-13", earned=3)   # made in an earlier session
    assert test_client.post("/api/add", json=dict(GRANT)).status_code == 409
    assert recorder.calls == []


def test_whitespace_and_datetime_forms_dont_hide_a_duplicate(add_client):
    test_client, recorder = add_client
    recorder._add(7, "  Vacation   service credits  ", "2026-06-12T00:00:00.000Z", "2026-06-13T09:00:00.000Z", earned=3.0)
    body = {**GRANT, "description": "Vacation service credits"}
    assert test_client.post("/api/add", json=body).status_code == 409


@pytest.mark.parametrize("change", [
    {"description": "Vacation service credits, SO No. 1674 s. 2026"},   # another order
    {"date_to": "2026-06-14"},
    {"earned": "3.5"},
    {"employee_id": 8},                                                  # another teacher
])
def test_a_genuinely_different_record_is_not_refused(add_client, change):
    test_client, recorder = add_client
    test_client.post("/api/add", json=dict(GRANT))
    assert test_client.post("/api/add", json={**GRANT, **change}).status_code == 200


def test_nothing_is_submitted_when_the_ledger_cant_be_checked(add_client):
    """Fail closed: not knowing whether it is a duplicate is not a reason to write."""
    test_client, recorder = add_client
    recorder.fail_ledger = True
    resp = test_client.post("/api/add", json=dict(GRANT))
    assert resp.status_code == 502 and "nothing was submitted" in resp.get_json()["error"]
    assert recorder.calls == []


def test_leave_records_are_held_to_the_same_rule(add_client):
    test_client, recorder = add_client
    assert test_client.post("/api/add", json=dict(ADD_BODY)).status_code == 200
    assert test_client.post("/api/add", json=dict(ADD_BODY)).status_code == 409
    # A half day and a full day on the same date are different records.
    assert test_client.post("/api/add", json={**ADD_BODY, "used": "0.5"}).status_code == 200


# --- Finding and removing duplicates ------------------------------------------

def _doubled(recorder, employees=(7, 8, 9)):
    """SO 1673 filed twice for each teacher: ids 1001/1002, 1003/1004, ..."""
    for emp in employees:
        for _ in range(2):
            recorder._add(emp, GRANT["description"], "2026-06-12", "2026-06-13", earned=3)


def _rows(*employees):
    return [{"employee_id": e, "employee_name": f"Teacher {e}", "date_from": "2026-06-12",
             "date_to": "2026-06-13", "earned": 3} for e in employees]


def test_the_finder_lists_each_doubled_record_and_keeps_the_oldest(add_client):
    test_client, recorder = add_client
    _doubled(recorder)
    groups = test_client.post("/api/duplicates", json={"rows": _rows(7, 8, 9)}).get_json()["groups"]
    assert [(g["employee_id"], g["keep"]["id"], [d["id"] for d in g["duplicates"]]) for g in groups] == [
        (7, 1001, [1002]), (8, 1003, [1004]), (9, 1005, [1006])]
    assert all(g["same_wording"] for g in groups)
    assert recorder.deleted == []          # read-only


def test_the_finder_ignores_duplicates_unrelated_to_the_upload(add_client):
    """An old identical pair in a ledger is not this mistake's to clean up."""
    test_client, recorder = add_client
    recorder._add(7, "NMRAA MEET 2026", "2026-01-17", "2026-03-24", earned=28.5)
    recorder._add(7, "NMRAA MEET 2026", "2026-01-17", "2026-03-24", earned=28.5)
    recorder._add(7, GRANT["description"], "2026-06-12", "2026-06-13", earned=3)
    assert test_client.post("/api/duplicates", json={"rows": _rows(7)}).get_json()["groups"] == []


def test_the_finder_reports_nothing_when_nothing_is_doubled(add_client):
    test_client, recorder = add_client
    test_client.post("/api/add", json=dict(GRANT))
    assert test_client.post("/api/duplicates", json={"rows": _rows(7)}).get_json()["groups"] == []


def test_removing_a_duplicate_deletes_it_and_leaves_the_original(add_client):
    test_client, recorder = add_client
    _doubled(recorder, (7,))
    resp = test_client.post("/api/duplicates/remove", json={"removals": [{"employee_id": 7, "record_id": 1002}]})
    assert resp.get_json()["results"] == [{"record_id": 1002, "ok": True}]
    assert [r["id"] for r in recorder.ledger[7]] == [1001]
    assert recorder.deleted == [(1002, False)]


def test_the_original_is_never_removed(add_client):
    """Asking for the oldest copy — or a record with no older copy — deletes nothing."""
    test_client, recorder = add_client
    _doubled(recorder, (7,))
    recorder._add(7, "NMRAA MEET 2026", "2026-01-17", "2026-03-24", earned=28.5)   # unique
    for record_id in (1001, 1003):
        result = test_client.post("/api/duplicates/remove",
                                  json={"removals": [{"employee_id": 7, "record_id": record_id}]}).get_json()["results"][0]
        assert result["ok"] is False and "not a duplicate" in result["error"]
    assert recorder.deleted == [] and len(recorder.ledger[7]) == 3


def test_removing_a_record_from_the_wrong_employee_deletes_nothing(add_client):
    test_client, recorder = add_client
    _doubled(recorder, (7, 8))
    result = test_client.post("/api/duplicates/remove",
                              json={"removals": [{"employee_id": 8, "record_id": 1002}]}).get_json()["results"][0]
    assert result["ok"] is False and "isn't in this employee's ledger" in result["error"]
    assert recorder.deleted == []


def test_removal_stops_at_the_first_failure(add_client):
    test_client, recorder = add_client
    _doubled(recorder, (7, 8, 9))
    removals = [{"employee_id": 7, "record_id": 1002}, {"employee_id": 8, "record_id": 9999},
                {"employee_id": 9, "record_id": 1006}]
    results = test_client.post("/api/duplicates/remove", json={"removals": removals}).get_json()["results"]
    assert [r["ok"] for r in results] == [True, False]
    assert recorder.deleted == [(1002, False)]         # the third was never attempted


def test_a_delete_hris_accepts_but_doesnt_perform_is_reported(add_client):
    test_client, recorder = add_client
    _doubled(recorder, (7,))
    recorder.ignore_deletes = True
    result = test_client.post("/api/duplicates/remove",
                              json={"removals": [{"employee_id": 7, "record_id": 1002}]}).get_json()["results"][0]
    assert result["ok"] is False and "still in the ledger" in result["error"]


def test_the_delete_is_a_dry_run_when_submitting_is_off(add_client, monkeypatch):
    test_client, recorder = add_client
    monkeypatch.setattr(app_module, "SUBMIT_ENABLED", False)
    _doubled(recorder, (7,))
    body = test_client.post("/api/duplicates/remove",
                            json={"removals": [{"employee_id": 7, "record_id": 1002}]}).get_json()
    assert body["dry_run"] is True and recorder.deleted == [(1002, True)] and len(recorder.ledger[7]) == 2


def test_the_same_grant_worded_differently_is_still_found(add_client):
    """Filed once before the SO number box existed, once after: the wording
    differs, the grant doesn't. The strict submit check can't see that; the
    finder shows both wordings and flags the difference."""
    test_client, recorder = add_client
    recorder._add(7, "Vacation service credits, Division Training, June 12-13, 2026 (16 hrs)", "2026-06-12", "2026-06-13", earned=3)
    recorder._add(7, GRANT["description"], "2026-06-12", "2026-06-13", earned=3)
    [group] = test_client.post("/api/duplicates", json={"rows": _rows(7)}).get_json()["groups"]
    assert group["keep"]["id"] == 1001 and [d["id"] for d in group["duplicates"]] == [1002]
    assert group["same_wording"] is False
    assert "SO No. 1673" in group["duplicates"][0]["description"]
    removed = test_client.post("/api/duplicates/remove", json={"removals": [{"employee_id": 7, "record_id": 1002}]})
    assert removed.get_json()["results"][0]["ok"] is True


def test_a_different_grant_on_other_dates_is_never_a_duplicate(add_client):
    test_client, recorder = add_client
    recorder._add(7, GRANT["description"], "2026-06-12", "2026-06-13", earned=3)
    recorder._add(7, GRANT["description"], "2026-06-12", "2026-06-14", earned=3)
    recorder._add(7, GRANT["description"], "2026-06-12", "2026-06-13", earned=3.5)
    assert test_client.post("/api/duplicates", json={"rows": _rows(7)}).get_json()["groups"] == []
    result = test_client.post("/api/duplicates/remove",
                              json={"removals": [{"employee_id": 7, "record_id": 1002}]}).get_json()["results"][0]
    assert result["ok"] is False and recorder.deleted == []
