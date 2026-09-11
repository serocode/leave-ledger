"""
Thin client for the DepEd HRIS backend (v2.depedcdo.online).

This talks directly to the same REST API the HRIS web app uses
(a Strapi-style backend), instead of driving a browser. That makes it
far more reliable than clicking through the UI: no dropdowns failing
to open, no modals silently discarding, no viewport-shift misclicks.

Reverse-engineered from real (authorized, logged-in) browser traffic on
2026-09-10. The `create_leave_credit` payload shape below is a best
guess based on the GET response's field names — CONFIRM IT against a
real request payload (captured via DevTools Network tab on a real,
legitimate submission) before relying on it. See the TODO near the
bottom.

Credentials are supplied by you, live only in your own environment
(e.g. entered at a prompt or read from a local .env file you create),
and are never sent anywhere but the official HRIS login endpoint.
"""

from __future__ import annotations

import getpass
import sys
from dataclasses import dataclass
from typing import Any

import requests

BASE_URL = "https://v2.depedcdo.online/api"
FRONTEND_ORIGIN = "https://hris.depedcdo.online"


class HrisError(RuntimeError):
    pass


@dataclass
class Employee:
    id: int
    last_name: str
    first_name: str
    middle_name: str
    suffix: str
    position: str | None
    school: str | None

    @property
    def full_name(self) -> str:
        parts = [self.last_name + ",", self.first_name]
        if self.middle_name:
            parts.append(self.middle_name)
        if self.suffix:
            parts.append(self.suffix)
        return " ".join(parts)


class HrisClient:
    def __init__(self) -> None:
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Content-Type": "application/json",
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0 Safari/537.36"
                ),
                "Origin": FRONTEND_ORIGIN,
                "Referer": f"{FRONTEND_ORIGIN}/",
            }
        )
        self._jwt: str | None = None
        self._user_id: int | None = None

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------
    def login(self, identifier: str, password: str) -> None:
        """Log in and store the JWT for subsequent requests."""
        resp = self._session.post(
            f"{BASE_URL}/auth/local",
            json={"identifier": identifier, "password": password},
        )
        if resp.status_code != 200:
            raise HrisError(f"Login failed ({resp.status_code}): {resp.text[:300]}")
        data = resp.json()
        jwt = data.get("jwt")
        if not jwt:
            raise HrisError(f"Login response had no jwt: {data}")
        self._jwt = jwt
        self._session.headers["Authorization"] = f"Bearer {jwt}"
        user = data.get("user") or {}
        self._user_id = user.get("id")

    @property
    def user_id(self) -> int | None:
        """The logged-in HRIS user's own id — required as `hrisBy` on any
        record you create (it records who submitted the entry)."""
        return self._user_id

    def login_interactive(self) -> None:
        """Prompt locally for credentials (never sent to me / anyone but HRIS)."""
        identifier = input("HRIS username/email: ").strip()
        password = getpass.getpass("HRIS password: ")
        self.login(identifier, password)

    def _require_auth(self) -> None:
        if not self._jwt:
            raise HrisError("Not logged in. Call login() or login_interactive() first.")

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------
    def search_employees(self, name_query: str) -> list[Employee]:
        """Search for an employee by name.

        A plain query ("DELA CRUZ") is matched against last/first/middle name.
        A "Last, First M.I." query — the shape a Form 6 row gives us — is
        split on the comma and matched as last name AND first name instead:
        matching that whole string against any single field finds nobody, and
        the given name is what separates two teachers who share a surname.
        Only the first given-name word is used, since HRIS stores the middle
        name separately. If that pair finds nothing (a misread given name,
        say), the surname alone is retried so a usable list still comes back.
        """
        self._require_auth()
        last, comma, given = name_query.partition(",")
        first = given.strip().split(" ")[0].strip() if given.strip() else ""
        if comma and not first:
            # "Surname," with the given name unread — drop the comma so it
            # isn't searched for literally.
            name_query = last.strip()
        if first:
            matches = self._search_request(
                {
                    "filters[last_name][$containsi]": last.strip(),
                    "filters[first_name][$containsi]": first,
                }
            )
            if matches:
                return matches
            name_query = last.strip()

        return self._search_request(
            {
                "filters[$or][0][last_name][$containsi]": name_query,
                "filters[$or][1][first_name][$containsi]": name_query,
                "filters[$or][2][middle_name][$containsi]": name_query,
            }
        )

    def _search_request(self, filters: dict[str, str]) -> list[Employee]:
        params = {
            "populate[]": ["plantilla", "plantilla.position", "station"],
            "pagination[pageSize]": 10,
            **filters,
        }
        resp = self._session.get(f"{BASE_URL}/employees/", params=params)
        if resp.status_code != 200:
            raise HrisError(f"Search failed ({resp.status_code}): {resp.text[:300]}")
        raw = resp.json().get("data", [])
        results = []
        for row in raw:
            # Strapi v5 flattens attributes onto the object directly;
            # v4 nests them under "attributes". Handle both defensively.
            attrs = row.get("attributes", row)
            plantilla = attrs.get("plantilla") or {}
            position = (
                (plantilla.get("attributes", plantilla).get("position") or {})
                .get("attributes", {})
                .get("position")
            )
            station = attrs.get("station") or {}
            school = (station.get("attributes", station) or {}).get("station_name")
            results.append(
                Employee(
                    id=row.get("id"),
                    last_name=attrs.get("last_name", ""),
                    first_name=attrs.get("first_name", ""),
                    middle_name=attrs.get("middle_name", ""),
                    suffix=attrs.get("suffix", ""),
                    position=position,
                    school=school,
                )
            )
        return results

    def get_leave_credits(self, employee_id: int) -> list[dict[str, Any]]:
        """Fetch an employee's leave-credit ledger (for baseline/verification)."""
        self._require_auth()
        params = {
            "populate[]": ["employee", "leave_credit_type", "leave_credit_so", "hrisBy"],
            "pagination[pageSize]": 10000,
            "sort[]": "id:asc",
            "filters[employee][id][$eq]": employee_id,
            "filters[is_published][$eq]": "true",
        }
        resp = self._session.get(f"{BASE_URL}/leave-credits/", params=params)
        if resp.status_code != 200:
            raise HrisError(f"Fetch leave credits failed ({resp.status_code}): {resp.text[:300]}")
        return resp.json().get("data", [])

    def current_service_credit_used(self, employee_id: int) -> float:
        """Sum of 'used' across Service Credit records — the baseline you'd
        eyeball in the UI before adding a new entry."""
        records = self.get_leave_credits(employee_id)
        total = 0.0
        for r in records:
            attrs = r.get("attributes", r)
            lct = attrs.get("leave_credit_type") or {}
            lct_attrs = lct.get("attributes", lct) or {}
            if lct_attrs.get("leave_credit_type") == "Service Credit":
                total += attrs.get("used") or 0
        return total

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------
    def create_leave_credit(
        self,
        *,
        employee_id: int,
        leave_credit_type_id: int,
        description: str,
        date_from: str | None,  # "YYYY-MM-DD"
        date_to: str | None,  # "YYYY-MM-DD"
        used: float | str = 0,
        earned: float = 0,
        wo_pay: float = 0,
        hris_by: int | None = None,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        """
        Create a new leave-credit record.

        Payload shape confirmed 2026-09-10 against a real captured
        "Add Record" submission (Sick leave, single day):

            {"data": {
                "date_from": "2026-07-02T00:00:00.000Z",
                "date_to":   "2026-07-02T09:00:00.000Z",
                "earned": 0,
                "used": "01",
                "wo_pay": 0,
                "description": "Sick leave July 2, 2026",
                "employee": 5192,
                "leave_credit_type": 4,
                "is_published": true,
                "hrisBy": 5845
            }}

        Notable, non-obvious details this confirms:
          - `date_from`/`date_to` are full ISO datetime strings, not bare
            dates. The real submission used 00:00:00.000Z for date_from and
            09:00:00.000Z for date_to on the *same* calendar date — that
            looks like "start of day" / "5pm Philippine time" (UTC+8)
            converted to UTC. This is only confirmed for a single-day entry;
            it has NOT been confirmed yet for a genuine multi-day range
            (different date_from/date_to dates) — treat that case as
            unverified until a real multi-day submission is captured.
          - `used` is sent as a STRING (`"01"`), not a number.
          - Leave taken WITHOUT pay belongs in `wo_pay`, not `used` — it
            consumes no leave credit, so putting it in `used` would wrongly
            deduct from the employee's balance. Unlike `used`, a non-zero
            `wo_pay` goes as a NUMBER, not a string (confirmed 2026-09-11
            against a real record moved from `used` to `wo_pay`).
          - `hrisBy` is REQUIRED — it's the *submitting* HRIS user's own id
            (i.e. `client.user_id` after login), not the employee the leave
            is being recorded for. This was missing entirely from the
            original guessed payload.

        `dry_run=True` (the default) prints the payload and does NOT send
        it — pass `dry_run=False` to actually submit.
        """
        self._require_auth()
        submitting_user_id = hris_by if hris_by is not None else self._user_id
        if submitting_user_id is None:
            raise HrisError(
                "hrisBy is required and no user_id is known — pass hris_by= explicitly, "
                "or log in with login()/login_interactive() first."
            )

        def _iso_start_of_day(d: str) -> str:
            return f"{d}T00:00:00.000Z"

        def _iso_end_of_day(d: str) -> str:
            # Matches the real captured payload: 09:00 UTC == 5pm Philippine
            # time (UTC+8) on the same calendar date.
            return f"{d}T09:00:00.000Z"

        payload = {
            "data": {
                "date_from": _iso_start_of_day(date_from) if date_from else None,
                "date_to": _iso_end_of_day(date_to) if date_to else None,
                "earned": earned,
                "used": str(used),
                "wo_pay": wo_pay,
                "description": description,
                "employee": employee_id,
                "leave_credit_type": leave_credit_type_id,
                "is_published": True,
                "hrisBy": submitting_user_id,
            }
        }
        if dry_run:
            print("[DRY RUN] Would POST to /api/leave-credits/ with payload:")
            print(payload)
            return payload

        resp = self._session.post(f"{BASE_URL}/leave-credits/", json=payload)
        if resp.status_code not in (200, 201):
            raise HrisError(f"Create failed ({resp.status_code}): {resp.text[:500]}")
        return resp.json()

    def update_leave_credit(
        self,
        record_id: int,
        *,
        used: float | str,
        wo_pay: float,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        """Move an existing record's days between the `used` and `wo_pay`
        columns, for entries filed under the wrong one.

        Only those two fields are sent: Strapi merges a PUT into the existing
        record, so dates, description and employee stay as they are and the
        record keeps its id and history. That is why this is preferable to
        deleting and re-creating — a delete that succeeds followed by a
        create that fails would lose the entry outright.

        Like create_leave_credit, `used` goes as a string and `wo_pay` as a
        number. Confirmed 2026-09-11 against a real record: the PUT is
        permitted and HRIS shows the days under Without Pay afterward.
        `dry_run=True` (the default) returns the payload without sending it.
        """
        self._require_auth()
        payload = {"data": {"used": str(used), "wo_pay": wo_pay}}
        if dry_run:
            print(f"[DRY RUN] Would PUT to /api/leave-credits/{record_id} with payload:")
            print(payload)
            return payload

        resp = self._session.put(f"{BASE_URL}/leave-credits/{record_id}", json=payload)
        if resp.status_code not in (200, 201):
            raise HrisError(f"Update failed ({resp.status_code}): {resp.text[:500]}")
        return resp.json()


# Known leave credit type IDs, captured from real data — extend as you
# confirm more (e.g. "Vacation Leave Credit", "Sick Leave Credit", "CTO Leave Credit").
LEAVE_CREDIT_TYPE_IDS = {
    "Service Credit": 4,
}


def _demo() -> None:
    """Manual smoke test: login, search, view balances. Run directly:
        python hris_client.py
    Nothing here writes any data.
    """
    client = HrisClient()
    client.login_interactive()

    query = input("Search employee (surname): ").strip()
    matches = client.search_employees(query)
    if not matches:
        print("No matches.")
        return

    for i, emp in enumerate(matches):
        print(f"[{i}] {emp.full_name} — {emp.position or '?'} @ {emp.school or '?'}")

    idx = int(input("Pick an index to view leave balance: "))
    emp = matches[idx]
    used = client.current_service_credit_used(emp.id)
    print(f"{emp.full_name}: Service Credit used so far = {used}")


if __name__ == "__main__":
    try:
        _demo()
    except HrisError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
