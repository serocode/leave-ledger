"""Tests for hris_client.py — mocked responses only, never live HRIS calls.

Run with:
    pytest test_hris_client.py -v
"""

from unittest.mock import MagicMock, patch

import pytest

from hris_client import HrisClient, HrisError, LEAVE_CREDIT_TYPE_IDS


# ---- Fixtures ----

@pytest.fixture
def client():
    """An HrisClient with a fake JWT + user_id, ready for authenticated calls."""
    c = HrisClient()
    c._jwt = "fake-jwt-for-testing"
    c._user_id = 9999
    c._session.headers["Authorization"] = "Bearer fake-jwt-for-testing"
    return c


# ---- Login ----

def test_login_success():
    client = HrisClient()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "jwt": "real-jwt-token",
        "user": {"id": 5845, "username": "admin"},
    }
    with patch.object(client._session, "post", return_value=mock_resp):
        client.login("user@example.com", "password123")
    assert client._jwt == "real-jwt-token"
    assert client.user_id == 5845


def test_login_failure():
    client = HrisClient()
    mock_resp = MagicMock()
    mock_resp.status_code = 400
    mock_resp.text = "Invalid identifier or password"
    with patch.object(client._session, "post", return_value=mock_resp):
        with pytest.raises(HrisError, match="Login failed"):
            client.login("bad@example.com", "wrongpass")


def test_login_no_jwt():
    client = HrisClient()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"user": {"id": 1}}  # no jwt field
    with patch.object(client._session, "post", return_value=mock_resp):
        with pytest.raises(HrisError, match="no jwt"):
            client.login("user@example.com", "password123")


# ---- Auth guard ----

def test_require_auth_not_logged_in():
    client = HrisClient()
    with pytest.raises(HrisError, match="Not logged in"):
        client.search_employees("test")


# ---- Search employees ----

# Strapi v4 response shape (attributes nested)
SEARCH_RESPONSE_V4 = {
    "data": [
        {
            "id": 621,
            "attributes": {
                "last_name": "DELA CRUZ",
                "first_name": "ANA",
                "middle_name": "REYES",
                "suffix": "",
                "plantilla": {
                    "attributes": {
                        "position": {
                            "attributes": {"position": "Teacher I"},
                        }
                    }
                },
                "station": {
                    "attributes": {"station_name": "Macanhan ES"},
                },
            },
        }
    ]
}

# Strapi v5 response shape (flat)
SEARCH_RESPONSE_V5 = {
    "data": [
        {
            "id": 621,
            "last_name": "DELA CRUZ",
            "first_name": "ANA",
            "middle_name": "REYES",
            "suffix": "",
            "plantilla": {
                "position": {
                    "attributes": {"position": "Teacher I"},
                }
            },
            "station": {"station_name": "Macanhan ES"},
        }
    ]
}


def test_search_employees_v4(client):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = SEARCH_RESPONSE_V4
    with patch.object(client._session, "get", return_value=mock_resp):
        results = client.search_employees("DELA CRUZ")
    assert len(results) == 1
    emp = results[0]
    assert emp.id == 621
    assert emp.last_name == "DELA CRUZ"
    assert emp.first_name == "ANA"
    assert emp.middle_name == "REYES"
    assert emp.school == "Macanhan ES"
    assert "DELA CRUZ" in emp.full_name


def test_search_employees_v5(client):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = SEARCH_RESPONSE_V5
    with patch.object(client._session, "get", return_value=mock_resp):
        results = client.search_employees("DELA CRUZ")
    assert len(results) == 1
    emp = results[0]
    assert emp.id == 621
    assert emp.last_name == "DELA CRUZ"
    assert emp.school == "Macanhan ES"


def test_search_employees_empty(client):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"data": []}
    with patch.object(client._session, "get", return_value=mock_resp):
        results = client.search_employees("ZZZZZZZ")
    assert results == []


def test_search_employees_splits_last_comma_first(client):
    """A Form 6 row gives "Last, First M.I." — that must become a last-name
    AND first-name filter, not a literal search for the whole string."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = SEARCH_RESPONSE_V4
    with patch.object(client._session, "get", return_value=mock_resp) as get:
        client.search_employees("Santos, Juana R.")
    params = get.call_args.kwargs["params"]
    assert params["filters[last_name][$containsi]"] == "Santos"
    # Only the first given-name word: HRIS holds the middle name separately.
    assert params["filters[first_name][$containsi]"] == "Juana"


def test_search_employees_falls_back_to_surname(client):
    """A misread given name shouldn't leave the user with nothing to pick."""
    empty = MagicMock(status_code=200)
    empty.json.return_value = {"data": []}
    found = MagicMock(status_code=200)
    found.json.return_value = SEARCH_RESPONSE_V4
    with patch.object(client._session, "get", side_effect=[empty, found]) as get:
        results = client.search_employees("Dela Cruz, Xxxx")
    assert len(results) == 1
    params = get.call_args.kwargs["params"]
    assert params["filters[$or][0][last_name][$containsi]"] == "Dela Cruz"


def test_search_employees_surname_only_query(client):
    """The plain surname box still searches across all three name fields."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = SEARCH_RESPONSE_V4
    with patch.object(client._session, "get", return_value=mock_resp) as get:
        client.search_employees("DELA CRUZ")
    params = get.call_args.kwargs["params"]
    assert params["filters[$or][1][first_name][$containsi]"] == "DELA CRUZ"
    assert "filters[last_name][$containsi]" not in params


def test_search_employees_trailing_comma(client):
    """An unread given name leaves "Surname," — the comma must not be searched."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"data": []}
    with patch.object(client._session, "get", return_value=mock_resp) as get:
        client.search_employees("Lim,")
    params = get.call_args.kwargs["params"]
    assert params["filters[$or][0][last_name][$containsi]"] == "Lim"


def test_search_employees_error(client):
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.text = "Internal Server Error"
    with patch.object(client._session, "get", return_value=mock_resp):
        with pytest.raises(HrisError, match="Search failed"):
            client.search_employees("test")


# ---- Leave credits / balance ----

LEAVE_CREDITS_RESPONSE = {
    "data": [
        {
            "id": 1,
            "attributes": {
                "date_from": "2026-01-15T00:00:00.000Z",
                "date_to": "2026-01-15T09:00:00.000Z",
                "earned": 0,
                "used": 1,
                "description": "Sick leave January 15, 2026",
                "leave_credit_type": {
                    "attributes": {"leave_credit_type": "Service Credit"},
                },
            },
        },
        {
            "id": 2,
            "attributes": {
                "date_from": "2026-03-10T00:00:00.000Z",
                "date_to": "2026-03-11T09:00:00.000Z",
                "earned": 0,
                "used": 2,
                "description": "Sick leave March 10-11, 2026",
                "leave_credit_type": {
                    "attributes": {"leave_credit_type": "Service Credit"},
                },
            },
        },
        {
            "id": 3,
            "attributes": {
                "date_from": "2026-06-01T00:00:00.000Z",
                "date_to": "2026-06-01T09:00:00.000Z",
                "earned": 1.25,
                "used": 0,
                "description": "Earned leave June 2026",
                "leave_credit_type": {
                    "attributes": {"leave_credit_type": "Vacation Leave Credit"},
                },
            },
        },
    ]
}


def test_get_leave_credits(client):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = LEAVE_CREDITS_RESPONSE
    with patch.object(client._session, "get", return_value=mock_resp):
        records = client.get_leave_credits(621)
    assert len(records) == 3


def test_current_service_credit_used(client):
    """Should sum only Service Credit used values (1 + 2 = 3), not Vacation."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = LEAVE_CREDITS_RESPONSE
    with patch.object(client._session, "get", return_value=mock_resp):
        total = client.current_service_credit_used(621)
    assert total == 3.0


def test_get_leave_credits_error(client):
    mock_resp = MagicMock()
    mock_resp.status_code = 401
    mock_resp.text = "Unauthorized"
    with patch.object(client._session, "get", return_value=mock_resp):
        with pytest.raises(HrisError, match="Fetch leave credits failed"):
            client.get_leave_credits(621)


# ---- Create leave credit ----

def test_create_leave_credit_dry_run(client):
    """dry_run=True should return the payload without making any HTTP call."""
    result = client.create_leave_credit(
        employee_id=5192,
        leave_credit_type_id=LEAVE_CREDIT_TYPE_IDS["Service Credit"],
        description="Sick leave July 2, 2026",
        date_from="2026-07-02",
        date_to="2026-07-02",
        used=1,
        dry_run=True,
    )
    data = result["data"]
    # Dates are full ISO datetime strings
    assert data["date_from"] == "2026-07-02T00:00:00.000Z"
    assert data["date_to"] == "2026-07-02T09:00:00.000Z"
    # used is a string, not a number
    assert data["used"] == "1"
    assert isinstance(data["used"], str)
    # hrisBy is the submitting user's id
    assert data["hrisBy"] == 9999
    # Other expected fields
    assert data["employee"] == 5192
    assert data["leave_credit_type"] == 4
    assert data["is_published"] is True
    assert data["description"] == "Sick leave July 2, 2026"


def test_create_leave_credit_hris_by_explicit(client):
    """Explicit hris_by should override the logged-in user's id."""
    result = client.create_leave_credit(
        employee_id=5192,
        leave_credit_type_id=4,
        description="Test",
        date_from="2026-01-01",
        date_to="2026-01-01",
        used=1,
        hris_by=1234,
        dry_run=True,
    )
    assert result["data"]["hrisBy"] == 1234


def test_create_leave_credit_no_user_id():
    """Should raise if no user_id and no explicit hris_by."""
    client = HrisClient()
    client._jwt = "fake"
    client._user_id = None
    with pytest.raises(HrisError, match="hrisBy is required"):
        client.create_leave_credit(
            employee_id=1,
            leave_credit_type_id=4,
            description="test",
            date_from="2026-01-01",
            date_to="2026-01-01",
            used=1,
            dry_run=True,
        )


def test_create_leave_credit_real_submit(client):
    """dry_run=False should actually POST and return the response."""
    mock_resp = MagicMock()
    mock_resp.status_code = 201
    mock_resp.json.return_value = {"data": {"id": 99999}}
    with patch.object(client._session, "post", return_value=mock_resp) as mock_post:
        result = client.create_leave_credit(
            employee_id=5192,
            leave_credit_type_id=4,
            description="Sick leave July 2, 2026",
            date_from="2026-07-02",
            date_to="2026-07-02",
            used=1,
            dry_run=False,
        )
    assert result == {"data": {"id": 99999}}
    mock_post.assert_called_once()
    # Verify the payload sent
    call_kwargs = mock_post.call_args
    sent_payload = call_kwargs[1]["json"]
    assert sent_payload["data"]["used"] == "1"
    assert sent_payload["data"]["hrisBy"] == 9999


def test_create_leave_credit_submit_failure(client):
    mock_resp = MagicMock()
    mock_resp.status_code = 400
    mock_resp.text = "Bad request"
    with patch.object(client._session, "post", return_value=mock_resp):
        with pytest.raises(HrisError, match="Create failed"):
            client.create_leave_credit(
                employee_id=5192,
                leave_credit_type_id=4,
                description="Test",
                date_from="2026-01-01",
                date_to="2026-01-01",
                used=1,
                dry_run=False,
            )


# ---- Employee model ----

def test_employee_full_name():
    from hris_client import Employee
    emp = Employee(id=1, last_name="DELA CRUZ", first_name="ANA",
                   middle_name="REYES", suffix="", position=None, school=None)
    assert emp.full_name == "DELA CRUZ, ANA REYES"

    emp2 = Employee(id=2, last_name="SANTOS", first_name="JUAN",
                    middle_name="", suffix="JR", position="T-I", school="Test ES")
    assert emp2.full_name == "SANTOS, JUAN JR"


# ---- Known leave credit type IDs ----

def test_known_leave_credit_type_ids():
    assert LEAVE_CREDIT_TYPE_IDS["Service Credit"] == 4
