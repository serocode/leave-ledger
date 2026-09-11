"""Tests for ocr_form6.py that don't need a scan or a Tesseract install.

Run with:
    pytest test_ocr_form6.py -v
"""

from unittest.mock import patch

import numpy as np
import pytest

import ocr_form6


def _blank_cell():
    return np.full((40, 160, 3), 255, dtype=np.uint8)


def test_whitelist_with_space_survives_as_one_argument():
    """The Windows regression this guards against.

    pytesseract splits its `config` string with shlex in non-POSIX mode on
    Windows, which cannot parse a quoted value containing a space — it raises
    "No closing quotation", and every cell using one of the space-bearing
    whitelists ("SICK LEAVE", "June 25-26,2026") failed. Passing argv as a
    list keeps the whitelist in exactly one element on every platform.
    """
    captured = {}

    class Result:
        returncode = 0
        stdout = b"SICK LEAVE"
        stderr = b""

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return Result()

    with patch.object(ocr_form6.subprocess, "run", fake_run):
        ocr_form6._run_tesseract(_blank_cell(), 6, "ABC abc/- ")

    cmd = captured["cmd"]
    assert "-c" in cmd
    whitelist_arg = cmd[cmd.index("-c") + 1]
    assert whitelist_arg == "tessedit_char_whitelist=ABC abc/- "
    # The space must live inside a single argv element, never split across two.
    assert sum(a.startswith("tessedit_char_whitelist=") for a in cmd) == 1
    # And no shell quoting should have been added around the value.
    assert '"' not in whitelist_arg


def test_every_shipped_whitelist_round_trips():
    """Any whitelist added later must also survive being passed as argv."""
    names = [n for n in dir(ocr_form6) if n.endswith("_WHITELIST")]
    assert names, "expected some whitelists to exist"
    for name in names:
        value = getattr(ocr_form6, name)
        arg = f"tessedit_char_whitelist={value}"
        assert arg.split("=", 1)[1] == value, name


def test_missing_tesseract_reports_a_readable_error():
    def boom(cmd, **kwargs):
        raise FileNotFoundError(cmd[0])

    with patch.object(ocr_form6.subprocess, "run", boom):
        with pytest.raises(ValueError, match="Tesseract OCR is not installed"):
            ocr_form6._run_tesseract(_blank_cell(), 6, None)


def test_tesseract_failure_surfaces_stderr():
    class Result:
        returncode = 1
        stdout = b""
        stderr = b"read_params_file: parameter not found"

    with patch.object(ocr_form6.subprocess, "run", lambda cmd, **kw: Result()):
        with pytest.raises(ValueError, match="parameter not found"):
            ocr_form6._run_tesseract(_blank_cell(), 6, None)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Q.5", 0.5),   # leading 0 misread as Q
        ("3.9", 3.5),   # trailing 5 misread as 9
        ("1.5", 1.5),
        ("2", 2.0),
        ("", None),
    ],
)
def test_day_counts_tolerate_small_print_misreads(raw, expected):
    """Half days are the only fraction these forms use, so a decimal point is
    the half marker regardless of what the digit after it came out as."""
    assert ocr_form6._normalize_total_days(raw) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        # One day.
        ("June 30, 2026", [("2026-06-30", "2026-06-30", 1.0)]),
        # A parenthesized half-day marker.
        ("June 29(pm), 2026", [("2026-06-29", "2026-06-29", 0.5)]),
        # "&"-joined ranges stay separate runs (non-adjacent days).
        ("July 1-3 & 6-7, 2026", [("2026-07-01", "2026-07-03", 3.0),
                                  ("2026-07-06", "2026-07-07", 2.0)]),
        # Two months in one cell, ";"-separated, second day a half.
        ("June 30, 2026; July 2(pm), 2026", [("2026-06-30", "2026-06-30", 1.0),
                                             ("2026-07-02", "2026-07-02", 0.5)]),
        # Adjacent days merge into one run, half-day included in the count.
        ("July 1(pm) & 2, 2026", [("2026-07-01", "2026-07-02", 1.5)]),
        ("June 25 & 30, 2026", [("2026-06-25", "2026-06-25", 1.0),
                                ("2026-06-30", "2026-06-30", 1.0)]),
        ("July 3 & 6-7, 2026", [("2026-07-03", "2026-07-03", 1.0),
                                ("2026-07-06", "2026-07-07", 2.0)]),
        # A run that crosses a month boundary is still one contiguous run.
        ("June 30 & July 1, 2026", [("2026-06-30", "2026-07-01", 2.0)]),
    ],
)
def test_absence_dates_cover_the_masterson_cell_shapes(raw, expected):
    runs, warning = ocr_form6._parse_date_segments_absence(raw, 2026)
    assert warning is None
    assert [(f.isoformat(), t.isoformat(), u) for f, t, u in runs] == expected


def test_absence_dates_prefer_the_printed_year_over_the_page_header():
    runs, warning = ocr_form6._parse_date_segments_absence("June 30, 2025", 2026)
    assert warning is None
    assert runs[0][0].year == 2025


def test_absence_dates_report_unreadable_cells_instead_of_guessing():
    for raw in ("", "wwe", "July 7-3, 2026", "June 31, 2026"):
        runs, warning = ocr_form6._parse_date_segments_absence(raw, 2026)
        assert runs == [] and warning, raw


def test_absence_dates_without_any_year_say_so():
    runs, warning = ocr_form6._parse_date_segments_absence("June 30", None)
    assert runs == []
    assert "year" in warning
