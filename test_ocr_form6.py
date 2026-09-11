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
def test_month_name_dates_cover_every_school_cell_shape(raw, expected):
    runs, warning = ocr_form6._parse_date_segments_monthname(raw, 2026)
    assert warning is None
    assert [(f.isoformat(), t.isoformat(), u) for f, t, u in runs] == expected


def test_month_name_dates_prefer_the_printed_year_over_the_page_header():
    runs, warning = ocr_form6._parse_date_segments_monthname("June 30, 2025", 2026)
    assert warning is None
    assert runs[0][0].year == 2025


def test_month_name_dates_report_unreadable_cells_instead_of_guessing():
    for raw in ("", "wwe", "July 7-3, 2026", "June 31, 2026"):
        runs, warning = ocr_form6._parse_date_segments_monthname(raw, 2026)
        assert runs == [] and warning, raw


def test_month_name_dates_without_any_year_say_so():
    runs, warning = ocr_form6._parse_date_segments_monthname("June 30", None)
    assert runs == []
    assert "year" in warning


def _draw_table(rows, cols, cell_w=220, cell_h=95):
    """A plain bordered grid — enough for the cell-detection step, which
    works off grid lines and never looks at the text inside."""
    img = np.full((rows * cell_h + 20, cols * cell_w + 20, 3), 255, dtype=np.uint8)
    for r in range(rows + 1):
        y = 10 + r * cell_h
        img[y - 1:y + 1, 10:10 + cols * cell_w] = 0
    for c in range(cols + 1):
        x = 10 + c * cell_w
        img[10:10 + rows * cell_h, x - 1:x + 1] = 0
    return img


@pytest.mark.parametrize("rows", [3, 8, 21])
def test_cells_are_found_on_short_and_long_tables_alike(rows):
    """The Man-Ai regression: cell detection used to drop any cell bigger
    than 5% of the table's area. On a transmittal with a header and two data
    rows every cell is legitimately a tenth of the table, so its widest
    columns vanished and a 6-column form arrived looking like a 3-column one.
    """
    boxes = ocr_form6._find_cell_boxes(_draw_table(rows, 6))
    assert len(boxes) == rows * 6, f"{rows}-row table lost cells"
    # And every row really has all six, not just the right total.
    grouped = ocr_form6._group_into_rows(boxes)
    assert [len(r) for r in grouped] == [6] * rows


@pytest.mark.parametrize(
    "header,field",
    [
        ("NO.", "no"), ("No.", "no"), ("Seq. No.", "no"),
        ("TEACHERS' NAME", "name"), ("Employee Name", "name"), ("Name", "name"),
        ("Last Name", "last"), ("First Name", "first"), ("M.I.", "mi"),
        ("NO. OF DAYS", "days"), ("No. of Day/s", "days"),
        ("DATE", "dates"), ("Date/s of Absence", "dates"), ("Inclusive Dates", "dates"),
        ("Leave Inclusive dates", "dates"),
        ("TYPE OF LEAVE", "type"), ("Leave Type", "type"), ("CAUSE", "type"),
        ("ACTION", "action"), ("Action Taken", "action"), ("Remarks", "action"),
        ("Position", "position"),
        ("", None), ("Signature", None),
    ],
)
def test_headers_from_real_transmittals_map_to_fields(header, field):
    """Every school labels its columns; those labels are what the reader now
    dispatches on instead of counting columns."""
    assert ocr_form6._header_field(header) == field


def test_no_of_days_is_a_day_column_not_a_number_column():
    """"NO. OF DAYS" contains "NO." — order in the table matters."""
    assert ocr_form6._header_field("NO. OF DAYS") == "days"


def test_a_header_spanning_three_columns_labels_all_three():
    """Macanhan prints one "Employee Name" over Last / First / M.I., and that
    span is exactly how those three columns are recognised."""
    header_boxes = [(0, 0, 100, 40), (100, 0, 300, 40), (400, 0, 100, 40)]
    header = {0: "no", 100: "name", 400: "days"}
    row_boxes = [(0, 50, 100, 40), (100, 50, 100, 40), (200, 50, 100, 40),
                 (300, 50, 100, 40), (400, 50, 100, 40)]
    assert ocr_form6._fields_for_row(header, header_boxes, row_boxes) == [
        "no", "name", "name", "name", "days",
    ]


def test_a_row_of_words_is_not_mistaken_for_a_header():
    """Mapping against a data row would be worse than falling back."""
    with patch.object(ocr_form6, "_ocr_cell", lambda *a, **k: "Santos, Juana"):
        assert ocr_form6._map_header_row(_blank_cell(), [(0, 0, 10, 10)]) is None


@pytest.mark.parametrize(
    "raw,expected",
    [
        # A spaced "&" joins two ranges...
        ("July 1-3 & 6-7, 2026", "July 1-3 & 6-7, 2026"),
        ("June 25 & 30, 2026", "June 25 & 30, 2026"),
        # ...anything tighter is an 8 that Tesseract read as an ampersand.
        ("June &pm,9,10", "June 8pm,9,10"),
        ("June &, 2026", "June 8, 2026"),
        ("June 1&, 2026", "June 18, 2026"),
    ],
)
def test_ampersands_are_told_apart_from_eights(raw, expected):
    assert ocr_form6._disambiguate_ampersands(raw) == expected
