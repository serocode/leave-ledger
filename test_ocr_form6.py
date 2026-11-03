"""Tests for ocr_form6.py that don't need a scan or a Tesseract install.

Run with:
    pytest test_ocr_form6.py -v
"""

from collections import Counter
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


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("0.5 DAY", 0.5),   # the unit word hid the decimal: this read as 0
        ("0.5DAY", 0.5),
        ("05 DAY", 0.5),    # point lost in OCR: this read as 5
        ("1 DAY", 1.0),
        ("| DAY", 1.0),     # a 1 read as a bar
        ("2 DAYS", 2.0),
        ("Half day", 0.5),
        ("DAY", None),      # the number itself unreadable
    ],
)
def test_day_counts_with_a_unit_word(raw, expected):
    assert ocr_form6._normalize_total_days(raw) == expected


def test_solo_parent_leave_is_recognised():
    assert ocr_form6._normalize_leave_type("SOLO P") == "SPL"
    assert ocr_form6._normalize_leave_type("Solo Parent Leave") == "SPL"


def test_a_printed_leave_type_is_never_relabelled_as_sick_leave():
    """Bulua row 28: "SOLO P" / "WP". The no-type-column assumption must not
    reach a form that prints its type — that would file the wrong leave."""
    assert ocr_form6._resolve_leave_type("SOLO P", "WP") == ("SPL", "WP")
    assert ocr_form6._resolve_leave_type("CTO", "WP") == ("CTO", "WP")


def test_sick_leave_is_only_assumed_when_there_is_no_type_column():
    assert ocr_form6._resolve_leave_type(None, "w/ pay") == ("SL", "WP")
    assert ocr_form6._resolve_leave_type(None, "w/o pay") == ("SL", "WOP")
    assert ocr_form6._resolve_leave_type(None, "SL-WP") == ("SL", "WP")
    assert ocr_form6._resolve_leave_type(None, "") == ("", "")


@pytest.mark.parametrize(
    "text,kind",
    [
        ("25", "no"), ("l", "no"),
        ("LUCONAN, JOAN Y.", "name"), ("Abing, Kathlyn Grace C.", "name"),
        ("7/3/2026", "dates"), ("7/16-17/2026", "dates"), ("7[2/2026", "dates"),
        ("June 30, 2026", "dates"),
        ("1 DAY", "days"), ("0.5 DAY", "days"), ("2 DAYS", "days"), ("| DAY", "days"),
        ("SL", "type"), ("SOLO P", "type"), ("Sick Leave", "type"),
        ("WP", "action"), ("WOP", "action"), ("w/ pay", "action"),
        ("T-I", "position"), ("MT-II", "position"), ("SPET-I", "position"),
        ("", None),
    ],
)
def test_cells_are_classified_by_what_they_hold(text, kind):
    assert ocr_form6._classify_cell(text) == kind


def test_a_headerless_bulua_page_gets_its_columns_from_content():
    """The continuation page that failed: same six columns as page 1, no
    header row. Counting columns guessed the Masterson layout and read the
    "1 DAY" column as dates."""
    columns = [
        ["25", "28", "30", "33"],
        ["LUCONAN, JOAN Y.", "MUGOT, JANE D.", "PACTURAN, KENNY LYN R.", "SACLOT, ERALYN T."],
        ["7/3/2026", "7/16-17/2026", "7/7/2026", "7/13/2026"],
        ["1 DAY", "2 DAYS", "0.5 DAY", "1 DAY"],
        ["SL", "SOLO P", "SL", "SL"],
        ["WP", "WP", "WP", "WP"],
    ]
    assert ocr_form6._labels_from_samples(columns) == [
        "no", "name", "dates", "days", "type", "action",
    ]


def test_a_table_without_people_and_dates_is_not_given_labels():
    columns = [["1", "2"], ["Pencils", "Paper"], ["40", "12"]]
    assert ocr_form6._labels_from_samples(columns) is None


def test_one_mislabelled_cell_doesnt_decide_a_column():
    columns = [
        ["SMITH, JAN P.", "CRUZ, ANA B.", "7/1/2026"],   # one stray read
        ["7/1/2026", "7/2/2026", "7/3/2026"],
    ]
    assert ocr_form6._labels_from_samples(columns) == ["name", "dates"]


def test_a_row_missing_its_last_cell_gets_it_back_from_the_grid():
    """Bulua page 1, rows 23-24: the Action cell's border wasn't detected, so
    the row arrived five cells wide and its WP was lost."""
    rows = [
        [(10, 0, 90, 40), (105, 0, 495, 40), (605, 0, 356, 40),
         (965, 0, 209, 40), (1179, 0, 168, 40), (1351, 0, 175, 40)]
        for _ in range(4)
    ]
    grid = ocr_form6._column_grid(rows)
    short = [(8, 500, 92, 39), (105, 500, 494, 40), (604, 500, 358, 40),
             (966, 500, 210, 39), (1181, 500, 169, 39)]
    filled = ocr_form6._fill_row_to_grid(short, grid)
    assert len(filled) == 6
    x, y, w, h = filled[-1]
    assert (x, x + w) == (1351, 1526) and y == 500


def test_a_wide_header_cell_is_not_split_by_the_grid():
    """Macanhan's one "Employee Name" header spans three data columns — it
    covers them, so nothing gets synthesised under it."""
    grid = [(0, 100), (100, 200), (200, 300), (300, 400), (400, 500)]
    header = [(0, 0, 100, 40), (100, 0, 300, 40), (400, 0, 100, 40)]
    assert ocr_form6._fill_row_to_grid(header, grid) == header


def _dates(result):
    text, segments, warning = result
    return [(f.isoformat(), t.isoformat()) for f, t in segments], warning


def test_agreeing_reads_are_taken_without_a_warning():
    """Saclot, Bulua p2 — a real 13th reads the same every way."""
    segs, warning = _dates(ocr_form6._consensus_numeric_dates(["7/13/2026", "7/13/2026", "7/13/2026"]))
    assert segs == [("2026-07-13", "2026-07-13")] and warning is None


def test_an_unparseable_alternate_is_not_a_disagreement():
    """Obsioma — one alternate read garbage; that isn't evidence against 7/1."""
    segs, warning = _dates(ocr_form6._consensus_numeric_dates(["7/1/2026", "/1/2026", "7/1/2026"]))
    assert segs == [("2026-07-01", "2026-07-01")] and warning is None


def test_a_phantom_one_with_a_majority_against_it_is_corrected_and_flagged():
    """Cadavos, Bulua p1 — paper says 7/3; the primary read 7/13."""
    segs, warning = _dates(ocr_form6._consensus_numeric_dates(["7/13/2026", "-7/3/2026", "7/3/2026"]))
    assert segs == [("2026-07-03", "2026-07-03")]
    assert "July 3 or July 13" in warning and "check it against the paper" in warning


def test_a_tied_disagreement_is_left_blank_rather_than_guessed():
    """Luconan, Bulua p2 — 7/13 vs 7/3 with the third read unusable. This was
    submitted as July 13 without any warning before."""
    segs, warning = _dates(ocr_form6._consensus_numeric_dates(["7/13/2026", "-7/3/2026", "7/3/9096"]))
    assert segs == []
    assert "July 13 or July 3" in warning or "July 3 or July 13" in warning
    assert "enter it from the paper" in warning


def test_a_failed_primary_is_recovered_when_two_other_reads_agree():
    """Ceballos, Bulua p1 — "719-10/2026" can't parse; both alternates agree."""
    segs, warning = _dates(ocr_form6._consensus_numeric_dates(["719-10/2026", "-7/9-10/2026", "7/9-10/2026"]))
    assert segs == [("2026-07-09", "2026-07-10")] and warning is None


def test_a_recovery_resting_on_one_read_is_flagged():
    """Salvaña, Bulua p2 — only one of three reads parsed at all."""
    segs, warning = _dates(ocr_form6._consensus_numeric_dates(["73/2026", "7/3/2026", "73/2026"]))
    assert segs == [("2026-07-03", "2026-07-03")]
    assert "second pass only" in warning


def test_nothing_parseable_asks_for_manual_entry():
    segs, warning = _dates(ocr_form6._consensus_numeric_dates(["73/2026", "", "7/"]))
    assert segs == [] and "enter manually" in warning


def test_a_misread_year_is_not_a_date():
    """"7/3/2026" read as "7/3/9096" parsed as July 3 of the year 9096 — a
    date that could reach HRIS, and that made one read look like two votes."""
    segments, warning = ocr_form6._parse_date_segments_numeric("7/3/9096")
    assert segments == [] and warning


def test_a_garbage_year_doesnt_pad_a_disagreement():
    """Luconan, Bulua p2: the warning read "July 13 or July 3 or July 3"."""
    _, segments, warning = ocr_form6._consensus_numeric_dates(["7/13/2026", "-7/3/2026", "7/3/9096"])
    assert segments == []
    assert warning.count("July 3") == 1 and "July 13" in warning


def _row(**overrides):
    from datetime import date
    args = dict(
        row_no="7", last_name="Cadavos", first_name="Mansueta", middle_initial="B",
        position_raw="", dates_raw="7/3/2026", days_raw="1 DAY",
        leave_type="SL", action_taken="WP", in_scope=True,
        segments=[(date(2026, 7, 3), date(2026, 7, 3))],
        total_override=1.0, unparsed_warning=None,
    )
    args.update(overrides)
    return ocr_form6._rows_from_segments(**args)


def test_a_filled_in_date_keeps_its_check_this_warning():
    """Cadavos, Bulua p1: the majority reading was filled in with a warning to
    check it — and the warning was being dropped because dates existed, so the
    row reached the review table unflagged."""
    rows = _row(unparsed_warning="OCR disagreed on this date (July 3 or July 13) — check it.")
    assert rows[0].date_from == "2026-07-03"
    assert "OCR disagreed" in rows[0].parse_warning


def test_a_clean_row_still_has_no_warning():
    assert _row()[0].parse_warning is None


def test_both_a_read_warning_and_a_total_mismatch_are_kept():
    from datetime import date
    rows = _row(
        unparsed_warning="Date read on a second pass only.",
        segments=[(date(2026, 6, 25), date(2026, 6, 25)), (date(2026, 6, 29), date(2026, 6, 29))],
        total_override=1.5,
    )
    assert "second pass" in rows[0].parse_warning and "printed total" in rows[0].parse_warning


def test_a_disagreement_only_in_the_year_names_the_years():
    """Tubaon, Macanhan: the warning read "June 26 or June 26" — two readings
    differing only in year, with the year left out of the message."""
    _, segments, warning = ocr_form6._consensus_numeric_dates(["6/26/2026", "6/26/2028", "6/26/2026"])
    assert segments and segments[0][0].year == 2026
    assert "2026" in warning and "2028" in warning


def test_ordinary_disagreements_stay_short():
    _, _, warning = ocr_form6._consensus_numeric_dates(["7/13/2026", "7/3/2026", "7/3/2026"])
    assert "July 3 or July 13" in warning and "2026" not in warning


# --- Vacation service credits -----------------------------------------------

@pytest.mark.parametrize(
    "header,field",
    [
        ("No. of hours Served", "hours"),
        ("No. of Vacation Service Credits Granted", "credits"),
        ("No.", "no"),
        ("No. of Days", "days"),
        ("Action Taken", "action"),
    ],
)
def test_grant_headers_dont_collide_with_leave_headers(header, field):
    """Both grant headers start "No. of", and "VACATION" contains "ACTION" —
    they used to land on the row-number and Action columns."""
    assert ocr_form6._header_field(header) == field


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Kenneth Joh Conde", ("Conde", "Kenneth Joh", "")),
        ("Krezel Marie M. Salva", ("Salva", "Krezel Marie", "M.")),
        ("Ana Marie A. Tres Reyes", ("Tres Reyes", "Ana Marie", "A.")),   # compound surname
        ("Danny P. Quilatan Jr", ("Quilatan", "Danny", "P.")),            # suffix dropped
        ("Vanessa Grace E. Alimios", ("Alimios", "Vanessa Grace", "E.")),
        ("LUCONAN, JOAN Y.", ("Luconan", "Joan", "Y.")),                  # comma order still works
    ],
)
def test_names_in_either_order(raw, expected):
    assert ocr_form6._split_any_name(raw) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [("138", 138.0), ("18 HRS", 18.0), ("4HRS.& 53MINS.", 4.883), ("10HRS&27MINS", 10.45), ("", None)],
)
def test_hours_served(raw, expected):
    assert ocr_form6._normalize_hours(raw) == expected


def test_the_stated_formula():
    """8 hours rendered on a vacation day, weekend or holiday = 1.50 days."""
    assert ocr_form6.vsc_days(8) == 1.5
    assert ocr_form6.vsc_days(138) == 25.875


@pytest.mark.parametrize(
    "hours,printed",
    [(138, 25.944), (122, 22.936), (105, 19.74), (66, 12.408), (141, 26.508),   # Pedro Oloy N. Roa, 0.188/hr
     (138, 25.875), (8, 1.5)],                                                   # the exact rate
)
def test_grants_at_either_legitimate_rate_pass(hours, printed):
    earned, warning = ocr_form6._resolve_vsc_credits(hours, printed)
    assert earned == printed and warning is None


@pytest.mark.parametrize(
    "hours,printed",
    [
        (8, 1.5), (18, 3.0), (4.883, 0.5), (6.95, 1.0),          # Kauswagan: truncated to half-days
        (32, 6.0), (31.617, 5.732), (39.5, 7.219), (22.217, 4.0),  # Sacred Heart: below the formula
    ],
)
def test_offices_that_grant_below_the_formula_are_not_flagged(hours, printed):
    """Each office works its grants out its own way. A figure at or under what
    the hours can earn is the order's to decide, not a misread."""
    earned, warning = ocr_form6._resolve_vsc_credits(hours, printed)
    assert earned == printed and warning is None


@pytest.mark.parametrize(
    "hours,printed",
    [(136, 29.568),   # Quilatan: 25.568 with a 5 read as 9
     (31.617, 9.132)],  # Agbon: 5.732 with a 5 read as 9
)
def test_a_grant_above_what_the_hours_can_earn_is_never_prefilled(hours, printed):
    """Flagged isn't enough: a prefilled row stays ticked for submission."""
    earned, warning = ocr_form6._resolve_vsc_credits(hours, printed)
    assert earned is None and "more than" in warning


def test_an_unreadable_grant_is_left_blank_not_computed():
    """The formula is exactly what these offices don't consistently use — a
    computed figure filled in would be a guess."""
    earned, warning = ocr_form6._resolve_vsc_credits(138, None)
    assert earned is None and "enter the figure from the paper" in warning and "25.875" in warning


def test_a_grant_period_spans_months():
    assert ocr_form6._period_from_dates("May 6 to June 2, 2026", None)[:2] == ("2026-05-06", "2026-06-02")
    assert ocr_form6._period_from_dates("May 12 to June 2, 2026", None)[:2] == ("2026-05-12", "2026-06-02")


def test_a_headerless_grant_page_is_labelled_from_content():
    columns = [
        ["12", "13", "14"],
        ["Dyesebel G. Eparwa", "Vanessa Grace E. Alimios", "Ellenor F. Questadio"],
        ["Teacher I", "Teacher I", "Teacher I"],
        ["May 6 to June 2, 2026", "May 6 to June 2, 2026", "May 6 to June 2, 2026"],
        ["120", "114", "141"],
        ["22.56", "21.432", "26.508"],
    ]
    labels = ocr_form6._labels_from_samples(columns)
    assert labels[4] == "hours" and labels[5] == "credits" and labels[0] == "no"


def test_a_lost_decimal_point_cannot_pass_as_a_grant():
    """Scan 2026-09-09: "7.21" read as 721 for June 1-5 — prefilled and ticked
    before this, with only a "not cross-checked" note."""
    earned, warning = ocr_form6._resolve_vsc_credits(None, 721, period_days=5)
    assert earned is None and "more than a 5-day period" in warning


def test_a_believable_grant_within_its_period_is_kept():
    earned, warning = ocr_form6._resolve_vsc_credits(None, 9.975, period_days=5)
    assert earned == 9.975 and "weren't checked" in warning


def test_the_ceiling_is_24_hours_a_day_at_time_and_a_half():
    assert ocr_form6.MAX_VSC_DAYS_PER_CALENDAR_DAY == 4.5
    assert ocr_form6._resolve_vsc_credits(None, 22.5, period_days=5)[0] == 22.5
    assert ocr_form6._resolve_vsc_credits(None, 22.6, period_days=5)[0] is None


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("2-14-26", ("2026-02-14", "2026-02-14")),
        ("1-31-26, 2-7-26", ("2026-01-31", "2026-02-07")),
        ("02/07/2026", ("2026-02-07", "2026-02-07")),
        ("May 6 to June 2, 2026", ("2026-05-06", "2026-06-02")),
    ],
)
def test_grant_periods_in_every_date_style(raw, expected):
    assert ocr_form6._period_from_dates(raw, None)[:2] == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("4HRS.& S3MINS.", 4.883),   # Alfonso: the 5 read as S — this came out 4.05
        ("SHRS.& 27MINS.", 5.45),    # Bunanebunane: came out 0.45
        ("1OHRS&27MINS", 10.45),     # O for 0
        ("18 HRS", 18.0),
    ],
)
def test_hours_survive_letter_lookalikes(raw, expected):
    assert ocr_form6._normalize_hours(raw) == expected


def test_half_a_day_written_with_a_leading_point():
    """Kauswagan prints ".5" — it was being read as 5 days."""
    assert ocr_form6._normalize_credits(".5") == 0.5
    assert ocr_form6._normalize_credits("1.5") == 1.5
    assert ocr_form6._normalize_credits("3") == 3.0


def test_hours_and_minutes_spelled_out():
    assert ocr_form6._normalize_hours("31 HOURS & 37 MINUTES") == 31.617
    assert ocr_form6._normalize_hours("32 HOURS") == 32.0


def test_credit_reads_that_agree_are_taken():
    assert ocr_form6._consensus_credits(["5.732", "5.732", "5.732"]) == (5.732, None)


def test_a_credit_misread_downward_is_caught_by_disagreement():
    """No upper bound can catch 25.568 read as 23.568 — disagreeing reads can."""
    value, warning = ocr_form6._consensus_credits(["23.568", "25.568", "25.568"])
    assert value == 25.568 and "disagreed" in warning


def test_a_tied_credit_read_is_left_blank():
    value, warning = ocr_form6._consensus_credits(["9.132", "5.732", ""])
    assert value is None and "enter it from the paper" in warning


@pytest.mark.parametrize(
    "text,kind",
    [("30 HOURS & 13 MINUTES", "hours"), ("40 HOURS", "hours"), ("4HRS.& 53MINS.", "hours"),
     ("5.575", "credits"), ("MASTER TEACHER I", "position")],
)
def test_grant_continuation_page_cells(text, kind):
    """Sacred Heart pages 3-4 repeat the table with no header, and spell the
    hours out — "HOURS", not "HRS"."""
    assert ocr_form6._classify_cell(text) == kind


def test_a_majority_above_the_hours_is_blank_with_a_hint():
    """Agbon, Sacred Heart: reads 9.132, 9.132, 5.732; the paper says 5.732.
    The hours veto the majority, but don't get to pick the minority."""
    ceiling = ocr_form6.vsc_ceiling(31.617)
    value, warning = ocr_form6._consensus_credits(["9.132", "9.132", "5.732"], ceiling)
    assert value is None and "Only 5.732 fits" in warning


def test_misread_hours_cannot_choose_a_wrong_credit():
    """Uy, Kauswagan: paper 9h25m and 1.5 days; hours read as 5h25m, whose
    ceiling (1.02) excludes the true 1.5 — this used to fill in a wrong 1."""
    ceiling = ocr_form6.vsc_ceiling(5.417)
    value, warning = ocr_form6._consensus_credits(["1.5", "1.5", "1"], ceiling)
    assert value is None and "enter it from the paper" in warning


def test_readings_that_all_exceed_the_hours_are_left_blank():
    """Cabiloque: 9.975 / 9.575 for 30.2 hours (ceiling 5.68); paper 5.575."""
    value, warning = ocr_form6._consensus_credits(["9.975", "9.575", "9.975"], ocr_form6.vsc_ceiling(30.217))
    assert value is None and "enter it from the paper" in warning


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("2/ HOURS & 40 MINUTES", 27.667),   # Gaid: a 7 read as "/" — this came out 0.667
        ("/ HOURS & 31 MINUTES", 7.517),
        ("HOURS & 40 MINUTES", None),        # the hour figure itself unreadable: blank, not 40 minutes
        ("27 HOURS & MINUTES", None),
    ],
)
def test_a_half_read_hours_cell_is_not_a_smaller_number(raw, expected):
    assert ocr_form6._normalize_hours(raw) == expected


def _fake_pages(page_outcomes, tables=1):
    """Drive extract_form6_with_notes with canned per-page results."""
    from collections import Counter
    outcomes = iter(page_outcomes)

    def fake_extract(img, seq, earlier_event="", context=None):
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        rows, dropped = outcome
        return rows, seq, Counter({6: dropped}) if dropped else Counter()

    return (
        patch.object(ocr_form6, "_load_pages", lambda path: [None] * len(page_outcomes)),
        patch.object(ocr_form6, "_extract_page", fake_extract),
        patch.object(ocr_form6, "_count_table_regions", lambda img: tables),
        patch.object(ocr_form6, "_upright", lambda img: img),
        patch.object(ocr_form6, "_deskew", lambda img: img),
    )


def _some_row():
    return ocr_form6.Form6Row("1", "Cruz", "Ana", "", "", "", "", "SL", "WP")


def test_a_page_that_yields_nothing_is_reported():
    """Sacred Heart page 4: ten grants silently missing from an upload that
    otherwise read fine."""
    load, extract, count, upright, deskew = _fake_pages([([_some_row()], 1), ([], 11)])
    with load, extract, count, upright, deskew:
        rows, notes = ocr_form6.extract_form6_with_notes("x.pdf")
    assert len(rows) == 1
    assert len(notes) == 1 and notes[0].startswith("Page 2") and "11 rows" in notes[0]


def test_a_page_losing_many_rows_is_reported():
    load, extract, count, upright, deskew = _fake_pages([([_some_row()] * 5, 4)])
    with load, extract, count, upright, deskew:
        _, notes = ocr_form6.extract_form6_with_notes("x.jpg")
    assert len(notes) == 1 and "4 table rows couldn't be read" in notes[0]


def test_a_header_and_a_blank_line_are_not_worth_a_note():
    load, extract, count, upright, deskew = _fake_pages([([_some_row()] * 20, 2)])
    with load, extract, count, upright, deskew:
        _, notes = ocr_form6.extract_form6_with_notes("x.jpg")
    assert notes == []


def test_a_region_covering_many_cells_does_not_swallow_them():
    """Sacred Heart page 4: one contour spanned the left four columns, and
    every real cell inside it was dropped as a nested stray mark."""
    cells = [(x, y, 100, 40) for y in (0, 45, 90) for x in (0, 105, 210)]
    region = (0, 0, 315, 135)
    kept = ocr_form6._drop_nested_boxes(cells + [region])
    assert region not in kept and sorted(kept) == sorted(cells)


def test_a_stray_mark_inside_one_cell_is_still_dropped():
    cell, speck = (0, 0, 200, 40), (20, 10, 30, 15)
    assert ocr_form6._drop_nested_boxes([cell, speck]) == [cell]


def test_a_merged_hours_and_credits_header_names_both_fields():
    fields = ocr_form6._header_fields_all("No. of Hours Served No. of Vacation Service Credits Granted")
    assert {"hours", "credits"} <= fields


def test_employee_name_over_three_columns_names_one_field():
    assert ocr_form6._header_fields_all("Employee Name") == {"name"}


def test_a_merged_header_is_split_along_the_grid():
    """Kauswagan: one header box over the hours and credits columns."""
    grid = [(5, 118), (123, 646), (651, 955), (960, 1414), (1419, 1825), (1830, 2173)]
    parts = ocr_form6._split_header_box((1419, 3, 754, 114), grid)
    assert [(x, x + w) for x, _, w, _ in parts] == [(1419, 1825), (1830, 2173)]


def test_a_merged_header_maps_each_column():
    """No. | Name | Inclusive Dates | [No. of Hours Served + No. of VSC Granted]"""
    grid = [(0, 100), (100, 300), (300, 450), (450, 600), (600, 800)]
    header_boxes = [(0, 0, 100, 40), (100, 0, 200, 40), (300, 0, 150, 40), (450, 0, 350, 40)]

    def fake_ocr(img, box, psm, wl):
        x, _, w, _ = box
        return {
            (0, 100): "No.", (100, 200): "Name", (300, 150): "Inclusive Dates",
            (450, 350): "No. of Hours Served No. of Vacation Service Credits Granted",
            (450, 150): "No. of Hours Served", (600, 200): "No. of Vacation Service Credits Granted",
        }[(x, w)]

    with patch.object(ocr_form6, "_ocr_cell", fake_ocr):
        mapping, boxes = ocr_form6._map_header_row(None, header_boxes, grid)
    assert mapping == {0: "no", 100: "name", 300: "dates", 450: "hours", 600: "credits"}
    assert len(boxes) == 5


# --- Kauswagan Annex D: open bottom, ruleless columns, "DO" -----------------

def _page_with_text(height, width, texts):
    """A white page with black text drawn at (x, baseline y)."""
    import cv2
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    for text, x, y in texts:
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    return img


def test_two_columns_with_no_rule_between_are_split_at_the_gap():
    """"16" and "3.000" sit in one detected cell per row, 200px apart."""
    rows = [(0, 60), (60, 60), (120, 60)]
    img = _page_with_text(180, 500, [(t, x, y + 42) for y, _ in rows for t, x in (("16", 60), ("3.000", 330))])
    cells = [(0, y, 500, h) for y, h in rows]
    split = ocr_form6._find_ruleless_split(img, cells)
    assert split is not None and 110 < split < 330


def test_a_word_gap_is_not_a_column():
    rows = [(0, 60), (60, 60), (120, 60)]
    img = _page_with_text(180, 500, [("1 DAY", 200, y + 42) for y, _ in rows])
    assert ocr_form6._find_ruleless_split(img, [(0, y, 500, h) for y, h in rows]) is None


def test_rows_below_the_last_closed_cell_are_recovered():
    """Three closed rows at a 60px pitch, then 120px of table whose column rules
    continue but whose row rules are missing, holding two more rows of text."""
    import cv2
    grid = [(0, 100), (100, 400)]
    grouped = [[(0, y, 100, 60), (100, y, 300, 60)] for y in (0, 60, 120)]
    img = _page_with_text(300, 400, [("7", 30, 222), ("CLERIGO, IRENE", 120, 222),
                                     ("8", 30, 282), ("LABISCASE, S", 120, 282)])
    cv2.line(img, (100, 180), (100, 299), (0, 0, 0), 2)      # the column rule, running on down
    out = ocr_form6._extend_open_bottom(img, grouped, grid)
    assert len(out) == 5
    assert [row[0][1] for row in out[3:]] == [180, 240]


def test_footer_text_below_a_closed_table_is_not_a_row():
    """East Gusa's page footer sits inside the dewarped area: it has ink, but no
    column rules, and was read as a teacher ("Wi, Peed Address: Fr...")."""
    grid = [(0, 100), (100, 400)]
    grouped = [[(0, y, 100, 60), (100, y, 300, 60)] for y in (0, 60, 120)]
    img = _page_with_text(300, 400, [("Address: Fr. William F.", 20, 222), ("Mobile No: +63 975 6403", 20, 282)])
    assert ocr_form6._extend_open_bottom(img, grouped, grid) == grouped


def test_a_closed_table_gains_no_rows():
    grid = [(0, 100), (100, 400)]
    grouped = [[(0, y, 100, 60), (100, y, 300, 60)] for y in (0, 60, 120)]
    img = _page_with_text(182, 400, [])
    assert ocr_form6._extend_open_bottom(img, grouped, grid) == grouped


@pytest.mark.parametrize("raw", ["DO", "do", "D0", "-do-", "Ditto", '"DO"'])
def test_ditto_marks(raw):
    assert ocr_form6._DITTO_RE.match(raw)


@pytest.mark.parametrize("raw", ["June 12-13, 2026", "", "DOE", "06/12/2026"])
def test_not_ditto_marks(raw):
    assert not ocr_form6._DITTO_RE.match(raw)


def test_a_row_number_breaking_the_count_is_repaired():
    """Kauswagan's serif 5 reads as 9, between a 4 and a 6. A form row split
    into several Form6Rows keeps one number across all of them."""
    def row(no):
        return ocr_form6.Form6Row(
            row_no=no, last_name="X", first_name="Y", middle_initial="", position_raw="",
            dates_raw="", days_raw="", leave_type="SL", action_taken="WP",
            date_from=None, date_to=None, description="", used="", parse_warning=None,
        )
    rows = [row("4"), row("9"), row("9"), row("6"), row("9")]
    ocr_form6._repair_row_numbers(rows)
    assert [r.row_no for r in rows] == ["4", "5", "5", "6", "9"]


@pytest.mark.parametrize("raw,expected", [
    ("TEACHER |", "Teacher I"), ("TEACHER lil", "Teacher III"),
    ("TEACHER UHI", "Teacher"), ("MASTER TEACHER |", "Master Teacher I"), ("T-III", "T-III"),
])
def test_normalize_position(raw, expected):
    assert ocr_form6._normalize_position(raw) == expected


# --- "-do-" in hours and credits; photos of several sheets -----------------

def _vsc_row_from(texts, previous=None, img=None):
    """Run _extract_vsc_row with each column's OCR text given directly."""
    boxes = {"name": (0, 0, 10, 10), "dates": (10, 0, 10, 10), "hours": (20, 0, 10, 10), "credits": (30, 0, 10, 10)}
    by_box = {box: texts[field] for field, box in boxes.items()}
    cells = {field: [("", box)] for field, box in boxes.items()}

    def fake_ocr(img, box, psm, wl, pad=4):
        text = by_box[box]
        if wl == "0123456789.,":
            # The digits whitelist can't represent "-do-"; it comes back a 0.
            return "0" if ocr_form6._DITTO_RE.match(text) else text
        return text

    with patch.object(ocr_form6, "_ocr_cell", fake_ocr), \
         patch.object(ocr_form6, "_ocr_cell_plain", fake_ocr), \
         patch.object(ocr_form6, "_ocr_date_cell", lambda img, box: (by_box[box], 0)):
        return ocr_form6._extract_vsc_row(img, cells, 2026, 1, previous)


def test_ditto_hours_and_credits_repeat_the_row_above():
    """Lapasan's Annex D: row 1 prints May 14-15, 2026 / 16 / 2.5 and every
    row under it says "-do-" in all three columns."""
    first = _vsc_row_from({"name": "ALIMATAR, APIDA P.", "dates": "May 14-15, 2026", "hours": "16", "credits": "2.5"})[0]
    second = _vsc_row_from({"name": "ALON III, NELSON E.", "dates": "-do-", "hours": "-do-", "credits": "-do-"}, first)[0]
    third = _vsc_row_from({"name": "ROCAMORA, RICA B.", "dates": "-do-", "hours": "-do-", "credits": "-do-"}, second)[0]
    for row in (second, third):
        assert (row.date_from, row.date_to) == ("2026-05-14", "2026-05-15")
        assert row.hours == 16 and row.earned == "2.5"
        assert row.parse_warning is None and row.in_scope


def test_ditto_credits_are_never_read_as_zero():
    """With no row above to repeat, "-do-" is left blank and flagged — the
    digits-only read would otherwise record 0 credits."""
    row = _vsc_row_from({"name": "ALON III, NELSON E.", "dates": "May 14-15, 2026", "hours": "16", "credits": "-do-"})[0]
    assert row.earned == "" and not row.in_scope
    assert "-do-" in row.parse_warning


def test_several_sheets_in_one_photo_are_counted():
    import cv2
    page = np.full((1400, 900, 3), 255, dtype=np.uint8)
    table = _draw_table(3, 4, cell_w=200, cell_h=80)
    page[20:20 + table.shape[0], 20:20 + table.shape[1]] = table
    assert ocr_form6._count_table_regions(page) == 1
    page[700:700 + table.shape[0], 20:20 + table.shape[1]] = table
    assert ocr_form6._count_table_regions(page) == 2


def test_a_photo_of_several_sheets_says_so():
    """Rows read off one sheet still leave the others' teachers missing."""
    load, extract, count, upright, deskew = _fake_pages([([_some_row()] * 3, 0)], tables=3)
    with load, extract, count, upright, deskew:
        rows, notes = ocr_form6.extract_form6_with_notes("x.jpg")
    assert len(rows) == 3
    assert len(notes) == 1 and "3 separate tables" in notes[0]


def test_a_photo_of_several_sheets_that_reads_nothing_explains_why():
    load, extract, count, upright, deskew = _fake_pages([([], 4)], tables=3)
    with load, extract, count, upright, deskew:
        with pytest.raises(ValueError, match="3 separate tables.*Photograph each page"):
            ocr_form6.extract_form6_with_notes("x.jpg")


# --- Sideways photos, decimal hours, blank pre-printed rows -----------------

@pytest.mark.parametrize("raw,expected", [("72.52", 72.52), ("113.29", 113.29), ("10.63", 10.63), ("10.00", 10.0), ("l6", 16.0)])
def test_decimal_hours_keep_their_point(raw, expected):
    """Kauswagan and Taglimao print hours to two decimals; "72.52" was read
    as 7252 hours, which lifted the credit ceiling instead of tripping it."""
    assert ocr_form6._normalize_hours(raw) == expected


@pytest.mark.parametrize("osd,shape", [
    ("Rotate: 270\nOrientation confidence: 15.90\n", (10, 20)),
    ("Rotate: 90\nOrientation confidence: 17.86\n", (10, 20)),
    ("Rotate: 180\nOrientation confidence: 3.48\n", (20, 10)),   # Bulua p2 is upright
    ("Rotate: 0\nOrientation confidence: 22.83\n", (20, 10)),
])
def test_only_a_confident_orientation_turns_the_page(osd, shape):
    page = np.zeros((20, 10, 3), dtype=np.uint8)
    with patch.object(ocr_form6, "_run_tesseract", lambda img, psm, wl: osd):
        assert ocr_form6._upright(page).shape[:2] == shape


def test_a_missing_orientation_pack_leaves_the_page_alone():
    page = np.zeros((20, 10, 3), dtype=np.uint8)
    def boom(img, psm, wl):
        raise ValueError("Tesseract failed: Failed loading language 'osd'")
    with patch.object(ocr_form6, "_run_tesseract", boom):
        assert ocr_form6._upright(page) is page


def test_paper_grain_and_rules_are_not_text():
    rng = np.random.default_rng(0)
    cell = np.clip(150 + rng.normal(0, 4, (90, 300, 3)), 0, 255).astype(np.uint8)
    cell[45:47, :] = 0   # a rule dragged through the cell by a crooked row
    assert not ocr_form6._has_text(cell, (0, 0, 300, 90))
    written = _page_with_text(90, 300, [("DOTAROT", 20, 60)])
    assert ocr_form6._has_text(written, (0, 0, 300, 90))


def test_a_one_teacher_order_splits_hours_from_credits():
    """Kauswagan SS 1671: a header and one row, with a signature crossing
    the row below — too few rows for a grid, and ink in that row's gap."""
    img = _page_with_text(300, 700, [("No. of Hours", 40, 60), ("No. Vacation", 400, 60),
                                     ("16", 90, 160), ("2.50", 470, 160)])
    import cv2
    cv2.line(img, (0, 250), (700, 230), (0, 0, 0), 6)   # the signature
    cells = [(0, 0, 700, 100), (0, 100, 700, 100), (0, 200, 700, 100)]
    split = ocr_form6._find_ruleless_split(img, cells)
    assert split is not None and 240 < split < 400


# --- The event a grant was earned for --------------------------------------

@pytest.mark.parametrize("paragraph,event", [
    ("The following teachers of KAUSWAGAN NATIONAL HIGH SCHOOL is hereby granted ser-\nvice credits for services "
     "rendered during DIVISION TRAINING FOR THE STRENGTHENED\nSENIOR HIGH SCHOOL CURRICULUM on JUNE 12-13. 2026 Held in "
     "Cagayan de Oro", "Division Training for the Strengthened Senior High School Curriculum"),
    # "ON THE" belongs to the name; "ON JUNE 1-5" ends it.
    ("service credits for services rendered during REGIONAL TRAINING OF TEACHERS ON THE STRENGTHENED SENIOR HIGH "
     "SCHOOL CURRICULUM ON JUNE 1-5. 2026.", "Regional Training of Teachers on the Strengthened Senior High School Curriculum"),
    ("rendered during CAPACITY BUILDING FOR CAMPUS JOURNALISM HELD AT FR. WILLIAM F. MASTERSON S.J. ELEMENTARY "
     "SCHOOL on May 25-26. 2026.", "Capacity Building for Campus Journalism"),
    # Pedro Roa's "during" OCR'd in pieces.
    ("vacation service credits for services rendered duri : 3 uring 2026 SUMMER REMEDIATION PROGRAMS (SRP) from May "
     "6 to June 2, 2026.", "2026 Summer Remediation Programs (SRP)"),
    ("service credit rendered during the participation in the conduct of Officiating in the Advance Training Course "
     "for Troop Leaders on May 11-17,2026 at Baikingon NHS", "Officiating in the Advance Training Course for Troop Leaders"),
    ("rendered during the Early Registration for SY 2026-2027 on Janauary 31 - February 27, 2026.",
     "Early Registration for SY 2026-2027"),
    # A continuation page: a table and no paragraph.
    ("No. Name Position Inclusive Dates 16 AZURA, LIEZEL A. Teacher I May 6 to May 26, 2026 93 17.484", ""),
])
def test_event_from_the_order_paragraph(paragraph, event):
    assert ocr_form6._event_from_text(paragraph) == event


def test_description_names_the_event():
    assert ocr_form6._vsc_description("Aral Summer Program", "2026-05-06", "2026-05-31", 113.29) == \
        "Vacation service credits, Aral Summer Program, May 6-31, 2026 (113.29 hrs)"
    # Without one, the format is what it always was.
    assert ocr_form6._vsc_description("", "2026-05-06", "2026-06-02", 138) == \
        "Vacation service credits May 6 - June 2, 2026 (138 hrs)"


# --- Tilted scans, pen strokes through cells, blank rows --------------------

def _tilted_table(degrees):
    """A ruled table (long horizontal and vertical rules) tilted by `degrees`."""
    import cv2
    img = _draw_table(6, 5, cell_w=300, cell_h=120)
    img = cv2.copyMakeBorder(img, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    h, w = img.shape[:2]
    return cv2.warpAffine(img, cv2.getRotationMatrix2D((w / 2, h / 2), -degrees, 1.0), (w, h),
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))


@pytest.mark.parametrize("degrees", [2.5, -2.0, 1.5])
def test_a_tilted_page_is_measured_and_straightened(degrees):
    """SO 1891 tilts 1.5-2.7 degrees; its rules broke into fragments."""
    img = _tilted_table(degrees)
    assert ocr_form6._skew_degrees(img) == pytest.approx(degrees, abs=0.4)
    assert abs(ocr_form6._skew_degrees(ocr_form6._deskew(img))) < 0.4


@pytest.mark.parametrize("degrees", [0.0, 0.4, -0.5])
def test_a_page_barely_tilted_is_left_exactly_as_scanned(degrees):
    """Every sample that read correctly before sits under 0.6 degrees, so none of
    them may change."""
    img = _tilted_table(degrees)
    assert ocr_form6._deskew(img) is img


def test_a_page_with_no_lines_is_left_alone():
    blank = np.full((400, 300, 3), 255, dtype=np.uint8)
    assert ocr_form6._skew_degrees(blank) is None and ocr_form6._deskew(blank) is blank


def _cell_with_digits(stroke):
    import cv2
    img = np.full((160, 340, 3), 235, dtype=np.uint8)
    cv2.putText(img, "60", (150, 95), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (90, 90, 90), 2)
    if stroke:
        cv2.line(img, (140, 0), (128, 160), (10, 10, 10), 9)     # a signature's downstroke
    return img


def test_a_pen_stroke_through_a_cell_is_detected():
    box = (0, 0, 340, 160)
    assert ocr_form6._has_pen_stroke(_cell_with_digits(True), [box])
    assert not ocr_form6._has_pen_stroke(_cell_with_digits(False), [box])


def test_a_stroke_splitting_a_cell_into_two_boxes_is_still_found():
    """The stroke acts as a rule, so the cell arrives as two boxes."""
    assert ocr_form6._has_pen_stroke(_cell_with_digits(True), [(0, 0, 130, 160), (130, 0, 210, 160)])


def test_a_stroke_makes_hours_unread_but_keeps_the_credits():
    """SO 1891 row 5: "60" under a signature reads as "1" hours, which used to
    refuse the correctly read 11.251 credits and put "(1 hrs)" in the description."""
    hours_box = (20, 0, 10, 10)
    with patch.object(ocr_form6, "_has_pen_stroke", lambda img, boxes: boxes == [hours_box]):
        [row] = _vsc_row_from(
            {"name": "HAGONAO, WINDY G.", "dates": "May 4-29, 2026", "hours": "1", "credits": "11.251"},
            img=np.zeros((10, 10, 3), dtype=np.uint8),
        )
    assert row.hours is None and row.earned == "11.251"
    assert "hrs" not in row.description
    assert "pen mark crosses the hours cell" in row.parse_warning
    assert row.in_scope


def test_faint_two_digit_figures_are_not_a_blank_row():
    """"96" and "18" in wide cells are 0.7-0.9% ink each — under the old
    per-cell cutoff, so the first teacher of SO 1891 vanished."""
    import cv2
    page = np.full((120, 900, 3), 235, dtype=np.uint8)
    cv2.putText(page, "96", (120, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (70, 70, 70), 2)
    cv2.putText(page, "18", (620, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (70, 70, 70), 2)
    boxes = [(0, 0, 450, 120), (450, 0, 450, 120)]
    assert sum(ocr_form6._ink_percent(page, b) for b in boxes) >= ocr_form6._MIN_FIGURES_INK_PERCENT
    blank = np.full((120, 900, 3), 235, dtype=np.uint8)
    assert sum(ocr_form6._ink_percent(blank, b) for b in boxes) < ocr_form6._MIN_FIGURES_INK_PERCENT


def test_a_number_broken_at_a_page_edge_is_repaired_across_pages():
    def grant(no):
        return ocr_form6.Form6Row(no, "X", "Y", "", "", "", "", "VSC", "", kind="vsc")
    rows = [grant("4"), grant("6"), grant("6"), grant("7")]    # page 1 ends on a misread 5
    ocr_form6._repair_row_numbers(rows)
    assert [r.row_no for r in rows] == ["4", "5", "6", "7"]


def test_a_page_with_no_table_is_reported_when_others_read():
    """SO 1891's page 3 vanished without a word: five teachers gone."""
    load, extract, count, upright, deskew = _fake_pages([([_some_row()], 0), ValueError("no table")])
    with load, extract, count, upright, deskew:
        rows, notes = ocr_form6.extract_form6_with_notes("x.pdf")
    assert len(rows) == 1
    assert any(n.startswith("Page 2: no table could be found") for n in notes)


def test_bold_digits_in_a_short_row_are_not_a_pen_stroke():
    """One scan's rows are 38px tall and its bold "4.883 HRS" 24px: over the
    60%-of-height line first used, so every row's hours were dropped."""
    import cv2
    cell = np.full((38, 400, 3), 235, dtype=np.uint8)
    cv2.putText(cell, "4.883 HRS", (120, 29), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3)
    assert not ocr_form6._has_pen_stroke(cell, [(0, 0, 400, 38)])


def test_a_cells_own_crooked_rule_is_not_a_pen_stroke():
    cell = _cell_with_digits(False)
    cell[:, 2:9] = 10          # a thick rule hugging the left edge, full height
    assert not ocr_form6._has_pen_stroke(cell, [(0, 0, 340, 160)])


def test_two_digit_reads_that_agree_outrank_a_misread_seven():
    """SO 1891 row 7: the free-text read gave "T1" (one hour); the two
    digit-oriented reads both gave 71."""
    boxes = {"name": (0, 0, 10, 10), "dates": (10, 0, 10, 10), "hours": (20, 0, 10, 10), "credits": (30, 0, 10, 10)}
    cells = {field: [("", box)] for field, box in boxes.items()}

    def fake_ocr(img, box, psm, wl, pad=4):
        return {(0, 0, 10, 10): "JOAQUIN, DISA L.", (10, 0, 10, 10): "May 4-29, 2026", (30, 0, 10, 10): "13.319",
                (20, 0, 10, 10): "71" if wl == "0123456789." else "T1"}[box]

    def fake_plain(img, box, psm, wl, pad=4):
        return "71" if box == (20, 0, 10, 10) else fake_ocr(img, box, psm, wl)

    with patch.object(ocr_form6, "_ocr_cell", fake_ocr), patch.object(ocr_form6, "_ocr_cell_plain", fake_plain), \
         patch.object(ocr_form6, "_ocr_date_cell", lambda img, box: (fake_ocr(img, box, 6, None), 0)):
        [row] = ocr_form6._extract_vsc_row(None, cells, 2026, 7)
    assert row.hours == 71 and row.earned == "13.319" and row.parse_warning is None


def test_a_tilted_page_is_straightened_only_after_it_reads_nothing_as_scanned():
    straight, tilted = object(), object()
    calls = []

    def extract(img, seq, earlier_event="", context=None):
        calls.append(img)
        if img is tilted:
            return [], seq, Counter()          # its rules broke into fragments
        return [_some_row()], seq + 1, Counter()

    with patch.object(ocr_form6, "_load_pages", lambda path: [tilted]), \
         patch.object(ocr_form6, "_extract_page", extract), \
         patch.object(ocr_form6, "_count_table_regions", lambda img: 1), \
         patch.object(ocr_form6, "_upright", lambda img: img), \
         patch.object(ocr_form6, "_deskew", lambda img: straight if img is tilted else img):
        rows, _ = ocr_form6.extract_form6_with_notes("x.pdf")
    assert len(rows) == 1 and calls == [tilted, straight]


def test_a_page_that_reads_as_scanned_is_never_straightened():
    """Straightening resamples every digit; a page that read fine must not change."""
    page = object()
    with patch.object(ocr_form6, "_load_pages", lambda path: [page]), \
         patch.object(ocr_form6, "_extract_page", lambda img, seq, ev="", ctx=None: ([_some_row()], seq + 1, Counter())), \
         patch.object(ocr_form6, "_count_table_regions", lambda img: 1), \
         patch.object(ocr_form6, "_upright", lambda img: img), \
         patch.object(ocr_form6, "_deskew", lambda img: pytest.fail("straightened a page that read fine")):
        rows, _ = ocr_form6.extract_form6_with_notes("x.pdf")
    assert len(rows) == 1


def test_a_header_split_uses_a_two_row_grid_when_no_three_row_grid_exists():
    """SO 1891 page 4: a header and two teachers. Too few rows to trust a grid
    for reading, but enough to say where "Hours" and "Credits" divide."""
    row = lambda y: [(0, y, 100, 50), (100, y, 300, 50), (400, y, 200, 50), (600, y, 200, 50)]
    grouped = [row(0), row(50), row(100)]
    assert ocr_form6._column_grid(grouped) is not None                        # 3 rows: the usual grid
    assert ocr_form6._column_grid(grouped[:2]) is None
    assert ocr_form6._column_grid(grouped[:2], min_rows=2) == [(0, 100), (100, 400), (400, 600), (600, 800)]


# --- SO 1892 (East Gusa): whole-number columns, stated period, two-pass dates --

@pytest.mark.parametrize("paragraph,period", [
    ("services rendered during the Early Registration for SY 2026-2027 on Janauary 31 - February 27, 2026.",
     ("2026-01-31", "2026-02-27")),
    ("services rendered during BRIGADA ESKWELA from June 1-5, 2026.", ("2026-06-01", "2026-06-05")),
    ("services rendered during DIVISION TRAINING FOR THE STRENGTHENED SENIOR HIGH SCHOOL CURRICULUM on JUNE 12-13. 2026 Held in",
     ("2026-06-12", "2026-06-13")),
    ("granted vacation services rendered during the 2026 SUMMER ACADEMIC REMEDIAL PROGRAM (SARP), on May 4 to 29, 2026.",
     ("2026-05-04", "2026-05-29")),
    ("rendered during 2026 SUMMER REMEDIATION PROGRAMS (SRP) from May 6 to June 2, 2026.", ("2026-05-06", "2026-06-02")),
    ("rendered during CAPACITY BUILDING HELD AT FR. WILLIAM F. MASTERSON on May 25-26. 2026.", ("2026-05-25", "2026-05-26")),
    # No stated period — or none that parses — means no check, never a guess.
    ("No. Name Position Inclusive Dates 16 AZURA, LIEZEL A. Teacher I May 6 to May 26, 2026", None),
    ("rendered during ORIENTATION ON THE STRENGTHENED SHS for the year", None),
])
def test_the_orders_stated_period(paragraph, period):
    assert ocr_form6._period_from_text(paragraph, 2026) == period


def _grant(no, date_from, date_to, **kw):
    return ocr_form6.Form6Row(no, "X", "Y", "", "", "", "", "VSC", "", date_from=date_from, date_to=date_to,
                              kind="vsc", hours=40, earned="5", in_scope=True, event="Brigada Eskwela",
                              description="Vacation service credits, Brigada Eskwela", **kw)


def test_a_date_outside_the_orders_period_is_flagged_and_unticked_not_changed():
    """SO 1892 read a 5 as a 9 in "June 1,2,3,4,5": June 1-9 against an order for June 1-5."""
    rows = [_grant("1", "2026-06-01", "2026-06-05"), _grant("2", "2026-06-01", "2026-06-09"),
            _grant("3", "2026-06-05", "2026-06-05"), _grant("4", "2026-06-01", "2026-06-05")]
    ocr_form6._apply_period(rows, ("2026-06-01", "2026-06-05"))
    assert [r.in_scope for r in rows] == [True, False, True, True]
    assert (rows[1].date_from, rows[1].date_to) == ("2026-06-01", "2026-06-09")      # kept: the doubt may be the paragraph's
    assert "outside the order's period (June 1-5, 2026)" in rows[1].parse_warning
    assert all(r.parse_warning is None for r in (rows[0], rows[2], rows[3]))


def test_if_much_of_the_page_is_outside_it_is_the_paragraph_that_was_misread():
    rows = [_grant("1", "2026-06-01", "2026-06-05"), _grant("2", "2026-07-01", "2026-07-05"),
            _grant("3", "2026-07-01", "2026-07-05"), _grant("4", "2026-07-02", "2026-07-05")]
    ocr_form6._apply_period(rows, ("2026-06-01", "2026-06-05"))
    assert all(r.in_scope and r.parse_warning is None for r in rows)


def test_no_stated_period_means_no_check():
    rows = [_grant("1", "2026-06-01", "2026-06-25")]
    ocr_form6._apply_period(rows, None)
    assert rows[0].in_scope and rows[0].parse_warning is None


PERIOD = ("2026-06-01", "2026-06-05")


@pytest.mark.parametrize("first,second,expect_from,expect_to,warns", [
    ("June 1,2,3,4,5, 2026", "June 1,2,3,4,5, 2026", "2026-06-01", "2026-06-05", False),   # agree
    ("June 1,2,3,4,5,42026", "June 1,2,3,4,5,2026", "2026-06-01", "2026-06-05", False),   # only the second parses
    ("June 1,2,3,4,9,", "June 1,2,3,4,5,", "2026-06-01", "2026-06-05", False),            # 5 misread as 9: the in-period read wins
    ("June 1,2,3,4,5,", "June 1,2,3,4,9,", "2026-06-01", "2026-06-05", False),
    ("June 1,2,3,4,9,", "June 1,23,4,5", "2026-06-01", "2026-06-09", True),               # both out of period: first, and say so
])
def test_two_reads_of_a_date_cell(first, second, expect_from, expect_to, warns):
    raw, date_from, date_to, warning = ocr_form6._choose_dates(first, second, 2026, PERIOD)
    assert (date_from, date_to) == (expect_from, expect_to)
    assert bool(warning) == warns
    if warns:
        assert "OCR disagreed on the dates" in warning


def test_two_reads_of_a_date_cell_agree_when_there_is_no_second_read():
    assert ocr_form6._choose_dates("June 5, 2026", "", 2026, None)[1:3] == ("2026-06-05", "2026-06-05")


def test_whole_number_hours_and_credits_are_told_from_the_row_number():
    """East Gusa's continuation pages have no header, and "40" and "5" look just like "16"."""
    columns = [["16", "17", "18"], ["DUBLIN, FLORDILYN RABUYO", "DUCA, JESSA LALUNA", "EMBATE, EMMANUEL BACARRO"],
               ["Teacher I", "Teacher I", "Teacher II"], ["June 1,2,3,4,5, 2026", "June 5, 2026", "June 1,2,3,5, 2026"],
               ["40", "8", "32"], ["5", "1", "4"]]
    assert ocr_form6._labels_from_samples(columns) == ["no", "name", "position", "dates", "hours", "credits"]


def test_a_leave_forms_numbers_are_not_mistaken_for_hours_and_credits():
    columns = [["1", "2", "3"], ["CRUZ, ANA", "DIAZ, JOSE", "LIM, SAM"], ["June 5, 2026", "June 8, 2026", "June 9, 2026"],
               ["1 day", "1 day", "2 days"], ["SL", "SL", "SL"], ["WP", "WP", "WP"]]
    labels = ocr_form6._labels_from_samples(columns)
    assert "hours" not in labels and "credits" not in labels


@pytest.mark.parametrize("text,name", [
    ("SALALIMA, RALPHA.", "SALALIMA, RALPHA."),
    ("  SALALIMA,   RALPHA. ", "SALALIMA, RALPHA."),
    ("Nnn eee an NNN", None),          # paper grain read as letters
    ("ta aes", None),
    ("OS", None),
    ("", None),
])
def test_a_teacher_listed_with_no_entries_is_noted_but_grain_is_not(text, name):
    """SO 1892 row 47 is "SALALIMA, RALPHA." and nothing else; Macabalan's blank
    pre-printed rows read as letters, and must not be reported as teachers."""
    assert ocr_form6._listed_name(text) == name


def test_hours_are_left_out_of_the_description_when_the_credits_were_refused_for_them():
    """SO 1891 row 7: "1 hrs" for a teacher who worked 71 would otherwise go to HRIS."""
    [row] = _vsc_row_from({"name": "JOAQUIN, DISA L.", "dates": "May 4-29, 2026", "hours": "1", "credits": "13.319"})
    assert row.earned == "" and "hrs" not in row.description
    [fine] = _vsc_row_from({"name": "MAGALSO, DENNIS M.", "dates": "May 4-29, 2026", "hours": "98", "credits": "18.375"})
    assert fine.earned == "18.375" and "(98 hrs)" in fine.description


def test_a_lone_digit_is_told_from_a_number_under_a_pen_stroke():
    """Only a lone digit is read cropped to it: "6.299" read that way as "6" would be wrong."""
    import cv2
    lone = np.full((90, 300, 3), 235, dtype=np.uint8)
    cv2.putText(lone, "5", (140, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (40, 40, 40), 3)
    many = np.full((90, 300, 3), 235, dtype=np.uint8)
    cv2.putText(many, "6.299", (80, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (40, 40, 40), 3)
    assert ocr_form6._glyph_count(lone, (0, 0, 300, 90)) == 1
    assert ocr_form6._glyph_count(many, (0, 0, 300, 90)) >= 4


# --- SO 1893: hidden "-do-" cells, page layout carried over, two-pass reads --

def _ditto_row_above():
    """A clean row whose hours, credits and dates were all "-do-"."""
    return ocr_form6.Form6Row("8", "Miso", "Czarina", "", "Teacher", "June 1-4, 2026", "", "VSC", "",
                              date_from="2026-06-01", date_to="2026-06-04", kind="vsc", hours=32, earned="6",
                              in_scope=True, inherited=("credits", "dates", "hours"))


def test_a_ditto_cell_hidden_by_a_pen_mark_is_taken_from_above_but_not_ticked():
    """SO 1893 rows 29-31: a signature runs down the dates column, whose cells
    read as "" or "-" instead of "-do-". In a "-do-" column the row above is the
    answer, but it was never seen, so the row isn't ticked."""
    [row] = _vsc_row_from({"name": "DORIA, LILIBETH Q.", "dates": "-", "hours": "-do-", "credits": "-do-"},
                          previous=_ditto_row_above())
    assert (row.date_from, row.date_to, row.hours, row.earned) == ("2026-06-01", "2026-06-04", 32, "6")
    assert "hidden by a pen mark" in row.parse_warning
    assert not row.in_scope


def test_a_blank_cell_is_not_inherited_outside_a_ditto_column():
    above = _ditto_row_above()
    above = ocr_form6.Form6Row(**{**above.__dict__, "inherited": ()})
    [row] = _vsc_row_from({"name": "DORIA, LILIBETH Q.", "dates": "", "hours": "32", "credits": "6"}, previous=above)
    assert row.date_from is None and not row.in_scope


def test_stray_digits_for_ditto_hours_give_way_to_the_row_above():
    """SO 1893 row 11: "-do-" under a pen mark read as "4" hours — and 6 credits
    can't come from 4 hours. The credits match the row above; the hours are the misread."""
    [row] = _vsc_row_from({"name": "PARAISO, JACKELINE C.", "dates": "-do-", "hours": "4", "credits": "-do-"},
                          previous=_ditto_row_above())
    assert row.hours == 32 and row.earned == "6"
    assert "misread under a pen mark" in row.parse_warning and not row.in_scope


def test_a_row_with_no_dates_is_never_ticked():
    [row] = _vsc_row_from({"name": "TAN, KAREN M.", "dates": "--", "hours": "32", "credits": "6"})
    assert row.date_from is None and not row.in_scope


def test_row_numbers_that_go_backwards_within_an_order_are_renumbered():
    def grant(no, event="Training"):
        return ocr_form6.Form6Row(no, "X", "Y", "", "", "", "", "VSC", "", kind="vsc", event=event)
    rows = [grant("49"), grant("30"), grant("1"), grant("2"), grant("06"), grant("58")]
    ocr_form6._enforce_increasing_numbers(rows)
    assert [r.row_no for r in rows] == ["49", "50", "51", "52", "53", "58"]
    # Two orders in one PDF each start at 1: left alone.
    two = [grant("1", "A"), grant("2", "A"), grant("1", "B"), grant("2", "B")]
    ocr_form6._enforce_increasing_numbers(two)
    assert [r.row_no for r in two] == ["1", "2", "1", "2"]


def test_a_school_column_beside_the_name_is_not_read_as_part_of_it():
    columns = [["13", "15", "17"], ["EVELYN C. ALERIA", "ANGELA MAE H. BAYAWA", "DARWIN D. ESTOQUE"],
               ["BUGO NHS", "TABLON NHS", "PUERTO NHS"], ["Teacher", "-do-", "-do-"],
               ["June 1-4, 2026", "-do-", "-do-"], ["32", "-do-", "-do-"], ["6", "-do-", "-do-"]]
    labels = ocr_form6._labels_from_samples(columns)
    assert labels[1] == "name" and labels[2] is None


def test_a_continuation_page_of_ditto_columns_uses_the_first_pages_layout():
    grid = [(0, 100), (100, 400), (400, 600), (600, 700), (700, 900), (900, 1000), (1000, 1100)]
    labels = ["no", "name", None, "position", "dates", "hours", "credits"]
    header = ocr_form6._header_from_labels(labels, grid)
    assert ocr_form6._labels_of(header, grid) == labels
    assert ocr_form6._doubtful(None, grid)
    assert ocr_form6._doubtful(ocr_form6._header_from_labels(["no", "name", "name", None, "dates", None, None], grid), grid)
    assert not ocr_form6._doubtful(header, grid)


def test_rows_are_rebuilt_from_the_number_column_when_the_grid_falls_apart():
    """SO 1893 page 4: a signature across the rows cuts the cells into fragments."""
    grid = [(0, 100), (100, 400), (400, 600)]
    boxes = [(2, 0, 96, 80), (2, 85, 96, 60), (2, 86, 96, 59), (150, 90, 30, 40), (2, 150, 96, 80), (420, 20, 30, 30)]
    rows = ocr_form6._rebuild_rows(boxes, grid)
    assert [(r[0][1], r[0][3]) for r in rows] == [(0, 80), (85, 60), (150, 80)]
    assert all(len(r) == 3 and r[2] == (400, r[0][1], 200, r[0][3]) for r in rows)


def test_two_readings_of_a_page_that_disagree_are_flagged():
    """SO 1892 row 49: "5" unreadable as scanned, read "3" once straightened."""
    def grant(no, name, earned, df="2026-06-01", dt="2026-06-05", hours=40):
        return ocr_form6.Form6Row(no, name, "Y", "", "", "", "", "VSC", "", date_from=df, date_to=dt,
                                  kind="vsc", hours=hours, earned=earned, in_scope=True)
    kept = [grant("48", "Salolog", "5"), grant("49", "Sialana", "3"), grant("50", "Soliano", "5")]
    other = [grant("48", "Salolog", "5"), grant("49", "Sialana", ""), grant("50", "Soliano", "4")]
    ocr_form6._flag_passes_that_disagree(kept, other)
    assert [r.in_scope for r in kept] == [True, True, False]          # an unread figure isn't a disagreement
    assert "disagree on the credits (5 or 4)" in kept[2].parse_warning


# --- SO 1898: pen strokes over credits, rows unlike their order, page footers --

def test_a_pen_stroke_through_the_credits_leaves_them_unread():
    """SO 1898 row 27, Gabitano: a "6" crossed by the signature read as "2" on
    every pass — under what 32 hours can earn, so nothing refused it."""
    credits_box = (30, 0, 10, 10)
    with patch.object(ocr_form6, "_has_pen_stroke", lambda img, boxes: boxes == [credits_box]):
        [row] = _vsc_row_from(
            {"name": "GABITANO, FAITH DYAN C.", "dates": "June 1-4, 2026", "hours": "32", "credits": "2"},
            img=np.zeros((10, 10, 3), dtype=np.uint8),
        )
    assert row.earned == "" and not row.in_scope
    assert "pen mark crosses the credits cell" in row.parse_warning
    assert row.hours == 32 and "(32 hrs)" in row.description



def test_a_figure_the_hours_earn_exactly_is_kept_under_a_pen_stroke():
    """Pedro Roa rows 12-15: the signature runs down the credits column beside a
    clean "22.56" — 120 hours at 0.188. No misread lands on that by chance."""
    credits_box = (30, 0, 10, 10)
    with patch.object(ocr_form6, "_has_pen_stroke", lambda img, boxes: boxes == [credits_box]):
        [row] = _vsc_row_from(
            {"name": "EPARWA, DYESEBEL", "dates": "May 6 - June 2, 2026", "hours": "120", "credits": "22.56"},
            img=np.zeros((10, 10, 3), dtype=np.uint8),
        )
    assert row.earned == "22.56" and row.in_scope and row.parse_warning is None


def _order_row(hours, earned, event="Grade 9 training", no="1"):
    return ocr_form6.Form6Row(no, "X", "Y", "", "", "", "", "VSC", "", date_from="2026-06-01",
                              date_to="2026-06-04", kind="vsc", hours=hours, earned=earned,
                              in_scope=True, event=event)


def test_a_row_unlike_the_rest_of_its_order_is_flagged_and_unticked():
    """SO 1898: 32 hrs for 6 credits on 109 of 111 rows. Aniñon read 32 for 2,
    and Aluro 34 for 6 — each believable alone, each ticked for submission."""
    usual = [_order_row(32, "6") for _ in range(6)]
    two_credits, more_hours = _order_row(32, "2"), _order_row(34, "6")
    shorter, unread = _order_row(24, "4.5"), _order_row(None, "6")      # a different grant; hours not read
    rows = usual + [two_credits, more_hours, shorter, unread]
    ocr_form6._flag_rows_unlike_their_order(rows)
    assert not two_credits.in_scope and "reads 32 hrs for 2" in two_credits.parse_warning
    assert not more_hours.in_scope and "reads 34 hrs for 6" in more_hours.parse_warning
    assert all(r.in_scope and r.parse_warning is None for r in usual + [shorter, unread])
    assert all(r.earned in ("2", "6") for r in (two_credits, more_hours))    # flagged, never changed


def test_an_order_without_a_clear_norm_is_left_alone():
    few = [_order_row(32, "6") for _ in range(4)] + [_order_row(32, "2")]         # too few rows to set a norm
    ocr_form6._flag_rows_unlike_their_order(few)
    # Offices below the formula by varying amounts: no pair is most of the order.
    varied = [_order_row(h, c) for h, c in [(30.217, "5.575"), (31.617, "5.732"), (32, "6"), (32, "6"),
                                         (32, "6"), (32, "5.9"), (28, "5"), (40, "7.5")]]
    ocr_form6._flag_rows_unlike_their_order(varied)
    assert all(r.in_scope and r.parse_warning is None for r in few + varied)


def test_an_office_granting_below_the_formula_may_give_the_same_credits_for_other_hours():
    """SO 1892 (East Gusa) grants 5 for 40 hours — below the formula's 7.5 — and
    the paper gives Paican, June 1-4, 5 for 32. Sialana's 40 for 3 was a misread."""
    usual = [_order_row(40, "5") for _ in range(6)]
    paican, sialana = _order_row(32, "5"), _order_row(40, "3")
    ocr_form6._flag_rows_unlike_their_order(usual + [paican, sialana])
    assert paican.in_scope and paican.parse_warning is None
    assert not sialana.in_scope and "reads 40 hrs for 3" in sialana.parse_warning


def test_one_orders_norm_does_not_judge_another():
    rows = [_order_row(32, "6", "Grade 9") for _ in range(6)] + [_order_row(32, "5", "Aral Program")]
    ocr_form6._flag_rows_unlike_their_order(rows)
    assert all(r.in_scope for r in rows)


# SO 1898 page 1, dewarped: the table's columns, and rows as the grid finder saw them.
_SO1898_GRID = [(6, 121), (126, 621), (626, 985), (990, 1302), (1306, 1592), (1596, 1788), (1792, 2021)]


def test_the_page_footer_joined_to_the_table_by_a_pen_line_is_not_a_row():
    """The "Doc. Ref. Code | Rev" box read as a teacher named "Eit Iedh Tan
    Ig I", pushing every row number after it up by two."""
    footer = [(1458, 1746, 199, 32), (1662, 1745, 93, 33), (1764, 1745, 115, 33),
              (1884, 1746, 76, 32), (1964, 1746, 86, 32)]
    assert not ocr_form6._sits_on_grid(footer, _SO1898_GRID)


def test_real_rows_sit_on_the_grid_even_when_cut_or_merged():
    row_10 = [(6, 1335, 115, 117), (126, 1335, 495, 117), (626, 1335, 358, 117), (990, 1335, 311, 118),
              (1306, 1335, 285, 118), (1596, 1335, 156, 118), (1759, 1335, 28, 118),   # hours split by the pen line
              (1792, 1335, 228, 118)]
    short = [(6, 252, 114, 108), (125, 253, 496, 107), (1792, 253, 229, 108)]
    spanning = [(6, 5, 1586, 243), (1596, 5, 425, 243)]
    for row in (row_10, short, spanning):
        assert ocr_form6._sits_on_grid(row, _SO1898_GRID)


# --- SO 1905: a day number broken by the cell's wrap, lone digits read cropped --

def test_a_day_number_broken_by_the_cells_line_wrap_is_rejoined():
    """SO 1905 rows 1-2: seventeen days in a narrow cell, "...28,2" / "9, 2026"."""
    raw = "May 6,7,8,11,12,13,14,15,18 ,19,20,21,22,25,26,28,2 9, 2026"
    assert ocr_form6._period_from_dates(raw, 2026) == ("2026-05-06", "2026-05-29", None)


@pytest.mark.parametrize("raw", [
    "May 1,2 9, 2026",       # 2 continues the list: two days, not a broken 29
    "May 6, 7, 8, 2026",     # spaced commas are separators
])
def test_days_that_continue_the_list_are_not_joined(raw):
    assert ocr_form6._join_wrapped_days(raw) == raw


def _cell_with(text, speck=False):
    import cv2
    cell = np.full((95, 190, 3), 235, dtype=np.uint8)
    cv2.putText(cell, text, (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (40, 40, 40), 3)
    if speck:
        cv2.circle(cell, (170, 80), 5, (40, 40, 40), -1)     # a blot far from the digit
    return cell


def _cropped_digits(cell):
    """The images _lone_digit_reads hands Tesseract, less their 20px border."""
    seen = []
    with patch.object(ocr_form6, "_run_tesseract", lambda img, psm, wl: seen.append(img) or "5"):
        reads = ocr_form6._lone_digit_reads(cell, (0, 0, 190, 95))
    return reads, [img[20:-20, 20:-20] for img in seen]


def test_a_lone_digit_is_read_cropped_to_its_ink():
    """SO 1905: a clean "5" alone in a tall, wide cell read blank, "9" or "8"
    whole. Each read gets the digit alone, at a set height."""
    reads, crops = _cropped_digits(_cell_with("5"))
    assert reads == ["5", "5", "5"]
    assert sorted(c.shape[0] for c in crops) == [36, 50, 50]
    assert all(c.shape[1] < c.shape[0] for c in crops)        # a digit's width, not the cell's


def test_a_mark_away_from_the_digit_is_left_out_of_the_crop():
    _, alone = _cropped_digits(_cell_with("5"))
    _, blotted = _cropped_digits(_cell_with("5", speck=True))
    assert [c.shape for c in blotted] == [c.shape for c in alone]


def test_the_point_of_a_half_is_kept_in_the_crop():
    """Kauswagan prints ".5": the point is too small to count as a digit, but
    cropped away it would read as 5."""
    _, five = _cropped_digits(_cell_with("5"))
    _, half = _cropped_digits(_cell_with(".5"))
    assert half[0].shape[1] > five[0].shape[1]


def test_a_one_digit_credits_cell_uses_the_cropped_reads():
    with patch.object(ocr_form6, "_glyph_count", lambda img, box, pad=6: 1), \
         patch.object(ocr_form6, "_lone_digit_reads", lambda img, box: ["5", "5", "5"]):
        [row] = _vsc_row_from(
            {"name": "EMBATE, EMMANUEL JR. B.", "dates": "May 6,11,18,25,30, 2026", "hours": "40", "credits": ""},
            img=np.zeros((10, 10, 3), dtype=np.uint8),
        )
    assert row.earned == "5" and row.parse_warning is None and row.in_scope


# --- SO 1849: numbers read at the size Tesseract reads best ------------------

def test_scaled_credits_are_taken_when_the_whole_cell_reads_are_impossible():
    """SO 1849 row 2: 5.252 for 28 hours, read whole as 9.2902 and 9.292."""
    old = [9.2902, 9.2902, 9.292]
    printed, warning = ocr_form6._consensus_credits(["9.2902", "9.2902", "9.292"], ocr_form6.vsc_ceiling(28))
    assert ocr_form6._settle_credits(old, printed, warning, 5.252, ocr_form6.vsc_ceiling(28)) == (5.252, None)


def test_scaled_credits_are_taken_when_nothing_else_read():
    assert ocr_form6._settle_credits([None, None, None], None, None, 7.5, ocr_form6.vsc_ceiling(40)) == (7.5, None)


def test_two_believable_credit_readings_that_disagree_are_left_blank():
    """SO 1849 row 25, Morata: 5.202 whole, 5.252 scaled — both under 28 hours' ceiling."""
    value, warning = ocr_form6._settle_credits([None, 5.202, None], 5.202, None, 5.252, ocr_form6.vsc_ceiling(28))
    assert value is None and "5.252 or 5.202" in warning


def test_a_scaled_reading_above_the_hours_changes_nothing():
    assert ocr_form6._settle_credits([3.0, 3.0, 3.0], 3.0, None, 13.0, ocr_form6.vsc_ceiling(16)) == (3.0, None)


def test_scaled_credits_agreeing_with_the_whole_cell_majority_settle_it():
    """SO 1891's Magalso, 18.375 for 98 hours: whole-cell reads 18.375, 18.379, 18.375."""
    old = [18.375, 18.379, 18.375]
    printed, warning = ocr_form6._consensus_credits(["18.375", "18.379", "18.375"], ocr_form6.vsc_ceiling(98))
    assert warning and ocr_form6._settle_credits(old, printed, warning, 18.375, ocr_form6.vsc_ceiling(98)) == (18.375, None)


def test_the_order_period_bounds_the_hours():
    assert ocr_form6._most_hours(("2026-06-01", "2026-06-05")) == 120
    assert ocr_form6._most_hours(None) is None


@pytest.mark.parametrize("hours,digit_reads,scaled,most,expected", [
    (395, (395, None), 35, 120, (35, None)),           # SO 1849 Salcedo: more hours than five days hold
    (None, (None, None), 27, 120, (27, None)),         # SO 1849 Sagosoy: read whole as "at"
    (2, (21, 27), 27, 120, (27, None)),                # SO 1849 Jugalbot: one whole-cell read agrees
    (90.97, (90.97, 90.97), 50.97, None, (None, "90.97 or 50.97")),   # SO 1896 Dubria: both believable
    (40, (40, 40), 40, 120, (40, None)),
    (40, (40, 40), None, 120, (40, None)),             # no scaled agreement: nothing changes
    (40, (40, 40), 400, 120, (40, None)),              # an impossible scaled read changes nothing
])
def test_scaled_hours_settle(hours, digit_reads, scaled, most, expected):
    assert ocr_form6._settle_hours(hours, digit_reads, scaled, most) == expected


def test_disputed_hours_are_left_out_and_the_row_unticked():
    with patch.object(ocr_form6, "_scaled_reads", lambda img, boxes, wl: ["50.97", "50.97"] if "," not in wl else []):
        [row] = _vsc_row_from({"name": "DUBRIA, ANA", "dates": "May 6 - June 2, 2026", "hours": "90.97", "credits": "10"})
    assert row.hours is None and "hrs" not in row.description
    assert "disagreed on the hours served (90.97 or 50.97)" in row.parse_warning
    assert row.earned == "10" and not row.in_scope


def _number_cell(text):
    import cv2
    cell = np.full((190, 250, 3), 255, dtype=np.uint8)
    cv2.putText(cell, text, (60, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (110, 110, 110), 1)     # small, thin and pale
    return cell


def test_a_number_is_cropped_close_and_read_at_two_heights():
    seen = []
    with patch.object(ocr_form6, "_run_tesseract", lambda img, psm, wl: seen.append(img) or "5.252"):
        assert ocr_form6._scaled_reads(_number_cell("5.252"), [(0, 0, 250, 190)], "0123456789.,") == ["5.252", "5.252"]
    assert [img.shape[0] - 40 for img in seen] == [24, 32]
    assert all(img.shape[1] > img.shape[0] for img in seen)        # the number's shape, not the tall cell's


def test_a_cell_split_by_a_pen_stroke_gets_no_scaled_reads():
    assert ocr_form6._scaled_reads(_number_cell("5.252"), [(0, 0, 120, 190), (120, 0, 130, 190)], "0123456789.,") == []


def test_a_ditto_cell_read_as_zero_is_not_a_scaled_value():
    assert ocr_form6._agreed(["0", "0."], ocr_form6._normalize_credits) is None
    assert ocr_form6._agreed(["5.252", "5.252"], ocr_form6._normalize_credits) == 5.252
    assert ocr_form6._agreed(["5.252", "9.252"], ocr_form6._normalize_credits) is None


# ---------------------------------------------------------------------------
# SO 2159 (Cagayan de Oro NHS Junior High, Annex D): numeric day lists with no
# year, a signature in blue across four rows, thin serif figures, and grants
# above the x1.50 formula.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    # The year is only in the paragraph; the days change month part way.
    ("5/6-8,11-15,18- 22,25,26,28,29, 6/1,2", ("2026-05-06", "2026-06-02")),
    ("5/7,8,11- 15,18,22,25,26,28,29", ("2026-05-07", "2026-05-29")),
    # The space before the next month lost: "29 6/1,2" read as "296/1,2".
    ("5/6-8,11-15,18- 22,25,26,28,296/1,2", ("2026-05-06", "2026-06-02")),
    # "to" and an en dash are the range, not noise to strip.
    ("5/6to6/2/2026", ("2026-05-06", "2026-06-02")),
    ("5/6 – 6/2/2026", ("2026-05-06", "2026-06-02")),
])
def test_numeric_day_lists_with_no_year(raw, expected):
    assert ocr_form6._period_from_dates(raw, 2026)[:2] == expected


def test_days_after_a_month_change_are_that_months():
    segments, warning = ocr_form6._parse_date_segments_numeric("5/28,29,6/1,2/2026")
    assert warning is None
    assert segments[-1] == (ocr_form6.date(2026, 6, 2), ocr_form6.date(2026, 6, 2))


def test_a_to_between_dates_survives_the_digits_reread():
    whitelists = []

    def fake_ocr(img, box, psm, wl, pad=4):
        whitelists.append(wl)
        return "56 to 62)2026" if len(whitelists) == 1 else "5/6to6/2/2026"

    with patch.object(ocr_form6, "_ocr_cell", fake_ocr):
        text, _ = ocr_form6._ocr_date_cell(None, (0, 0, 10, 10))
    assert "to" in whitelists[1] and text == "5/6to6/2/2026"


_SO2159_TOP = (
    "SPECIAL ORDER FOR VACATION SERVICE CREDITS Date: July 7, 2026 SPECIAL ORDER SS No. 2159 s. 2026 "
    "The teacher/s of CAGAYAN DE ORO NATIONAL HIGH SCHOOL JUNIOR HIGH is hereby granted vacation service "
    "credits for services rendered during the 2026 SUMMER REMEDIATION PROGRAMS (SRP) May 6 to June 2, 2026."
)


def test_a_period_with_no_on_or_from_and_a_title_before_it():
    """The title says "service credits" before the order's date line — which
    read as the period (July 7) — and the period itself has no "on"/"from"."""
    assert ocr_form6._period_from_text(_SO2159_TOP, 2026) == ("2026-05-06", "2026-06-02")
    assert ocr_form6._event_from_text(_SO2159_TOP) == "2026 Summer Remediation Programs (SRP)"


def test_a_word_that_isnt_a_month_doesnt_start_a_period():
    assert ocr_form6._period_from_text("services rendered during ANNEX 2 activities", 2026) is None


def test_a_confirmed_grant_above_the_formula_is_filled_in_with_a_warning():
    earned, warning = ocr_form6._resolve_vsc_credits_unbounded(114, 28.5, confirmed=True)
    assert earned == 28.5 and "not ticked" in warning
    # Unconfirmed, it is refused as before; confirmed, a lost decimal still is.
    assert ocr_form6._resolve_vsc_credits_unbounded(114, 28.5)[0] is None
    assert ocr_form6._resolve_vsc_credits_unbounded(114, 285, confirmed=True)[0] is None


def _so2159_row(credits_whole, close_up, thick, hours="114"):
    def scaled(img, boxes, wl):
        return [close_up, close_up] if "," in wl else [hours, hours]

    def thick_read(img, boxes, wl):
        return thick if "," in wl else hours

    with patch.object(ocr_form6, "_scaled_reads", scaled), patch.object(ocr_form6, "_thick_read", thick_read):
        [row] = _vsc_row_from({"name": "ALILIN, JOYLEN A", "dates": "5/6 - 6/2/2026", "hours": hours,
                                "credits": credits_whole})
    return row


def test_two_other_ways_of_reading_outvote_the_whole_cell_reads():
    """SO 2159's thin 5 reads as 9 across the whole cell; close up and with
    thickened strokes it reads right."""
    row = _so2159_row("28.9", "28.5", "28.5")
    assert row.earned == "28.5" and not row.in_scope       # above x1.50: shown, never ticked
    # A whole-cell reading the hours also allow is a believable rival: flagged.
    row = _so2159_row("5.9", "5.5", "5.5", hours="40")
    assert row.earned == "5.5" and not row.in_scope
    assert "5.5 close up and 5.9 across the whole cell" in row.parse_warning


def test_a_grant_above_the_formula_is_shown_but_not_ticked():
    row = _so2159_row("28.5", "28.5", "28.5")
    assert row.earned == "28.5" and row.hours == 114
    assert not row.in_scope and row.trusted
    assert "more than 114 hours earn at x1.50" in row.parse_warning


def test_without_agreement_the_formula_still_refuses():
    row = _so2159_row("28.9", "28.5", "")
    assert row.earned == "" and not row.in_scope


def test_the_credits_settle_disputed_hours_only_at_the_formula():
    assert ocr_form6._earns_as_printed(75, 14.06)          # 14.0625 to the cent
    assert not ocr_form6._earns_as_printed(79, 14.06)
    assert ocr_form6._earns_as_printed(100, 18.75)
    assert not ocr_form6._earns_as_printed(114, 28.5)


def _page_with_pen(blue=(200, 60, 20)):
    import cv2
    page = np.full((120, 200, 3), 255, dtype=np.uint8)
    cv2.line(page, (0, 60), (199, 60), (0, 0, 0), 2)                 # a row's rule
    cv2.putText(page, "28.5", (40, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (20, 20, 20), 2)
    cv2.line(page, (120, 0), (130, 119), blue, 4)                     # a signature's downstroke
    return page


def test_blue_pen_ink_is_removed_and_print_kept():
    page = _page_with_pen()
    clean = ocr_form6._drop_pen_ink(page)
    b, r = clean[..., 0].astype(int), clean[..., 2].astype(int)
    assert not ((b - r) > 50).any()                                   # no blue left
    assert (clean[30:50, 40:110] < 100).any()                         # the figure survives
    assert (clean[60, 125] == 255).all()                              # the crossing went too...
    rules = ocr_form6._drop_pen_ink(page, keep_rules=True)
    assert (rules[60, 125] == 0).all()                                # ...unless rules are kept


def test_a_page_with_no_colour_is_unchanged():
    page = _page_with_pen(blue=(40, 40, 40))
    assert ocr_form6._drop_pen_ink(page) is page


def _position_cell(numerals):
    import cv2
    cell = np.full((80, 160, 3), 255, dtype=np.uint8)
    cv2.rectangle(cell, (20, 15), (50, 20), (0, 0, 0), -1)            # the T's bar
    cv2.rectangle(cell, (33, 15), (37, 60), (0, 0, 0), -1)            # its stem
    for i in range(numerals):
        x = 60 + 14 * i
        cv2.rectangle(cell, (x, 15), (x + 4, 60), (0, 0, 0), -1)
        cv2.rectangle(cell, (x - 5, 15), (x + 9, 18), (0, 0, 0), -1)  # serifs that touch the next I
        cv2.rectangle(cell, (x - 5, 57), (x + 9, 60), (0, 0, 0), -1)
    return cell


@pytest.mark.parametrize("numerals", [1, 2, 3])
def test_a_positions_numeral_is_counted_by_its_strokes(numerals):
    assert ocr_form6._count_stems(_position_cell(numerals), (0, 0, 160, 80)) == numerals + 1


def test_the_column_split_on_the_first_page_is_remembered():
    before = [(0, 100), (100, 400), (400, 900)]
    after = [(0, 100), (100, 400), (400, 650), (650, 900)]
    assert ocr_form6._split_columns(before, after) == [2]
    assert ocr_form6._split_columns(before, before) == []


def test_a_name_with_no_comma_reads_like_the_rest_of_its_order():
    rows = [
        ocr_form6.Form6Row("1", "Abella", "Jeffry", "Q.", "", "", "", "VSC", "", kind="vsc", name_raw="ABELLA, JEFFRY Q."),
        ocr_form6.Form6Row("2", "Adaya", "Kristine", "L.", "", "", "", "VSC", "", kind="vsc", name_raw="ADAYA, KRISTINE L."),
        ocr_form6.Form6Row("3", "Omar", "Asnorah", "M.", "", "", "", "VSC", "", kind="vsc", name_raw="OMAR, ASNORAH M."),
        ocr_form6.Form6Row("4", "Philip", "Olarte Roy", "", "", "", "", "VSC", "", kind="vsc", name_raw="OLARTE ROY PHILIP"),
    ]
    notes = ocr_form6._surname_first_like_the_rest(rows)
    assert (rows[3].last_name, rows[3].first_name) == ("Olarte", "Roy Philip")
    assert len(notes) == 1 and "OLARTE ROY PHILIP" in notes[0]


def test_a_first_name_first_order_is_left_alone():
    rows = [ocr_form6.Form6Row(str(i), "Conde", "Kenneth", "", "", "", "", "VSC", "", kind="vsc",
                               name_raw="Kenneth Joh Conde") for i in range(4)]
    assert ocr_form6._surname_first_like_the_rest(rows) == []
    assert rows[0].last_name == "Conde"


@pytest.mark.parametrize("raw,expected", [
    ("OLARTE ROY PHILIP", ("Olarte", "Roy Philip", "")),
    ("DELA VEGA. ARA NICOLE BACONG", ("Dela Vega", "Ara Nicole Bacong", "")),   # SO 1892: a period for the comma
    ("PANAL REVERIANO JR. PUTI-AN", ("Panal", "Reveriano Jr. Puti-An", "")),   # "JR." is no comma
    ("DE LOS SANTOS JUAN A.", ("De Los Santos", "Juan", "A.")),
])
def test_a_comma_less_surname_first_name(raw, expected):
    assert ocr_form6._split_combined_name(ocr_form6._surname_comma(raw.split())) == expected


def test_table_header_text_is_never_the_event():
    """SO 1849's page 6 OCR'd its paragraph out of order: what followed
    "during" was the table's header, up to the first row's "June 1-5"."""
    text = ("granted vacation services rendered during the 2026 Brigada Eskwela No. of Inclusive No. of Vacation "
            "No. Name Position Dates Hours Service Served Credits Granted YANA, JUDY ANN C. T21. June 1-5, 2026 7 1.313")
    assert ocr_form6._event_from_text(text) == ""
