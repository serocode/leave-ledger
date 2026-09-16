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


def _fake_pages(page_outcomes):
    """Drive extract_form6_with_notes with canned per-page results."""
    from collections import Counter
    outcomes = iter(page_outcomes)

    def fake_extract(img, seq):
        rows, dropped = next(outcomes)
        return rows, seq, Counter({6: dropped}) if dropped else Counter()

    return (
        patch.object(ocr_form6, "_load_pages", lambda path: [None] * len(page_outcomes)),
        patch.object(ocr_form6, "_extract_page", fake_extract),
    )


def _some_row():
    return ocr_form6.Form6Row("1", "Cruz", "Ana", "", "", "", "", "SL", "WP")


def test_a_page_that_yields_nothing_is_reported():
    """Sacred Heart page 4: ten grants silently missing from an upload that
    otherwise read fine."""
    load, extract = _fake_pages([([_some_row()], 1), ([], 11)])
    with load, extract:
        rows, notes = ocr_form6.extract_form6_with_notes("x.pdf")
    assert len(rows) == 1
    assert len(notes) == 1 and notes[0].startswith("Page 2") and "11 rows" in notes[0]


def test_a_page_losing_many_rows_is_reported():
    load, extract = _fake_pages([([_some_row()] * 5, 4)])
    with load, extract:
        _, notes = ocr_form6.extract_form6_with_notes("x.jpg")
    assert len(notes) == 1 and "4 table rows couldn't be read" in notes[0]


def test_a_header_and_a_blank_line_are_not_worth_a_note():
    load, extract = _fake_pages([([_some_row()] * 20, 2)])
    with load, extract:
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
