"""
OCR extraction for DepEd Form 6 (Sick Leave) transmittal tables.

Turns a scanned/photographed Form 6 transmittal into a list of structured
draft rows — this is a DRAFT ONLY. It is meant to pre-fill the review table
in app.py so you correct mistakes before anything is submitted, not to be
trusted blindly. OCR on a phone photo of a table WILL make mistakes,
especially on short codes and Roman numerals.

Pipeline (tuned and verified against a real Macanhan Elementary School Form
6 transmittal on 2026-09-10 — see samples/form6_sample.jpg):

  1. Detect the table's four outer corners and perspective-correct
     ("dewarp") the photo so the grid is axis-aligned. Phone photos of a
     sheet of paper are essentially never perfectly parallel to the camera,
     so skipping this step makes row/column detection unreliable.
  2. Detect grid lines (morphological open on a binary threshold) and pull
     out individual cell bounding boxes.
  3. Group cells into rows (by y-position) and sort each row left-to-right;
     the row's column count picks the template to read it with (see the
     COLUMN_CONFIGS* tables); a header row and any stray/blank rows
     naturally fall out of this.
  4. OCR each cell individually rather than the whole table at once — much
     more reliable for a bordered form, and lets each column use tuned
     settings (see COLUMN_CONFIGS below; psm 8 in particular does badly on
     these short two-character codes, psm 6 does much better — found by
     testing directly against the real sample, not assumed).
  5. Normalize each column's raw OCR text (digit confusions, M.I. cleanup,
     a best-effort Roman-numeral fix for Position) and parse the
     "Inclusive Dates" column into real date_from/date_to values plus a
     ready-made description string, matching this project's established
     convention ("Sick leave <date(s)>").

Verified accuracy on the real sample (19 real employee rows): Last/First
name, dates, day count, Type of Leave, and Action Taken were all correct.
Position (T-I / T-III) was wrong on ~7/19 rows — left as best-effort since
Position isn't sent to HRIS at all, it's just there for you to visually
cross-check the right employee during review.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass, field, replace
from datetime import date, timedelta

import cv2
import numpy as np
import pymupdf
import pytesseract

# The Windows installer puts tesseract.exe in Program Files but does not add
# it to PATH unless the user ticks a box, so pytesseract's default lookup
# fails on an otherwise correct install. Point it at the standard locations
# when the binary isn't already resolvable.
if os.name == "nt" and not shutil.which("tesseract"):
    for _candidate in (
        os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "Tesseract-OCR", "tesseract.exe"),
        os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"), "Tesseract-OCR", "tesseract.exe"),
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Tesseract-OCR", "tesseract.exe"),
    ):
        if os.path.isfile(_candidate):
            pytesseract.pytesseract.tesseract_cmd = _candidate
            break

# Rasterize PDF pages at this resolution before running the same
# cell-detection pipeline used for a photo — too low and small print (M.I.,
# leave-type codes) becomes illegible; 300 DPI matches a decent phone photo.
_PDF_RENDER_DPI = 300

MONTH_NAMES = [
    "", "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


@dataclass
class Form6Row:
    row_no: str
    last_name: str
    first_name: str
    middle_initial: str
    position_raw: str
    dates_raw: str
    days_raw: str
    leave_type: str
    action_taken: str

    date_from: str | None = None  # "YYYY-MM-DD"
    date_to: str | None = None  # "YYYY-MM-DD"
    description: str = ""
    used: str = ""
    parse_warning: str | None = None
    in_scope: bool = False  # True when leave_type == "SL" and action_taken in {"WP", "WOP"}
    # "leave" deducts days (used / wo_pay); "vsc" is a vacation service credit
    # grant, which adds them (earned) — different column, opposite direction.
    kind: str = "leave"
    hours: float | None = None
    earned: str = ""
    # What a grant was earned for, as its paragraph names it ("Aral Summer
    # Program"). Empty when the page doesn't say.
    event: str = ""
    # Columns of this grant taken from the row above because the cell said
    # "-do-" (or was hidden under a pen mark in a column that does): the sign
    # that the column is a "-do-" column.
    inherited: tuple[str, ...] = ()
    # A grant flagged only because it grants more than the formula, every
    # reading having agreed: not a misread, so still "the row above" that a
    # "-do-" below it repeats.
    trusted: bool = False
    # The name cell as read, for re-splitting once the whole order is known.
    name_raw: str = ""

    @property
    def full_name_search(self) -> str:
        """Best string to feed into HrisClient.search_employees()."""
        return f"{self.last_name} {self.first_name}".strip()


# ---------------------------------------------------------------------------
# Step 1-3: locate and rectify the table, find cell grid
# ---------------------------------------------------------------------------

def _binary_threshold(gray: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(
        ~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2
    )


def _grid_mask(thr: np.ndarray) -> np.ndarray:
    h, w = thr.shape
    horiz_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 30, 1))
    horiz = cv2.dilate(cv2.erode(thr, horiz_structure), horiz_structure)
    vert_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 60))
    vert = cv2.dilate(cv2.erode(thr, vert_structure), vert_structure)
    mask = cv2.add(horiz, vert)
    return cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))


def _order_corners(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return tl, tr, br, bl


def _count_table_regions(img: np.ndarray) -> int:
    """How many separate tables of comparable size the page holds.

    _dewarp_table reads only the largest, which is right for a scan: the
    runner-up region on every sample so far is a stamp or a stray rule, under
    3% of the table's size. A photo of sheets fanned over one another
    (Lapasan's three-page Annex D) has one table per sheet at nearly equal
    sizes, and reading only one of them drops the others' teachers without
    a word."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(_grid_mask(_binary_threshold(gray)), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = sorted((cv2.contourArea(c) for c in contours), reverse=True)
    if not areas:
        return 0
    page = img.shape[0] * img.shape[1]
    return sum(1 for a in areas if a >= 0.4 * areas[0] and a >= 0.01 * page)


def _dewarp_table(img: np.ndarray, *also: np.ndarray):
    """Find the table's outer border and perspective-correct it flat. With
    `also` (images the same size), returns all of them, warped identically."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thr = _binary_threshold(gray)
    mask = _grid_mask(thr)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Could not find a table grid in this image.")
    big = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(big, True)
    approx = cv2.approxPolyDP(big, 0.02 * peri, True).reshape(-1, 2)
    if len(approx) != 4:
        # Fall back to the rotated bounding rect if the border isn't a clean quad.
        rect = cv2.minAreaRect(big)
        approx = cv2.boxPoints(rect)

    tl, tr, br, bl = _order_corners(approx.astype(np.float32))
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if width < 50 or height < 50:
        raise ValueError("Detected table region is too small — is this really a Form 6 scan?")

    src = np.array([tl, tr, br, bl], dtype=np.float32)
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, matrix, (width, height))
    if not also:
        return warped
    return (warped, *(cv2.warpPerspective(a, matrix, (width, height)) for a in also))


def _find_cell_boxes(img: np.ndarray) -> list[tuple[int, int, int, int]]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thr = _binary_threshold(gray)
    mask = _grid_mask(thr)
    inv = cv2.bitwise_not(mask)
    contours, _ = cv2.findContours(inv, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    h, w = mask.shape
    boxes = []
    for c in contours:
        x, y, ww, hh = cv2.boundingRect(c)
        area = ww * hh
        if area < 1200 or ww < 20 or hh < 15:
            continue
        # What needs dropping here is the whole-table contour and any
        # whole-row one, and those are recognisable by *spanning* the table
        # rather than by covering some fixed share of its area. An area cap
        # can't tell the two apart: on a short transmittal (a header and two
        # data rows) every cell is legitimately a tenth of the table, so a
        # cap tuned against a 20-row form silently dropped that form's three
        # widest columns and left a 6-column table looking like a 3-column one.
        if ww > 0.75 * w or hh > 0.75 * h:
            continue
        boxes.append((x, y, ww, hh))
    return _split_merged_rows(_drop_nested_boxes(boxes))


def _split_merged_rows(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    """Recover cells whose separating rule didn't survive thresholding.

    A box measuring a clean multiple of the usual row height is ambiguous: it
    is either several rows welded together by a rule that didn't survive
    thresholding (seen at a scanned page's top edge, where the rule is faint
    or cropped), or one genuinely tall row whose text wraps. Left unsplit the
    first case costs a row two columns, so it matches no template and those
    employees vanish; wrongly splitting the second truncates a cell's text.

    The two are told apart by looking beside the box: a welded box sits
    alongside normal-height cells that still form n separate rows, whereas a
    tall row's neighbours are tall too. Only the former is split."""
    heights = sorted(h for _, _, _, h in boxes)
    if not heights:
        return boxes
    median = heights[len(heights) // 2]
    if median <= 0:
        return boxes

    normal_centers = [y + h / 2 for _, y, _, h in boxes if round(h / median) == 1]

    out = []
    for x, y, w, h in boxes:
        n = round(h / median)
        if n >= 2 and abs(h / median - n) <= 0.15:
            inside = sorted(c for c in normal_centers if y <= c <= y + h)
            rows_beside = 0
            last = None
            for c in inside:
                if last is None or c - last > median / 2:
                    rows_beside += 1
                last = c
            if rows_beside == n:
                step = h / n
                out.extend((x, int(y + i * step), w, int(step)) for i in range(n))
                continue
        out.append((x, y, w, h))
    return out


def _drop_nested_boxes(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    """Real table cells never nest inside one another. A stray mark or bit of
    handwriting inside a cell (seen on the Lumbia sample, inside the Employee
    Name column) can get picked up as its own small contour — drop any box
    that sits (almost) entirely inside a larger one rather than let it throw
    off column counting for that row."""
    def _contained(inner: tuple[int, int, int, int], outer: tuple[int, int, int, int]) -> bool:
        ix, iy, iw, ih = inner
        ox, oy, ow, oh = outer
        return ix >= ox - 2 and iy >= oy - 2 and ix + iw <= ox + ow + 2 and iy + ih <= oy + oh + 2

    # A box holding two or more others isn't a cell at all — it's a region of
    # the page the grid-line mask failed to split (seen on a scan where one
    # contour covered the left four columns of a whole table). Left in, it
    # would count as the "larger box" below, and every genuine cell inside it
    # would be dropped as a stray mark — ten rows gone without a trace.
    regions = {
        i for i, b in enumerate(boxes)
        if sum(1 for j, other in enumerate(boxes) if i != j and other != b and _contained(other, b)) >= 2
    }
    boxes = [b for i, b in enumerate(boxes) if i not in regions]

    kept = []
    for i, b in enumerate(boxes):
        if any(i != j and b != other and _contained(b, other) for j, other in enumerate(boxes)):
            continue
        kept.append(b)
    return kept


def _group_into_rows(boxes: list[tuple[int, int, int, int]], y_tol: int = 15) -> list[list[tuple[int, int, int, int]]]:
    boxes = sorted(boxes, key=lambda b: b[1] + b[3] / 2)
    rows: list[list[tuple[int, int, int, int]]] = []
    current: list[tuple[int, int, int, int]] = []
    last_yc = None
    for b in boxes:
        yc = b[1] + b[3] / 2
        if last_yc is not None and (yc - last_yc) > y_tol:
            rows.append(current)
            current = []
        current.append(b)
        last_yc = yc
    if current:
        rows.append(current)
    return rows


# ---------------------------------------------------------------------------
# Step 4: per-cell OCR
# ---------------------------------------------------------------------------

_DATE_WHITELIST = "0123456789/-,&"
# Some transmittals spell Leave Type / Action out in full words ("Sick
# leave", "w/ pay") instead of 2-letter codes ("SL", "WP") — allow both
# cases plus punctuation here and fold it down to a canonical code in
# _normalize_leave_type / _normalize_action afterward.
_WORD_OR_CODE_WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz/- "
_MI_WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ."
# For the "Month D,D,Dpm,..." day-list date format (no numeric month/slashes).
# The "-" is what lets a day range survive OCR: without it tesseract renders
# "June 25-26,2026" as "June 2526,2026", which no longer reads as a range.
_MONTHDAY_WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789,- "

# "No. of Day/s" is a small-print numeric cell; restricting it to digits and a
# decimal point keeps stray letter-shaped noise out. An unreadable cell comes
# back empty, which means "no printed total to cross-check against" rather
# than a wrong total.
_DAYS_WHITELIST = "0123456789."
# For a "<date-list>, YYYY (N day(s))" cell that bundles dates, half-day
# markers, and the day count all in one (Iponan Elementary School template).
_COMBINED_DATES_WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),./ "
# For a "Date/s of Absence" cell that can hold several months, day ranges,
# "&"/";" separators and parenthesized half-day markers all at once
# ("July 1-3 & 6-7, 2026", "June 30, 2026; July 2(pm), 2026"). Every one of
# those separators has to survive OCR or the cell reads as the wrong days.
_ABSENCE_DATES_WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),;&- "

# (tesseract --psm, whitelist or None) per column, in left-to-right order.
# Found by testing directly against a real sample, not assumed defaults —
# psm 8 ("single word") in particular reliably drops trailing characters on
# these short two-letter codes; psm 6/7 do not.
COLUMN_CONFIGS = [
    (7, None),                      # 0: No.
    (6, None),                      # 1: Last name (psm 6: some cells wrap to 2 lines)
    (6, None),                      # 2: First name (ditto)
    (6, _MI_WHITELIST),             # 3: M.I.
    (7, None),                      # 4: Position (normalized after, best-effort)
    (6, _DATE_WHITELIST),           # 5: Inclusive dates (psm 6: multi-range cells wrap)
    (7, None),                      # 6: No. of days
    (6, _WORD_OR_CODE_WHITELIST),   # 7: Type of leave
    (6, _WORD_OR_CODE_WHITELIST),   # 8: Action taken
]

# A second, differently-shaped transmittal template (seen from Lumbia
# Central School): one combined "Last, First M.I." name column, no Position
# column, and dates given as "<Month> <day>[am|pm],<day>,..." instead of
# MM/DD/YYYY. Detected purely by row column-count (5 vs COLUMN_CONFIGS' 9).
COLUMN_CONFIGS_5COL = [
    (6, None),                      # 0: Employee Name ("Last, First M.I.")
    (6, _MONTHDAY_WHITELIST),       # 1: Leave Inclusive dates
    (7, None),                      # 2: No. of Days
    (6, _WORD_OR_CODE_WHITELIST),   # 3: Leave Type
    (6, _WORD_OR_CODE_WHITELIST),   # 4: Action
]

# A third template (seen from Iponan Elementary School): just 3 columns —
# combined "<No>. Last, First M.I." name (the row number lives inside this
# cell, not its own column), a "No. of Days" cell that bundles the whole
# date list + half-day (AM)/(PM) markers + a trailing "(N day(s))" count,
# and a combined "SL-WP" leave-type/action code under "Remarks".
COLUMN_CONFIGS_3COL = [
    (6, None),                          # 0: "<No>. Last, First M.I."
    (6, _COMBINED_DATES_WHITELIST),     # 1: No. of Days (dates + markers + count, combined)
    (6, _WORD_OR_CODE_WHITELIST),       # 2: Remarks ("SL-WP")
]

# A fifth template (seen from Fr. William F. Masterson S.J. ES): a
# "Summary of Absences" transmittal — No./Name/Position/Date/s of Absence/
# Days/Remarks. It has no Leave Type column at all (see _extract_6col_row),
# and its date cell is the most free-form of the lot: several months in one
# cell, day ranges, "&" and ";" separators, and parenthesized "(am)"/"(pm)"
# half-day markers ("June 30, 2026; July 2(pm), 2026").
COLUMN_CONFIGS_6COL = [
    (7, None),                          # 0: No.
    (6, None),                          # 1: Teachers' Name ("Last, First M.I.")
    (7, None),                          # 2: Position
    (6, _ABSENCE_DATES_WHITELIST),      # 3: Date/s of Absence
    (7, _DAYS_WHITELIST),               # 4: Days
    (6, _WORD_OR_CODE_WHITELIST),       # 5: Remarks ("w/ pay" / "w/o pay")
]

# A fourth template (seen from Lumbia National High School): its own Seq No.
# column, a combined "Last, First M.I." name, Position, and "<Month> D[-D][am|pm],
# ...,YYYY" dates — i.e. the 5-column layout plus Seq No. and Position, with
# leave type/action spelled out ("SICK LEAVE" / "WITH PAY") rather than coded.
COLUMN_CONFIGS_7COL = [
    (7, None),                      # 0: Seq. No.
    (6, None),                      # 1: Name ("Last, First M.I.")
    (6, _MONTHDAY_WHITELIST),       # 2: Inclusive Date
    (7, None),                      # 3: Position
    (7, _DAYS_WHITELIST),           # 4: No. of Day/s
    (6, _WORD_OR_CODE_WHITELIST),   # 5: Type of Leave
    (6, _WORD_OR_CODE_WHITELIST),   # 6: Remarks (action taken)
]


# ---------------------------------------------------------------------------
# Header-driven column mapping
#
# Matching a layout by how many columns it has does not survive contact with
# real schools: Man-Ai and Masterson are both 6-column forms whose columns are
# in a different order and mean different things, so a count alone read one
# form's "2 days" as the other's Position. Every one of these transmittals
# labels its columns, though — so read the header row and let it say which
# column is which. A school that invents a new column order, or drops a
# column, then needs no new template at all.
#
# Header cells are matched to data cells by horizontal overlap rather than by
# index, because a header can span several data columns ("Employee Name" over
# Last / First / M.I.) — that span is exactly how those three are recognised.
# ---------------------------------------------------------------------------

# (field, patterns) in priority order — the first field whose pattern appears
# in a cleaned header wins, so "NO. OF DAYS" lands on days rather than on the
# "NO." row-number column.
_HEADER_FIELDS: list[tuple[str, tuple[str, ...]]] = [
    ("last", ("LASTNAME", "SURNAME")),
    ("first", ("FIRSTNAME", "GIVENNAME")),
    # A service-credit grant's columns — ahead of everything below, because
    # "No. of hours Served" would otherwise land on the row-number column and
    # "No. of Vacation Service Credits Granted" on Action (VAC-ACTION).
    ("hours", ("HOUR", "HRS")),
    ("credits", ("CREDIT", "GRANTED", "SERVICE", "EARNED")),
    ("days", ("DAY",)),
    ("dates", ("DATE", "INCLUSIVE")),
    ("type", ("TYPEOFLEAVE", "LEAVETYPE", "TYPE", "CAUSE", "KIND")),
    ("action", ("ACTION", "REMARK", "STATUS")),
    ("position", ("POSITION", "DESIGNATION")),
    ("name", ("NAME", "PERSONNEL", "TEACHER", "EMPLOYEE")),
    ("no", ("NO", "SEQ")),
]


def _header_field(raw: str) -> str | None:
    """Fold one header cell's text down to a canonical field name."""
    cleaned = re.sub(r"[^A-Za-z]", "", raw).upper()
    if not cleaned:
        return None
    if cleaned in ("MI", "MIDDLEINITIAL") or "MIDDLE" in cleaned:
        return "mi"
    for field, patterns in _HEADER_FIELDS:
        if any(p in cleaned for p in patterns):
            return field
    return None


def _header_fields_all(raw: str) -> set[str]:
    """Every field a header's text mentions, not just the first by priority —
    so a header cell that is really two cells can be told apart."""
    cleaned = re.sub(r"[^A-Za-z]", "", raw).upper()
    found = {field for field, patterns in _HEADER_FIELDS if any(p in cleaned for p in patterns)}
    # "No." and the "action" inside "vacation" ride along on almost anything.
    return found - {"no", "action"}


def _split_header_box(box: tuple[int, int, int, int], grid: list[tuple[int, int]] | None) -> list[tuple[int, int, int, int]]:
    """The grid columns one header box spans, as boxes of their own."""
    if not grid:
        return [box]
    x, y, w, h = box
    inside = [(left, right) for left, right in grid if min(x + w, right) - max(x, left) >= 0.5 * (right - left)]
    if len(inside) < 2:
        return [box]
    return [(left, y, right - left, h) for left, right in inside]


def _map_header_row(
    img: np.ndarray, header_boxes: list[tuple[int, int, int, int]], grid: list[tuple[int, int]] | None = None,
) -> tuple[dict[int, str], list[tuple[int, int, int, int]]] | None:
    """Read a candidate header row. Returns ({box x: field}, the header boxes
    to match data cells against), or None when the row isn't a header.

    A header cell whose vertical border didn't detect arrives merged with its
    neighbour — Kauswagan's "No. of Hours Served" and "No. of Vacation Service
    Credits Granted" came through as one cell, mapped to hours, and the credits
    column under it was never read. A cell that names more than one field and
    spans more than one grid column is read column by column instead. One that
    names a single field keeps its span: "Employee Name" over Last / First /
    M.I. is exactly how those three are recognised."""
    fields: dict[int, str] = {}
    boxes: list[tuple[int, int, int, int]] = []
    for box in header_boxes:
        text = _ocr_cell(img, box, 6, None)
        parts = _split_header_box(box, grid) if len(_header_fields_all(text)) >= 2 else [box]
        if len(parts) == 1:
            parts_text = [text]
        else:
            parts_text = [_ocr_cell(img, part, 6, None) for part in parts]
        for part, part_text in zip(parts, parts_text):
            boxes.append(part)
            field = _header_field(part_text)
            if field:
                fields[part[0]] = field
    # A real header names at least a person and a date; anything less is a
    # data row that happened to contain a word, and mapping against it would
    # be worse than falling back to the fixed templates.
    named = set(fields.values())
    if not named & {"name", "last"} or not named & {"dates", "days"}:
        return None
    return fields, boxes


def _fields_for_row(
    header: dict[int, str], header_boxes: list[tuple[int, int, int, int]],
    row_boxes: list[tuple[int, int, int, int]],
) -> list[str | None]:
    """Label each cell of a data row with the field of the header cell it sits
    under — by overlap, so one wide header covers every column beneath it."""
    labels: list[str | None] = []
    for x, _, w, _ in row_boxes:
        best, best_overlap = None, 0
        for hx, _, hw, _ in header_boxes:
            overlap = min(x + w, hx + hw) - max(x, hx)
            if overlap > best_overlap:
                best, best_overlap = header.get(hx), overlap
        labels.append(best)
    return labels


def _ocr_cell(img: np.ndarray, box: tuple[int, int, int, int], psm: int, whitelist: str | None, pad: int = 4) -> str:
    x, y, w, h = box
    crop = img[y + pad : y + h - pad, x + pad : x + w - pad]
    if crop.size == 0:
        return ""
    crop = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, binarized = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binarized = cv2.copyMakeBorder(binarized, 14, 14, 14, 14, cv2.BORDER_CONSTANT, value=255)
    # Collapse a wrapped 2-line cell ("PHOEBE\nCHARMENNE") to one line — every
    # caller wants flat text, and the numeric/date parsers strip whitespace
    # anyway, so this is safe everywhere.
    return " ".join(_run_tesseract(binarized, psm, whitelist).split())


def _ocr_cell_plain(img: np.ndarray, box: tuple[int, int, int, int], psm: int, whitelist: str | None, pad: int = 4) -> str:
    """_ocr_cell without its binarization: a 2x greyscale upscale read as-is.
    Deliberately different preprocessing, used as an independent second
    opinion on cells where one read can be plausibly wrong (see
    _consensus_numeric_dates) — two reads that share every step also share
    their mistakes."""
    x, y, w, h = box
    crop = img[y + pad : y + h - pad, x + pad : x + w - pad]
    if crop.size == 0:
        return ""
    crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    return " ".join(_run_tesseract(crop, psm, whitelist).split())


def _run_tesseract(img: np.ndarray, psm: int, whitelist: str | None) -> str:
    """Run tesseract on one prepared cell image and return its text.

    This calls the binary directly instead of going through
    pytesseract.image_to_string because that splits its `config` string with
    `shlex.split(config, posix=not_windows)`. Several of our whitelists
    contain a space (they have to: "SICK LEAVE", "June 25-26,2026"), and on
    Windows the non-POSIX mode cannot parse a quoted value containing one —
    it raises "No closing quotation", so every such cell failed and the whole
    page came back empty. Passing argv as a list has no quoting step at all,
    so the whitelist arrives intact on every platform.
    """
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("Could not encode a table cell for OCR.")

    # Via a temp file rather than stdin: that is the path pytesseract already
    # used successfully here, so quoting is the only thing being changed. A
    # temp path containing spaces ("C:\\Users\\Juan Dela Cruz\\...") is fine
    # precisely because argv is a list.
    fd, path = tempfile.mkstemp(suffix=".png")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(buf.tobytes())

        cmd = [pytesseract.pytesseract.tesseract_cmd, path, "stdout", "--psm", str(psm)]
        if whitelist:
            cmd += ["-c", f"tessedit_char_whitelist={whitelist}"]

        # Keep a console window from flashing on Windows, once per cell.
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        try:
            proc = subprocess.run(cmd, capture_output=True, creationflags=creationflags)
        except FileNotFoundError:
            raise ValueError(
                "Tesseract OCR is not installed, or not where this tool can find it. "
                "Everything except reading Form 6 scans works without it."
            ) from None
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass

    if proc.returncode != 0:
        raise ValueError(f"Tesseract failed: {proc.stderr.decode('utf-8', 'replace')[:200]}")
    return proc.stdout.decode("utf-8", "replace")


# ---------------------------------------------------------------------------
# Step 5: normalization
# ---------------------------------------------------------------------------

# Tesseract reliably reads this specific serif "1" glyph as one of these
# letter-like characters instead of a digit — found by inspecting the actual
# misread cell image, not guessed. Any of these, alone, means "1".
_ONE_LOOKALIKES = {"l", "I", "|", "!", "i", "L", "]", "[", "j"}


def _normalize_digits(raw: str) -> str:
    raw = raw.strip()
    if not raw:
        return raw
    # Replace each lookalike character in place (not just a whole-string
    # match) — a two-digit "11" can OCR as one real "1" plus one lookalike,
    # e.g. "1l", and both characters need to become "1".
    lookalike_pattern = "[" + re.escape("".join(_ONE_LOOKALIKES)) + "]"
    substituted = re.sub(lookalike_pattern, "1", raw)
    digits = re.findall(r"\d+", substituted)
    return "".join(digits) if digits else raw


def _normalize_mi(raw: str) -> str:
    letters = re.findall(r"[A-Za-z]", raw)
    return f"{letters[0].upper()}." if letters else raw.strip()


def _normalize_position(raw: str) -> str:
    """Best-effort only — Position is not sent to HRIS, just shown for the
    user's own cross-check, so this doesn't need to be perfect."""
    cleaned = raw.strip().upper().replace(" ", "").rstrip(".,-_]['\"")
    # "Teacher I" .. "Teacher V", as grant forms spell the position out.
    spelled = re.fullmatch(r"(MASTER)?TEACHER([IVX1L|!]+)", cleaned)
    title = "Master Teacher" if cleaned.startswith("MASTER") else "Teacher"
    if spelled:
        roman = re.sub(r"[1L|!]", "I", spelled.group(2))
        return f"{title} {roman}"
    if re.match(r"(MASTER)?TEACHER", cleaned):
        # The numeral under a wrapped "TEACHER" reads as anything ("tt",
        # "UHI"); a bare "Teacher" beats passing that noise on.
        return title
    # Adjacent "I"s fuse into an "H" often enough to be worth folding in
    # ("T-HI" is always "T-III", never a real position).
    cleaned = re.sub(r"^T([-_]?)H(?=[I1L|!])", r"T\g<1>II", cleaned)
    m = re.match(r"^T[-_]?([I1L|!]+)$", cleaned)
    if m:
        n = min(len(m.group(1)), 4)
        return "T-IV" if n == 4 else "T-" + "I" * n
    return cleaned


def _normalize_leave_type(raw: str) -> str:
    """Fold either a 2-3 letter code ("SL") or a spelled-out word ("Sick
    Leave") down to the same canonical code."""
    cleaned = re.sub(r"[^A-Za-z]", "", raw).upper()
    if cleaned in ("SL", "PL", "VL", "ML", "SPL"):
        return cleaned
    if "SICK" in cleaned:
        return "SL"
    if "PATERNITY" in cleaned:
        return "PL"
    if "MATERNITY" in cleaned:
        return "ML"
    if "VACATION" in cleaned:
        return "VL"
    if "SOLO" in cleaned:
        return "SPL"
    return cleaned


_LEAVE_TYPE_LABELS = {
    "SL": "Sick leave", "PL": "Paternity leave", "ML": "Maternity leave",
    "VL": "Vacation leave", "SPL": "Solo parent leave",
}


def _normalize_action(raw: str) -> str:
    """Fold either a code ("WP"/"WOP") or spelled-out text ("w/ pay",
    "w/o pay") down to the same canonical code."""
    cleaned = re.sub(r"[^A-Za-z]", "", raw).upper()
    if cleaned in ("WP", "WOP", "D"):
        return cleaned
    if not cleaned:
        return cleaned
    if "WITHOUT" in cleaned or cleaned.startswith("WO"):
        return "WOP"
    if cleaned.startswith("W"):
        return "WP"
    return cleaned


# A crop's top/right edge occasionally catches a sliver of the next cell's
# grid border, which the digit-only whitelist reads as 1-2 stray trailing
# digits (e.g. "06/30/2026 7"). Ignore up to this many leftover characters
# after a full date match rather than failing the whole parse over noise.
_MAX_TRAILING_NOISE = 2

_YEAR_IN_GROUP_RE = re.compile(r"/(\d{4})")


def _parse_day_token(token: str, month: int, year: int) -> tuple[date, date]:
    """One comma-separated piece of a numeric date cell: "D", "D-D" (same
    month range), or "D-M/D" (cross-month range)."""
    if "-" in token:
        left, right = token.split("-", 1)
        d1 = int(left)
        if "/" in right:
            m2_str, d2_str = right.split("/", 1)
            m2, d2 = int(m2_str), int(d2_str)
        else:
            m2, d2 = month, int(right)
    else:
        d1 = d2 = int(token)
        m2 = month
    return date(year, month, d1), date(year, m2, d2)


def _parse_date_segments_numeric(raw: str) -> tuple[list[tuple[date, date]], str | None]:
    """Parse a "MM/DD/YYYY"-style Inclusive Dates cell into one or more
    contiguous (date_from, date_to) ranges.

    Handles: "MM/DD/YYYY" (single day), "MM/DD-DD/YYYY" (same-month range),
    "MM/DD-MM/DD/YYYY" (cross-month range), "MM/DD-DD,DD-DD/YYYY" (several
    ranges in one month, comma-separated), and "MM/DD/YYYY & MM/DD/YYYY"
    (fully independent dates/ranges, ampersand-separated — each gets its own
    month/year). Returns ([], warning) if the text couldn't be parsed — the
    review UI should flag that row for manual entry rather than guess.
    """
    # "5/6 to 6/2/2026" and "5/6 – 6/2/2026" are ranges; stripping the word or
    # the en dash as noise glued them into "5/66/2/2026".
    dashed = re.sub(r"\s*(?:(?<=\d)\s*to\s*(?=\d)|[–—])\s*", "-", raw.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"[^0-9/\-,&]", "", dashed).strip("-,")
    fail = ([], f"Could not parse dates from {raw!r} — enter manually.")
    if not cleaned:
        return fail

    segments: list[tuple[date, date]] = []
    for group in (g for g in cleaned.split("&") if g):
        year_m = _YEAR_IN_GROUP_RE.search(group)
        if not year_m or len(group) - year_m.end() > _MAX_TRAILING_NOISE:
            return fail
        year = int(year_m.group(1))
        if not 2000 <= year <= 2099:
            # "2026" read as "9096" still parses as a valid date — in the year
            # 9096. A year that far out is a misread, never a real leave.
            return fail
        chunks = [c for c in group[: year_m.start()].split(",") if c]
        if not chunks or "/" not in chunks[0]:
            return fail
        try:
            m1_str, rest1 = chunks[0].split("/", 1)
            month = int(m1_str)
            segments.append(_parse_day_token(rest1, month, year))
            for chunk in chunks[1:]:
                if "/" in chunk:
                    m_str, token = chunk.split("/", 1)
                    # Days after a month change are that month's: SO 2159's
                    # "...28,29, 6/1,2" is June 1 and 2, not June 1 and May 2.
                    month = int(m_str)
                    segments.append(_parse_day_token(token, month, year))
                else:
                    segments.append(_parse_day_token(chunk, month, year))
        except (ValueError, IndexError):
            return fail

    return (segments, None) if segments else fail


_MONTH_NAME_TO_NUM = {name.lower(): i for i, name in enumerate(MONTH_NAMES) if name}


_MONTHDAY_TOKEN_RE = re.compile(
    r"(?P<year>20\d{2})"
    r"|(?P<word>[A-Za-z]+)"
    r"|(?P<d1>\d{1,2})\s*(?:\(?\s*(?P<h1>am|pm)\s*\)?)?"
    r"(?:\s*-\s*(?P<d2>\d{1,2})\s*(?:\(?\s*(?P<h2>am|pm)\s*\)?)?)?",
    re.IGNORECASE,
)


def _disambiguate_ampersands(raw: str) -> str:
    """Tell a separator "&" from a digit 8 misread as one.

    The two glyphs are close enough that Tesseract swaps them freely, and a
    date cell needs both: "July 1-3 & 6-7" joins two ranges, while "June
    8pm,9,10" starts with an eight. Printed separators are spaced out on
    every one of these forms, so an "&" with whitespace on both sides is a
    separator and any other one is the digit it was misread from — without
    this, one form loses a day off the front of a range and the other gains
    a day that was never there.
    """
    return re.sub(
        r"&",
        lambda m: "&" if (
            m.start() > 0 and raw[m.start() - 1].isspace()
            and m.end() < len(raw) and raw[m.end()].isspace()
        ) else "8",
        raw,
    )


# "28,2 9," — a day, then two single digits a space apart before a comma.
_WRAPPED_DAY_RE = re.compile(r"\b(\d{1,2}),\s*(\d) (\d)(?=\s*,)")


def _join_wrapped_days(raw: str) -> str:
    """Rejoin a day number the cell's line wrap broke in two. SO 1905 lists
    seventeen days in a narrow cell, and "...26,28,2" / "9, 2026" reads as
    "28,2 9," — days 2 and 9, which made the period May 2-28 instead of May
    6-29. Listed days ascend, so the halves are joined only when the parts
    break the order and the whole continues it: "1,2 9," stays 2 and 9."""
    def join(m: re.Match) -> str:
        before, first, whole = int(m.group(1)), int(m.group(2)), int(m.group(2) + m.group(3))
        if first < before < whole <= 31:
            return f"{m.group(1)},{whole}"
        return m.group(0)
    return _WRAPPED_DAY_RE.sub(join, raw)


def _parse_date_segments_monthname(
    raw: str, default_year: int | None
) -> tuple[list[tuple[date, date, float]], str | None]:
    """Parse any month-name date cell. Every school writes these a little
    differently and one parser covers all of it: "June 26, 2026",
    "June 8pm,9,10,11,22,23", "July 1-3 & 6-7, 2026", "June 29(pm), 2026",
    "June 30, 2026; July 2(pm), 2026".

    The month is tracked as the cell is scanned, so one cell may span more
    than one; "D-D" ranges are expanded rather than read as their two
    endpoints; and a half-day marker counts whether or not it is bracketed
    ("19pm" and "19(pm)" alike). The year is usually printed *after* the days
    it belongs to, so days are held pending until a year token arrives,
    falling back to the page header's year at the end.

    Consecutive calendar days become one contiguous (date_from, date_to,
    used) run each, an am/pm day counting as half.
    """
    fail = ([], f"Could not parse dates from {raw!r} — enter manually.")
    raw = _join_wrapped_days(_disambiguate_ampersands(raw))
    month: int | None = None
    pending: list[tuple[int, int, bool]] = []  # (month, day, is_half), year not yet known
    dated: list[tuple[date, bool]] = []

    def _flush(year: int) -> bool:
        for mo, day, is_half in pending:
            try:
                dated.append((date(year, mo, day), is_half))
            except ValueError:
                return False
        pending.clear()
        return True

    for m in _MONTHDAY_TOKEN_RE.finditer(raw):
        if m.group("year"):
            if not _flush(int(m.group("year"))):
                return [], f"Invalid date parsed from {raw!r} — enter manually."
        elif m.group("word"):
            # Anything that isn't a month name is OCR noise (a stray letter,
            # a descender bleeding in from the row above) — ignore it.
            month = _MONTH_NAME_TO_NUM.get(m.group("word").lower(), month)
        elif m.group("d1"):
            if month is None:
                return fail
            d1, d2 = int(m.group("d1")), int(m.group("d2") or m.group("d1"))
            if d2 < d1:
                return fail
            for day in range(d1, d2 + 1):
                # A half-day marker only ever attaches to an endpoint.
                is_half = bool(m.group("h1") if day == d1 else None) or (
                    day == d2 and bool(m.group("h2"))
                )
                pending.append((month, day, is_half))

    if pending:
        if default_year is None:
            return [], f"Could not determine the year for {raw!r} — enter manually."
        if not _flush(default_year):
            return [], f"Invalid date parsed from {raw!r} — enter manually."
    if not dated:
        return fail

    runs: list[tuple[date, date, float]] = []
    start = end = dated[0][0]
    used = 0.5 if dated[0][1] else 1.0
    for day, is_half in dated[1:]:
        if day == end + timedelta(days=1):
            end = day
            used += 0.5 if is_half else 1.0
        else:
            runs.append((start, end, used))
            start = end = day
            used = 0.5 if is_half else 1.0
    runs.append((start, end, used))
    return runs, None


_HALF_WORD_RE = re.compile(r"half", re.IGNORECASE)
_HALF_SYMBOL_RE = re.compile(r"[½]|\b1\s*/\s*2\b")


def _normalize_total_days(raw: str) -> float | None:
    """Parse a "No. of Days" cell into a float: "1 day" -> 1.0, "5 ½ days"
    -> 5.5, "½ day" -> 0.5, "Half" -> 0.5, "6" -> 6.0, "0.5 DAY" -> 0.5."""
    # The unit word has to go first: every rule below anchors on the number
    # ending the cell, so "0.5 DAY" used to miss the decimal rule entirely and
    # come back as its first integer — zero days for a half-day absence.
    s = re.sub(r"\s*days?\.?\s*$", "", raw.strip(), flags=re.IGNORECASE).strip()
    if not s:
        return None
    # A count printed with a leading zero is a half day whose decimal point
    # didn't survive OCR ("0.5" -> "05"); no form writes five days as "05".
    if re.fullmatch(r"0\s*[59]", s):
        return 0.5

    # Decimal form ("0.5", "3.5", "1.5"). A half-day is the only fraction
    # these forms ever use, so the decimal point itself is the half marker —
    # the digit after it is read as .5 regardless of what tesseract made of
    # it. That matters because this column's print is small and the trailing
    # "5" frequently comes back as "9" (and a leading "0" as "Q"/"a"), which
    # would otherwise turn 0.5 into a bogus 5 or 0 and fire a false
    # "doesn't match the form's printed total" warning on a correct row.
    decimal = re.match(r"^(.*?)\.\s*\d?$", s)
    if decimal:
        whole = _normalize_digits(decimal.group(1))
        return (int(whole) if whole.isdigit() else 0) + 0.5

    has_half = bool(_HALF_SYMBOL_RE.search(s)) or bool(_HALF_WORD_RE.search(s))
    digits = re.findall(r"\d+", s)
    if not digits and not has_half:
        # No plain digit and no half-word/symbol — this is likely a purely
        # numeric cell where tesseract misread "1" as a look-alike glyph
        # (e.g. "|"). Only try that fallback here: applying it whenever a
        # half-word like "Half" is present would wrongly turn its "l" into
        # "1" (see _ONE_LOOKALIKES).
        lookalike_pattern = "[" + re.escape("".join(_ONE_LOOKALIKES)) + "]"
        digits = re.findall(r"\d+", re.sub(lookalike_pattern, "1", s))
    if not digits:
        return 0.5 if has_half else None
    return int(digits[0]) + (0.5 if has_half else 0.0)


def _format_used(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


# The closing ")" is sometimes misread as a stray trailing letter (e.g.
# "(1 dayj" for "(1 day)") — tolerate a dropped/garbled close-paren rather
# than fail the whole cell over it.
_TRAILING_COUNT_RE = re.compile(r"\(\s*([\d.\/]+)\s*days?\)?[a-zA-Z]{0,2}\s*$", re.IGNORECASE)
# One comma-separated piece of a combined date-list cell: a day number
# (with an optional leading month name and/or trailing (AM)/(PM) half-day
# marker, e.g. "June 30", "30", "3(PM)") — or a bare month name on its own
# ("June" in "June, 29, 2026"), which just sets the month for later chunks.
_COMBINED_CHUNK_RE = re.compile(r"^(?:([A-Za-z]+)\s+)?(\d{1,2})\s*(?:\((AM|PM)\))?$", re.IGNORECASE)
_MONTH_ONLY_CHUNK_RE = re.compile(r"^([A-Za-z]+)$")


def _parse_fraction(text: str) -> float | None:
    text = text.strip()
    if "/" in text:
        num, _, den = text.partition("/")
        try:
            return float(num) / float(den)
        except (ValueError, ZeroDivisionError):
            return None
    try:
        return float(text)
    except ValueError:
        return None


def _parse_date_segments_combined(raw: str) -> tuple[list[tuple[date, date, float]], float | None, str | None]:
    """Parse a "<date-list>, <year> (<count> day(s))" cell (seen on the
    Iponan Elementary School template) where dates, half-day (AM)/(PM)
    markers, and the total day count are all bundled into one "No. of
    Days" cell instead of separate date/day columns.

    A date list may switch months mid-cell without repeating the year, e.g.
    "June 29, 30, July 1, 2, 3, 2026 (5 days)" — bare day numbers inherit
    whichever month most recently appeared. Consecutive calendar days
    (across a month boundary too, e.g. June 30 -> July 1) are grouped into
    one contiguous run each, same as the Lumbia day-list format.

    Returns (runs, embedded_total, warning) — embedded_total is the
    cell's own trailing "(N day(s))" figure, kept separate so the caller
    can use it only as a cross-check (same reasoning as elsewhere: the
    per-run count computed from actual date arithmetic is more trustworthy
    than free-text OCR of a number).
    """
    fail_warning = f"Could not parse dates from {raw!r} — enter manually."
    # A stray "{"/"}"/"["/"]" occasionally bleeds into the "(AM)"/"(PM)"
    # parenthetical (grid-line or crop-edge noise) — drop those brackets.
    text = re.sub(r"[{}\[\]]", "", raw.strip())

    embedded_total = None
    count_m = _TRAILING_COUNT_RE.search(text)
    if count_m:
        embedded_total = _parse_fraction(count_m.group(1))
        text = text[: count_m.start()]

    year_m = re.search(r"\b(20\d{2})\b", text)
    if not year_m:
        return [], embedded_total, fail_warning
    year = int(year_m.group(1))
    text = text[: year_m.start()] + text[year_m.end() :]

    chunks = [c.strip() for c in text.split(",")]
    chunks = [c for c in chunks if c]
    if not chunks:
        return [], embedded_total, fail_warning

    entries: list[tuple[date, bool]] = []
    current_month: int | None = None
    for chunk in chunks:
        month_only = _MONTH_ONLY_CHUNK_RE.match(chunk)
        if month_only:
            month_key = month_only.group(1).lower()
            if month_key not in _MONTH_NAME_TO_NUM:
                return [], embedded_total, fail_warning
            current_month = _MONTH_NAME_TO_NUM[month_key]
            continue

        m = _COMBINED_CHUNK_RE.match(chunk)
        if not m:
            return [], embedded_total, fail_warning
        month_name, day_str, marker = m.groups()
        if month_name:
            month_key = month_name.lower()
            if month_key not in _MONTH_NAME_TO_NUM:
                return [], embedded_total, fail_warning
            current_month = _MONTH_NAME_TO_NUM[month_key]
        if current_month is None:
            return [], embedded_total, fail_warning
        try:
            entries.append((date(year, current_month, int(day_str)), bool(marker)))
        except ValueError:
            return [], embedded_total, f"Invalid date parsed from {raw!r} — enter manually."

    runs: list[tuple[date, date, float]] = []
    run_start = run_end = entries[0][0]
    run_len, run_half = 1, (1 if entries[0][1] else 0)
    for d, is_half in entries[1:]:
        if d == run_end + timedelta(days=1):
            run_end = d
            run_len += 1
            run_half += 1 if is_half else 0
        else:
            runs.append((run_start, run_end, run_len - 0.5 * run_half))
            run_start = run_end = d
            run_len, run_half = 1, (1 if is_half else 0)
    runs.append((run_start, run_end, run_len - 0.5 * run_half))

    return runs, embedded_total, None


def _split_combined_remarks(raw: str) -> tuple[str, str]:
    """Split a combined "SL-WP" Remarks cell into (leave_type, action)."""
    left, sep, right = raw.partition("-")
    if not sep:
        return _normalize_leave_type(raw), ""
    return _normalize_leave_type(left), _normalize_action(right)


def _clean_name_part(raw: str) -> str:
    """Strip stray leading/trailing OCR marks (a watermark fleck, a curly
    quote picked up from a nearby border) that aren't part of the name
    itself, without touching legitimate interior punctuation like O'Brien."""
    return re.sub(r"[^A-Za-z]+$", "", re.sub(r"^[^A-Za-z]+", "", raw.strip()))


def _split_combined_name(raw: str) -> tuple[str, str, str]:
    """Split a "Last, First M.I." combined name cell (used on the Lumbia
    template, which has no separate Last/First/M.I. columns)."""
    last, _, rest = raw.partition(",")
    tokens = rest.strip().split()
    middle_initial = ""
    if tokens and re.match(r"^[A-Za-z]\.?$", tokens[-1]):
        middle_initial = _normalize_mi(tokens[-1])
        tokens = tokens[:-1]
    return _clean_name_part(last).title(), _clean_name_part(" ".join(tokens)).title(), middle_initial


def _format_period(date_from: str, date_to: str | None) -> str:
    df = date.fromisoformat(date_from)
    if not date_to or date_to == date_from:
        return f"{MONTH_NAMES[df.month]} {df.day}, {df.year}"
    dt = date.fromisoformat(date_to)
    if df.month == dt.month and df.year == dt.year:
        return f"{MONTH_NAMES[df.month]} {df.day}-{dt.day}, {df.year}"
    return f"{MONTH_NAMES[df.month]} {df.day} - {MONTH_NAMES[dt.month]} {dt.day}, {dt.year}"


def _build_description(leave_type: str, date_from: str | None, date_to: str | None) -> str:
    label = _LEAVE_TYPE_LABELS.get(leave_type, "Leave")
    if not date_from:
        return label
    df = date.fromisoformat(date_from)
    if not date_to or date_to == date_from:
        return f"{label} {MONTH_NAMES[df.month]} {df.day}, {df.year}"
    dt = date.fromisoformat(date_to)
    if df.month == dt.month and df.year == dt.year:
        return f"{label} {MONTH_NAMES[df.month]} {df.day}-{dt.day}, {df.year}"
    return f"{label} {MONTH_NAMES[df.month]} {df.day} - {MONTH_NAMES[dt.month]} {dt.day}, {dt.year}"


def _page_text(img: np.ndarray) -> str:
    """The whole page OCR'd once (not the dewarped table): the header and
    paragraph around the table say things its cells don't."""
    return pytesseract.image_to_string(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))


def _detect_default_year(text: str) -> int | None:
    """Some templates (e.g. Lumbia) never print a year in the dates column
    itself, only once in the page header ("For the Month of June 2026")."""
    m = re.search(r"month\s+of\s+\w+\s+(\d{4})", text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"\b(20\d{2})\b", text)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def _rows_from_segments(
    *,
    row_no: str,
    last_name: str,
    first_name: str,
    middle_initial: str,
    position_raw: str,
    dates_raw: str,
    days_raw: str,
    leave_type: str,
    action_taken: str,
    in_scope: bool,
    segments: list[tuple[date, date]],
    total_override: float | None,
    unparsed_warning: str | None,
    default_used: list[float] | None = None,
) -> list[Form6Row]:
    """Shared plumbing: turn parsed date segment(s) for one raw table row
    into one Form6Row per segment (per the project's convention that a
    non-contiguous leave — e.g. two separate date ranges in one cell —
    becomes two separate, independently-submittable draft rows).

    `default_used` is each segment's day count absent an explicit printed
    total to prefer — plain calendar-day counting unless the caller already
    computed something more precise (e.g. a half-day adjustment)."""
    common = dict(
        row_no=row_no, last_name=last_name, first_name=first_name,
        middle_initial=middle_initial, position_raw=position_raw,
        dates_raw=dates_raw, days_raw=days_raw,
        leave_type=leave_type, action_taken=action_taken, in_scope=in_scope,
    )

    if not segments:
        row = Form6Row(
            date_from=None, date_to=None,
            used=_format_used(total_override) if total_override is not None else "",
            parse_warning=unparsed_warning,
            **common,
        )
        row.description = _build_description(leave_type, None, None)
        return [row]

    # A caller-supplied default_used (e.g. the Lumbia date parser's am/pm-
    # aware per-run counts) is more trustworthy than the free-text "No. of
    # Days" cell, which has no structure to anchor OCR against and can
    # misread a "½" glyph as a stray digit. Only fall back to preferring the
    # printed total when we have nothing better than blind calendar-day
    # counting (the 9-column numeric-date layout).
    trust_default = default_used is not None
    if default_used is None:
        default_used = [(dt - df).days + 1 for df, dt in segments]

    seg_used: list[float]
    # A parser can fill the dates in *and* ask for them to be checked (a
    # disagreement between OCR reads settled by majority). That warning has to
    # reach the row: dropping it here once dates existed is what left a
    # corrected-but-uncertain date unflagged in the review table.
    warnings = [unparsed_warning] if unparsed_warning else []
    if trust_default or len(segments) > 1:
        seg_used = default_used
        computed_total = sum(seg_used)
        if total_override is not None and abs(computed_total - total_override) > 0.01:
            warnings.append(
                f"Computed total ({_format_used(computed_total)} days) doesn't match the "
                f"form's printed total ({_format_used(total_override)} days) — double-check dates."
            )
    else:
        seg_used = [total_override if total_override is not None else default_used[0]]

    rows = []
    for (df, dt), used in zip(segments, seg_used):
        row = Form6Row(
            date_from=df.isoformat(), date_to=dt.isoformat(),
            used=_format_used(used), parse_warning=" ".join(warnings) or None,
            **common,
        )
        row.description = _build_description(leave_type, df.isoformat(), dt.isoformat())
        rows.append(row)
    return rows


def _ocr_date_cell(img: np.ndarray, box: tuple[int, int, int, int]) -> tuple[str, int]:
    """A mapped date column can hold either style — "06/30/2026" or
    "June 30, 2026" — so read it with the month-name whitelist first, then
    re-read a numeric cell under the tight digits-and-slashes one it was
    tuned for. A numeric cell read this way comes back without its slashes
    ("06302026"), which is still enough to tell the two styles apart.

    Returns (text, half-day markers). The numeric re-read can't represent
    letters, so a "7/7PM/2026" would lose its PM there; the count is taken
    from the first pass, which keeps them. Month-name cells report 0 — their
    parser reads the markers itself."""
    raw = _ocr_cell(img, box, 6, _ABSENCE_DATES_WHITELIST)
    if _MONTH_IN_TEXT_RE.search(raw) or _DITTO_RE.match(raw):
        return raw, 0
    halves = len(re.findall(r"\d\s*\(?\s*(?:AM|PM)\b", raw, re.IGNORECASE))
    # No month name: a numeric cell, whose slashes the first whitelist can't
    # even represent ("06/17/2026" comes back "06172026"). Read it again
    # under the digits-and-slashes whitelist that style was tuned for.
    # A "to" between figures ("5/6 to 6/2/2026", SO 2159) is the range itself,
    # and without it the re-read glues both dates into "5/66/2/2026".
    allowed = _DATE_WHITELIST + ("to" if re.search(r"\d\s*to\s*\d", raw, re.IGNORECASE) else "")
    return _ocr_cell(img, box, 6, allowed), halves


# "DO" / "ditto" / "-do-": same as the row above. The digits-only re-read
# can't represent it, so it is caught on the first pass.
_DITTO_RE = re.compile(r"^\W*(?:D[O0Q]|DITTO)\W*$", re.IGNORECASE)

_MONTH_IN_TEXT_RE = re.compile(
    r"\b(" + "|".join(MONTH_NAMES[1:]) + r"|" + "|".join(m[:3] for m in MONTH_NAMES[1:]) + r")",
    re.IGNORECASE,
)


def _describe_segments(segments: list[tuple[date, date]], with_year: bool = False) -> str:
    parts = []
    for f, t in segments:
        day = f"{f.day}" if f == t else f"{f.day}-{t.day}"
        parts.append(f"{MONTH_NAMES[f.month]} {day}" + (f", {f.year}" if with_year else ""))
    return "; ".join(parts) if with_year else ", ".join(parts)


def _consensus_numeric_dates(reads: list[str]) -> tuple[str, list[tuple[date, date]], str | None]:
    """Settle a numeric date cell from several independent OCR reads, the
    first being the primary one. Returns (text, segments, warning).

    One read of "7/3/2026" can come back "7/13/2026": the slash gains a
    phantom 1, and the result is a perfectly plausible date that parses
    cleanly and would be submitted without a second look. A single read can't
    tell that from a real 13th — but reads made with different settings don't
    share the mistake, so disagreement between them is the signal.

      - Every read that parses agrees: take it.
      - The primary read failed but the others agree: take theirs. With only
        one read to go on, flag it for a check against the paper.
      - Reads disagree: never pick silently. A strict majority is filled in
        and flagged with the alternatives; a tie is left blank to be entered.
    """
    primary = reads[0]
    parsed: list[tuple[str, tuple[tuple[date, date], ...]]] = []
    for text in reads:
        segments, _ = _parse_date_segments_numeric(text)
        if segments:
            parsed.append((text, tuple(segments)))
    if not parsed:
        return primary, [], f"Could not parse dates from {primary!r} — enter manually."

    votes = Counter(segs for _, segs in parsed)
    if len(votes) == 1:
        segs, count = next(iter(votes.items()))
        text = next(t for t, sg in parsed if sg == segs)
        if parsed[0][0] == primary or count >= 2:
            return text, list(segs), None
        return text, list(segs), (
            f"Date read as {_describe_segments(list(segs))} on a second pass only — "
            "check it against the paper."
        )

    # Readings that differ only in the year describe identically without it
    # ("June 26 or June 26"), which hides the very thing in dispute.
    plain = [_describe_segments(list(sg)) for sg, _ in votes.most_common()]
    with_year = len(set(plain)) < len(plain)
    options = " or ".join(_describe_segments(list(sg), with_year) for sg, _ in votes.most_common())
    (top, top_votes), (_, runner_up) = votes.most_common(2)
    if top_votes > runner_up:
        text = next(t for t, sg in parsed if sg == top)
        return text, list(top), (
            f"OCR disagreed on this date ({options}) — the more common reading is "
            "filled in; check it against the paper."
        )
    return primary, [], f"OCR disagreed on this date ({options}) — enter it from the paper."


def _read_numeric_dates(img: np.ndarray, box: tuple[int, int, int, int], primary: str):
    return _consensus_numeric_dates([
        primary,
        _ocr_cell(img, box, 13, _DATE_WHITELIST),
        _ocr_cell_plain(img, box, 7, _DATE_WHITELIST),
    ])


# ---------------------------------------------------------------------------
# Vacation service credits
#
# A grant form ("the following teachers are hereby granted vacation service
# credits for services rendered during ...") runs the other way from a Form 6:
# it adds credit to the ledger's `earned` column instead of deducting leave.
# Its rows carry the hours served and the credits granted for them.
# ---------------------------------------------------------------------------

# Work rendered during summer or Christmas vacation, weekends or holidays is
# credited at time-and-a-half: VSC days = hours x 1.50 / 8.
VSC_CONVERSION_FACTOR = 1.50
# The most a confirmed reading may grant per hour (see _resolve_vsc_credits_unbounded).
VSC_MAX_CONFIRMED_FACTOR = 2.00
HOURS_PER_DAY = 8

_NAME_SUFFIXES = {"JR", "SR", "II", "III", "IV", "V"}


# Where the event's name ends: "on"/"from" followed by a date ("on JUNE 12-13",
# "from May 6") or "held" ("HELD AT FR. WILLIAM F. MASTERSON ..."). A bare "on"
# isn't enough: "Orientation on the Strengthened SHS" is part of the name. The
# date's month isn't matched by name, since OCR misspells it ("Janauary").
# A month's name or abbreviation, as a whole word: what lets a period open with
# no "on"/"from" — SO 2159's "...PROGRAMS (SRP) May 6 to June 2, 2026."
_MONTH_WORD = r"(?:" + "|".join(sorted({*MONTH_NAMES[1:], *(m[:3] for m in MONTH_NAMES[1:])}, key=len, reverse=True)) + r")\b"

_EVENT_END_RE = re.compile(
    r"\s+(?:(?:on|from)\s+[A-Za-z]{3,}\.?\s*\d|held\b|" + _MONTH_WORD + r"\.?\s*\d{1,2}\b)", re.IGNORECASE,
)
# "during", including OCR'd remnants of it: Pedro Roa's came back "duri : 3 uring".
_EVENT_START_RE = re.compile(r"\b(?:d?uring)\s+", re.IGNORECASE)
_EVENT_FILLER_RE = re.compile(r"^(?:(?:the|participation\s+in|conduct\s+of)\s+)+", re.IGNORECASE)
_EVENT_SMALL_WORDS = {"A", "AN", "AND", "AT", "BY", "CUM", "FOR", "IN", "OF", "ON", "OR", "THE", "TO", "WITH"}


# "on JUNE 12-13. 2026", "from June 1-5, 2026", "on May 4 to 29, 2026",
# "from May 6 to June 2, 2026": the period a special order grants credits for.
_PERIOD_RE = re.compile(
    r"(?:\b(?:on|from)\s+|(?<![A-Za-z])(?=" + _MONTH_WORD + r"))(?P<m1>[A-Za-z]{3,10})\.?\s*(?P<d1>\d{1,2})\b"
    r"(?:\s*(?:-|\u2013|to)\s*(?:(?P<m2>[A-Za-z]{3,10})\.?\s*)?(?P<d2>\d{1,2})\b)?"
    r"[,.\s]*(?P<y>20\d\d)?",
    re.IGNORECASE,
)


def _month_number(word: str) -> int | None:
    prefix = word[:3].lower()
    return next((i for i, name in enumerate(MONTH_NAMES) if i and name[:3].lower() == prefix), None)


def _grant_sentence(flat: str) -> re.Match | None:
    """Where the order's grant sentence starts: "rendered" ("...for services
    rendered during..."), or failing that, "service credits". "rendered" first:
    SO 2159's title, "SPECIAL ORDER FOR VACATION SERVICE CREDITS", comes before
    the order's date line, which then read as the period (July 7)."""
    return re.search(r"rendered", flat, re.IGNORECASE) or re.search(r"service\s+credits?", flat, re.IGNORECASE)


def _period_from_text(text: str, default_year: int | None) -> tuple[str, str] | None:
    """The period the order's paragraph says the credits are for, as ISO
    dates — or None when it isn't stated cleanly. Only ever used to *doubt* a
    row, so a paragraph that doesn't parse costs nothing."""
    flat = " ".join(re.sub(r"-\s*\n\s*", "", text).split())
    anchor = _grant_sentence(flat)
    if not anchor:
        return None
    for m in _PERIOD_RE.finditer(flat[anchor.end():anchor.end() + 500]):
        month1 = _month_number(m.group("m1"))
        if not month1:
            continue        # "on THE STRENGTHENED..." isn't a date
        month2 = _month_number(m.group("m2")) if m.group("m2") else month1
        year = int(m.group("y")) if m.group("y") else default_year
        if not month2 or not year:
            return None
        try:
            start = date(year, month1, int(m.group("d1")))
            end = date(year, month2, int(m.group("d2") or m.group("d1")))
        except ValueError:
            return None
        return (start.isoformat(), end.isoformat()) if start <= end else None
    return None


def _apply_period(rows: list[Form6Row], period: tuple[str, str] | None) -> None:
    """Doubt a grant whose dates fall outside the period its order states.

    SO 1892 read a 5 as a 9 in "June 1,2,3,4,5" on four rows (June 1-9), and a
    pen stroke turned another into June 1-25 — plausible ranges, quietly wrong;
    the order says June 1-5. The dates stay, since the doubt may be the
    paragraph's (one order's rows run a day past its printed period), but the
    row is flagged and unticked: it can't be submitted without a deliberate
    tick. If a good share of the page is "outside", it is the paragraph that
    was misread, not every row — and nothing is touched."""
    grants = [r for r in rows if r.kind == "vsc" and r.date_from and r.date_to]
    if not period or not grants:
        return
    lo, hi = period
    outside = [r for r in grants if not (lo <= r.date_from and r.date_to <= hi)]
    if not outside or len(outside) > 0.4 * len(grants):
        return
    for r in outside:
        note = (
            f"Dates read as {_format_period(r.date_from, r.date_to)}, outside the order's period "
            f"({_format_period(lo, hi)}) — check them against the paper before ticking."
        )
        r.parse_warning = f"{r.parse_warning} {note}" if r.parse_warning else note
        r.in_scope = False


def _event_from_text(text: str) -> str:
    """The event a grant was earned for, from the order's paragraph: "...
    for services rendered during DIVISION TRAINING FOR THE STRENGTHENED
    SENIOR HIGH SCHOOL CURRICULUM on JUNE 12-13, 2026". Empty when the page
    has no such sentence (a continuation page) or it didn't OCR cleanly."""
    flat = " ".join(re.sub(r"-\s*\n\s*", "", text).split())   # rejoin "ser-\nvice"
    anchor = _grant_sentence(flat)
    if not anchor:
        return ""
    tail = flat[anchor.end():anchor.end() + 400]
    end = _EVENT_END_RE.search(tail)
    if not end:
        return ""
    starts = list(_EVENT_START_RE.finditer(tail, 0, end.start()))
    if not starts:
        return ""
    event = _EVENT_FILLER_RE.sub("", tail[starts[-1].end():end.start()]).strip(" .,;:")
    if not 3 <= len(event) <= 200 or _TABLE_HEADER_TEXT_RE.search(event):
        # SO 1849's page 6 OCR'd its paragraph out of order, and what ran on
        # from "during" was the table's header and first row.
        return ""
    return _event_case(event)


_TABLE_HEADER_TEXT_RE = re.compile(r"\bNo\.\s*of\b|\bInclusive\b|\bName\s+Position\b|\bCredits\s+Granted\b", re.IGNORECASE)


def _event_case(event: str) -> str:
    """Readable casing for an event printed in capitals; mixed case is kept
    as printed. Short words with no vowel are acronyms (SHS, SY, NHS) and
    stay capitals — "OLD" and "DAY" don't — as does anything in parentheses
    or with digits or dots ("(SRP)", "S.J.")."""
    letters = [c for c in event if c.isalpha()]
    if not letters or sum(c.isupper() for c in letters) < 0.8 * len(letters):
        return event
    words = []
    for i, word in enumerate(event.split()):
        core = word.strip("(),")
        if not core.isalpha():
            words.append(word)
        elif core in _EVENT_SMALL_WORDS and i > 0:
            words.append(word.lower())
        elif len(core) <= 4 and not re.search("[AEIOU]", core) or word.startswith("("):
            words.append(word)
        else:
            words.append(word.replace(core, core.capitalize()))
    return " ".join(words)


def _vsc_description(event: str, date_from: str | None, date_to: str | None, hours: float | None) -> str:
    """"Vacation service credits, Aral Summer Program, May 6-31, 2026 (113.29 hrs)"."""
    parts = ["Vacation service credits"]
    if event:
        parts.append(event)
    if date_from:
        parts.append(_format_period(date_from, date_to))
    description = ", ".join(parts) if event else " ".join(parts)
    return description + (f" ({hours:g} hrs)" if hours is not None else "")


def _split_name_first_last(raw: str) -> tuple[str, str, str]:
    """Split a "First M. Last" name — the order a grant form prints it in.

    The middle initial is the boundary when there is one, which is what keeps
    a compound surname whole ("Ana Marie A. Tres Reyes" -> Tres Reyes). With
    no initial the last word is taken as the surname ("Kenneth Joh Conde").
    A trailing suffix (Jr, III) is dropped: it isn't part of how HRIS is
    searched, and leaving it in would make it the surname."""
    tokens = raw.replace(",", " ").split()
    while len(tokens) > 2 and re.sub(r"[^A-Za-z]", "", tokens[-1]).upper() in _NAME_SUFFIXES:
        tokens.pop()
    initials = [i for i, t in enumerate(tokens) if re.fullmatch(r"[A-Za-z]\.", t)]
    if initials and 0 < initials[-1] < len(tokens) - 1:
        i = initials[-1]
        first, middle, last = tokens[:i], tokens[i], tokens[i + 1:]
        mi = _normalize_mi(middle)
    elif len(tokens) >= 2:
        first, last, mi = tokens[:-1], tokens[-1:], ""
    else:
        return "", "", ""
    return (
        _clean_name_part(" ".join(last)).title(),
        _clean_name_part(" ".join(first)).title(),
        mi,
    )


def _split_any_name(raw: str) -> tuple[str, str, str]:
    """(last, first, M.I.) from a name in either order: "Last, First M.I."
    when there's a comma, "First M.I. Last" when there isn't."""
    return _split_combined_name(raw) if "," in raw else _split_name_first_last(raw)


def _normalize_hours(raw: str) -> float | None:
    """Hours served: "138" -> 138.0, "18 HRS" -> 18.0, and the
    hours-and-minutes form some grants use, "4HRS.& 53MINS." -> 4.883."""
    s = raw.upper()
    # The number in front of each unit can come back with letter look-alikes
    # ("S3MINS" for 53, "SHRS" for 5). Only that number is folded back to
    # digits — never the unit, whose own S would otherwise become a 5.
    # "/" too: a 7 comes back as one ("27 HOURS" -> "2/ HOURS").
    lookalike = "[0-9SOQ/" + re.escape("".join(_ONE_LOOKALIKES).upper()) + "]"
    fold = str.maketrans({"S": "5", "O": "0", "Q": "0", "/": "7", **{c.upper(): "1" for c in _ONE_LOOKALIKES}})
    hrs = re.search(rf"({lookalike}+)\s*H(?:RS?|OURS?)\b", s)
    mins = re.search(rf"({lookalike}+)\s*MIN(?:UTE)?S?\b", s)
    has_hour_unit = re.search(r"H(?:RS?|OURS?)\b", s)
    has_minute_unit = re.search(r"MIN(?:UTE)?S?\b", s)
    if has_hour_unit or has_minute_unit:
        # A unit whose number didn't read makes the whole cell unreadable —
        # not zero of that unit. "2/ HOURS & 40 MINUTES" used to come back as
        # 40 minutes, a quietly wrong figure instead of an honest blank.
        if (has_hour_unit and not hrs) or (has_minute_unit and not mins):
            return None
        def number(m):
            digits = m.group(1).translate(fold) if m else "0"
            return int(digits) if digits.isdigit() else None
        h, mi = number(hrs), number(mins)
        if h is None or mi is None:
            return None
        return round(h + mi / 60, 3)
    # Not _normalize_digits: it joins every digit run, so "72.52" became 7252
    # hours — and a figure that large lifts the credit ceiling instead of
    # tripping it. Look-alikes are folded to 1 in place, keeping the point.
    folded = re.sub("[" + re.escape("".join(_ONE_LOOKALIKES)) + "]", "1", s.strip())
    m = re.search(r"\d+(?:[.,]\d+)?", folded)
    return float(m.group(0).replace(",", ".")) if m else None


def _normalize_credits(raw: str) -> float | None:
    """A printed credit figure: "25.944" -> 25.944. A comma read in place of
    the decimal point ("25,944") is taken as one, since these are never in
    the thousands."""
    s = raw.replace(",", ".")
    # A leading-point figure (".5", as Kauswagan prints half a day) has to be
    # matched whole — a plain digit run would read it as 5.
    m = re.search(r"\d*\.\d+|\d+", s)
    return float(m.group(0)) if m else None


def _blank_text(raw: str) -> bool:
    """Nothing legible: empty, or only marks ("-", "--", ","). What a cell of
    "-do-" reads as when a pen line runs through it."""
    return not re.search(r"[A-Za-z0-9]", raw)


def vsc_days(hours: float, factor: float = VSC_CONVERSION_FACTOR) -> float:
    """VSC days = hours x factor / 8, to the 3 decimals the ledger keeps."""
    return round(hours * factor / HOURS_PER_DAY, 3)


# The most credit one calendar day can possibly earn: 24 hours at x1.50.
MAX_VSC_DAYS_PER_CALENDAR_DAY = 24 * VSC_CONVERSION_FACTOR / HOURS_PER_DAY


def _resolve_vsc_credits(
    hours: float | None, printed: float | None, period_days: int | None = None, confirmed: bool = False,
) -> tuple[float | None, str | None]:
    """_resolve_vsc_credits_unbounded, then refused outright if the result is
    more than the period could physically earn. A lost decimal point turns
    7.21 into 721 — a figure that flagging alone would still leave prefilled
    and ticked for submission."""
    earned, warning = _resolve_vsc_credits_unbounded(hours, printed, confirmed)
    if earned is not None and period_days:
        ceiling = period_days * MAX_VSC_DAYS_PER_CALENDAR_DAY
        if earned > ceiling:
            return None, (
                f"Credits read as {earned:g} — more than a {period_days}-day period can earn "
                f"(at most {ceiling:g} even at 24 hours a day). Likely a misread; enter it from the paper."
            )
    return earned, warning


def _resolve_vsc_credits_unbounded(
    hours: float | None, printed: float | None, confirmed: bool = False,
) -> tuple[float | None, str | None]:
    """The credit to record for one row, and a warning when it needs a look.

    The printed figure is what the special order grants, so it is what gets
    recorded — and offices work it out differently: one at 0.188 per hour,
    one truncating to half-days, one below the formula by varying amounts.
    Checking for an exact match with hours x 1.50 / 8 would flag nearly every
    correct row of those orders, and a flag on everything gets ignored.

    What does hold across all of them is that a grant never exceeds what the
    hours can earn at x1.50 — while OCR misreads push the other way (a 5 read
    as a 9, a lost decimal point). So the hours are an upper bound, not a
    target. Downward misreads are caught separately, by reading the cell more
    than one way (_consensus_credits).

    Except that some orders do grant more. SO 2159 prints 28.5 for 114 hours
    on seven rows (x2.00) and 27.188 for 112. So a figure above the hours'
    ceiling that every independent reading agrees on (`confirmed`) is filled
    in as printed — but with a warning, and the caller leaves it unticked.
    Up to x2.00 only: a lost decimal point is ten times over, not a third."""
    if printed is None:
        ceiling = f" (at most {vsc_days(hours):g} for {hours:g} hours at x{VSC_CONVERSION_FACTOR:.2f})" if hours is not None else ""
        # Not computed and filled in: the formula is exactly what these
        # offices *don't* consistently use, so it would be a guess.
        return None, f"Credits granted couldn't be read — enter the figure from the paper{ceiling}."
    if hours is None:
        return printed, "Hours served couldn't be read, so the credits granted weren't checked against them."
    # The rounded hourly rate (0.188) sits a hair above the exact one (0.1875)
    # and is used in practice, so it is the ceiling.
    ceiling = hours * round(VSC_CONVERSION_FACTOR / HOURS_PER_DAY, 3) + 0.0015
    if printed > ceiling and confirmed and printed <= vsc_days(hours, VSC_MAX_CONFIRMED_FACTOR) + 0.0015:
        return printed, (
            f"Credits printed as {printed:g} — more than {hours:g} hours earn at x{VSC_CONVERSION_FACTOR:.2f} "
            f"({vsc_days(hours):g}). Read alike two independent ways, so it's filled in as printed, but not ticked: "
            "confirm the paper grants it before submitting."
        )
    if printed > ceiling:
        # Not filled in, even flagged: no office can grant more than the hours
        # earn, so this is a misread — and a prefilled misread stays ticked.
        return None, (
            f"Credits granted read as {printed:g} — more than {hours:g} hours can earn at "
            f"x{VSC_CONVERSION_FACTOR:.2f} ({vsc_days(hours):g}). Likely a misread; enter it from the paper."
        )
    return printed, None


def _earned_at_formula(hours: float | None, credits: float | None) -> bool:
    """Whether `credits` is exactly what `hours` earn, at the exact rate or the
    rounded one (0.188) some offices use."""
    if hours is None or credits is None:
        return False
    rates = (VSC_CONVERSION_FACTOR / HOURS_PER_DAY, round(VSC_CONVERSION_FACTOR / HOURS_PER_DAY, 3))
    return any(abs(credits - hours * rate) < 0.0015 for rate in rates)


def _earns_as_printed(hours: float, credits: float) -> bool:
    """Whether `credits` is what `hours` earn at the formula, to the precision
    it's printed at: 14.06 for 75 hours' 14.0625, as well as the exact rates
    _earned_at_formula checks."""
    decimals = len(f"{credits:g}".partition(".")[2])
    exact = hours * VSC_CONVERSION_FACTOR / HOURS_PER_DAY
    return _earned_at_formula(hours, credits) or (decimals >= 2 and abs(round(exact, decimals) - credits) < 1e-9)


def vsc_ceiling(hours: float | None) -> float | None:
    """The most credit `hours` can earn, at the rounded rate some offices use."""
    if hours is None:
        return None
    return hours * round(VSC_CONVERSION_FACTOR / HOURS_PER_DAY, 3) + 0.0015


def _consensus_credits(reads: list[str], ceiling: float | None = None) -> tuple[float | None, str | None]:
    """Settle a printed credit figure from several independent OCR reads.

    Filled in only when a strict majority of the readable reads agree *and*
    that value fits under what the hours can earn. The hours can veto a
    reading, but never pick one: they are OCR'd too. Uy's "9 HRS" read as 5
    made the paper's 1.5 look impossible, and letting the hours choose among
    the readings filled in a wrong 1. So any disagreement the majority can't
    settle within the ceiling is left blank — with a hint naming the reading
    that does fit, since that one is usually right and easy to confirm."""
    values = [v for v in (_normalize_credits(r) for r in reads) if v is not None]
    if not values:
        return None, None
    votes = Counter(values)
    if len(votes) == 1:
        return values[0], None
    options = " or ".join(f"{v:g}" for v, _ in votes.most_common())
    (top, top_votes), (_, runner_up) = votes.most_common(2)
    if top_votes > runner_up and (ceiling is None or top <= ceiling):
        return top, (
            f"OCR disagreed on the credits granted ({options}) — the more common reading "
            "is filled in; check it against the paper."
        )
    hint = ""
    if ceiling is not None:
        fits = [v for v in votes if v <= ceiling]
        if len(fits) == 1:
            hint = f" Only {fits[0]:g} fits the hours served."
    return None, f"OCR disagreed on the credits granted ({options}) — enter it from the paper.{hint}"


def _agreed(reads: list[str], parse) -> float | None:
    """The value both scaled reads give, or None if they don't agree. Never
    zero: under a digits-only whitelist "-do-" reads as "0", and no grant
    is for nothing."""
    values = [parse(r) for r in reads]
    return values[0] if len(values) == 2 and values[0] and values[0] == values[1] else None


def _settle_credits(
    old: list[float | None], printed: float | None, warning: str | None,
    scaled: float | None, ceiling: float | None,
) -> tuple[float | None, str | None]:
    """Weigh the scaled reads (_scaled_reads) against the whole-cell reads.

    The scaled value is taken where the whole-cell reads offer nothing the
    hours allow — SO 1849's "9.2902" and "9.292" for a 5.252 earned by 28
    hours — or agree with it. Where a whole-cell read the hours allow says
    something else, two believable figures disagree and neither is filled
    in: SO 1849's Morata read as 5.202 whole and 5.252 scaled, both under
    28 hours' ceiling, and only the paper can say."""
    if scaled is None or (ceiling is not None and scaled > ceiling):
        return printed, warning
    believable = {v for v in old if v is not None and (ceiling is None or v <= ceiling)}
    if believable <= {scaled} or printed == scaled:
        return scaled, None
    options = " or ".join(f"{v:g}" for v in [scaled, *sorted(believable - {scaled})])
    return None, f"OCR disagreed on the credits granted ({options}) — enter it from the paper."


def _most_hours(period: tuple[str, str] | None) -> float | None:
    """Every hour of the order's stated period: no teacher serves more."""
    if not period:
        return None
    return 24 * ((date.fromisoformat(period[1]) - date.fromisoformat(period[0])).days + 1)


def _settle_hours(
    hours: float | None, digit_reads: tuple[float | None, ...], scaled: float | None, most: float | None,
) -> tuple[float | None, str | None]:
    """The hours, weighing the scaled reads against the whole-cell ones, and
    the readings that disagreed if they can't be settled ("90.97 or 50.97").

    The scaled value is taken where the whole-cell reads found nothing, or
    more hours than the order's period has — SO 1849's "395" for 35 in a
    five-day order — or where one of them read it too. Otherwise two
    believable figures disagree, and the row isn't ticked: SO 1896's 50.97
    read whole as 90.97, ticked, and "(90.97 hrs)" would have gone to HRIS."""
    possible = lambda v: v is not None and (most is None or v <= most)
    if not possible(scaled) or scaled == hours:
        return hours, None
    if not possible(hours) or scaled in digit_reads:
        return scaled, None
    return None, f"{hours:g} or {scaled:g}"


def _period_from_dates(raw: str, default_year: int | None) -> tuple[str | None, str | None, str | None]:
    """A grant's Inclusive Dates is one period ("May 6 to June 2, 2026"), not
    a list of absence days — the first and last dates mentioned bound it."""
    short = re.findall(r"\b(\d{1,2})-(\d{1,2})-(\d{2}|\d{4})\b", raw)
    if short and not _MONTH_IN_TEXT_RE.search(raw):
        # "1-31-26, 2-7-26": month-day-year with a two-digit year. These list
        # the separate days worked; the grant's period runs first to last.
        dates = []
        for m, d, y in short:
            year = int(y) + 2000 if len(y) == 2 else int(y)
            try:
                dates.append(date(year, int(m), int(d)))
            except ValueError:
                return None, None, f"Invalid date parsed from {raw!r} — enter manually."
        warning = None
    elif _MONTH_IN_TEXT_RE.search(raw):
        runs, warning = _parse_date_segments_monthname(raw, default_year)
        dates = [d for f, t, _ in runs for d in (f, t)]
    else:
        segments, warning = _parse_date_segments_numeric(_yearless_day_list(raw, default_year))
        dates = [d for f, t in segments for d in (f, t)]
    if not dates:
        return None, None, warning or f"Could not parse dates from {raw!r} — enter manually."
    return min(dates).isoformat(), max(dates).isoformat(), None


# "296/1": a day and the next month's number with the space between lost.
_LOST_SPACE_BEFORE_MONTH_RE = re.compile(r"(?<![\d/])(\d{2})(\d{1,2})/")


def _yearless_day_list(raw: str, default_year: int | None) -> str:
    """A numeric day list as _parse_date_segments_numeric reads it.

    SO 2159 lists the days worked by month with no year at all — "5/6-8,11-15,
    18-22,25,26,28,29, 6/1,2" — the year is in the order's paragraph. That
    year goes on the end. And where the cell's space before the next month
    was lost ("...28,296/1,2"), the run of digits is split back into the day
    and the month — only when it can't be a month itself."""
    def split(m: re.Match) -> str:
        day, month = int(m.group(1)), int(m.group(2))
        return f"{m.group(1)},{m.group(2)}/" if 1 <= day <= 31 and 1 <= month <= 12 else m.group(0)

    text = _LOST_SPACE_BEFORE_MONTH_RE.sub(split, raw)
    if default_year and "/" in text and not re.search(r"(?:19|20)\d{2}", text):
        text = f"{text.rstrip(' ,')}/{default_year}"
    return text


def _choose_dates(
    first: str, second: str, default_year: int | None, period: tuple[str, str] | None,
) -> tuple[str, str | None, str | None, str | None]:
    """(raw text, date_from, date_to, warning) from two independent reads of a
    month-name date cell. The reads fail in different places — SO 1892's
    "June 1,2,3,4,5, 2026" came back "...4,9," on one pass and "...4,5," on the
    other, and "...5,42026" on one where the other was clean.

    Both agree: fine. Only one parses: take it. Both parse but differ: take the
    one inside the order's period if only one is, since a 5 misread as 9 lands
    outside it; failing that, take the first and say the reads disagreed."""
    a_from, a_to, a_warn = _period_from_dates(first, default_year)
    if not second.strip() or second.strip() == first.strip():
        return first, a_from, a_to, a_warn
    b_from, b_to, b_warn = _period_from_dates(second, default_year)
    if a_from and b_from:
        if (a_from, a_to) == (b_from, b_to):
            return first, a_from, a_to, a_warn
        inside = lambda f, t: period is None or (period[0] <= f and t <= period[1])
        a_in, b_in = inside(a_from, a_to), inside(b_from, b_to)
        if a_in and not b_in:
            return first, a_from, a_to, a_warn
        if b_in and not a_in:
            return second, b_from, b_to, b_warn
        return first, a_from, a_to, (
            f"OCR disagreed on the dates ({_format_period(a_from, a_to)} or {_format_period(b_from, b_to)}) "
            "— check them against the paper."
        )
    if b_from:
        return second, b_from, b_to, b_warn
    return first, a_from, a_to, a_warn


def _extract_vsc_row(
    img: np.ndarray, cells: dict[str, list[tuple[str, tuple[int, int, int, int]]]],
    default_year: int | None, seq: int, previous: Form6Row | None = None, event: str = "",
    period: tuple[str, str] | None = None, inked: np.ndarray | None = None,
) -> list[Form6Row]:
    """previous: the grant row above, for "-do-" cells to inherit from.
    inked: the table as scanned, pen and all (img has blue pen removed). A
    "-do-" hidden under a pen mark is told by the mark, so it is looked for
    there; whether digits can be read through it is a question for img.
    event: what the page's paragraph says the grants were earned for.
    period: the period it states, to tell a misread date from a real one."""
    def read_all(field: str, psm: int, whitelist: str | None) -> str:
        # A pen tick or margin line through a column can split one cell into
        # two boxes; reading them left to right keeps the whole value.
        return " ".join(
            _ocr_cell(img, box, psm, whitelist).strip() for _, box in cells.get(field) or []
        ).strip()

    name_raw = read_all("name", 6, None)
    last_name, first_name, middle_initial = _split_any_name(name_raw)
    if not last_name or not first_name:
        return []

    ditto_notes = []
    # psm 6: "31 HOURS & 37 MINUTES" wraps over three lines in its cell.
    hours_raw = read_all("hours", 6, None)
    inherited: list[str] = []
    hidden = False      # a figure taken from the row above without seeing it
    hours_disputed = False
    hour_options: list[float] = []
    is_ditto_column = lambda name: bool(previous and name in previous.inherited)
    inked = img if inked is None else inked
    hours_stroke = img is not None and _has_pen_stroke(img, [box for _, box in cells.get("hours") or []])
    hours_marked = inked is not None and _has_pen_stroke(inked, [box for _, box in cells.get("hours") or []])
    if _DITTO_RE.match(hours_raw):
        hours = previous.hours if previous else None
        inherited.append("hours")
        if hours is None:
            ditto_notes.append("Hours are \"-do-\" but the row above had none to repeat — enter them from the paper.")
    elif previous and previous.hours is not None and is_ditto_column("hours") and (hours_marked or _blank_text(hours_raw)):
        # A "-do-" column, and this cell is hidden: a pen mark reads as "-",
        # nothing, or stray digits ("4"). Taken from above, but never quietly.
        hours = previous.hours
        inherited.append("hours")
        hidden = True
        ditto_notes.append("Hours are hidden by a pen mark; taken from the row above (\"-do-\") — check them against the paper.")
    elif hours_stroke:
        # Left unread, not guessed: the credits (a separate cell) are still
        # taken as printed, and no invented "(1 hrs)" reaches the description.
        hours = None
        ditto_notes.append("A pen mark crosses the hours cell, so hours weren't read — check them against the paper.")
    else:
        hours = _normalize_hours(hours_raw)
        if not re.search(r"[HM]", hours_raw.upper()):
            # A plain number ("71"), not "4 HRS & 53 MINS": the free-text read
            # can turn a 7 into a T ("T1" -> 1 hour) where a read told to
            # expect digits does not. Two digit reads that agree outrank one
            # that doesn't; a single dissent, or reads that disagree, change
            # nothing.
            first = " ".join(_ocr_cell(img, box, 7, "0123456789.").strip() for _, box in cells.get("hours") or [])
            second = " ".join(_ocr_cell_plain(img, box, 6, None).strip() for _, box in cells.get("hours") or [])
            digit_reads = (_normalize_hours(first), _normalize_hours(second))
            if digit_reads[0] is not None and digit_reads[0] == digit_reads[1] != hours:
                hours = digit_reads[0]
            # A letter that is no digit's look-alike means the free-text read
            # isn't a reading of these digits: SO 2159's "75" came back "T9",
            # and 9 hours stood because the two digit reads (79, 75) differed.
            stray = re.sub(r"[^A-Z]|[SOQ" + re.escape("".join(_ONE_LOOKALIKES).upper()) + "]", "", hours_raw.upper())
            # Not in a "-do-" column, where stray marks are the pen over a "-do-"
            # and the row above is what settles it (further down).
            stray = stray and not is_ditto_column("hours")
            if stray and digit_reads[0] != digit_reads[1]:
                hours = None
            hour_boxes = [box for _, box in cells.get("hours") or []]
            scaled = _agreed(_scaled_reads(img, hour_boxes, "0123456789."), _normalize_hours)
            thick = _normalize_hours(_thick_read(img, hour_boxes, "0123456789."))
            if scaled is not None and thick == scaled and (_most_hours(period) is None or scaled <= _most_hours(period)):
                # Two ways of reading it apart from the whole-cell reads agree.
                hours, disputed = scaled, None
            else:
                hours, disputed = _settle_hours(hours, digit_reads, scaled, _most_hours(period))
            if hours is None and not disputed and stray:
                options = sorted({v for v in digit_reads if v is not None})
                if len(options) > 1:
                    disputed = " or ".join(f"{v:g}" for v in options)
            if disputed:
                hours_disputed = True
                hour_options = [float(v) for v in re.findall(r"\d+(?:\.\d+)?", disputed)]
                ditto_notes.append(f"OCR disagreed on the hours served ({disputed}) — check them against the paper.")
    credit_boxes = [box for _, box in cells.get("credits") or []]
    credits_confirmed = False   # read alike by every independent way of reading it
    credits_doubt = False       # ...but the whole-cell reads all said otherwise
    credit_wl = "0123456789.,"
    # Read once without the digits whitelist first: under it "-do-" comes
    # back as "0", which would be filled in as zero credits.
    credits_text = read_all("credits", 7, None) if credit_boxes else ""
    credits_stroke = img is not None and credit_boxes and _has_pen_stroke(img, credit_boxes)
    credits_marked = inked is not None and credit_boxes and _has_pen_stroke(inked, credit_boxes)
    if credit_boxes and _DITTO_RE.match(credits_text):
        printed, credits_warning = None, None
        inherited.append("credits")
        if previous and previous.earned:
            printed = float(previous.earned)
        else:
            ditto_notes.append("Credits are \"-do-\" but the row above had none to repeat — enter them from the paper.")
    elif credit_boxes and previous and previous.earned and is_ditto_column("credits") and (credits_marked or _blank_text(credits_text)):
        printed, credits_warning = float(previous.earned), None
        inherited.append("credits")
        hidden = True
        ditto_notes.append("Credits are hidden by a pen mark; taken from the row above (\"-do-\") — check them against the paper.")
    else:
        glyphs = [_glyph_count(img, b) for b in credit_boxes] if img is not None else []
        if sum(glyphs) == 1:
            # One digit ("5"): read cropped to it. Whole-cell reads miss it or
            # guess, and psm 8 over the whole cell read SO 1905's 5s as 8s.
            reads = _lone_digit_reads(img, credit_boxes[glyphs.index(1)])
        else:
            reads = [
                " ".join(_ocr_cell(img, b, 7, credit_wl) for b in credit_boxes),
                " ".join(_ocr_cell(img, b, 6, credit_wl) for b in credit_boxes),
                " ".join(_ocr_cell_plain(img, b, 7, credit_wl) for b in credit_boxes),
            ]
        printed, credits_warning = _consensus_credits(reads, vsc_ceiling(hours))
        if sum(glyphs) != 1:
            scaled = _agreed(_scaled_reads(img, credit_boxes, credit_wl), _normalize_credits)
            thick = _normalize_credits(_thick_read(img, credit_boxes, credit_wl))
            whole = {_normalize_credits(r) for r in reads} - {None}
            if scaled is not None and thick == scaled:
                # Two ways of reading it apart from the whole-cell reads agree:
                # SO 2159's "28.5" read whole as 28.9, 28.9, 28.5. Taken as
                # printed, even above what the hours earn at x1.50 (see
                # _resolve_vsc_credits).
                printed, credits_warning, credits_confirmed = scaled, None, True
                if len(whole) == 1 and scaled not in whole and next(iter(whole)) <= (vsc_ceiling(hours) or float("inf")):
                    # Every whole-cell read says something else. That is how
                    # this font's 5 misreads (its "28.5" under a signature read
                    # 28.9 three times), but it's three reads against three:
                    # filled in, not ticked. Unless the hours rule it out.
                    (other,) = whole
                    credits_doubt = True
                    credits_warning = (
                        f"Credits read as {scaled:g} close up and {other:g} across the whole cell — "
                        f"{scaled:g} is filled in; check it against the paper."
                    )
            else:
                printed, credits_warning = _settle_credits(
                    [_normalize_credits(r) for r in reads], printed, credits_warning, scaled, vsc_ceiling(hours))
        if credits_stroke and not _earned_at_formula(hours, printed):
            # As with the hours: digits under a pen stroke aren't a reading.
            # SO 1898's Gabitano: a "6" crossed by the signature read as "2" on
            # all three passes, and 2 fits under what the hours can earn. Only
            # the hours earning exactly the figure vouch for it — Pedro Roa's
            # signature runs down the credits column beside a clean "22.56"
            # for 120 hours — since no misread lands on that by chance.
            printed = None
            credits_warning = "A pen mark crosses the credits cell, so the credits weren't read — enter them from the paper."
    if hours_disputed and printed is not None and not hidden:
        # The credits settle it when exactly one of the readings earns them at
        # the formula: SO 2159's "75" read as 75 or 79, and 14.06 is 75 hours'
        # 14.0625 to the cent. A misread doesn't land on that by chance.
        earning = [h for h in hour_options if _earns_as_printed(h, printed)]
        if len(earning) == 1:
            hours, hours_disputed = earning[0], False
            ditto_notes = [n for n in ditto_notes if not n.startswith("OCR disagreed on the hours served")]
    ceiling_now = vsc_ceiling(hours)
    if (printed is not None and ceiling_now is not None and printed > ceiling_now and is_ditto_column("hours")
            and previous and previous.hours is not None and previous.earned
            and abs(printed - float(previous.earned)) < 1e-9 and (vsc_ceiling(previous.hours) or 0) >= printed):
        # The credits are the row above's ("-do-") and the hours are not, in a
        # column of "-do-": a pen mark turned this row's "-do-" into stray
        # digits ("4"), and 6 credits can't come from 4 hours. The hours are
        # the misread, not the credits.
        hours = previous.hours
        inherited.append("hours")
        hidden = True
        ditto_notes.append("Hours were misread under a pen mark; taken from the row above (\"-do-\") — check them against the paper.")
    if hours is None and printed is None and not credits_warning and not ditto_notes:
        # No figures at all: a header remnant or a note row, not a grant.
        return []

    # The same two-pass read as a leave row's dates: a whitelist without "/"
    # would turn "02/07/2026" into "02072026".
    dates_raw = _ocr_date_cell(img, cells["dates"][0][1])[0] if cells.get("dates") else ""
    if previous and previous.dates_raw and _DITTO_RE.match(dates_raw):
        # Carried down as the text itself, so a run of "DO"s chains.
        dates_raw = previous.dates_raw
        inherited.append("dates")
    elif (previous and previous.dates_raw and cells.get("dates") and is_ditto_column("dates")
          and not _MONTH_IN_TEXT_RE.search(dates_raw)
          and (_blank_text(dates_raw) or _has_pen_stroke(inked, [box for _, box in cells["dates"]]))):
        dates_raw = previous.dates_raw
        inherited.append("dates")
        hidden = True
        ditto_notes.append("Dates are hidden by a pen mark; taken from the row above (\"-do-\") — check them against the paper.")
    if cells.get("dates") and _MONTH_IN_TEXT_RE.search(dates_raw) and not (previous and dates_raw == previous.dates_raw):
        second = _ocr_cell_plain(img, cells["dates"][0][1], 6, _ABSENCE_DATES_WHITELIST)
        dates_raw, date_from, date_to, date_warning = _choose_dates(dates_raw, second, default_year, period)
    else:
        date_from, date_to, date_warning = _period_from_dates(dates_raw, default_year)
    period_days = (
        (date.fromisoformat(date_to) - date.fromisoformat(date_from)).days + 1 if date_from else None
    )
    earned, warning = _resolve_vsc_credits(hours, printed, period_days, confirmed=credits_confirmed)
    # Granted above what the hours earn at x1.50, and filled in only because
    # every reading agreed: shown, but never ticked without a look.
    above_formula = earned is not None and hours is not None and earned > vsc_ceiling(hours)
    if credits_warning:
        warning = credits_warning if earned is None and not credits_doubt or not warning else f"{credits_warning} {warning}"

    row_no = _normalize_digits(read_all("no", 7, None))
    if not row_no.isdigit():
        row_no = str(seq)

    # Credits refused as more than the hours can earn: the hours are as likely
    # the misread ("1" for "71"), and a wrong "(1 hrs)" would go to HRIS.
    described_hours = None if (printed is not None and earned is None) else hours
    description = _vsc_description(event, date_from, date_to, described_hours)
    position_text = read_all("position", 6, None)
    position_raw = previous.position_raw if previous and _DITTO_RE.match(position_text) else _normalize_position(position_text)
    position_boxes = [box for _, box in cells.get("position") or []]
    if img is not None and len(position_boxes) == 1 and _STEMMED_POSITION_RE.match(position_text.upper().replace(" ", "")):
        stems = _count_stems(img, position_boxes[0])
        if stems is not None and 2 <= stems <= 4:
            position_raw = "T-" + "I" * (stems - 1)

    return [Form6Row(
        row_no=row_no,
        last_name=last_name,
        first_name=first_name,
        middle_initial=middle_initial,
        # psm 6: a grant form wraps "TEACHER / III" onto two lines. "-do-"
        # repeats the row above.
        position_raw=position_raw,
        dates_raw=dates_raw,
        days_raw="",
        leave_type="VSC",
        action_taken="",
        date_from=date_from,
        date_to=date_to,
        description=description,
        used="",
        parse_warning=" ".join(w for w in (date_warning, *ditto_notes, warning) if w) or None,
        # Ticked for submission only when something corroborates the figure:
        # the hours it was checked against, or at least a period that bounded
        # it. A lone number read off a scan with neither isn't pre-ticked.
        # A row with no dates can't be submitted, and one whose figures were
        # taken from above without being seen must be confirmed, not ticked.
        in_scope=(earned is not None and date_from is not None and not hidden and not hours_disputed
                  and not above_formula and not credits_doubt
                  and (hours is not None or period_days is not None)),
        kind="vsc",
        hours=hours,
        earned=_format_used(earned) if earned is not None else "",
        event=event,
        inherited=tuple(inherited),
        trusted=above_formula and not (date_warning or ditto_notes or credits_warning),
        name_raw=name_raw,
    )]


def _extract_mapped_row(
    img: np.ndarray, row_boxes: list[tuple[int, int, int, int]],
    labels: list[str | None], default_year: int | None, seq: int,
    previous: Form6Row | None = None, event: str = "", period: tuple[str, str] | None = None,
    inked: np.ndarray | None = None,
) -> list[Form6Row]:
    """Read one data row using the header's own column labels, whatever order
    the school chose to print them in."""
    cells: dict[str, list[tuple[str, tuple[int, int, int, int]]]] = {}
    for label, box in zip(labels, row_boxes):
        if label:
            cells.setdefault(label, []).append(("", box))

    if cells.get("hours") or cells.get("credits"):
        return _extract_vsc_row(img, cells, default_year, seq, previous, event, period, inked)

    def read(field: str, psm: int, whitelist: str | None, index: int = 0) -> str:
        entries = cells.get(field) or []
        if index >= len(entries):
            return ""
        return _ocr_cell(img, entries[index][1], psm, whitelist).strip()

    # One "name" header spanning three columns is how Last / First / M.I. are
    # recognised; a single one holds "Last, First M.I." all together.
    name_cells = cells.get("name") or []
    if cells.get("last"):
        last_name = _clean_name_part(read("last", 6, None)).title()
        first_name = _clean_name_part(read("first", 6, None)).title()
        middle_initial = _normalize_mi(read("mi", 6, _MI_WHITELIST))
    elif len(name_cells) >= 3:
        last_name = _clean_name_part(read("name", 6, None, 0)).title()
        first_name = _clean_name_part(read("name", 6, None, 1)).title()
        middle_initial = _normalize_mi(read("name", 6, _MI_WHITELIST, 2))
    else:
        name_raw = read("name", 6, None)
        # Some templates put the row number inside the name cell ("1. Cruz, J.").
        m = re.match(r"^\s*(\d{1,2})[.)]\s*(.+)$", name_raw)
        if m:
            name_raw = m.group(2)
        last_name, first_name, middle_initial = _split_combined_name(name_raw)

    if not last_name or (not first_name and "," not in (last_name or "")):
        # Header remnants, a section label ("TEACHING PERSONNEL:"), a blank
        # row — anything without a readable person on it.
        return []

    has_dates_col = bool(cells.get("dates"))
    days_raw = (
        read("days", 6, _COMBINED_DATES_WHITELIST) if not has_dates_col
        else read("days", 7, None)
    )
    date_halves = 0
    if has_dates_col:
        dates_raw, date_halves = _ocr_date_cell(img, cells["dates"][0][1])
    else:
        dates_raw = days_raw

    raw_action = read("action", 6, _WORD_OR_CODE_WHITELIST)
    leave_type, action_taken = _resolve_leave_type(
        read("type", 6, _WORD_OR_CODE_WHITELIST) if cells.get("type") else None,
        raw_action,
    )
    if leave_type not in _LEAVE_TYPE_LABELS and action_taken not in ("WP", "WOP"):
        # Neither column says anything a leave form would say — this is some
        # other table that happens to have people's names in it.
        return []

    used_override: float | None
    if has_dates_col:
        if _MONTH_IN_TEXT_RE.search(dates_raw):
            runs, warning = _parse_date_segments_monthname(dates_raw, default_year)
            segments, per_run_used = [(f, t) for f, t, _ in runs], [u for _, _, u in runs]
        else:
            dates_raw, segments, warning = _read_numeric_dates(img, cells["dates"][0][1], dates_raw)
            per_run_used = []
            if date_halves and len(segments) == 1:
                # "7/7PM/2026": a half day the numeric format can only mark
                # with letters. Counted here so it cross-checks the printed
                # day count instead of silently deferring to it.
                f, t = segments[0]
                per_run_used = [max(0.5, (t - f).days + 1 - 0.5 * date_halves)]
        used_override = _normalize_total_days(days_raw)
    else:
        runs, used_override, warning = _parse_date_segments_combined(days_raw)
        segments, per_run_used = [(f, t) for f, t, _ in runs], [u for _, _, u in runs]

    row_no = _normalize_digits(read("no", 7, None))
    if not row_no.isdigit():
        row_no = str(seq)

    return _rows_from_segments(
        row_no=row_no,
        last_name=last_name,
        first_name=first_name,
        middle_initial=middle_initial,
        position_raw=_normalize_position(read("position", 7, None)),
        dates_raw=dates_raw,
        days_raw=days_raw,
        leave_type=leave_type,
        action_taken=action_taken,
        in_scope=(leave_type == "SL" and action_taken in ("WP", "WOP")),
        segments=segments,
        total_override=used_override,
        unparsed_warning=warning,
        default_used=per_run_used or None,
    )


def _extract_9col_row(img: np.ndarray, row_boxes: list[tuple[int, int, int, int]]) -> list[Form6Row]:
    """The original bordered 9-column layout (No./Last/First/M.I./Position/
    Dates/Days/Type/Action) — also matches the Bongbongon Elementary
    template, which uses the same column layout with spelled-out Leave
    Type/Action text instead of 2-letter codes."""
    raw_values = [_ocr_cell(img, box, psm, wl) for box, (psm, wl) in zip(row_boxes, COLUMN_CONFIGS)]

    row_no = _normalize_digits(raw_values[0])
    if not row_no.isdigit():
        # Not a real numbered data row (header remnants, the "***NF***"
        # marker row, a trailing blank row) — skip it rather than guess.
        return []

    dates_raw = raw_values[5].strip()
    days_raw = raw_values[6].strip()
    leave_type = _normalize_leave_type(raw_values[7])
    action_taken = _normalize_action(raw_values[8])
    segments, warning = _parse_date_segments_numeric(dates_raw)

    return _rows_from_segments(
        row_no=row_no,
        last_name=_clean_name_part(raw_values[1]).title(),
        first_name=_clean_name_part(raw_values[2]).title(),
        middle_initial=_normalize_mi(raw_values[3]),
        position_raw=_normalize_position(raw_values[4]),
        dates_raw=dates_raw,
        days_raw=days_raw,
        leave_type=leave_type,
        action_taken=action_taken,
        in_scope=(leave_type == "SL" and action_taken in ("WP", "WOP")),
        segments=segments,
        total_override=_normalize_total_days(days_raw),
        unparsed_warning=warning,
    )


def _extract_5col_row(
    img: np.ndarray, row_boxes: list[tuple[int, int, int, int]], default_year: int | None, seq: int
) -> list[Form6Row]:
    """The Lumbia Central School layout: combined Employee Name, no
    Position column, and "<Month> D[am|pm],D,..." dates with no row-number
    column of its own — rows are numbered by top-to-bottom position."""
    raw_values = [_ocr_cell(img, box, psm, wl) for box, (psm, wl) in zip(row_boxes, COLUMN_CONFIGS_5COL)]

    name_raw = raw_values[0].strip()
    if len(name_raw) < 3 or "," not in name_raw:
        # Not a real data row (header remnants, a blank/noise row).
        return []

    last_name, first_name, middle_initial = _split_combined_name(name_raw)
    dates_raw = raw_values[1].strip()
    days_raw = raw_values[2].strip()
    leave_type = _normalize_leave_type(raw_values[3])
    action_taken = _normalize_action(raw_values[4])
    segments, warning = _parse_date_segments_monthname(dates_raw, default_year)

    return _rows_from_segments(
        row_no=str(seq),
        last_name=last_name,
        first_name=first_name,
        middle_initial=middle_initial,
        position_raw="",
        dates_raw=dates_raw,
        days_raw=days_raw,
        leave_type=leave_type,
        action_taken=action_taken,
        in_scope=(leave_type == "SL" and action_taken in ("WP", "WOP")),
        segments=[(f, t) for f, t, _ in segments],
        total_override=_normalize_total_days(days_raw),
        unparsed_warning=warning,
        default_used=[u for _, _, u in segments] or None,
    )


def _extract_7col_row(
    img: np.ndarray, row_boxes: list[tuple[int, int, int, int]], default_year: int | None, seq: int
) -> list[Form6Row]:
    """The Lumbia National High School layout: Seq No./Name/Inclusive Date/
    Position/Days/Type/Remarks, with a combined name cell and month-name
    dates (the 5-column date style) rather than MM/DD/YYYY."""
    raw_values = [_ocr_cell(img, box, psm, wl) for box, (psm, wl) in zip(row_boxes, COLUMN_CONFIGS_7COL)]

    name_raw = raw_values[1].strip()
    if len(name_raw) < 3 or "," not in name_raw:
        # Not a real data row (header remnants, a section label such as
        # "TEACHING PERSONNEL:", a blank/noise row). The name cell decides,
        # not the Seq. No. cell — that one is small-print and misreads often,
        # and it would be wrong to drop a whole employee's leave over a
        # cosmetic number that is never sent to HRIS.
        return []

    leave_type = _normalize_leave_type(raw_values[5])
    if leave_type not in _LEAVE_TYPE_LABELS:
        # No recognizable leave type in the column this template reserves for
        # one. Other DepEd forms are also 7-column tables of teacher names
        # (Annex D vacation service credits, for one), and without this they
        # would come through as bogus leave rows.
        return []

    row_no = _normalize_digits(raw_values[0])
    if not row_no.isdigit():
        row_no = str(seq)

    last_name, first_name, middle_initial = _split_combined_name(name_raw)
    dates_raw = raw_values[2].strip()
    days_raw = raw_values[4].strip()
    action_taken = _normalize_action(raw_values[6])
    segments, warning = _parse_date_segments_monthname(dates_raw, default_year)

    return _rows_from_segments(
        row_no=row_no,
        last_name=last_name,
        first_name=first_name,
        middle_initial=middle_initial,
        position_raw=_normalize_position(raw_values[3]),
        dates_raw=dates_raw,
        days_raw=days_raw,
        leave_type=leave_type,
        action_taken=action_taken,
        in_scope=(leave_type == "SL" and action_taken in ("WP", "WOP")),
        segments=[(f, t) for f, t, _ in segments],
        total_override=_normalize_total_days(days_raw),
        unparsed_warning=warning,
        default_used=[u for _, _, u in segments] or None,
    )


def _extract_6col_row(
    img: np.ndarray, row_boxes: list[tuple[int, int, int, int]], default_year: int | None, seq: int
) -> list[Form6Row]:
    """The Masterson ES "Summary of Absences" layout: No./Name/Position/
    Date/s of Absence/Days/Remarks.

    This template prints no leave type — the Remarks column carries only
    "w/ pay" / "w/o pay" — so the type is taken as SL, which is what this
    tool records and what these transmittals are in practice. The draft rows
    still land in the review table with their leave type shown, so anything
    that was really another type can be unticked before submitting.

    That missing column also means the Remarks cell is the only thing
    separating this from any other 6-column table of teacher names, so a row
    whose Remarks doesn't read as with/without pay is dropped.
    """
    raw_values = [_ocr_cell(img, box, psm, wl) for box, (psm, wl) in zip(row_boxes, COLUMN_CONFIGS_6COL)]

    name_raw = raw_values[1].strip()
    if len(name_raw) < 3 or "," not in name_raw:
        # Header remnants, a section label, a blank/noise row.
        return []

    action_taken = _normalize_action(raw_values[5])
    if action_taken not in ("WP", "WOP"):
        return []

    row_no = _normalize_digits(raw_values[0])
    if not row_no.isdigit():
        # Small-print and often misread; it is never sent to HRIS, so fall
        # back to counting position rather than dropping the row.
        row_no = str(seq)

    last_name, first_name, middle_initial = _split_combined_name(name_raw)
    dates_raw = raw_values[3].strip()
    days_raw = raw_values[4].strip()
    segments, warning = _parse_date_segments_monthname(dates_raw, default_year)

    return _rows_from_segments(
        row_no=row_no,
        last_name=last_name,
        first_name=first_name,
        middle_initial=middle_initial,
        position_raw=_normalize_position(raw_values[2]),
        dates_raw=dates_raw,
        days_raw=days_raw,
        leave_type="SL",
        action_taken=action_taken,
        in_scope=True,
        segments=[(f, t) for f, t, _ in segments],
        total_override=_normalize_total_days(days_raw),
        unparsed_warning=warning,
        default_used=[u for _, _, u in segments] or None,
    )


def _extract_3col_row(img: np.ndarray, row_boxes: list[tuple[int, int, int, int]]) -> list[Form6Row]:
    """The Iponan Elementary School layout: just 3 columns (Name / No. of
    Days / Remarks) — the row number lives inside the Name cell text
    ("1. Solidum, Koren Joy C.") rather than its own column, dates/half-day
    markers/day-count are all bundled into one "No. of Days" cell, and
    leave-type+action are a single combined "SL-WP" code under Remarks."""
    raw_values = [_ocr_cell(img, box, psm, wl) for box, (psm, wl) in zip(row_boxes, COLUMN_CONFIGS_3COL)]

    name_cell = raw_values[0].strip()
    m = re.match(r"^(\d+)\.\s*(.+)$", name_cell)
    if not m or "," not in m.group(2):
        # Not a real numbered data row (header remnants, a blank/noise row).
        return []
    row_no, name_raw = m.group(1), m.group(2)

    last_name, first_name, middle_initial = _split_combined_name(name_raw)
    days_cell = raw_values[1].strip()
    leave_type, action_taken = _split_combined_remarks(raw_values[2])
    runs, embedded_total, warning = _parse_date_segments_combined(days_cell)

    return _rows_from_segments(
        row_no=row_no,
        last_name=last_name,
        first_name=first_name,
        middle_initial=middle_initial,
        position_raw="",
        dates_raw=days_cell,
        days_raw=days_cell,
        leave_type=leave_type,
        action_taken=action_taken,
        in_scope=(leave_type == "SL" and action_taken in ("WP", "WOP")),
        segments=[(f, t) for f, t, _ in runs],
        total_override=embedded_total,
        unparsed_warning=warning,
        default_used=[u for _, _, u in runs] or None,
    )


def _load_pages(path: str) -> list[np.ndarray]:
    """Load a Form 6 transmittal as a list of BGR page images — a single
    photo/scan (JPG/PNG/...) is one page; a PDF is rasterized page by page
    at _PDF_RENDER_DPI so the same cell-detection pipeline can run on it."""
    if path.lower().endswith(".pdf"):
        pages = []
        with pymupdf.open(path) as doc:
            zoom = _PDF_RENDER_DPI / 72
            matrix = pymupdf.Matrix(zoom, zoom)
            for page in doc:
                pix = page.get_pixmap(matrix=matrix, colorspace=pymupdf.csRGB)
                arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                pages.append(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
        if not pages:
            raise ValueError("This PDF has no pages.")
        return pages

    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return [img]


# Clockwise degrees, as Tesseract's orientation detection reports them.
_ROTATIONS = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
# Below this Tesseract is guessing. Bulua's upright Form 6 pages read as upside
# down at 2.8-3.5; the sideways orders photographed so far all score above 13.
_MIN_ORIENTATION_CONFIDENCE = 5.0


def _upright(img: np.ndarray) -> np.ndarray:
    """Turn a page photographed sideways or upside down the right way up.

    A phone held in portrait over a landscape-printed order gives a sideways
    table, whose rows then read as columns ("13-column (6 rows)"), so every
    Kauswagan, Macabalan and Taglimao order shot that way was refused.

    Uses Tesseract's orientation detection (the osd language pack, which the
    Windows installer includes). If that isn't installed, or isn't sure, the
    page is left as it is: the reader behaves exactly as it did before."""
    h, w = img.shape[:2]
    scale = min(1.0, 1600 / max(h, w))
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else img
    try:
        osd = _run_tesseract(small, 0, None)
    except ValueError:
        return img
    rotate = re.search(r"Rotate:\s*(\d+)", osd)
    confidence = re.search(r"Orientation confidence:\s*([\d.]+)", osd)
    if not rotate or not confidence or float(confidence.group(1)) < _MIN_ORIENTATION_CONFIDENCE:
        return img
    code = _ROTATIONS.get(int(rotate.group(1)))
    return cv2.rotate(img, code) if code is not None else img


# A page tilted less than this is left exactly as scanned: every sample that
# read correctly before sits under 0.6 degrees, so nothing that worked can
# change. Puerto NHS's SO 1891 tilts 1.5-2.7, and its tables' rules broke into
# fragments.
_MIN_SKEW_DEGREES = 1.0


def _skew_degrees(img: np.ndarray) -> float | None:
    """How far the page is rotated, from its long straight lines (the table's
    rules, the paper's edge): the length-weighted median deviation from
    horizontal/vertical. Positive means the lines run down to the right. None
    when there aren't enough lines to say."""
    h, w = img.shape[:2]
    scale = min(1.0, 1600 / max(h, w))
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else img
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 50, 150)
    sh, sw = gray.shape
    lines = cv2.HoughLinesP(edges, 1, np.pi / 720, threshold=120, minLineLength=int(0.12 * min(sh, sw)), maxLineGap=15)
    if lines is None:
        return None
    devs, lengths = [], []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        deviation = (np.degrees(np.arctan2(y2 - y1, x2 - x1)) + 45) % 90 - 45
        if abs(deviation) <= 8:     # more than that is a diagonal, not a rule
            devs.append(deviation)
            lengths.append(np.hypot(x2 - x1, y2 - y1))
    if len(devs) < 8:
        return None
    order = np.argsort(devs)
    devs, lengths = np.array(devs)[order], np.array(lengths)[order]
    return float(devs[np.searchsorted(np.cumsum(lengths), lengths.sum() / 2)])


def _deskew(img: np.ndarray) -> np.ndarray:
    """Straighten a page tilted more than _MIN_SKEW_DEGREES; leave others be."""
    angle = _skew_degrees(img)
    if angle is None or abs(angle) < _MIN_SKEW_DEGREES:
        return img
    h, w = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))


def _resolve_leave_type(type_raw: str | None, raw_action: str) -> tuple[str, str]:
    """(leave type, action) for one row. `type_raw` is None when the form has
    no Type of Leave column at all.

    Only that case assumes sick leave — a form printing just "w/ pay" means
    sick leave in practice (see the README). A type the form *does* print is
    kept exactly as read, even one this tool doesn't record: relabelling
    "SOLO P" as sick leave would file the wrong leave against someone."""
    action = _normalize_action(raw_action)
    if type_raw is not None:
        return _normalize_leave_type(type_raw), action
    if "-" in raw_action:
        # One Remarks column carrying both, combined ("SL-WP").
        return _split_combined_remarks(raw_action)
    return ("SL" if action in ("WP", "WOP") else ""), action


# ---------------------------------------------------------------------------
# Column grid
#
# Grid-line detection occasionally misses one border, and a row then arrives
# a cell short — an Action cell gone means its WP/WOP is gone and the row
# comes through unticked. The columns themselves sit at the same x on every
# row of a printed table, so the grid those rows agree on is used to put the
# missing cell back.
# ---------------------------------------------------------------------------

Box = tuple[int, int, int, int]


def _column_grid(grouped: list[list[Box]], min_rows: int = 3) -> list[tuple[int, int]] | None:
    """Column x-ranges (left, right) the table's most common row shape agrees
    on, or None when there aren't enough complete rows to trust."""
    counts = Counter(len(r) for r in grouped if len(r) > 1)
    if not counts:
        return None
    width, seen = counts.most_common(1)[0]
    if seen < min_rows:
        return None
    rows = [r for r in grouped if len(r) == width]
    grid = []
    for i in range(width):
        lefts = sorted(r[i][0] for r in rows)
        rights = sorted(r[i][0] + r[i][2] for r in rows)
        grid.append((lefts[len(lefts) // 2], rights[len(rights) // 2]))
    return grid


def _sits_on_grid(row: list[Box], grid: list[tuple[int, int]]) -> bool:
    """Whether a row's cells stand on the table's column rules — at least half
    of them starting or ending where a column does. SO 1898 page 1: the
    signer's pen line joins the table to the page footer's "Doc. Ref. Code |
    Rev" box, so that box lands inside the table; its five cells match none of
    the table's columns, and filled out to the grid they read as a teacher. A
    cell split by a pen stroke keeps one edge on a rule, and a cell spanning
    several columns keeps both."""
    tolerance = max(8, 0.01 * (grid[-1][1] - grid[0][0]))
    lefts = [left for left, _ in grid]
    rights = [right for _, right in grid]
    on_rule = sum(
        min(abs(x - l) for l in lefts) <= tolerance or min(abs(x + w - r) for r in rights) <= tolerance
        for x, _, w, _ in row
    )
    return 2 * on_rule >= len(row)


def _fill_row_to_grid(row: list[Box], grid: list[tuple[int, int]]) -> list[Box]:
    """Put back any cell a short row lost, cut from its column's x-range and
    the row's own height. Rows that already have every column — or more,
    like a header whose one cell spans several — are returned untouched."""
    if len(row) >= len(grid) or len(row) < len(grid) // 2:
        return row
    y = sorted(b[1] for b in row)[len(row) // 2]
    h = sorted(b[3] for b in row)[len(row) // 2]
    filled = list(row)
    for left, right in grid:
        covered = any(
            min(bx + bw, right) - max(bx, left) >= 0.5 * min(bw, right - left)
            for bx, _, bw, _ in row
        )
        if not covered:
            filled.append((left, y, right - left, h))
    return sorted(filled, key=lambda b: b[0])


def _ink_profile(img: np.ndarray, box: Box, pad: int = 6) -> np.ndarray | None:
    """Per-x "is there text here" for one cell, or None for a blank cell.
    The pad keeps the cell's own rules out of it."""
    x, y, w, h = box
    crop = img[y + pad : y + h - pad, x + pad : x + w - pad]
    if crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if gray.std() < 12:
        # Otsu on a blank cell turns paper grain into "ink".
        return None
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Two pixels a column, so a lone speck doesn't count as text.
    return (ink > 0).sum(axis=0) >= 2


def _ink_percent(img: np.ndarray, box: Box, pad: int = 6) -> float:
    """Share of a cell (in percent) that is writing: dark against the cell's
    own paper tone, with the table's rules removed — long straight strokes a
    crooked row can drag into a cell. Paper grain and a rule score about 0."""
    x, y, w, h = box
    crop = img[y + pad : y + h - pad, x + pad : x + w - pad]
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    dark = (gray < np.median(gray) - 60).astype(np.uint8)
    ch, cw = dark.shape
    lines = cv2.morphologyEx(dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, cw // 3), 1)))
    lines |= cv2.morphologyEx(dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(1, int(ch * 0.8)))))
    return float((dark & (1 - lines)).mean()) * 100


def _has_text(img: np.ndarray, box: Box, min_percent: float = 1.0) -> bool:
    """Whether a cell holds writing, not just paper grain or a stray rule.

    Tesseract finds letters in noise: the empty pre-printed rows of a grainy
    Macabalan photo came back as "Nnn, Nnn Eee An"."""
    return _ink_percent(img, box) >= min_percent


# The hours and credits cells of a real grant together hold at least 1.6%
# (SO 1891's "96" and "18", the faintest so far — each cell alone is under
# 1.0); a blank pre-printed row's total is 0.0.
_MIN_FIGURES_INK_PERCENT = 0.8


def _cell_marks(img: np.ndarray, box: Box, pad: int = 6) -> tuple[np.ndarray, np.ndarray] | None:
    """A cell's ink with its rules removed, and the stats of each mark in it
    (cv2.connectedComponentsWithStats, background dropped)."""
    x, y, w, h = box
    crop = img[y + pad : y + h - pad, x + pad : x + w - pad]
    if crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    dark = (gray < np.median(gray) - 60).astype(np.uint8)
    ch, cw = dark.shape
    lines = cv2.morphologyEx(dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, cw // 3), 1)))
    lines |= cv2.morphologyEx(dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(1, int(ch * 0.8)))))
    ink = dark & (1 - lines)
    _, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    return ink, stats[1:]


def _glyph_count(img: np.ndarray, box: Box, pad: int = 6) -> int:
    """How many digit-sized marks a cell holds (rules removed). A cell of one
    lone digit ("5") has one; "6.299" has several, even under a pen stroke."""
    marks = _cell_marks(img, box, pad)
    if marks is None:
        return 0
    ink, stats = marks
    return len(_digit_sized(stats, ink.shape[0]))


def _digit_sized(stats: np.ndarray, cell_height: int) -> list:
    """The marks tall enough to be digits — not a point, a speck, or a rule."""
    return [s for s in stats if 0.2 * cell_height <= s[3] <= 0.95 * cell_height
            and s[2] >= 0.03 * cell_height and s[4] >= 12]


def _lone_digit_reads(img: np.ndarray, box: Box, pad: int = 6) -> list[str]:
    """Three reads of a cell holding one digit, cropped to its ink and scaled
    to a set height. Read whole, a lone digit is a small mark in a wide,
    mostly empty cell, and Tesseract misses it or guesses: SO 1905's clean
    serif "5" came back blank, "9", or "8" on every whole-cell pass. Cropped
    tight and 50px tall it reads right on every one-digit cell of SO 1892 and
    SO 1905 — including Sialana's 5, read as 3 whole. The crop takes the
    digit and only the marks beside it (the point of ".5"), so a speck
    elsewhere in the cell can't shrink the digit."""
    marks = _cell_marks(img, box, pad)
    if marks is None:
        return []
    ink, stats = marks
    digits = _digit_sized(stats, ink.shape[0])
    if not digits:
        return []
    top, bottom = min(s[1] for s in digits), max(s[1] + s[3] for s in digits)
    left, right = min(s[0] for s in digits), max(s[0] + s[2] for s in digits)
    reach = bottom - top
    kept = [s for s in stats if s[1] >= top - 2 and s[1] + s[3] <= bottom + 0.3 * reach
            and s[0] + s[2] >= left - reach and s[0] <= right + reach]
    x0, y0 = min(s[0] for s in kept), min(s[1] for s in kept)
    x1, y1 = max(s[0] + s[2] for s in kept), max(s[1] + s[3] for s in kept)
    digit = np.where(ink[y0:y1, x0:x1] > 0, 0, 255).astype(np.uint8)
    reads = []
    for height, psm in ((50, 8), (36, 8), (50, 10)):
        scale = height / digit.shape[0]
        sized = cv2.resize(digit, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
        _, sized = cv2.threshold(sized, 127, 255, cv2.THRESH_BINARY)
        sized = cv2.copyMakeBorder(sized, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
        reads.append(" ".join(_run_tesseract(sized, psm, "0123456789.,").split()))
    return reads


def _number_crop(img: np.ndarray, box: Box, pad: int = 6) -> np.ndarray | None:
    """A cell's number, cut close and binarized as _ocr_cell binarizes. The
    digits are the marks of the commonest height (by ink), so a pen tick or
    a speck of grain beside them is left out; a point or comma between them
    is kept. The marks only locate the number — the crop is thresholded from
    the greyscale page, since the stricter mark threshold breaks a thin, pale
    "3" into four pieces that read as "2"."""
    marks = _cell_marks(img, box, pad)
    if marks is None:
        return None
    _, stats = marks
    stats = [s for s in stats if s[4] >= 12]
    if not stats:
        return None
    alike = lambda height: [s for s in stats if 0.75 * height <= s[3] <= 1.33 * height]
    digits = alike(max(stats, key=lambda s: sum(t[4] for t in alike(s[3])))[3])
    top, bottom = min(s[1] for s in digits), max(s[1] + s[3] for s in digits)
    left, right = min(s[0] for s in digits), max(s[0] + s[2] for s in digits)
    reach = bottom - top
    kept = [s for s in stats if s[1] >= top - 0.15 * reach and s[1] + s[3] <= bottom + 0.3 * reach
            and s[0] + s[2] >= left - reach and s[0] <= right + reach]
    x0, y0 = min(s[0] for s in kept), min(s[1] for s in kept)
    x1, y1 = max(s[0] + s[2] for s in kept), max(s[1] + s[3] for s in kept)
    cx, cy = box[0] + pad, box[1] + pad
    gray = cv2.cvtColor(img[max(0, cy + y0 - 2) : cy + y1 + 2, max(0, cx + x0 - 2) : cx + x1 + 2], cv2.COLOR_BGR2GRAY)
    _, binarized = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binarized


def _scaled_reads(img: np.ndarray | None, boxes: list[Box], whitelist: str) -> list[str]:
    """Two reads of a number cell, cut close and scaled so the digits are 24
    and 32px tall. Every other read scales the whole cell 3x, whatever size
    its print — SO 1849's thin, pale digits come out 80px tall, and Tesseract
    reads "5.252" as "9.2902" at that size. The same 5-for-9 misread blanked
    or mis-ticked rows of SO 1870, 1897 and 1903 and of Pedro Roa. At
    24-32px they read right. A cell split in two by a pen stroke isn't one
    number to crop, and gets no reads."""
    if img is None or len(boxes) != 1:
        return []
    number = _number_crop(img, boxes[0])
    if number is None or number.size == 0:
        return []
    reads = []
    for height in (24, 32):
        scale = height / number.shape[0]
        sized = cv2.resize(number, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
        _, sized = cv2.threshold(sized, 127, 255, cv2.THRESH_BINARY)
        sized = cv2.copyMakeBorder(sized, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
        reads.append(" ".join(_run_tesseract(sized, 7, whitelist).split()))
    return reads


# "TI", "THI", "TUHI": a T and what OCR made of the strokes after it. No V or
# X — those aren't stems — and no spelled-out "TEACHER".
_STEMMED_POSITION_RE = re.compile(r"^T[-_]?[IHUL1|!\]\[]+$")


def _count_stems(img: np.ndarray, box: Box, pad: int = 6) -> int | None:
    """How many upright strokes a "TIII"-style position cell has — the T's and
    one per I. Serif I's touch, and OCR fuses them into H's and U's: SO 2159's
    TIII came back "THI", "TUHI" and "TII", and its TII as "TI". Counted
    across a band below the T's bar and above where a hyphen sits, every
    stroke is one run of ink, however its serifs join it to the next."""
    x, y, w, h = box
    crop = img[y + pad : y + h - pad, x + pad : x + w - pad]
    if crop.size == 0:
        return None
    _, ink = cv2.threshold(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    marks = [s for s in stats[1:] if s[3] >= 0.3 * crop.shape[0] or s[4] > 40]
    if not marks:
        return None
    tallest = max(s[3] for s in marks)
    glyphs = [s for s in marks if s[3] >= 0.6 * tallest]
    top, bottom = min(s[1] for s in glyphs), max(s[1] + s[3] for s in glyphs)
    left, right = min(s[0] for s in glyphs), max(s[0] + s[2] for s in glyphs)
    height = bottom - top
    band = ink[top + int(0.25 * height) : top + int(0.42 * height) + 1, left : right + 1] > 0
    if band.size == 0:
        return None
    inked = band.mean(axis=0) > 0.5
    return int(inked[0]) + int(np.count_nonzero(inked[1:] & ~inked[:-1]))


def _thick_read(img: np.ndarray | None, boxes: list[Box], whitelist: str) -> str:
    """One more read of a number cell, made a third way: scaled 2x and its
    strokes thickened. Thin serif print (SO 2159's) reads 5 as 9 and 7 as 1
    under the 3x whole-cell reads — and two of those share one binarization,
    so they make the same mistake twice and outvote the right answer. Thick
    strokes keep the 5's open top and the 7's bar: 43 of SO 2159's 44 figures
    read right this way, and the 44th came back empty, not wrong."""
    if img is None or len(boxes) != 1:
        return ""
    x, y, w, h = boxes[0]
    crop = img[y + 4 : y + h - 4, x + 4 : x + w - 4]
    if crop.size == 0:
        return ""
    gray = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, binarized = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thick = cv2.erode(binarized, np.ones((3, 3), np.uint8))     # black ink on white: eroding white thickens it
    thick = cv2.copyMakeBorder(thick, 14, 14, 14, 14, cv2.BORDER_CONSTANT, value=255)
    return " ".join(_run_tesseract(thick, 7, whitelist).split())


def _drop_pen_ink(img: np.ndarray, keep_rules: bool = False) -> np.ndarray:
    """The page with blue pen ink turned to paper.

    Orders are printed in black; what's blue on a colour scan is a pen — the
    signature, initials, the hand-written SO number. SO 2159's signature line
    runs down through four rows' hours and credits ("114", "28.5"), and in
    greyscale it is just more ink: those rows read as "728.9" or not at all.
    By colour the stroke and the print come apart cleanly.

    Blue means blue well above red and green and not dark: in near-black
    print the channels are almost equal, and saturation alone is useless
    there (it's noisy wherever the brightest channel is tiny). A greyscale
    scan has no such pixels and comes back unchanged.

    keep_rules: leave the pen's crossing of a horizontal rule printed black,
    for finding cells. A signature runs down the page, so the long horizontals
    under it are rules and its own strokes never are."""
    if img.ndim != 3:
        return img
    b, g, r = (img[..., i].astype(np.int16) for i in range(3))
    pen = (b - r > 50) & (b - g > 15) & (b > 90)
    if not pen.any():
        return img
    ink = pen
    # The stroke's anti-aliased edge is a shade less blue than its core — but
    # only bluish pixels are its edge. Growing into neutral ones bit the top
    # off the "2" of a "28.5" it crossed, which then read as 28.9.
    pen = (cv2.dilate(pen.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & (b - r > 15)
    out = img.copy()
    out[pen] = 255
    if keep_rules:
        out[_printed_rules(img, out, pen, ink)] = 0
    return out


def _printed_rules(img: np.ndarray, clean: np.ndarray, pen: np.ndarray, ink: np.ndarray) -> np.ndarray:
    """Where the table's rules ran under pen ink that has been removed. Losing
    even a few of those pixels opens a cell's corner: SO 2159's last column
    merged five rows deep where its signature crossed, and SO 1898's "-do-"
    joined the hours beside it where initials were scrawled on the corner.

    Two ways, since each misses what the other finds. A rule segment of the
    page as scanned that is mostly not pen ink is printed, and its pixels under
    the pen are kept — but a thick scrawl thresholds hollow, so the rule
    inside it isn't seen. Then the gaps left between collinear rule segments
    of the cleaned page, where the pen was, are bridged. A signature's
    downstroke is neither — mostly pen, and nothing on either side of a gap —
    so it never becomes a column rule (kept, it cut SO 2159's "75" in half)."""
    h, w = pen.shape
    rules = np.zeros(pen.shape, dtype=bool)
    thr = _binary_threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    for size in ((w // 30, 1), (1, h // 60)):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, size)
        lines = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(lines, connectivity=8)
        if n > 1:
            printed = np.bincount(labels[ink], minlength=n) < 0.5 * stats[:, cv2.CC_STAT_AREA]
            printed[0] = False
            rules |= printed[labels] & pen
    thr = _binary_threshold(cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)) | (rules * 255).astype(np.uint8)
    near_pen = cv2.dilate(pen.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    # Bridged 3px wide: a scanned rule wanders a pixel or two either side. And
    # from shorter segments than a rule is found by: what's left of one between
    # a scrawl and the row's corner can be 30px (SO 1898's Estrella-Turnuas).
    for size, bridge in (((w // 60, 1), (w // 15, 3)), ((1, h // 120), (3, h // 30))):
        lines = cv2.morphologyEx(thr, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, size))
        closed = cv2.morphologyEx(lines, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, bridge))
        rules |= (closed > 0) & near_pen
    return rules

def _has_pen_stroke(img: np.ndarray, boxes: list[Box], pad: int = 4) -> bool:
    """Whether a thick pen stroke crosses the cell (or the cells one split
    into): a mark that runs the cell's full height and is much thicker than
    any printed digit's stroke. A signature's downstroke is drawn straight through the
    rows of these orders, and what Tesseract makes of the digits under it is
    not a reading — "60" came back as "1", "16" as "6"."""
    if not boxes:
        return False
    x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    x1, y1 = max(b[0] + b[2] for b in boxes), max(b[1] + b[3] for b in boxes)
    crop = img[y0 + pad : y1 - pad, x0 + pad : x1 - pad]
    if crop.size == 0:
        return False
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    dark = (gray < np.median(gray) - 70).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    edge = max(6, int(0.03 * crop.shape[1]))
    for i in range(1, n):
        left, top, width, height, _ = stats[i]
        # Edge to edge: a digit is never as tall as its cell, even in a short
        # row (the bold "4.883 HRS" of one scan is 60% of its 38px cell).
        if height < 0.85 * crop.shape[0] or top > 0.1 * crop.shape[0] or top + height < 0.9 * crop.shape[0]:
            continue
        if left <= edge or left + width >= crop.shape[1] - edge:
            # The cell's own rule, crooked or thick — not a mark across it.
            continue
        thickness = cv2.distanceTransform((labels == i).astype(np.uint8), cv2.DIST_L2, 3).max()
        if thickness >= 2.5:
            return True
    return False


def _find_ruleless_split(img: np.ndarray, cells: list[Box]) -> int | None:
    """Where one detected column is really two with no rule printed between
    them — the x of a gap that is blank in most rows and has text on both
    sides of it in at least half. None if there's no such gap.

    "Most", not "every": a signature or sticker crossing the table's last row
    (Kauswagan's one-teacher orders) puts ink in the gap of that row alone."""
    profiles = [(b[0] + 6, p) for b in cells if (p := _ink_profile(img, b)) is not None]
    if not profiles:
        return None
    left = min(x for x, _ in profiles)
    right = max(x + len(p) for x, p in profiles)
    counts = np.zeros(right - left, dtype=int)
    for x, p in profiles:
        counts[x - left : x - left + len(p)] += p
    inked = np.flatnonzero(counts > len(profiles) / 3)
    if len(inked) < 2:
        return None
    blank = counts <= len(profiles) / 3
    best: tuple[int, int] | None = None
    start = None
    for i in range(inked[0], inked[-1] + 1):
        if blank[i] and start is None:
            start = i
        elif not blank[i] and start is not None:
            if best is None or i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    # A word gap ("1 DAY") is narrow; two columns' worth of centred text
    # leaves a wide one.
    if best is None or best[1] - best[0] < max(20, 0.08 * (right - left)):
        return None
    split = left + (best[0] + best[1]) // 2
    both_sides = sum(1 for x, p in profiles if p[: max(0, split - x)].any() and p[max(0, split - x):].any())
    return split if both_sides >= 0.5 * len(profiles) else None


def _split_ruleless_columns(
    img: np.ndarray, grouped: list[list[Box]], grid: list[tuple[int, int]] | None,
) -> tuple[list[list[Box]], list[tuple[int, int]] | None]:
    """Split a column whose header names two fields but that has no rule
    between them — Kauswagan's Annex D prints "No. of Hours Served" and "No.
    Vacation Service Credits Granted" side by side with nothing in between,
    so every row's "16" and "3.000" arrived as one cell and read as 163000
    hours. _map_header_row already splits a merged header along the grid;
    this is for when the grid itself has no line to split along, so the gap
    in the text beneath is used as the rule instead.

    Only a header naming two fields triggers it: a gap alone could be the
    space inside a column of identical "1 DAY" cells."""
    for index, candidate in enumerate(grouped[:3]):
        texts = [_ocr_cell(img, box, 6, None) for box in candidate]
        if not any(_header_field(t) for t in texts):
            continue
        splits = []
        for box, text in zip(candidate, texts):
            if len(_header_fields_all(text)) < 2 or len(_split_header_box(box, grid)) > 1:
                continue
            x, _, w, _ = box
            # The header cell counts too: its two labels have the same gap, and
            # an order for one teacher has only one data row to go on.
            column = [
                cell for row in grouped[index:] for cell in row
                if min(cell[0] + cell[2], x + w) - max(cell[0], x) >= 0.8 * cell[2]
            ]
            split = _find_ruleless_split(img, column)
            if split is not None:
                splits.append(split)
        if not splits:
            return grouped, grid

        def cut(boxes: list[Box]) -> list[Box]:
            out = []
            for bx, by, bw, bh in boxes:
                s = next((s for s in splits if bx + 10 < s < bx + bw - 10), None)
                if s is None:
                    out.append((bx, by, bw, bh))
                else:
                    out.extend([(bx, by, s - bx, bh), (s, by, bx + bw - s, bh)])
            return out

        new_grid = None
        if grid:
            new_grid = []
            for left, right in grid:
                s = next((s for s in splits if left + 10 < s < right - 10), None)
                new_grid.extend([(left, s), (s, right)] if s is not None else [(left, right)])
        return [cut(row) for row in grouped], new_grid
    return grouped, grid


def _split_columns(before: list[tuple[int, int]], after: list[tuple[int, int]]) -> list[int]:
    """Which columns of `before` _split_ruleless_columns cut in two to give `after`."""
    split, j = [], 0
    for i, (_, right) in enumerate(before):
        if j < len(after) and after[j][1] != right:
            split.append(i)
            j += 2
        else:
            j += 1
    return split


def _split_like_layout(
    img: np.ndarray, grouped: list[list[Box]], grid: list[tuple[int, int]], layout: dict,
) -> tuple[list[list[Box]], list[tuple[int, int]]]:
    """Split the columns an earlier page of the order split for having no rule
    between two fields, along this page's own gap in the text. Left as it was
    unless every one of them splits and the result has as many columns as the
    earlier page's labels — a page that doesn't line up is no continuation."""
    ruleless = layout["ruleless"]
    if len(grid) + len(ruleless) != len(layout["labels"]):
        return grouped, grid
    splits = []
    for i in ruleless:
        left, right = grid[i]
        column = [b for row in grouped for b in row if abs(b[0] - left) < 10 and abs(b[0] + b[2] - right) < 10]
        split = _find_ruleless_split(img, column)
        if split is None:
            return grouped, grid
        splits.append(split)

    def cut(boxes: list[Box]) -> list[Box]:
        out = []
        for bx, by, bw, bh in boxes:
            s = next((s for s in splits if bx + 10 < s < bx + bw - 10), None)
            out.extend([(bx, by, bw, bh)] if s is None else [(bx, by, s - bx, bh), (s, by, bx + bw - s, bh)])
        return out

    new_grid = []
    for left, right in grid:
        s = next((s for s in splits if left + 10 < s < right - 10), None)
        new_grid.extend([(left, s), (s, right)] if s is not None else [(left, right)])
    return [cut(row) for row in grouped], new_grid


def _has_vertical_rules(img: np.ndarray, grid: list[tuple[int, int]], y: int, height: int) -> bool:
    """Whether the column rules run through this strip — what makes it a table
    row and not the text below the table. A page's footer ("Address: Fr.
    William F. Masterson Ave...") has ink, but no rules."""
    if len(grid) < 2 or height < 8:
        return False
    strip = cv2.cvtColor(img[y : y + height, :], cv2.COLOR_BGR2GRAY)
    if strip.size == 0:
        return False
    dark = strip < np.median(strip) - 70
    boundaries = [right for _, right in grid[:-1]]
    with_rule = 0
    for x in boundaries:
        band = dark[:, max(0, x - 5) : x + 6]
        # A rule is dark down (nearly) the whole strip in at least one column.
        if band.size and band.mean(axis=0).max() >= 0.6:
            with_rule += 1
    return with_rule >= max(1, int(0.6 * len(boundaries)))


def _extend_open_bottom(img: np.ndarray, grouped: list[list[Box]], grid: list[tuple[int, int]]) -> list[list[Box]]:
    """Recover rows below the last closed cell of a table whose bottom rules
    weren't printed or didn't scan. Kauswagan's Annex D runs its column rules
    down past rows 7-9 with no horizontal line under any of them, so no cell
    closes there and those three teachers vanished.

    The dewarped page ends where the table's rules do, so whatever height is
    left below the last row is table. It's cut into rows at the pitch the
    closed rows keep, and a strip is kept only if it has text in it *and* the
    column rules run through it: East Gusa's page footer sits inside the
    dewarped area, and was read as a row."""
    rows = [r for r in grouped if len(r) >= 2]
    if len(rows) < 3:
        return grouped
    centers = [sorted(b[1] + b[3] / 2 for b in r)[len(r) // 2] for r in rows]
    pitch = sorted(b - a for a, b in zip(centers, centers[1:]))[(len(centers) - 1) // 2]
    last = rows[-1]
    bottom = sorted(b[1] + b[3] for b in last)[len(last) // 2]
    remaining = img.shape[0] - bottom
    if pitch <= 0:
        return grouped
    n = int((remaining + 0.35 * pitch) // pitch)
    if n < 1:
        return grouped
    step = remaining / n
    added = []
    for i in range(n):
        y = int(bottom + i * step)
        row = [(left, y, right - left, int(step)) for left, right in grid]
        if sum(_ink_profile(img, b) is not None for b in row) >= 2 and _has_vertical_rules(img, grid, y, int(step)):
            added.append(row)
    return grouped + added


# ---------------------------------------------------------------------------
# Content-based column inference
#
# A continuation page repeats the table but not its header, and reading the
# header is how columns get their meaning. Rather than guess a layout from
# how many columns there are, look at what each column actually holds: a
# column of "7/3/2026" is dates, "1 DAY" is a day count, "SL" or "SOLO P" is
# the leave type, "WP" is the action. That needs no knowledge of any
# particular school's layout, so a headerless page of a new form reads too.
# ---------------------------------------------------------------------------

_NUMERIC_DATE_RE = re.compile(r"\d{1,2}\s*[/\[\]{}()|\\]\s*\d{1,2}.*20\d{2}")
_DAYS_CELL_RE = re.compile(r"^\s*(?:\d+(?:\.\d)?|[|lI!]|half|½)\s*days?\b", re.IGNORECASE)
_POSITION_RE = re.compile(r"^(?:SPET|MT|HT|T|TEACHER|MASTER\s*TEACHER)\s*[-_]?\s*[IVX1L|!]+$", re.IGNORECASE)


def _classify_cell(text: str) -> str | None:
    """What kind of value one cell holds, judged only by its text."""
    t = " ".join(text.split())
    if not t:
        return None
    if re.search(r"\d\s*(?:HRS?|HOURS?|MIN(?:UTE)?S?)\b", t, re.IGNORECASE):
        return "hours"
    if re.fullmatch(r"\d{1,3}[.,]\d{2,3}", t):
        return "credits"
    if _DAYS_CELL_RE.search(t):
        return "days"
    if _NUMERIC_DATE_RE.search(t) or (_MONTH_IN_TEXT_RE.search(t) and re.search(r"\d", t)):
        return "dates"
    if "," in t and len(re.findall(r"[A-Za-z]{2,}", t)) >= 2:
        return "name"
    if _POSITION_RE.match(t.replace(" ", "")):
        return "position"
    if _normalize_leave_type(t) in _LEAVE_TYPE_LABELS:
        return "type"
    if len(t) <= 12 and _normalize_action(t) in ("WP", "WOP"):
        return "action"
    if re.fullmatch(r"[\dlI|]{1,3}[.)]?", t):
        return "no"
    # A name with no comma ("Kenneth Joh Conde", "Krezel Marie M. Salva") —
    # the order a grant form prints. Only after position, type and action have
    # had their turn, since "Teacher I" and "Sick Leave" are also two words.
    words = re.findall(r"[A-Za-z]{2,}", t)
    if len(words) >= 2 and not re.search(r"\d", t) and len(re.sub(r"[A-Za-z .'-]", "", t)) == 0:
        return "name"
    return None


def _labels_from_samples(columns: list[list[str]]) -> list[str | None] | None:
    """Label each column from sample texts of its cells (one list per column,
    left to right). A column takes the kind most of its readable cells agree
    on. None when the result doesn't describe a leave table — no person and
    no date means this isn't one, and guessing would be worse than the fixed
    fallback layouts."""
    labels: list[str | None] = []
    for samples in columns:
        kinds = [k for k in (_classify_cell(s) for s in samples) if k]
        if not kinds:
            labels.append(None)
            continue
        kind, votes = Counter(kinds).most_common(1)[0]
        labels.append(kind if votes * 2 >= len(kinds) else None)
    # A grant's hours column is a plain whole number, which reads just like a
    # row number. When there's a credits column, the leftmost number column is
    # the row number and any other is the hours.
    if "credits" in labels:
        numbered = [i for i, label in enumerate(labels) if label == "no"]
        for i in numbered[1:]:
            labels[i] = "hours"
    # A grant whose hours and credits are both whole numbers ("40", "5" — East
    # Gusa's Brigada Eskwela) has no decimal to give the credits away, so all
    # three number columns read alike. The first is the row number; the two
    # after the dates are hours and then credits, in the order every grant form
    # prints them. Not for a leave form: any days, type or action column means
    # the numbers are something else.
    if "dates" in labels and not {"credits", "hours", "days", "type", "action"} & set(labels):
        numbered = [i for i, label in enumerate(labels) if label == "no"]
        after_dates = [i for i in numbered if i > labels.index("dates")]
        if len(numbered) >= 3 and len(after_dates) >= 2:
            labels[after_dates[0]], labels[after_dates[1]] = "hours", "credits"
    # A grant with a School column (SO 1893: "MACASANDIG NHS") has two columns of
    # capitalised words side by side. The first is the teacher; reading both
    # as the name glued the school onto every teacher ("Arquero Macasandig
    # Nhs, Ma. Dorothy...") and no HRIS search finds that. Only for a grant:
    # a leave form's names come split by their headers, not by content.
    if {"hours", "credits"} & set(labels):
        name_columns = [i for i, label in enumerate(labels) if label == "name"]
        if len(name_columns) == 2 and name_columns[1] == name_columns[0] + 1:
            labels[name_columns[1]] = None
    found = set(labels)
    if "name" not in found or not found & {"dates", "days"}:
        return None
    return labels


def _infer_header_from_content(img: np.ndarray, grouped: list[list[Box]], grid: list[tuple[int, int]]):
    """Build a header mapping for a page with no readable header row, in the
    same shape _map_header_row returns, from a sample of its data rows."""
    rows = [r for r in grouped if len(r) == len(grid)]
    if len(rows) < 2:
        return None
    # A spread of rows, not just the first few: one odd row shouldn't decide.
    step = max(1, len(rows) // 6)
    sample = rows[::step][:6]
    columns = [[_ocr_cell(img, row[i], 6, None).strip() for row in sample] for i in range(len(grid))]
    labels = _labels_from_samples(columns)
    if not labels:
        return None
    header_boxes = [(left, 0, right - left, 1) for left, right in grid]
    mapping = {box[0]: label for box, label in zip(header_boxes, labels) if label}
    return mapping, header_boxes


def _header_from_labels(labels: list[str | None], grid: list[tuple[int, int]]):
    """A header mapping, in _map_header_row's shape, from one label per column."""
    header_boxes = [(left, 0, right - left, 1) for left, right in grid]
    return {box[0]: label for box, label in zip(header_boxes, labels) if label}, header_boxes


def _labels_of(header, grid: list[tuple[int, int]]) -> list[str | None]:
    """The label a header gives each grid column."""
    return _fields_for_row(header[0], header[1], [(left, 0, right - left, 1) for left, right in grid])


def _doubtful(header, grid: list[tuple[int, int]] | None) -> bool:
    """Whether labels read from a page's content can't be trusted: none at all,
    or two columns both called the name (a school beside the teacher)."""
    if header is None:
        return True
    return grid is not None and [l for l in _labels_of(header, grid) if l == "name"].__len__() > 1


def _rebuild_rows(boxes: list[Box], grid: list[tuple[int, int]]) -> list[list[Box]]:
    """Rows for a page whose own grid fell apart, from its number column.

    SO 1893's last page has a signature scrawled across its five rows: cells
    are cut into fragments, and the right-hand column has no closing border at
    the page's edge, so no row has all its columns. The number column, though,
    stays clean. Each of its cells gives a row's height, and the columns come
    from an earlier page of the same order, whose layout this page shares.
    Only used when the page has no header of its own."""
    left, right = grid[0]
    width = right - left
    numbers = sorted(
        (b for b in boxes if abs(b[0] - left) <= 0.25 * width and 0.6 * width <= b[2] <= 1.4 * width and b[3] >= 30),
        key=lambda b: b[1],
    )
    rows: list[Box] = []
    for b in numbers:
        if rows and b[1] < rows[-1][1] + 0.5 * rows[-1][3]:
            continue        # the same row cut twice
        rows.append(b)
    return [[(l, b[1], r - l, b[3]) for l, r in grid] for b in rows]


def _extract_page(
    img: np.ndarray, seq_start: int, earlier_event: str = "", context: dict | None = None,
) -> tuple[list[Form6Row], int, Counter[int]]:
    """Run the full per-page pipeline (table detection through row
    extraction) on one page image. Returns (rows, next seq_start, counts of
    rows no layout could read) — seq is the running row-number counter for
    layouts with no row-number column, threaded across pages so numbering
    stays continuous in a multi-page PDF.

    Columns get their meaning from, in order: the page's own header row; the
    content of its cells, for a continuation page with no header; and only
    then the fixed layouts this tool knew before either. Nothing is carried
    over from an earlier page — every page is dewarped to its own size and
    offset, so column positions from one don't line up with the next.

    `context` carries what the caller and the page pass to each other, without
    widening the return: in, "period" (the order's stated period, from an
    earlier page); out, "period" (this page's, if it states one) and "notes"
    (things the reader saw and skipped, for the user)."""
    context = context if context is not None else {}
    context.setdefault("notes", [])
    page_text = _page_text(img)
    default_year = _detect_default_year(page_text)
    event = _event_from_text(page_text)
    period = _period_from_text(page_text, default_year)
    if not re.search(r"hereby\s+granted|rendered|special\s+order", page_text, re.IGNORECASE):
        # A continuation page repeats the table but not the paragraph naming
        # the event and period. One that has its own order paragraph is a
        # different order, though, and never borrows: an unread event beats a
        # wrong one.
        event = event or earlier_event
        period = period or context.get("period")
    if default_year is None and period:
        # A continuation page may print no year anywhere, and SO 2159's dates
        # ("5/6-8,11-15") carry none of their own: the order's year is theirs.
        default_year = int(period[0][:4])
    context["period"] = period
    # Everything is read with pen ink removed (_drop_pen_ink). The cells are
    # found on a copy that keeps the rules under it: where SO 2159's signature
    # crossed a row's bottom rule, removing it all merged the last column into
    # one cell five rows tall — and keeping it all made its downstroke a
    # column rule, cutting Ecobiza's "75" in half.
    inked, lines, warped = _dewarp_table(img, _drop_pen_ink(img, keep_rules=True), _drop_pen_ink(img))
    boxes = _find_cell_boxes(lines)
    grouped = [sorted(row, key=lambda b: b[0]) for row in _group_into_rows(boxes)]

    grid = _column_grid(grouped)
    if grid:
        grouped = [_fill_row_to_grid(row, grid) for row in grouped if _sits_on_grid(row, grid)]
    # Needs no grid: a one-teacher order has too few rows to agree on one.
    unsplit_grid = grid
    grouped, grid = _split_ruleless_columns(warped, grouped, grid)
    layout = context.get("layout")
    if grid and grid == unsplit_grid and layout and layout.get("ruleless"):
        # A continuation page has no header to say a column is two fields, but
        # the first page's did — SO 2159's page 2 printed "76  14.252" in one
        # cell on every row and read nothing.
        grouped, grid = _split_like_layout(warped, grouped, grid, layout)
    if grid:
        grouped = _extend_open_bottom(warped, grouped, grid)

    # Read the column labels off the form itself. Only the first few rows are
    # considered — a header is at the top, and a data row that happens to
    # contain a word shouldn't be mistaken for one further down.
    header = None
    # A short page (a header and two teachers) has too few rows to trust a grid
    # for reading cells, but two are enough to say where a merged header —
    # "No. of Hours Served" and "No. Vacation Service Credits" — divides.
    header_grid = grid or _column_grid(grouped, min_rows=2)
    own_header = False
    for candidate in grouped[:3]:
        header = _map_header_row(warped, candidate, header_grid)
        if header:
            own_header = True
            break
    if header is None and grid:
        header = _infer_header_from_content(warped, grouped, grid)
        if layout and len(layout["labels"]) == len(grid) and _doubtful(header, grid):
            # A continuation page of columns of "-do-" says nothing about what
            # they are; the first page's header did.
            header = _header_from_labels(layout["labels"], grid)
    elif header is None and layout and not grid:
        scale = warped.shape[1] / layout["width"]
        carried = [(round(l * scale), round(r * scale)) for l, r in layout["grid"]]
        rebuilt = _rebuild_rows(boxes, carried)
        if rebuilt:
            grouped, grid = rebuilt, carried
            header = _header_from_labels(layout["labels"], grid)
    if own_header and grid:
        # What later pages of this order can use when they have no header.
        context["layout"] = {
            "labels": _labels_of(header, grid), "grid": grid, "width": warped.shape[1],
            "ruleless": _split_columns(unsplit_grid, grid) if unsplit_grid else [],
        }

    results: list[Form6Row] = []
    seq = seq_start
    unmatched: Counter[int] = Counter()
    previous: Form6Row | None = context.get("previous")
    for row_boxes in grouped:
        if header and len(row_boxes) > 1.5 * len(header[1]):
            # Several rows chained into one: a crooked photo's rows overlap in
            # height, and grouping by height links them. Reading it would mix
            # teachers' cells, so it is reported as unreadable instead.
            unmatched[len(row_boxes)] += 1
            continue
        if header and not any(_has_text(warped, box) for box in row_boxes[1:]):
            # A blank line — pre-numbered forms print a number and nothing else.
            continue
        if header:
            labels = _fields_for_row(header[0], header[1], row_boxes)
            figures = [box for box, label in zip(row_boxes, labels) if label in ("hours", "credits")]
            if figures and sum(_ink_percent(warped, box) for box in figures) < _MIN_FIGURES_INK_PERCENT:
                # Every grant states its figures, even if only as "-do-". A
                # row with neither is a blank line of a pre-printed form,
                # whatever its name cell's paper grain OCR'd as.
                listed = _listed_name(" ".join(
                    _ocr_cell(warped, box, 6, None) for box, label in zip(row_boxes, labels) if label == "name"
                ))
                if listed:
                    # A name, and nothing else: on the paper, not in the table.
                    context["notes"].append(f"{listed} is listed with no hours or credits, so isn't in the review table.")
                continue
            seq += 1
            row_results = _extract_mapped_row(warped, row_boxes, labels, default_year, seq, previous, event, period, inked)
        # No readable header (a cropped photo, a table with no header row on
        # this page): fall back to recognising the handful of layouts this
        # tool knew before headers were read, purely by column count.
        elif len(row_boxes) == len(COLUMN_CONFIGS):
            row_results = _extract_9col_row(warped, row_boxes)
        elif len(row_boxes) == len(COLUMN_CONFIGS_7COL):
            seq += 1
            row_results = _extract_7col_row(warped, row_boxes, default_year, seq)
        elif len(row_boxes) == len(COLUMN_CONFIGS_6COL):
            seq += 1
            row_results = _extract_6col_row(warped, row_boxes, default_year, seq)
        elif len(row_boxes) == len(COLUMN_CONFIGS_5COL):
            seq += 1
            row_results = _extract_5col_row(warped, row_boxes, default_year, seq)
        elif len(row_boxes) == len(COLUMN_CONFIGS_3COL):
            row_results = _extract_3col_row(warped, row_boxes)
        else:
            row_results = []

        if row_results:
            results.extend(row_results)
            if row_results[-1].kind == "vsc":
                row = row_results[-1]
                if previous is None or not row.parse_warning or row.trusted:
                    # Only a row that read cleanly is the row above: a flagged
                    # one may hold a misread ("4" for "-do-"), and every "-do-"
                    # below would repeat it. Which columns are "-do-" columns
                    # is remembered across rows.
                    seen = tuple(sorted(set(row.inherited) | set(previous.inherited if previous else ())))
                    previous = replace(row, inherited=seen)
        elif len(row_boxes) > 1:
            # A row nothing could read: an unlabelled layout, a header row, or
            # a different DepEd form that happens to be a table of names.
            # Counted so the caller can say so instead of silently reporting
            # "0 rows" — only ever read when the document yielded nothing.
            unmatched[len(row_boxes)] += 1

    _apply_period(results, context["period"])
    _repair_row_numbers(results)
    context["previous"] = previous
    return results, seq, unmatched


_LISTED_NAME_RE = re.compile(r"[A-Z][A-Za-z.'\- ]+,[A-Za-z.,'\- ]{3,}")


def _listed_name(text: str) -> str | None:
    """The name in a cell of a row that has no figures — "SALALIMA, RALPHA." —
    or None when the text isn't a name. A comma is required: paper grain read
    as letters ("Nnn eee an NNN") has none, and a pre-printed blank row must
    not be reported as a teacher."""
    text = " ".join(text.split())
    return text if _LISTED_NAME_RE.fullmatch(text) else None


def _repair_row_numbers(rows: list[Form6Row]) -> None:
    """A row number that breaks an otherwise unbroken count is a misread —
    Kauswagan's serif 5 reads as 9 under every OCR mode, between a 4 and a 6.
    Cosmetic (row numbers never reach HRIS), but a second "9" on a nine-row
    order makes a correct read look like a duplicate. One form row can yield
    several Form6Rows sharing its number, so runs are compared, not rows."""
    runs: list[list[Form6Row]] = []
    for row in rows:
        # A leave row can yield several records sharing its number; a grant
        # never does, so two grants in a row with one number are a misread.
        if runs and runs[-1][0].row_no == row.row_no and row.kind != "vsc":
            runs[-1].append(row)
        else:
            runs.append([row])
    for before, run, after in zip(runs, runs[1:], runs[2:]):
        prev_no, no, next_no = before[0].row_no, run[0].row_no, after[0].row_no
        if prev_no.isdigit() and next_no.isdigit() and int(next_no) == int(prev_no) + 2 and no != str(int(prev_no) + 1):
            for row in run:
                row.row_no = str(int(prev_no) + 1)


def _flag_passes_that_disagree(kept: list[Form6Row], other: list[Form6Row]) -> None:
    """Two reads of one page (as scanned, and straightened) are two independent
    readings. Where both gave a figure for a teacher and the figures differ,
    neither can be trusted: SO 1892's "5" was unreadable as scanned and read as
    "3" by every pass once straightened — silently wrong, the worst kind.
    Rows are matched by number and the start of the surname; a figure one pass
    couldn't read at all isn't a disagreement."""
    theirs = {(r.row_no, r.last_name[:3].lower()): r for r in other if r.kind == "vsc"}
    for row in kept:
        match = theirs.get((row.row_no, row.last_name[:3].lower())) if row.kind == "vsc" else None
        if not match:
            continue
        differ = []
        if row.earned and match.earned and abs(float(row.earned) - float(match.earned)) > 1e-6:
            differ.append(f"credits ({row.earned} or {match.earned})")
        if row.hours is not None and match.hours is not None and row.hours != match.hours:
            differ.append(f"hours ({row.hours:g} or {match.hours:g})")
        if row.date_from and match.date_from and (row.date_from, row.date_to) != (match.date_from, match.date_to):
            differ.append(f"dates ({_format_period(row.date_from, row.date_to)} or {_format_period(match.date_from, match.date_to)})")
        if differ:
            note = f"Two readings of this page disagree on the {', '.join(differ)} — check against the paper."
            row.parse_warning = f"{row.parse_warning} {note}" if row.parse_warning else note
            row.in_scope = False


# The fewest rows sharing one grant that make it the order's norm.
_ORDER_NORM_MIN_ROWS = 5


def _flag_rows_unlike_their_order(rows: list[Form6Row]) -> None:
    """Most orders grant most of their teachers the same thing: SO 1898 is 32
    hours for 6 credits on 109 of 111 rows. Where most rows of an order (rows
    sharing an event) agree on one pair, a row with the same hours and other
    credits has a misread — 32 hours can't earn 6 credits for one teacher and
    2 for the next, whatever the office's method.

    The same credits for other hours is a misread only when the order's pair
    is exactly the formula: SO 1898's 6 for 32 hours can't also come from 34.
    Offices that cap or cut their grants give the same figure for different
    hours — SO 1892 grants 5 for 40 hours and, rightly, 5 for Paican's 32.

    A row matching neither figure is a different grant (24 hours for 4.5) and
    is left alone, as is a row with either figure unread. Nothing is changed,
    only unticked and explained: SO 1898's Aniñon read as 2 credits and Aluro
    as 34 hours, both believable on their own and both ticked for submission."""
    orders: dict[str, list[Form6Row]] = {}
    for row in rows:
        if row.kind == "vsc" and row.hours is not None and row.earned:
            orders.setdefault(row.event, []).append(row)
    for order in orders.values():
        pairs = Counter((round(r.hours, 3), float(r.earned)) for r in order)
        (hours, credits), count = pairs.most_common(1)[0]
        if count < _ORDER_NORM_MIN_ROWS or 2 * count <= len(order):
            continue
        at_formula = _earned_at_formula(hours, credits)
        for row in order:
            same_hours = round(row.hours, 3) == hours
            same_credits = abs(float(row.earned) - credits) < 1e-9
            if same_hours == same_credits or (same_credits and not at_formula):
                continue
            note = (
                f"{count} of {len(order)} rows in this order are {hours:g} hrs for {credits:g} credits, "
                f"but this one reads {row.hours:g} hrs for {row.earned} — one of the two is likely misread; "
                "check both against the paper."
            )
            row.parse_warning = f"{row.parse_warning} {note}" if row.parse_warning else note
            row.in_scope = False


# The share of an order's names with a comma that makes it a "Last, First" order.
_COMMA_NAMES_SHARE = 0.75


# Words that begin a Filipino compound surname: "DELA VEGA", "DE LOS SANTOS".
_SURNAME_PARTICLES = {"DE", "DEL", "DELA", "DELOS", "LA", "LOS", "SAN", "STA", "STA.", "STO", "STO."}


def _surname_comma(tokens: list[str]) -> str:
    """A comma-less "LAST FIRST M." name with its comma put back. A period
    typed for the comma marks it ("DELA VEGA. ARA NICOLE BACONG", SO 1892) —
    after a word, not an initial or a suffix; otherwise the surname is the
    first word, with any particles that start it ("DELA VEGA")."""
    for i, token in enumerate(tokens[:-1]):
        word = token.rstrip(".")
        if token.endswith(".") and len(word) >= 3 and word.upper() not in _NAME_SUFFIXES | _SURNAME_PARTICLES:
            return f"{' '.join(tokens[:i])} {word}, {' '.join(tokens[i + 1:])}".strip()
    n = 1
    while n < len(tokens) - 1 and tokens[n - 1].upper() in _SURNAME_PARTICLES:
        n += 1
    return f"{' '.join(tokens[:n])}, {' '.join(tokens[n:])}"


def _surname_first_like_the_rest(rows: list[Form6Row]) -> list[str]:
    """Re-read a grant's name printed without its comma surname first, when
    the rest of its order is "LAST, FIRST M." — SO 2159's "OLARTE ROY PHILIP"
    among 21 such names was read "First Last", as a name with no comma
    usually is on a grant, and became Philip, Olarte Roy. Says so in a note,
    since only the paper (or HRIS) settles which word is the surname."""
    orders: dict[str, list[Form6Row]] = {}
    for row in rows:
        if row.kind == "vsc" and row.name_raw:
            orders.setdefault(row.event, []).append(row)
    notes = []
    for order in orders.values():
        commaless = [r for r in order if "," not in r.name_raw]
        if len(order) < 4 or len(order) - len(commaless) < _COMMA_NAMES_SHARE * len(order):
            continue
        for row in commaless:
            tokens = row.name_raw.split()
            if len(tokens) < 2:
                continue
            last, first, mi = _split_combined_name(_surname_comma(tokens))
            if not first:
                continue
            row.last_name, row.first_name, row.middle_initial = last, first, mi
            notes.append(
                f"Row {row.row_no}: \"{' '.join(tokens)}\" has no comma; read surname first "
                f"({last}, {first}) like the rest of the order — check it when matching to HRIS."
            )
    return notes


def _page_score(rows: list[Form6Row]) -> tuple[int, int]:
    """How well a page read: rows with no warning first, then rows at all."""
    return sum(1 for r in rows if not r.parse_warning), len(rows)


def _enforce_increasing_numbers(rows: list[Form6Row]) -> None:
    """Within one order, a grant's number that doesn't go up is a misread.

    SO 1893's last rows sit under the signature: 50-57 read as "30, 1, 2, 3,
    4, 5, 06" — the number column is cosmetic (it never reaches HRIS) but a
    review table numbered 30, 1, 2 looks like duplicates. A row that doesn't
    exceed the one above takes that plus one. Scoped to one order (rows sharing
    an event) so two orders in a PDF that each restart at 1 are left alone."""
    last: Form6Row | None = None
    for row in rows:
        if row.kind != "vsc" or not row.row_no.isdigit():
            continue
        if last is not None and last.event == row.event and int(row.row_no) <= int(last.row_no):
            row.row_no = str(int(last.row_no) + 1)
        last = row


# A page yielding nothing, or losing most of its rows, is reported rather than
# skipped. A few dropped rows are normal — the header, a blank line — so only
# more than that counts.
_DROPPED_ROWS_WORTH_NOTING = 3


def extract_form6_with_notes(image_path: str) -> tuple[list[Form6Row], list[str]]:
    """extract_form6, plus notes on anything the reader couldn't read that the
    user needs to know about.

    The rows alone can't say what's missing. A continuation page whose table
    didn't detect cleanly used to contribute nothing at all to an otherwise
    successful upload — ten teachers' grants, gone without a word, because the
    "couldn't read this table" error only fires when the whole document
    yields nothing."""
    pages = _load_pages(image_path)

    results: list[Form6Row] = []
    notes: list[str] = []
    seq = 0
    errors: list[str] = []
    unmatched: Counter[int] = Counter()
    several_tables: list[tuple[str, int]] = []
    failed_pages: list[int] = []
    event = ""
    period: tuple[str, str] | None = None
    # What one page of an order passes to the next: its column layout, and its
    # last grant (for "-do-" cells that open a page).
    carry: dict = {}
    for i, img in enumerate(pages, start=1):
        img = _upright(img)
        where = f"Page {i}" if len(pages) > 1 else "This page"
        tables = _count_table_regions(img)
        if tables > 1:
            several_tables.append((where, tables))
        try:
            def attempt(image, seq=seq, event=event, period=period, carry=carry):
                ctx: dict = {"period": period, "notes": [], **carry}
                try:
                    return _extract_page(image, seq, event, ctx), None, ctx
                except ValueError as e:
                    return None, e, ctx

            first, first_error, context = attempt(img)
            poor = first is None or not first[0] or bool(first[2]) or any(r.parse_warning for r in first[0])
            straight = _deskew(img) if poor else img
            if straight is not img:
                # A tilted page's rules break into fragments (SO 1891-1893 tilt
                # 1.5-3.2 degrees): nothing may read at all, or a cropped table
                # may read partly (SO 1892's credits column, cut to 68px).
                # Straighten it and read again, and keep whichever gives more
                # rows without a warning. Only when the first read was poor,
                # because straightening resamples every digit — a page that
                # read cleanly must keep reading exactly as it did.
                second, _, second_context = attempt(straight)
                if second and second[0]:
                    if first is None or not first[0]:
                        first, first_error, context = second, None, second_context
                    else:
                        if _page_score(second[0]) > _page_score(first[0]):
                            (first, first_error, context), other = (second, None, second_context), first
                        else:
                            other = second
                        _flag_passes_that_disagree(first[0], other[0])
            if first is None:
                raise first_error
            page_results, seq, page_unmatched = first
        except ValueError as e:
            # A page with no detectable table (e.g. a PDF cover/signature
            # page) shouldn't abort a multi-page document — skip it, but
            # surface the reason if it turns out no page had a table at all.
            errors.append(f"page {i}: {e}")
            failed_pages.append(i)
            continue
        unmatched.update(page_unmatched)
        results.extend(page_results)
        event = next((r.event for r in reversed(page_results) if r.event), event)
        period = context["period"] or period
        carry = {k: context[k] for k in ("layout", "previous") if context.get(k)} or carry
        notes.extend(f"{where}: {n}" for n in context["notes"])

        dropped = sum(page_unmatched.values())
        if tables > 1:
            notes.append(
                f"{where} shows {tables} separate tables — sheets laid over one another? "
                "Only one was read, so rows from the others are missing. "
                "Photograph each page flat on its own and upload them separately."
            )
        if not page_results and dropped >= _DROPPED_ROWS_WORTH_NOTING:
            notes.append(
                f"{where}: found a table with {dropped} rows but couldn't read any of them. "
                "Nothing from this page is in the review table — enter its rows by hand."
            )
        elif page_results and dropped >= _DROPPED_ROWS_WORTH_NOTING:
            notes.append(
                f"{where}: {dropped} table rows couldn't be read and aren't in the review table. "
                "Count this page's rows against the paper."
            )

    # Per page it can't see a number at a page's edge, whose neighbour is on the
    # next page. Harmless twice over: it only fixes a number breaking a count.
    _repair_row_numbers(results)
    _enforce_increasing_numbers(results)
    _flag_rows_unlike_their_order(results)
    notes.extend(_surname_first_like_the_rest(results))
    if results and failed_pages:
        # Not silently: a page whose table can't be found is teachers missing.
        # (A cover or signature page has no table either, and costs only this
        # one line.)
        notes.extend(
            f"Page {n}: no table could be found on it, so nothing from it is in the review table."
            for n in failed_pages
        )
    if not results:
        if several_tables:
            where, tables = several_tables[0]
            raise ValueError(
                f"{where} shows {tables} separate tables, overlapping or at different angles, "
                "and they couldn't be read together. Photograph each page flat on its own, "
                "with the whole table in frame, and upload them separately."
            )
        if unmatched:
            layouts = ", ".join(f"{n}-column ({c} rows)" for n, c in sorted(unmatched.items()))
            raise ValueError(
                f"Found a table, but its layout isn't one this tool knows how to read: {layouts}. "
                "The supported Form 6 templates have 3, 5, 6, 7, or 9 columns."
            )
        if errors:
            raise ValueError("; ".join(errors))
    return results, notes


def extract_form6(image_path: str) -> list[Form6Row]:
    """Extract every data row from a Form 6 transmittal — or a vacation
    service credit grant — into draft Form6Row objects. Accepts a photo/scan
    or a PDF, whose pages are each processed in order. See
    extract_form6_with_notes for what couldn't be read."""
    return extract_form6_with_notes(image_path)[0]


def _demo() -> None:
    import sys

    if len(sys.argv) < 2:
        print("Usage: python ocr_form6.py <path-to-form6-image>")
        sys.exit(1)
    rows, notes = extract_form6_with_notes(sys.argv[1])
    for note in notes:
        print(f"[NOTE] {note}")
    for r in rows:
        if r.kind == "vsc":
            warn = f"  [!] {r.parse_warning}" if r.parse_warning else ""
            print(
                f"{r.row_no:>3} {r.last_name}, {r.first_name} {r.middle_initial} "
                f"({r.position_raw}) — {r.description} — earned={r.earned} [VSC]{warn}"
            )
            continue
        flag = "" if r.in_scope else "  (skipped — not SL/WP/WOP)"
        warn = f"  [!] {r.parse_warning}" if r.parse_warning else ""
        print(
            f"{r.row_no:>3} {r.last_name}, {r.first_name} {r.middle_initial} "
            f"({r.position_raw}) — {r.description} — used={r.used} "
            f"[{r.leave_type}/{r.action_taken}]{flag}{warn}"
        )


if __name__ == "__main__":
    _demo()
