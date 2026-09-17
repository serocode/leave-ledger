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
from dataclasses import dataclass, field
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


def _dewarp_table(img: np.ndarray) -> np.ndarray:
    """Find the table's outer border and perspective-correct it flat."""
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
    return cv2.warpPerspective(img, matrix, (width, height))


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
    cleaned = re.sub(r"[^0-9/\-,&]", "", raw.strip()).strip("-,")
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
            default_month = int(m1_str)
            segments.append(_parse_day_token(rest1, default_month, year))
            for chunk in chunks[1:]:
                if "/" in chunk:
                    m_str, token = chunk.split("/", 1)
                    segments.append(_parse_day_token(token, int(m_str), year))
                else:
                    segments.append(_parse_day_token(chunk, default_month, year))
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
    raw = _disambiguate_ampersands(raw)
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
    return _ocr_cell(img, box, 6, _DATE_WHITELIST), halves


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
HOURS_PER_DAY = 8

_NAME_SUFFIXES = {"JR", "SR", "II", "III", "IV", "V"}


# Where the event's name ends: "on"/"from" followed by a date ("on JUNE 12-13",
# "from May 6") or "held" ("HELD AT FR. WILLIAM F. MASTERSON ..."). A bare "on"
# isn't enough: "Orientation on the Strengthened SHS" is part of the name. The
# date's month isn't matched by name, since OCR misspells it ("Janauary").
_EVENT_END_RE = re.compile(r"\s+(?:(?:on|from)\s+[A-Za-z]{3,}\.?\s*\d|held\b)", re.IGNORECASE)
# "during", including OCR'd remnants of it: Pedro Roa's came back "duri : 3 uring".
_EVENT_START_RE = re.compile(r"\b(?:d?uring)\s+", re.IGNORECASE)
_EVENT_FILLER_RE = re.compile(r"^(?:(?:the|participation\s+in|conduct\s+of)\s+)+", re.IGNORECASE)
_EVENT_SMALL_WORDS = {"A", "AN", "AND", "AT", "BY", "CUM", "FOR", "IN", "OF", "ON", "OR", "THE", "TO", "WITH"}


def _event_from_text(text: str) -> str:
    """The event a grant was earned for, from the order's paragraph: "...
    for services rendered during DIVISION TRAINING FOR THE STRENGTHENED
    SENIOR HIGH SCHOOL CURRICULUM on JUNE 12-13, 2026". Empty when the page
    has no such sentence (a continuation page) or it didn't OCR cleanly."""
    flat = " ".join(re.sub(r"-\s*\n\s*", "", text).split())   # rejoin "ser-\nvice"
    anchor = re.search(r"rendered|service\s+credits?", flat, re.IGNORECASE)
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
    if not 3 <= len(event) <= 200:
        return ""
    return _event_case(event)


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


def vsc_days(hours: float, factor: float = VSC_CONVERSION_FACTOR) -> float:
    """VSC days = hours x factor / 8, to the 3 decimals the ledger keeps."""
    return round(hours * factor / HOURS_PER_DAY, 3)


# The most credit one calendar day can possibly earn: 24 hours at x1.50.
MAX_VSC_DAYS_PER_CALENDAR_DAY = 24 * VSC_CONVERSION_FACTOR / HOURS_PER_DAY


def _resolve_vsc_credits(
    hours: float | None, printed: float | None, period_days: int | None = None,
) -> tuple[float | None, str | None]:
    """_resolve_vsc_credits_unbounded, then refused outright if the result is
    more than the period could physically earn. A lost decimal point turns
    7.21 into 721 — a figure that flagging alone would still leave prefilled
    and ticked for submission."""
    earned, warning = _resolve_vsc_credits_unbounded(hours, printed)
    if earned is not None and period_days:
        ceiling = period_days * MAX_VSC_DAYS_PER_CALENDAR_DAY
        if earned > ceiling:
            return None, (
                f"Credits read as {earned:g} — more than a {period_days}-day period can earn "
                f"(at most {ceiling:g} even at 24 hours a day). Likely a misread; enter it from the paper."
            )
    return earned, warning


def _resolve_vsc_credits_unbounded(hours: float | None, printed: float | None) -> tuple[float | None, str | None]:
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
    than one way (_consensus_credits)."""
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
    if printed > ceiling:
        # Not filled in, even flagged: no office can grant more than the hours
        # earn, so this is a misread — and a prefilled misread stays ticked.
        return None, (
            f"Credits granted read as {printed:g} — more than {hours:g} hours can earn at "
            f"x{VSC_CONVERSION_FACTOR:.2f} ({vsc_days(hours):g}). Likely a misread; enter it from the paper."
        )
    return printed, None


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
        segments, warning = _parse_date_segments_numeric(raw)
        dates = [d for f, t in segments for d in (f, t)]
    if not dates:
        return None, None, warning or f"Could not parse dates from {raw!r} — enter manually."
    return min(dates).isoformat(), max(dates).isoformat(), None


def _extract_vsc_row(
    img: np.ndarray, cells: dict[str, list[tuple[str, tuple[int, int, int, int]]]],
    default_year: int | None, seq: int, previous: Form6Row | None = None, event: str = "",
) -> list[Form6Row]:
    """previous: the grant row above, for "-do-" cells to inherit from.
    event: what the page's paragraph says the grants were earned for."""
    def read_all(field: str, psm: int, whitelist: str | None) -> str:
        # A pen tick or margin line through a column can split one cell into
        # two boxes; reading them left to right keeps the whole value.
        return " ".join(
            _ocr_cell(img, box, psm, whitelist).strip() for _, box in cells.get(field) or []
        ).strip()

    last_name, first_name, middle_initial = _split_any_name(read_all("name", 6, None))
    if not last_name or not first_name:
        return []

    ditto_notes = []
    # psm 6: "31 HOURS & 37 MINUTES" wraps over three lines in its cell.
    hours_raw = read_all("hours", 6, None)
    if _DITTO_RE.match(hours_raw):
        hours = previous.hours if previous else None
        if hours is None:
            ditto_notes.append("Hours are \"-do-\" but the row above had none to repeat — enter them from the paper.")
    else:
        hours = _normalize_hours(hours_raw)
    credit_boxes = [box for _, box in cells.get("credits") or []]
    credit_wl = "0123456789.,"
    # Read once without the digits whitelist first: under it "-do-" comes
    # back as "0", which would be filled in as zero credits.
    if credit_boxes and _DITTO_RE.match(read_all("credits", 7, None)):
        printed, credits_warning = None, None
        if previous and previous.earned:
            printed = float(previous.earned)
        else:
            ditto_notes.append("Credits are \"-do-\" but the row above had none to repeat — enter them from the paper.")
    else:
        printed, credits_warning = _consensus_credits([
            " ".join(_ocr_cell(img, b, 7, credit_wl) for b in credit_boxes),
            " ".join(_ocr_cell(img, b, 6, credit_wl) for b in credit_boxes),
            " ".join(_ocr_cell_plain(img, b, 7, credit_wl) for b in credit_boxes),
        ], vsc_ceiling(hours))
    if hours is None and printed is None and not credits_warning and not ditto_notes:
        # No figures at all: a header remnant or a note row, not a grant.
        return []

    # The same two-pass read as a leave row's dates: a whitelist without "/"
    # would turn "02/07/2026" into "02072026".
    dates_raw = _ocr_date_cell(img, cells["dates"][0][1])[0] if cells.get("dates") else ""
    if previous and previous.dates_raw and _DITTO_RE.match(dates_raw):
        # Carried down as the text itself, so a run of "DO"s chains.
        dates_raw = previous.dates_raw
    date_from, date_to, date_warning = _period_from_dates(dates_raw, default_year)
    period_days = (
        (date.fromisoformat(date_to) - date.fromisoformat(date_from)).days + 1 if date_from else None
    )
    earned, warning = _resolve_vsc_credits(hours, printed, period_days)
    if credits_warning:
        warning = credits_warning if earned is None or not warning else f"{credits_warning} {warning}"

    row_no = _normalize_digits(read_all("no", 7, None))
    if not row_no.isdigit():
        row_no = str(seq)

    description = _vsc_description(event, date_from, date_to, hours)

    return [Form6Row(
        row_no=row_no,
        last_name=last_name,
        first_name=first_name,
        middle_initial=middle_initial,
        # psm 6: a grant form wraps "TEACHER / III" onto two lines.
        position_raw=_normalize_position(read_all("position", 6, None)),
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
        in_scope=earned is not None and (hours is not None or period_days is not None),
        kind="vsc",
        hours=hours,
        earned=_format_used(earned) if earned is not None else "",
        event=event,
    )]


def _extract_mapped_row(
    img: np.ndarray, row_boxes: list[tuple[int, int, int, int]],
    labels: list[str | None], default_year: int | None, seq: int,
    previous: Form6Row | None = None, event: str = "",
) -> list[Form6Row]:
    """Read one data row using the header's own column labels, whatever order
    the school chose to print them in."""
    cells: dict[str, list[tuple[str, tuple[int, int, int, int]]]] = {}
    for label, box in zip(labels, row_boxes):
        if label:
            cells.setdefault(label, []).append(("", box))

    if cells.get("hours") or cells.get("credits"):
        return _extract_vsc_row(img, cells, default_year, seq, previous, event)

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


def _column_grid(grouped: list[list[Box]]) -> list[tuple[int, int]] | None:
    """Column x-ranges (left, right) the table's most common row shape agrees
    on, or None when there aren't enough complete rows to trust."""
    counts = Counter(len(r) for r in grouped if len(r) > 1)
    if not counts:
        return None
    width, seen = counts.most_common(1)[0]
    if seen < 3:
        return None
    rows = [r for r in grouped if len(r) == width]
    grid = []
    for i in range(width):
        lefts = sorted(r[i][0] for r in rows)
        rights = sorted(r[i][0] + r[i][2] for r in rows)
        grid.append((lefts[len(lefts) // 2], rights[len(rights) // 2]))
    return grid


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


def _has_text(img: np.ndarray, box: Box, min_percent: float = 1.0, pad: int = 6) -> bool:
    """Whether a cell holds writing, not just paper grain or a stray rule.

    Tesseract finds letters in noise: the empty pre-printed rows of a grainy
    Macabalan photo came back as "Nnn, Nnn Eee An". Dark pixels are counted
    against the cell's own paper tone, after removing long straight strokes —
    the table's rules, which a crooked row can drag into its cells."""
    x, y, w, h = box
    crop = img[y + pad : y + h - pad, x + pad : x + w - pad]
    if crop.size == 0:
        return False
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    dark = (gray < np.median(gray) - 60).astype(np.uint8)
    ch, cw = dark.shape
    lines = cv2.morphologyEx(dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, cw // 3), 1)))
    lines |= cv2.morphologyEx(dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(1, int(ch * 0.8)))))
    return float((dark & (1 - lines)).mean()) * 100 >= min_percent


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


def _extend_open_bottom(img: np.ndarray, grouped: list[list[Box]], grid: list[tuple[int, int]]) -> list[list[Box]]:
    """Recover rows below the last closed cell of a table whose bottom rules
    weren't printed or didn't scan. Kauswagan's Annex D runs its column rules
    down past rows 7-9 with no horizontal line under any of them, so no cell
    closes there and those three teachers vanished.

    The dewarped page ends where the table's rules do, so whatever height is
    left below the last row is table. It's cut into rows at the pitch the
    closed rows keep, and a strip is kept only if it has text in it."""
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
        if sum(_ink_profile(img, b) is not None for b in row) >= 2:
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


def _extract_page(img: np.ndarray, seq_start: int, earlier_event: str = "") -> tuple[list[Form6Row], int, Counter[int]]:
    """Run the full per-page pipeline (table detection through row
    extraction) on one page image. Returns (rows, next seq_start, counts of
    rows no layout could read) — seq is the running row-number counter for
    layouts with no row-number column, threaded across pages so numbering
    stays continuous in a multi-page PDF.

    Columns get their meaning from, in order: the page's own header row; the
    content of its cells, for a continuation page with no header; and only
    then the fixed layouts this tool knew before either. Nothing is carried
    over from an earlier page — every page is dewarped to its own size and
    offset, so column positions from one don't line up with the next."""
    page_text = _page_text(img)
    default_year = _detect_default_year(page_text)
    event = _event_from_text(page_text)
    if not event and not re.search(r"hereby\s+granted|rendered|special\s+order", page_text, re.IGNORECASE):
        # A continuation page repeats the table but not the paragraph naming
        # the event. One that has its own order paragraph is a different order,
        # though, and never borrows: an unread event beats a wrong one.
        event = earlier_event
    warped = _dewarp_table(img)
    boxes = _find_cell_boxes(warped)
    grouped = [sorted(row, key=lambda b: b[0]) for row in _group_into_rows(boxes)]

    grid = _column_grid(grouped)
    if grid:
        grouped = [_fill_row_to_grid(row, grid) for row in grouped]
    # Needs no grid: a one-teacher order has too few rows to agree on one.
    grouped, grid = _split_ruleless_columns(warped, grouped, grid)
    if grid:
        grouped = _extend_open_bottom(warped, grouped, grid)

    # Read the column labels off the form itself. Only the first few rows are
    # considered — a header is at the top, and a data row that happens to
    # contain a word shouldn't be mistaken for one further down.
    header = None
    for candidate in grouped[:3]:
        header = _map_header_row(warped, candidate, grid)
        if header:
            break
    if header is None and grid:
        header = _infer_header_from_content(warped, grouped, grid)

    results: list[Form6Row] = []
    seq = seq_start
    unmatched: Counter[int] = Counter()
    previous: Form6Row | None = None
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
            if figures and not any(_has_text(warped, box) for box in figures):
                # Every grant states its figures, even if only as "-do-". A
                # row with neither is a blank line of a pre-printed form,
                # whatever its name cell's paper grain OCR'd as.
                continue
            seq += 1
            row_results = _extract_mapped_row(warped, row_boxes, labels, default_year, seq, previous, event)
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
                previous = row_results[-1]
        elif len(row_boxes) > 1:
            # A row nothing could read: an unlabelled layout, a header row, or
            # a different DepEd form that happens to be a table of names.
            # Counted so the caller can say so instead of silently reporting
            # "0 rows" — only ever read when the document yielded nothing.
            unmatched[len(row_boxes)] += 1

    _repair_row_numbers(results)
    return results, seq, unmatched


def _repair_row_numbers(rows: list[Form6Row]) -> None:
    """A row number that breaks an otherwise unbroken count is a misread —
    Kauswagan's serif 5 reads as 9 under every OCR mode, between a 4 and a 6.
    Cosmetic (row numbers never reach HRIS), but a second "9" on a nine-row
    order makes a correct read look like a duplicate. One form row can yield
    several Form6Rows sharing its number, so runs are compared, not rows."""
    runs: list[list[Form6Row]] = []
    for row in rows:
        if runs and runs[-1][0].row_no == row.row_no:
            runs[-1].append(row)
        else:
            runs.append([row])
    for before, run, after in zip(runs, runs[1:], runs[2:]):
        prev_no, no, next_no = before[0].row_no, run[0].row_no, after[0].row_no
        if prev_no.isdigit() and next_no.isdigit() and int(next_no) == int(prev_no) + 2 and no != str(int(prev_no) + 1):
            for row in run:
                row.row_no = str(int(prev_no) + 1)


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
    event = ""
    for i, img in enumerate(pages, start=1):
        img = _upright(img)
        where = f"Page {i}" if len(pages) > 1 else "This page"
        tables = _count_table_regions(img)
        if tables > 1:
            several_tables.append((where, tables))
        try:
            page_results, seq, page_unmatched = _extract_page(img, seq, event)
        except ValueError as e:
            # A page with no detectable table (e.g. a PDF cover/signature
            # page) shouldn't abort a multi-page document — skip it, but
            # surface the reason if it turns out no page had a table at all.
            errors.append(f"page {i}: {e}")
            continue
        unmatched.update(page_unmatched)
        results.extend(page_results)
        event = next((r.event for r in reversed(page_results) if r.event), event)

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
