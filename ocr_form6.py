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
        if area < 1200 or area > 0.05 * (w * h):
            continue
        if ww < 20 or hh < 15:
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
    return cleaned


_LEAVE_TYPE_LABELS = {"SL": "Sick leave", "PL": "Paternity leave", "ML": "Maternity leave", "VL": "Vacation leave"}


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


def _parse_date_segments_monthname(
    raw: str, default_year: int | None
) -> tuple[list[tuple[date, date, float]], str | None]:
    """Parse a "<Month> D[am|pm],D,D..." Inclusive Dates cell (seen on the
    Lumbia Central School template — no numeric month, and a trailing
    "am"/"pm" on a day marks it a half-day). Some rows spell out a trailing
    ", <year>" (e.g. "June 26, 2026") even though most just rely on the
    page header's year — either way the year itself is stripped before day
    tokens are read, so it's never mistaken for two more days ("20", "26").

    Consecutive calendar days are grouped into one contiguous (date_from,
    date_to, used) run each; non-adjacent days become separate runs, e.g.
    "June 8pm,9,10,11,22,23" -> [(Jun 8, Jun 11, 3.5), (Jun 22, Jun 23, 2.0)].
    """
    fail = ([], f"Could not parse dates from {raw!r} — enter manually.")
    text = raw.strip()
    # Anchor on the first month name rather than the start of the cell: the
    # dates always begin at the month, so any leading characters are OCR
    # noise (e.g. a descender bleeding down from the row above).
    m = next(
        (
            mm
            for mm in re.finditer(r"\b([A-Za-z]+)\b", text)
            if mm.group(1).lower() in _MONTH_NAME_TO_NUM
        ),
        None,
    )
    if not m:
        return fail
    month = _MONTH_NAME_TO_NUM[m.group(1).lower()]
    if default_year is None:
        return [], f"Could not determine the year for {raw!r} — enter manually."

    rest = text[m.end():]
    year_m = re.search(r"\b(20\d{2})\b", rest)
    if year_m:
        rest = rest[: year_m.start()] + rest[year_m.end() :]

    tokens = re.findall(r"(\d{1,2})\s*(am|pm)?", rest, re.IGNORECASE)
    days = [(int(d), bool(half)) for d, half in tokens]
    if not days:
        return fail

    runs: list[tuple[date, date, float]] = []
    run_start = run_end = days[0][0]
    run_half_count = 1 if days[0][1] else 0
    run_len = 1
    for day, is_half in days[1:]:
        if day == run_end + 1:
            run_end = day
            run_len += 1
            run_half_count += 1 if is_half else 0
        else:
            runs.append(_finish_run(month, default_year, run_start, run_end, run_len, run_half_count))
            run_start = run_end = day
            run_len = 1
            run_half_count = 1 if is_half else 0
    runs.append(_finish_run(month, default_year, run_start, run_end, run_len, run_half_count))

    try:
        return [(f, t, u) for f, t, u in runs], None
    except ValueError:
        return [], f"Invalid date parsed from {raw!r} — enter manually."


_ABSENCE_TOKEN_RE = re.compile(
    r"(?P<year>20\d{2})"
    r"|(?P<word>[A-Za-z]+)"
    r"|(?P<d1>\d{1,2})\s*(?:\(\s*(?P<h1>am|pm)\s*\)?)?"
    r"(?:\s*-\s*(?P<d2>\d{1,2})\s*(?:\(\s*(?P<h2>am|pm)\s*\)?)?)?",
    re.IGNORECASE,
)


def _parse_date_segments_absence(
    raw: str, default_year: int | None
) -> tuple[list[tuple[date, date, float]], str | None]:
    """Parse a free-form "Date/s of Absence" cell (the Masterson ES
    template): "June 30, 2026", "July 1-3 & 6-7, 2026", "June 29(pm), 2026",
    "June 30, 2026; July 2(pm), 2026".

    Unlike the other month-name parser this one tracks the month as it scans,
    so a cell may span more than one, and it expands "D-D" ranges rather than
    reading their endpoints as two separate days. The year is usually printed
    *after* the days it belongs to, so days are held pending until a year
    token arrives (falling back to the page header's year at the end).

    Consecutive calendar days become one contiguous (date_from, date_to,
    used) run each, with an "(am)"/"(pm)" day counting as half.
    """
    fail = ([], f"Could not parse dates from {raw!r} — enter manually.")
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

    for m in _ABSENCE_TOKEN_RE.finditer(raw):
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


def _finish_run(month: int, year: int, start_day: int, end_day: int, length: int, half_count: int) -> tuple[date, date, float]:
    return date(year, month, start_day), date(year, month, end_day), length - 0.5 * half_count


_HALF_WORD_RE = re.compile(r"half", re.IGNORECASE)
_HALF_SYMBOL_RE = re.compile(r"[½]|\b1\s*/\s*2\b")


def _normalize_total_days(raw: str) -> float | None:
    """Parse a "No. of Days" cell into a float: "1 day" -> 1.0, "5 ½ days"
    -> 5.5, "½ day" -> 0.5, "Half" -> 0.5, "6" -> 6.0."""
    s = raw.strip()
    if not s:
        return None

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


def _detect_default_year(img: np.ndarray) -> int | None:
    """Some templates (e.g. Lumbia) never print a year in the dates column
    itself, only once in the page header ("For the Month of June 2026").
    OCR the raw page (not the dewarped table) once to recover it."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
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
    warning = None
    if trust_default or len(segments) > 1:
        seg_used = default_used
        computed_total = sum(seg_used)
        if total_override is not None and abs(computed_total - total_override) > 0.01:
            warning = (
                f"Computed total ({_format_used(computed_total)} days) doesn't match the "
                f"form's printed total ({_format_used(total_override)} days) — double-check dates."
            )
    else:
        seg_used = [total_override if total_override is not None else default_used[0]]

    rows = []
    for (df, dt), used in zip(segments, seg_used):
        row = Form6Row(
            date_from=df.isoformat(), date_to=dt.isoformat(),
            used=_format_used(used), parse_warning=warning,
            **common,
        )
        row.description = _build_description(leave_type, df.isoformat(), dt.isoformat())
        rows.append(row)
    return rows


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
    segments, warning = _parse_date_segments_absence(dates_raw, default_year)

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


def _extract_page(img: np.ndarray, seq_start: int) -> tuple[list[Form6Row], int, Counter[int]]:
    """Run the full per-page pipeline (table detection through row
    extraction) on one page image. Returns (rows, next seq_start, counts of
    rows whose column count matched no template) — seq is
    the running row-number counter for the row-number-less 5-column layout,
    threaded across pages so numbering stays continuous in a multi-page PDF."""
    default_year = _detect_default_year(img)
    warped = _dewarp_table(img)
    boxes = _find_cell_boxes(warped)
    grouped = _group_into_rows(boxes)

    results: list[Form6Row] = []
    seq = seq_start
    unmatched: Counter[int] = Counter()
    for row_boxes in grouped:
        row_boxes = sorted(row_boxes, key=lambda b: b[0])
        if len(row_boxes) == len(COLUMN_CONFIGS):
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
        elif len(row_boxes) > 1:
            # Either a column layout no template covers, or one that matched
            # a template by column count but that the template then rejected
            # (a header row, or a different DepEd form that happens to have
            # the same number of columns). Counted either way so the caller
            # can say so instead of silently reporting "0 rows" — it is only
            # ever read when the whole document yielded nothing.
            unmatched[len(row_boxes)] += 1

    return results, seq, unmatched


def extract_form6(image_path: str) -> list[Form6Row]:
    """Extract every data row from a Form 6 transmittal into draft Form6Row
    objects. Accepts a photo/scan (JPG, PNG, ...) or a PDF — a PDF's pages
    are each rasterized and processed the same way, in order. Does not
    filter by leave type/action — use `row.in_scope` (True for Type of
    Leave == SL and Action Taken in {WP, WOP}, matching this project's
    standing sick-leave-entry convention) or filter yourself in the caller.

    A non-contiguous leave cell (e.g. "June 8-11, 22-23") becomes multiple
    Form6Row entries, one per contiguous date range — a leave record can
    only carry a single start/end date, so this is the only way to keep
    every real date range submittable on its own.

    Supports five table layouts, dispatched purely by how many columns a
    detected row has (3, 5, 6, 7 or 9) — see _extract_3col_row and friends.
    """
    pages = _load_pages(image_path)

    results: list[Form6Row] = []
    seq = 0
    errors: list[str] = []
    unmatched: Counter[int] = Counter()
    for i, img in enumerate(pages, start=1):
        try:
            page_results, seq, page_unmatched = _extract_page(img, seq)
            unmatched.update(page_unmatched)
        except ValueError as e:
            # A page with no detectable table (e.g. a PDF cover/signature
            # page) shouldn't abort a multi-page document — skip it, but
            # surface the reason if it turns out no page had a table at all.
            errors.append(f"page {i}: {e}")
            continue
        results.extend(page_results)

    if not results:
        if unmatched:
            layouts = ", ".join(f"{n}-column ({c} rows)" for n, c in sorted(unmatched.items()))
            raise ValueError(
                f"Found a table, but its layout isn't one this tool knows how to read: {layouts}. "
                "The supported Form 6 templates have 3, 5, 6, 7, or 9 columns."
            )
        if errors:
            raise ValueError("; ".join(errors))
    return results


def _demo() -> None:
    import sys

    if len(sys.argv) < 2:
        print("Usage: python ocr_form6.py <path-to-form6-image>")
        sys.exit(1)
    rows = extract_form6(sys.argv[1])
    for r in rows:
        flag = "" if r.in_scope else "  (skipped — not SL/WP/WOP)"
        warn = f"  [!] {r.parse_warning}" if r.parse_warning else ""
        print(
            f"{r.row_no:>3} {r.last_name}, {r.first_name} {r.middle_initial} "
            f"({r.position_raw}) — {r.description} — used={r.used} "
            f"[{r.leave_type}/{r.action_taken}]{flag}{warn}"
        )


if __name__ == "__main__":
    _demo()
