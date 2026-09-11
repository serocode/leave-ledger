# HRIS Leave Credit Tool (local)

A small local app so you don't have to click through the HRIS website (or
ask an AI to) every time you record a sick leave entry.

It runs entirely on your own computer. Your HRIS password is typed into a
form on `http://127.0.0.1:5057` (a server running on your machine), which
forwards it directly to the real HRIS login endpoint. It is never sent to
me, Anthropic, or anywhere else, and is never written to disk.

## Install and run

You need [Python 3.9 or newer](https://www.python.org/downloads/). On
Windows, tick **"Add python.exe to PATH"** in the installer. Everything else
the setup script handles for you.

**macOS / Linux** — open a terminal and run:

```
git clone https://github.com/serocode/leave-ledger.git
cd leave-ledger
./setup.sh          # one time
./run.sh            # every time you want the tool
```

**Windows** — download the repo (Code → Download ZIP, then unzip; or
`git clone https://github.com/serocode/leave-ledger.git`), open the folder,
then:

1. Double-click **`setup.bat`** — one time only.
2. Double-click **`run.bat`** — whenever you want the tool.

Either way your browser opens at `http://127.0.0.1:5057`. Close the terminal
window (or press Ctrl+C in it) to stop the server.

### What setup does

The script checks your Python version, creates a `.venv` and installs the
packages into it, then checks for the **Tesseract OCR engine** and offers to
install it (Homebrew on macOS, winget on Windows). Tesseract is only needed
to read Form 6 scans — everything else works without it, so you can skip that
step and add it later.

If you'd rather do it by hand: `pip install -r requirements.txt`, plus
`brew install tesseract` (macOS), `sudo apt install tesseract-ocr` (Linux),
or the [UB-Mannheim build](https://github.com/UB-Mannheim/tesseract/wiki)
(Windows). The app finds Tesseract in its standard Windows install location
even if it isn't on your PATH.

## Using it

Log in with your HRIS credentials. The tool has three tabs:

- **Form 6 OCR Batch Upload** — upload a scanned transmittal, review the draft
  rows it reads, and submit them in one go.
- **Individual Employee Search & Record** — look someone up by surname to see
  their Service Credit balance and ledger, or add a single record by hand.
- **Without-Pay Audit (WOP)** — find and repair records filed under "used"
  that should have been without pay.

## Current status: submission is live

Adding a record on the "Add sick leave record" form now actually creates
the record in HRIS (`SUBMIT_ENABLED = True` in `app.py`). You'll get a
browser confirmation prompt showing the description/dates/used value before
anything is sent — nothing goes through on an accidental click.

This was confirmed against a real captured request on 2026-09-10 (a
**single-day** Sick leave entry). One gap remains: a genuine **multi-day**
entry (different start/end dates) hasn't been captured yet, so that case is
still unverified — if a multi-day submission behaves unexpectedly, that's
why. The same capture method can confirm it the next time you add one.

### With pay vs without pay

Leave taken **without pay** consumes no leave credit, so HRIS keeps it in a
separate `wo_pay` column. Recording it under "used" would wrongly deduct it
from the employee's balance. Form 6 rows marked `WITHOUT PAY` are routed
there automatically (and tagged as such in the review table); the manual
form has a "Without pay" checkbox for the same purpose. Both confirmation
prompts say which column the days are going into.

If some without-pay records were already filed under "used", the Form 6 card
has a **"Check submitted without-pay rows"** button. It's read-only: it lists
the affected records with their IDs, and each one gets its own button to move
the days across. It edits records in place rather than deleting them, so IDs
and history are preserved. Confirmed working against real records on
2026-09-11.

## Form 6 batch upload

Upload a photo or scan of a Form 6 sick-leave transmittal (JPG/PNG, or a
multi-page PDF) and it's OCR'd into a draft table: one row
per employee on the form, pre-filled with description/dates/days-used, with
Sick-Leave-and-"WP" rows pre-checked (matching how these entries have always
been filtered) and the two other leave types (e.g. "PL"/"WOP") left
unchecked. The header checkbox checks or unchecks every row at once.

Each row's name is looked up in HRIS automatically as soon as the scan is
read. A name that resolves to exactly one employee is matched for you;
anything ambiguous or unfound is left for you to decide, with its candidates
already listed so picking one is a single click. Names are de-duplicated
first, so an employee with six date ranges costs one lookup, not six.

**Nothing here is trusted blindly** — review every row, and for any row the
lookup couldn't settle, click "Find" to search HRIS and pick the actual
matching employee (the name read off the scan is a starting point, not a
guarantee) before hitting "Submit checked rows". A row flagged with a parse
warning (e.g. an unreadable date) needs its dates filled in by hand.

Schools lay this form out differently, so the reader recognises five column
layouts (3, 5, 6, 7 and 9 columns) and picks whichever one a page matches. If
a form's table matches none of them, the upload now says so outright instead
of quietly coming back with zero rows.

One of those layouts — the 6-column "Summary of Absences" transmittal
(No./Name/Position/Date/s of Absence/Days/Remarks) — prints no leave type at
all, only "w/ pay" or "w/o pay", so its rows are drafted as **sick leave**.
That is what these transmittals are in practice and what this tool records,
but it is an assumption: check the leave type shown on each row before
submitting, and untick anything that was really another type.

Verified against a real Macanhan Elementary School Form 6 transmittal
(2026-09-10, 19 employee rows): names, dates (including multi-day ranges),
day counts, leave type, and action taken all came out correct. The
"Position" column (T-I/T-III/etc.) is noisier — Roman numerals are
genuinely hard to OCR reliably — but that's low-stakes since Position isn't
sent to HRIS at all, it's only shown so you can eyeball the right employee.

## Files

- `hris_client.py` — the API client (login, search, read balances, create
  records). This is the reusable core; `app.py` is just a thin UI on top
  of it.
- `ocr_form6.py` — Form 6 image → draft entries. Also runnable standalone:
  `python ocr_form6.py path/to/scan.jpg`.
- `app.py` — the local web app / UI.
- `test_hris_client.py`, `test_app.py` — the test suite. All mocked; it never
  calls live HRIS. Run it with `.venv/bin/python -m pytest`.

No sample transmittals are kept in this repository. The scans this was
developed against are real Form 6 forms naming real teachers alongside their
leave dates and without-pay status, which is not ours to publish. Keep your
own scans outside the repo — `samples/` is git-ignored for that reason.

## Roadmap

- [x] Login via the real HRIS API (no browser automation)
- [x] Search employees
- [x] View leave balance / ledger
- [x] Confirm and enable actually submitting new records (single-day; multi-day unverified)
- [x] OCR draft from a scanned Form 6 → editable review table → submit
