# HRIS Service Credit Tool (local)

A small local app so you don't have to click through the HRIS website (or
ask an AI to) every time you record a Service Credit entry — leave taken,
which is deducted from it, or vacation service credits earned, which are
added to it.

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

Log in with your HRIS credentials. The tool has four tabs:

- **Form 6 OCR Batch Upload** — upload a scanned leave transmittal, review the
  draft rows it reads, and submit them in one go. Every row deducts.
- **Service Credits Earned (VSC)** — upload a special order granting vacation
  service credits, or record a single grant by hand. Every row adds to
  *earned*. See [Vacation service credits](#vacation-service-credits).
- **Individual Employee Search & Record** — look someone up by surname to see
  their Service Credit balance and ledger, or deduct a single leave by hand.
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

Schools lay this form out differently, so the reader **reads the table's own
header row** and maps each column by what it is labelled — "NO. OF DAYS",
"Date/s of Absence", "TYPE OF LEAVE", "Action Taken" and so on. A school that
orders its columns differently, or leaves one out, needs no change here. One
header spanning several columns (an "Employee Name" printed across Last /
First / M.I.) is matched by position, which is how those three are
recognised.

A continuation page — page 2 of a long transmittal, with the table but no
header row — is read from **what its columns contain** instead: a column of
`7/3/2026` is dates, `1 DAY` is the day count, `SL` or `SOLO P` is the leave
type, `WP` is the action. That needs no knowledge of any school's layout, so a
new form's headerless page reads too. Older fixed layouts (3, 5, 6, 7 and 9
columns) remain only as a last resort, and a table that matches nothing says
so outright rather than quietly coming back with zero rows.

A leave type the form prints is kept exactly as printed. Solo parent leave,
and anything else this tool doesn't record, comes through **unticked** rather
than being relabelled as sick leave.

**Numeric dates are read three ways.** A slash can pick up a phantom `1`, so
`7/3/2026` reads as `7/13/2026` — a perfectly plausible date that would
otherwise go straight through. Each date cell is read with different OCR
settings, and when the readings disagree the row is flagged rather than
guessed: a clear majority is filled in with a warning to check it, and a tie
is left blank for you to enter. Flagged rows are highlighted in the review
table.

A transmittal with no Type of Leave column — one printing only "w/ pay" or
"w/o pay" — has its rows drafted as **sick leave**. That is what these are in
practice and what this tool records, but it is an assumption: check the leave
type shown on each row before submitting, and untick anything that was really
another type.

### Vacation service credits

Credits earned have their own tab, **Service Credits Earned (VSC)**. Every record
there is the same **Service Credit** leave credit type, with its days in the
ledger's **earned** column instead of **used** — added, not deducted. The Form 6
tab only deducts; a document uploaded on the wrong tab is sent back with where it
belongs.

**By hand** — *Record a single grant*: find the employee, enter the hours rendered
and the conversion factor, and the credits are worked out as

    VSC days = hours rendered × conversion factor ÷ 8

to the three decimals the ledger keeps. Work rendered during summer or Christmas
vacation, weekends or holidays is credited at **×1.50**, so 8 hours is 1.50 days.
The figure stays editable, to match what the order actually granted.

**From a special order** — the table of teachers "hereby granted vacation service
credits" (No. / Name / Position / Inclusive Dates / No. of hours served / No. of
vacation service credits granted) uploads and reviews just like a Form 6,
continuation pages included.

Each row records the **printed** credits granted — that is what the order legally
grants, and offices work it out differently: one at 0.188 per hour, one truncating
to half-days, one a little under the formula. So the hours are not something the
grant has to match. They are a **ceiling** — no office grants more than
hours × 1.50 ÷ 8, while OCR misreads push figures up (a 5 read as a 9, a lost
decimal point turning 7.219 into 721). Each credit figure is also read three ways.
A figure is filled in only when:

- the readings agree, or a clear majority of them does, **and**
- it fits under what the hours can earn.

Anything else is **left blank and unticked** for you to enter, with a hint when
one reading fits the hours. The hours can rule a reading out but never pick one —
they are read off the scan too, and a misread "9 HRS" once made the paper's 1.5
look impossible. A blank is never replaced by a formula figure, since the formula
is exactly what these offices don't consistently use.

Checked against three real special orders (66 rows): 44 filled in correctly, 22
left blank to enter, none filled in wrong.

**Pages that can't be read are named.** If a page's table can't be read — or loses
more than a couple of rows — a note above the review table says which page, so a
missing page never passes for a complete upload.

One thing is not yet verified: HRIS has only ever been sent `earned: 0`. The
first real grant you record, open it in HRIS itself and confirm the credits
landed in the right column with the right value.

**Only rows inside the table's printed borders can be read.** A row added by
hand underneath the last ruled line is invisible to a grid-based reader (and
handwriting is not OCR-able anyway), so it will not appear — count the rows
against the paper and add any stragglers from the Individual Employee tab.

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
