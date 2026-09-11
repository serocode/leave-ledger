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
