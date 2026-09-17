"""
HRIS Service Credit Tool — a local desktop app for recording DepEd HRIS
Service Credit entries: leave deducted from it, and vacation service credits
earned into it.

Run this on your own computer:

    pip install flask requests
    python app.py

It starts a small web server on http://127.0.0.1:5057 and opens it in your
browser automatically. Everything — your HRIS password included — stays on
your machine: the server here just relays requests to the real HRIS API
(v2.depedcdo.online), the same one the HRIS website itself uses. Nothing is
sent to me or to any third party.

Ways to add entries:
  - Upload a scanned Form 6 transmittal (leave, deducted) or a vacation
    service credit special order (credits, earned) — each on its own tab. It
    gets OCR'd into a draft table you review and correct before anything is
    submitted (see ocr_form6.py).
  - Record one entry by hand: leave from the Individual Employee tab, a
    credit grant from the Service Credits Earned tab.

Submission is live (SUBMIT_ENABLED below) — confirmed against a real
captured HRIS request on 2026-09-10. You'll always get a confirmation
prompt before a record is actually sent.
"""

from __future__ import annotations

import os
import tempfile
import shutil
import threading
import webbrowser
from urllib.parse import urlparse

from flask import Flask, jsonify, redirect, render_template_string, request, session, url_for

from ocr_form6 import extract_form6_with_notes

from hris_client import BASE_URL, HrisClient, HrisError, LEAVE_CREDIT_TYPE_IDS

APP_VERSION = "1.0"
HOST, PORT = "127.0.0.1", 5057

# Confirmed 2026-09-10 against a real captured "Add Record" submission (see
# hris_client.create_leave_credit's docstring). Set back to False if HRIS
# ever changes its request shape and submissions start failing.
SUBMIT_ENABLED = True

app = Flask(__name__)
app.secret_key = "local-only-not-a-real-secret"  # only matters for this local, single-user server

# One client per browser session id. This app is meant to be run by a single
# person on their own machine, so a simple in-memory dict is fine — nothing
# here is written to disk.
_clients: dict[str, HrisClient] = {}


def _get_client() -> HrisClient | None:
    sid = session.get("sid")
    if not sid:
        return None
    return _clients.get(sid)


PAGE = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HRIS Service Credit Tool (local)</title>
<style>
  /* Design tokens from Design/DESIGN (5).md — DepEd institutional palette:
     navy authority, amber agency accent, slate neutrals, low-radius geometry. */
  :root {
    --navy: #0f3460; --navy-hover: #0a2342; --navy-active: #06162a;
    --amber: #d97706; --amber-bright: #f59e0b; --cyan: #0284c7;
    --canvas: #f8fafc; --surface: #ffffff; --subtle: #f1f5f9;
    --line: #e2e8f0; --line-strong: #cbd5e1;
    --text: #0f172a; --text-body: #334155; --text-muted: #64748b;
    --on-navy: #ffffff;
    --ok: #16a34a; --ok-bg: #f0fdf4; --ok-line: #bbf7d0; --ok-text: #166534;
    --warn: #d97706; --warn-bg: #fffbeb; --warn-line: #fde68a; --warn-text: #92400e;
    --err: #dc2626; --err-bg: #fef2f2; --err-line: #fecaca; --err-text: #991b1b;
    --r: 0.25rem; --r-lg: 0.5rem; --r-pill: 9999px;
    --e1: 0 1px 2px 0 rgba(15,23,42,0.05);
    --e2: 0 4px 6px -1px rgba(15,23,42,0.08), 0 2px 4px -2px rgba(15,23,42,0.04);
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --navy: #1a365d; --navy-hover: #234876; --navy-active: #2c5689;
      --canvas: #0b1220; --surface: #131c2e; --subtle: #1b2740;
      --line: #253352; --line-strong: #33456b;
      --text: #e8edf7; --text-body: #c2ccdf; --text-muted: #8798b5;
      --ok-bg: #10261a; --ok-line: #225033; --ok-text: #7fdca0;
      --warn-bg: #2b2008; --warn-line: #5c4410; --warn-text: #f0c674;
      --err-bg: #2e1414; --err-line: #5f2724; --err-text: #f2a29c;
      --e1: 0 1px 2px 0 rgba(0,0,0,0.4);
      --e2: 0 4px 6px -1px rgba(0,0,0,0.5), 0 2px 4px -2px rgba(0,0,0,0.3);
    }
  }
  * { box-sizing: border-box; }
  html { background: var(--canvas); }
  body { font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         margin: 0; color: var(--text-body); background: var(--canvas);
         font-size: 0.875rem; line-height: 1.25rem; -webkit-font-smoothing: antialiased; }
  /* Tabular figures keep day counts and balances decimal-aligned down a column. */
  .num, input[type=number], td.num { font-variant-numeric: tabular-nums;
        font-feature-settings: "tnum" 1; letter-spacing: -0.01em; }
  .wrap { max-width: 1360px; margin: 0 auto; padding: 0 1.5rem; }
  h1, h2, h3, h4 { margin: 0; color: var(--text); }
  a { color: var(--cyan); }

  /* ---- App bar ---- */
  .appbar { background: var(--surface); border-bottom: 1px solid var(--line); box-shadow: var(--e1); }
  .appbar-top { display: flex; align-items: center; gap: 1rem; flex-wrap: wrap; padding: 0.75rem 0; }
  .brand { display: flex; align-items: center; gap: 0.75rem; min-width: 0; }
  .brand-text h1 { font-size: 0.9375rem; font-weight: 700; letter-spacing: 0.01em;
                   text-transform: uppercase; line-height: 1.15rem; }
  .brand-text p { margin: 2px 0 0; font-size: 0.75rem; color: var(--text-muted); }
  .ver { font-size: 0.75rem; font-weight: 600; color: var(--navy); background: var(--subtle);
         border: 1px solid var(--line); border-radius: var(--r); padding: 2px 7px; }
  @media (prefers-color-scheme: dark) { .ver { color: var(--text); } }
  .meta-chips { display: flex; gap: 0.5rem; flex-wrap: wrap; margin-left: auto; align-items: center; }
  .meta-chip { display: flex; align-items: center; gap: 0.4rem; background: var(--subtle);
               border: 1px solid var(--line); border-radius: var(--r); padding: 5px 10px;
               font-size: 0.75rem; color: var(--text-muted); white-space: nowrap; }
  .meta-chip b { color: var(--text); font-weight: 600; }
  .meta-chip .live { color: var(--amber); font-weight: 700; }
  .dot { width: 7px; height: 7px; border-radius: var(--r-pill); background: var(--amber-bright);
         display: inline-block; flex: none; }
  .dot.on { background: var(--ok); }
  .who { text-align: right; font-size: 0.75rem; color: var(--text-muted); line-height: 1.1rem; }
  .who b { display: block; color: var(--text); font-size: 0.8125rem; }
  .icon-btn { display: inline-flex; align-items: center; justify-content: center; width: 34px;
              height: 34px; border-radius: var(--r); border: 1px solid var(--line);
              background: var(--surface); color: var(--text-muted); cursor: pointer; margin: 0; }
  .icon-btn:hover { background: var(--subtle); color: var(--err); border-color: var(--line-strong); }

  /* ---- Tabs ---- */
  .tabs { display: flex; gap: 0.25rem; overflow-x: auto; }
  .tab { appearance: none; background: none; border: 0; border-bottom: 2px solid transparent;
         padding: 0.6rem 0.9rem; margin: 0; font: inherit; font-size: 0.875rem; font-weight: 600;
         color: var(--text-muted); cursor: pointer; white-space: nowrap; border-radius: 0; }
  .tab:hover { color: var(--text); }
  .tab[aria-selected="true"] { color: var(--navy); border-bottom-color: var(--navy); }
  @media (prefers-color-scheme: dark) {
    .tab[aria-selected="true"] { color: var(--text); border-bottom-color: var(--amber); }
  }
  .tab .pill-primary { background: var(--amber-bright); color: #3b2300; font-size: 0.6875rem;
                       font-weight: 700; border-radius: var(--r-pill); padding: 2px 8px;
                       margin-left: 6px; text-transform: uppercase; letter-spacing: 0.03em; }

  main { padding: 1.5rem 0 5rem; }

  /* ---- Cards ---- */
  .card { background: var(--surface); border: 1px solid var(--line); border-radius: var(--r-lg);
          box-shadow: var(--e1); padding: 1rem 1.25rem; margin-bottom: 1rem; }
  .card > h4 { font-size: 1.125rem; font-weight: 600; letter-spacing: -0.01em; }
  .sub { color: var(--text-muted); font-size: 0.8125rem; margin: 0.35rem 0 0.85rem; }
  .banner { display: flex; gap: 0.85rem; align-items: flex-start; flex-wrap: wrap;
            background: var(--surface); border: 1px solid var(--line); border-radius: var(--r-lg);
            box-shadow: var(--e1); padding: 0.85rem 1.25rem; margin-bottom: 1rem; }
  .banner .grow { flex: 1; min-width: 240px; }
  .banner h4 { font-size: 0.9375rem; }

  /* ---- Buttons ---- */
  button { font: inherit; font-size: 0.8125rem; font-weight: 600; cursor: pointer;
           border-radius: var(--r); border: 1px solid var(--line-strong);
           background: var(--surface); color: var(--text-body); padding: 0 12px; height: 36px;
           display: inline-flex; align-items: center; gap: 6px; }
  button:hover:not(:disabled) { background: var(--subtle); color: var(--text); }
  button:disabled { opacity: 0.45; cursor: not-allowed; }
  button.primary { background: var(--navy); border-color: var(--navy); color: var(--on-navy); }
  button.primary:hover:not(:disabled) { background: var(--navy-hover); color: var(--on-navy); }
  button.accent { background: var(--amber); border-color: var(--amber); color: #fff; }
  button.accent:hover:not(:disabled) { filter: brightness(0.94); color: #fff; }
  button.dense { height: 30px; font-size: 0.75rem; padding: 0 9px; }
  :focus-visible { outline: 2px solid var(--cyan); outline-offset: 2px; }

  /* ---- Forms ---- */
  label { display: block; font-size: 0.8125rem; font-weight: 600; color: var(--text-body);
          margin-bottom: 4px; }
  input, select { width: 100%; height: 36px; padding: 0 10px; font: inherit; font-size: 0.875rem;
                  color: var(--text); background: var(--surface);
                  border: 1px solid var(--line-strong); border-radius: var(--r); }
  input:focus, select:focus { outline: none; border-color: var(--navy);
                              box-shadow: 0 0 0 2px rgba(15,52,96,0.13); }
  input[type=file] { padding: 6px 10px; height: auto; }
  input[type=checkbox] { width: auto; height: auto; accent-color: var(--navy);
                         transform: scale(1.1); cursor: pointer; }
  .field { margin-bottom: 0.75rem; }
  .field-row { display: flex; gap: 0.75rem; }
  .field-row > * { flex: 1; min-width: 0; }
  .hint { font-size: 0.75rem; color: var(--text-muted); margin-top: 4px; }

  /* ---- Status pills ---- */
  .pill { display: inline-flex; align-items: center; gap: 4px; border-radius: var(--r-pill);
          padding: 2px 8px; font-size: 0.75rem; font-weight: 600; letter-spacing: 0.025em;
          white-space: nowrap; border: 1px solid transparent; }
  .pill-ok   { background: var(--ok-bg);   color: var(--ok-text);   border-color: var(--ok-line); }
  .pill-warn { background: var(--warn-bg); color: var(--warn-text); border-color: var(--warn-line); }
  .pill-err  { background: var(--err-bg);  color: var(--err-text);  border-color: var(--err-line); }
  .pill-neutral { background: var(--subtle); color: var(--text-body); border-color: var(--line); }

  /* ---- KPI cards ---- */
  .kpis { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 1rem;
          margin-bottom: 1rem; }
  .kpi { background: var(--surface); border: 1px solid var(--line); border-radius: var(--r-lg);
         box-shadow: var(--e1); padding: 0.85rem 1rem; border-bottom: 3px solid var(--line-strong); }
  .kpi .k-label { font-size: 0.75rem; font-weight: 600; letter-spacing: 0.05em;
                  text-transform: uppercase; color: var(--text-muted); }
  .kpi .k-value { font-size: 1.75rem; font-weight: 700; letter-spacing: -0.02em; color: var(--text);
                  font-variant-numeric: tabular-nums; line-height: 2.25rem; }
  .kpi .k-foot { font-size: 0.75rem; color: var(--text-muted); }
  .kpi.k-navy  { border-bottom-color: var(--navy); }
  .kpi.k-amber { border-bottom-color: var(--amber); }
  .kpi.k-amber .k-value, .kpi.k-amber .k-label { color: var(--warn-text); }
  .kpi.k-err   { border-bottom-color: var(--err); }
  .kpi.k-err .k-value, .kpi.k-err .k-label { color: var(--err-text); }
  .kpi.k-fill { background: var(--navy); border-color: var(--navy); }
  .kpi.k-fill .k-label, .kpi.k-fill .k-foot { color: #a9c8fc; }
  .kpi.k-fill .k-value { color: #fff; }
  @media (max-width: 1000px) { .kpis { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
  @media (max-width: 560px)  { .kpis { grid-template-columns: 1fr; } }

  /* ---- Filter chips ---- */
  .summary { display: flex; gap: 0.5rem; flex-wrap: wrap; align-items: center; margin-bottom: 0.75rem; }
  .chip { border: 1px solid var(--line-strong); background: var(--surface); color: var(--text-body);
          border-radius: var(--r); padding: 0 11px; height: 32px; font-size: 0.8125rem;
          font-weight: 600; display: inline-flex; align-items: center; gap: 6px; }
  .chip:hover { background: var(--subtle); }
  .chip[aria-pressed="true"] { background: var(--navy); border-color: var(--navy); color: var(--on-navy); }
  .chip[aria-pressed="true"]:hover { background: var(--navy-hover); color: var(--on-navy); }
  .chip .n { font-variant-numeric: tabular-nums; opacity: 0.8; font-weight: 700; }
  .chip[aria-pressed="true"] .n { opacity: 1; }
  .chip.chip-warn .n { color: var(--warn-text); }
  .chip.chip-warn[aria-pressed="true"] .n { color: inherit; }
  .stat { color: var(--text-muted); font-size: 0.8125rem; }

  /* ---- Tables ---- */
  .table-scroll { overflow-x: auto; border: 1px solid var(--line); border-radius: var(--r-lg);
                  background: var(--surface); }
  table { width: 100%; border-collapse: collapse; font-size: 0.8125rem; }
  thead th { position: sticky; top: 0; z-index: 10; background: var(--canvas);
             color: var(--text-muted); font-size: 0.75rem; font-weight: 600;
             letter-spacing: 0.05em; text-transform: uppercase; text-align: left;
             padding: 0.6rem 0.75rem; border-bottom: 2px solid var(--line-strong); }
  tbody td { padding: 0.5rem 0.75rem; border-bottom: 1px solid var(--line);
             vertical-align: middle; color: var(--text-body); }
  tbody tr:hover td { background: var(--subtle); }
  td.num, th.num { text-align: right; font-variant-numeric: tabular-nums;
                   font-weight: 600; color: var(--text); }
  .review-table td { vertical-align: top; }
  .review-table input { height: 32px; font-size: 0.8125rem; }
  .review-table input[type=date] { min-width: 126px; }
  .review-table td.name-cell { min-width: 168px; }
  .review-table td.match-cell { min-width: 232px; }
  .review-table tbody tr.needs-attention td { background: var(--warn-bg); }
  .review-table tbody tr.row-done td { opacity: 0.5; }
  .rowno { color: var(--text-muted); font-variant-numeric: tabular-nums; font-weight: 600; }
  .scan-name { font-weight: 600; color: var(--text); }
  .scan-meta { display: flex; gap: 5px; flex-wrap: wrap; align-items: center; margin-top: 3px; }
  .scan-meta .mini { font-size: 0.6875rem; color: var(--text-muted); background: var(--subtle);
                     border: 1px solid var(--line); border-radius: var(--r); padding: 1px 5px; }
  .wop-text { color: var(--warn-text); font-weight: 600; font-size: 0.75rem; }

  /* Matched HRIS record block */
  .match-done { display: flex; gap: 8px; align-items: flex-start; background: var(--subtle);
                border: 1px solid var(--line); border-radius: var(--r); padding: 6px 8px; }
  .match-done .grow { flex: 1; min-width: 0; }
  .match-picked { font-weight: 600; color: var(--text); font-size: 0.8125rem; }
  .match-id { font-size: 0.6875rem; color: var(--text-muted); font-variant-numeric: tabular-nums;
              display: flex; gap: 6px; align-items: center; flex-wrap: wrap; margin-top: 3px; }
  .match-change { opacity: 0; transition: opacity .1s; flex: none; }
  tr:hover .match-change, .match-change:focus-visible { opacity: 1; }
  .match-row { display: flex; gap: 6px; }
  .match-row .match-input { flex: 1; min-width: 0; height: 32px; }
  .match-results { max-height: 168px; overflow-y: auto; margin-top: 5px; border-radius: var(--r); }
  .match-results:not(:empty) { border: 1px solid var(--line); box-shadow: var(--e2); }
  .match-results div, #results div { padding: 7px 9px; cursor: pointer;
                                     border-bottom: 1px solid var(--line); font-size: 0.8125rem; }
  .match-results div:hover, #results div:hover { background: var(--subtle); }
  #results:not(:empty) { border: 1px solid var(--line); border-radius: var(--r);
                         margin-top: 6px; box-shadow: var(--e2); }
  .row-status.ok  { color: var(--ok-text);  font-weight: 600; }
  .row-status.err { color: var(--err-text); font-weight: 600; }
  .warn2 { color: var(--warn-text); font-size: 0.75rem; margin-top: 3px; }

  /* ---- Docked batch action bar ---- */
  .actionbar { position: sticky; bottom: 0; z-index: 20; margin: 1rem -1.25rem -1rem;
               padding: 0.85rem 1.25rem; background: var(--surface);
               border-top: 1px solid var(--line-strong);
               border-radius: 0 0 var(--r-lg) var(--r-lg);
               display: flex; gap: 1rem; align-items: center; flex-wrap: wrap;
               box-shadow: 0 -4px 6px -4px rgba(15,23,42,0.08); }
  .actionbar .grow { flex: 1; min-width: 160px; }
  .bar-title { font-size: 0.875rem; font-weight: 600; color: var(--text); }
  .progress { height: 6px; background: var(--line); border-radius: var(--r-pill);
              overflow: hidden; margin-top: 6px; }
  .progress > div { height: 100%; width: 0; background: var(--navy); transition: width .2s; }

  /* ---- Employee record tab ---- */
  .emp-head { display: flex; gap: 1rem; align-items: center; flex-wrap: wrap; }
  .emp-avatar { width: 46px; height: 46px; border-radius: var(--r); background: var(--navy);
                color: #fff; display: flex; align-items: center; justify-content: center;
                font-weight: 700; font-size: 1rem; flex: none; }
  .emp-id { font-size: 0.75rem; font-weight: 600; color: var(--navy); background: var(--subtle);
            border: 1px solid var(--line); border-radius: var(--r); padding: 2px 7px;
            font-variant-numeric: tabular-nums; }
  @media (prefers-color-scheme: dark) { .emp-id { color: var(--text); } }
  .two-col { display: grid; grid-template-columns: minmax(0, 1.35fr) minmax(0, 1fr);
             gap: 1rem; align-items: start; }
  @media (max-width: 1000px) { .two-col { grid-template-columns: 1fr; } }

  .err-box { background: var(--err-bg); border: 1px solid var(--err-line); color: var(--err-text);
             padding: 9px 12px; border-radius: var(--r); font-size: 0.8125rem; margin-top: 8px; }
  .ok-box { background: var(--ok-bg); border: 1px solid var(--ok-line); color: var(--ok-text);
            padding: 9px 12px; border-radius: var(--r); font-size: 0.8125rem; margin-top: 8px; }
  .warn-box { background: var(--warn-bg); border: 1px solid var(--warn-line); color: var(--warn-text);
              padding: 9px 12px; border-radius: var(--r); font-size: 0.8125rem; }
  .info-box { background: var(--subtle); border: 1px solid var(--line); border-radius: var(--r);
              padding: 9px 12px; font-size: 0.8125rem; color: var(--text-body); }
  .err { color: var(--err-text); }
  .loading { color: var(--text-muted); font-style: italic; }

  /* A Form 6 upload runs two slow steps back to back — OCR on the scan, then
     one HRIS search per name — and both used to sit behind a single static
     line of text, which reads as a hung app. The spinner shows the step is
     alive; the bar is a real count of names checked during matching (the
     long step), and only falls back to an indeterminate sweep for OCR, where
     there is no progress to report. */
  .busy { display: flex; align-items: center; gap: 8px; font-size: 0.8125rem;
          color: var(--text-body); }
  .busy-count { margin-left: auto; flex: none; font-variant-numeric: tabular-nums;
                color: var(--text-muted); }
  .spinner { width: 13px; height: 13px; flex: none; border-radius: 50%;
             border: 2px solid var(--line); border-top-color: var(--navy);
             animation: spin 0.7s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .progress.indeterminate > div { width: 34%; transition: none;
                                  animation: sweep 1.2s ease-in-out infinite; }
  @keyframes sweep { from { margin-left: -34%; } to { margin-left: 100%; } }
  /* Rows resolve a few at a time, so a settling flash is what makes the
     progress visible on the table itself, not just in the status line. */
  @keyframes settle { from { background: var(--ok-bg); } to { background: transparent; } }
  .review-table tbody tr.just-matched td { animation: settle 1.2s ease-out; }
  @media (prefers-reduced-motion: reduce) {
    .spinner { animation-duration: 2.4s; }
    .progress.indeterminate > div { animation: none; width: 100%; opacity: 0.4; }
    .review-table tbody tr.just-matched td { animation: none; }
  }

  /* A spinner sitting inside a filled button or on a coloured row needs its
     track lightened, or the ring reads as a solid dot. */
  .spinner-sm { width: 11px; height: 11px; border-width: 2px; }
  .primary .spinner, .accent .spinner { border-color: rgba(255,255,255,0.35);
                                        border-top-color: #fff; }
  .row-status .spinner, .loading .spinner { vertical-align: -1px; margin-right: 5px;
                                            display: inline-block; }

  /* ---- Confirmation modal ----
     These confirmations are the last thing standing between a mis-typed day
     count and a real HRIS record, so they show the actual values being
     written rather than native confirm()'s wall of text — which can't show
     structure, can't be styled, and which browsers let a user suppress for
     the rest of the session ("don't ask again"). <dialog> is used so Esc,
     focus trapping and the inert backdrop come from the platform. */
  dialog.modal { border: 1px solid var(--line); border-radius: var(--r-lg); padding: 0;
                 background: var(--surface); color: var(--text-body); box-shadow: var(--e2);
                 width: min(460px, calc(100vw - 2rem)); }
  dialog.modal::backdrop { background: rgba(15, 23, 42, 0.45); }
  .modal-body { padding: 1.1rem 1.25rem 0; }
  .modal-body h3 { font-size: 1rem; margin: 0 0 0.35rem; color: var(--text); }
  .modal-intro { font-size: 0.8125rem; color: var(--text-body); margin: 0 0 0.75rem; }
  .modal-rows { margin: 0 0 0.85rem; display: grid; grid-template-columns: auto minmax(0, 1fr);
                gap: 5px 12px; font-size: 0.8125rem; border-top: 1px solid var(--line);
                padding-top: 0.75rem; }
  .modal-rows dt { color: var(--text-muted); }
  .modal-rows dd { margin: 0; font-weight: 600; color: var(--text); overflow-wrap: anywhere; }
  .modal-actions { display: flex; justify-content: flex-end; gap: 8px;
                   padding: 0.9rem 1.25rem 1.1rem; }
  @media (max-width: 480px) { .modal-actions { flex-direction: column-reverse; }
                              .modal-actions button { width: 100%; justify-content: center; } }
  .session-banner { background: var(--err-bg); border: 1px solid var(--err-line); color: var(--err-text);
                    padding: 11px 14px; border-radius: var(--r); font-size: 0.875rem;
                    margin-bottom: 1rem; text-align: center; display: none; }
  pre { background: var(--canvas); border: 1px solid var(--line); padding: 10px;
        border-radius: var(--r); overflow-x: auto; font-size: 0.75rem; }

  /* ---- Footer ---- */
  footer { border-top: 1px solid var(--line); background: var(--surface); padding: 1rem 0;
           font-size: 0.75rem; color: var(--text-muted); }
  .foot-grid { display: flex; gap: 1.5rem; flex-wrap: wrap; align-items: center; }

  /* ---- Login ---- */
  .login-wrap { max-width: 430px; margin: 3.5rem auto; padding: 0 1.5rem; }
  .login-brand { display: flex; align-items: center; gap: 0.75rem; justify-content: center;
                 margin-bottom: 1.25rem; }

  @media (max-width: 640px) {
    .wrap { padding: 0 1rem; }
    .actionbar { margin: 1rem -1rem -1rem; padding: 0.75rem 1rem; }
    .who, .meta-chips { display: none; }
  }
</style>
</head>
<body>

{% macro logo() %}
<svg width="34" height="34" viewBox="0 0 48 48" fill="none" aria-hidden="true" style="flex:none">
  <rect width="48" height="48" rx="11" fill="#1B3A5C"/>
  <rect x="12" y="15" width="20" height="4.2" rx="2.1" fill="#F5A623"/>
  <rect x="12" y="22.4" width="14" height="4.2" rx="2.1" fill="#F5A623"/>
  <rect x="12" y="29.8" width="9" height="4.2" rx="2.1" fill="#F5A623"/>
  <circle cx="30.5" cy="31.9" r="3.6" fill="#29ABE2"/>
</svg>
{% endmacro %}

{% if not logged_in %}
<div class="login-wrap">
  <div class="login-brand">
    {{ logo() }}
    <div class="brand-text">
      <h1>DepEd HRIS Service Credit Tool</h1>
      <p>Division Office ledger for leave taken and credits earned</p>
    </div>
  </div>
  <div class="card">
    <h4>Sign in</h4>
    <p class="sub">Log in with your own HRIS credentials. They go from your browser
    to the local server on this machine, then straight to the DepEd HRIS login
    endpoint — never anywhere else.</p>
    <form method="post" action="/login">
      <div class="field">
        <label>Username / email</label>
        <input name="identifier" required autofocus>
      </div>
      <div class="field">
        <label>Password</label>
        <input name="password" type="password" required>
      </div>
      {% if error %}<div class="err-box">{{ error }}</div>{% endif %}
      <button type="submit" class="primary" style="width:100%;justify-content:center;margin-top:6px">Log in</button>
    </form>
  </div>
  <p class="sub" style="text-align:center">Zero-cloud storage. Runs only on this computer.</p>
</div>
{% else %}

<header class="appbar">
  <div class="wrap">
    <div class="appbar-top">
      <div class="brand">
        {{ logo() }}
        <div class="brand-text">
          <h1>DepEd HRIS Service Credit Tool</h1>
          <p>Division Office ledger for leave taken and credits earned</p>
        </div>
        <span class="ver">v{{ version }}</span>
      </div>
      <div class="meta-chips">
        <span class="meta-chip"><span class="dot on"></span>Local server: <b>{{ local_url }}</b></span>
        <span class="meta-chip">Submission:
          {% if submit_enabled %}<span class="live">LIVE to {{ api_host }}</span>
          {% else %}<b>Dry run only</b>{% endif %}
        </span>
      </div>
      <div class="who">
        Logged in as<b>{{ user_label }}</b>
      </div>
      <a href="/logout" class="icon-btn" title="Log out" aria-label="Log out">
        <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor"
             stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/>
          <polyline points="16 17 21 12 16 7"/><line x1="21" y1="12" x2="9" y2="12"/>
        </svg>
      </a>
    </div>
    <nav class="tabs" role="tablist">
      <button class="tab" role="tab" data-tab="batch" aria-selected="true">
        Form 6 OCR Batch Upload<span class="pill-primary">Primary</span>
      </button>
      <button class="tab" role="tab" data-tab="vsc" aria-selected="false">
        Service Credits Earned (VSC)
      </button>
      <button class="tab" role="tab" data-tab="employee" aria-selected="false">
        Individual Employee Search &amp; Record
      </button>
      <button class="tab" role="tab" data-tab="wop" aria-selected="false">
        Without-Pay Audit (WOP)
      </button>
    </nav>
  </div>
</header>

<main class="wrap">

<div id="sessionBanner" class="session-banner">
  Your session has expired. <a href="/">Log in again.</a>
</div>

{% macro review_workspace(p, title, button, kpi4_label, kpi4_foot, amount_header) %}
  <div class="card">
    <h4>{{ title }}</h4>
    {{ caller() }}
    <div class="field-row" style="align-items:flex-end">
      <div style="flex:2"><input type="file" id="{{ p }}File" accept="image/*,application/pdf" multiple></div>
      <div style="flex:none"><button id="{{ p }}Upload" type="button" class="primary">{{ button }}</button></div>
    </div>
    <div id="{{ p }}Status" class="sub" style="margin-bottom:0"></div>
    <div id="{{ p }}Notes"></div>
    <div id="{{ p }}Sources"></div>
  </div>

  <div class="kpis" id="{{ p }}Kpis" style="display:none">
    <div class="kpi k-navy">
      <div class="k-label">Identified personnel</div>
      <div class="k-value" id="{{ p }}KpiPeople">0</div>
      <div class="k-foot" id="{{ p }}KpiPeopleFoot">From 0 rows</div>
    </div>
    <div class="kpi k-navy">
      <div class="k-label">Auto-matched HRIS</div>
      <div class="k-value" id="{{ p }}KpiAuto">0</div>
      <div class="k-foot" id="{{ p }}KpiAutoFoot">Resolved to one employee</div>
    </div>
    <div class="kpi k-amber">
      <div class="k-label">Needs attention</div>
      <div class="k-value" id="{{ p }}KpiAttention">0</div>
      <div class="k-foot">Requires manual alignment</div>
    </div>
    <div class="kpi {{ 'k-err' if p == 'form6' else 'k-navy' }}">
      <div class="k-label">{{ kpi4_label }}</div>
      <div class="k-value" id="{{ p }}Kpi4">0</div>
      <div class="k-foot">{{ kpi4_foot }}</div>
    </div>
  </div>

  <div class="card" id="{{ p }}ReviewCard" style="display:none">
    <h4>Review draft entries</h4>
    <p class="sub">Match each row to the right HRIS employee before submitting — the name
    read from the scan is a starting point, not a guarantee. Unmatched or unchecked rows
    are skipped.</p>
    <div class="summary" id="{{ p }}Summary"></div>

    <div class="table-scroll">
    <table id="{{ p }}Table" class="review-table">
      <thead><tr>
        <th style="width:34px"><input type="checkbox" id="{{ p }}CheckAll" title="Check / uncheck every visible row"></th>
        <th style="width:34px">#</th>
        <th>Name on scan / designation</th>
        <th>Matched HRIS record</th>
        <th>Description</th>
        <th>Start</th>
        <th>End</th>
        <th class="num">{{ amount_header }}</th>
        <th>Status</th>
      </tr></thead>
      <tbody></tbody>
    </table>
    </div>
    <p class="sub" id="{{ p }}Empty" style="display:none">No rows match this filter.</p>

    <div class="actionbar">
      <div class="grow">
        <div class="bar-title" id="{{ p }}BarTitle">Batch ready for ledger commit</div>
        <div class="progress" id="{{ p }}Progress"><div></div></div>
        <div class="stat" id="{{ p }}SubmitStatus"></div>
      </div>
      <button id="{{ p }}SubmitAll" type="button" class="primary">Submit checked rows</button>
    </div>
  </div>
{% endmacro %}

<!-- ===================== Form 6 batch tab ===================== -->
<section id="tab-batch" role="tabpanel">

  <div class="banner">
    <div class="grow">
      <h4>
        Submission pipeline:
        {% if submit_enabled %}LIVE HRIS production direct
          <span class="pill pill-warn">Safeguard armed</span>
        {% else %}Dry run <span class="pill pill-neutral">Nothing is sent</span>{% endif %}
      </h4>
      <p class="sub" style="margin-bottom:0">
        {% if submit_enabled %}Writes go to <b>{{ api_base }}</b>. Every submission asks
        you to confirm first — nothing is sent on an accidental click.
        {% else %}Submitting is disabled. Adding a record shows you the payload only.{% endif %}
      </p>
    </div>
    <span class="meta-chip">Local engine: <b>{{ local_url }}</b></span>
    <span class="meta-chip" id="ocrEngineChip">OCR engine: <b>checking…</b></span>
  </div>

  {% call review_workspace("form6", "Upload a Form 6 transmittal", "Read Form 6",
                            "Without pay (WOP)", "Routed to the wo_pay column", "Days") %}
    <p class="sub">A photo or scan (JPG/PNG) or a multi-page PDF. It is read into a
    draft table for you to review and correct — nothing reaches HRIS until you submit.
    Sick-leave rows (with or without pay) are pre-checked. Every row <b>deducts</b>:
    paid leave from <b>used</b>, leave without pay into <b>wo_pay</b>.</p>
    <p class="sub" style="margin-bottom:0.85rem"><b>Count the rows against the paper.</b>
    Only rows inside the table's printed borders can be read — a row written in by hand
    below the last line is invisible to the reader and will not appear here. Add those
    from the Individual Employee tab.</p>
  {% endcall %}
</section>

<!-- ===================== Service credits earned tab ===================== -->
<section id="tab-vsc" role="tabpanel" hidden>

  <div class="banner">
    <div class="grow">
      <h4>Vacation service credits — added to <b>earned</b></h4>
      <p class="sub" style="margin-bottom:0">
        Every record here is a <b>Service Credit</b> entry with its days in the
        <b>earned</b> column, never <b>used</b>. Upload a special order granting
        vacation service credits, or record a single grant by hand.
      </p>
    </div>
  </div>

  {% call review_workspace("vsc", "Upload a vacation service credit special order", "Read special order",
                           "Credits to record", "Days, across checked rows", "Earned") %}
    <p class="sub">The table of teachers "hereby granted vacation service credits" — No. /
    Name / Position / Inclusive Dates / No. of hours served / No. of vacation service
    credits granted. Continuation pages without a header read too.</p>
    <p class="sub" style="margin-bottom:0.85rem">Each row records the <b>credits granted as
    printed</b> — offices work them out differently, so the hours are only a ceiling.
    A figure above what the hours can earn, or one the reads don't agree on, is left
    blank for you to enter from the paper.</p>
  {% endcall %}

  <div class="card">
    <h4>Record a single grant</h4>
    <p class="sub">For one teacher, or a row the upload couldn't read.</p>
    <div class="field">
      <label>Employee</label>
      <div class="match-row">
        <input id="vscEmpQ" placeholder="Surname works best — e.g. Dela Cruz">
        <button type="button" id="vscEmpFind" class="dense">Find</button>
      </div>
      <div id="vscEmpResults" class="match-results"></div>
      <div id="vscEmpChosen" class="hint"></div>
    </div>
    <div class="field-row">
      <div class="field">
        <label>Hours rendered</label>
        <input id="vscHours" type="number" step="any" min="0" placeholder="8">
      </div>
      <div class="field">
        <label>Conversion factor</label>
        <input id="vscFactor" type="number" step="0.01" min="0" value="1.50">
        <div class="hint">&times;1.50 for work during summer or Christmas vacation, weekends or holidays.</div>
      </div>
      <div class="field">
        <label>Credits earned (days)</label>
        <input id="vscEarned" type="number" step="any" min="0.001">
        <div class="hint" id="vscFormula">Hours &times; factor &divide; 8 — edit it to match the order.</div>
      </div>
    </div>
    <div class="field-row">
      <div class="field">
        <label>Start date</label>
        <input id="vscFrom" type="date">
      </div>
      <div class="field">
        <label>End date</label>
        <input id="vscTo" type="date">
      </div>
    </div>
    <div class="field-row">
      <div class="field" style="flex:1">
        <label>Special order number</label>
        <input id="vscSo" placeholder="e.g. 1673 s. 2026">
      </div>
      <div class="field" style="flex:3">
        <label>Description &amp; authority</label>
        <input id="vscDescription" placeholder="Vacation service credits, SO No. 1673 s. 2026, May 6 - June 2, 2026 (138 hrs)">
      </div>
    </div>
    <button type="button" id="vscAddBtn" class="primary" style="width:100%;justify-content:center">
      {{ 'Add credits to HRIS' if submit_enabled else 'Preview payload (dry run)' }}
    </button>
    <div id="vscAddResult"></div>
  </div>
</section>

<!-- ===================== Individual employee tab ===================== -->
<section id="tab-employee" role="tabpanel" hidden>
  <div class="card">
    <label for="q">Search the DepEd personnel database</label>
    <input id="q" placeholder="Surname works best — e.g. Dela Cruz">
    <div id="results"></div>
  </div>

  <div id="employeeCard" style="display:none">
    <div class="card">
      <div class="emp-head">
        <div class="emp-avatar" id="empInitials">—</div>
        <div style="flex:1;min-width:0">
          <h3 id="empName" style="font-size:1.125rem"></h3>
          <div class="sub" style="margin:3px 0 0" id="empMeta"></div>
        </div>
        <span class="emp-id" id="empIdChip"></span>
      </div>
    </div>

    <div class="kpis">
      <div class="kpi k-navy">
        <div class="k-label">Earned credits</div>
        <div class="k-value" id="empEarned">—</div>
        <div class="k-foot">Service Credit accruals</div>
      </div>
      <div class="kpi k-amber">
        <div class="k-label">Used so far</div>
        <div class="k-value" id="empUsed">—</div>
        <div class="k-foot">Form 6 deductions</div>
      </div>
      <div class="kpi k-fill">
        <div class="k-label">Net available balance</div>
        <div class="k-value" id="empBalance">—</div>
        <div class="k-foot">Earned minus used</div>
      </div>
      <div class="kpi k-err">
        <div class="k-label">Without pay (WOP)</div>
        <div class="k-value" id="empWop">—</div>
        <div class="k-foot">Consumes no credit</div>
      </div>
    </div>

    <div class="two-col">
      <div class="card">
        <h4>Service credit transaction ledger</h4>
        <p class="sub">Every Service Credit posting HRIS holds for this employee.</p>
        <div id="balanceError"></div>
        <div class="table-scroll">
          <table id="ledger">
            <thead><tr>
              <th>Posting date</th><th>Description</th>
              <th class="num">Earned</th><th class="num">Used</th><th class="num">W/O pay</th>
            </tr></thead>
            <tbody></tbody>
          </table>
        </div>
        <p class="sub" id="ledgerCount" style="margin-bottom:0"></p>
      </div>

      <div class="card">
        <h4>Add sick leave record</h4>
        <p class="sub">Direct Form 6 transaction entry.</p>
        <form id="addForm">
          <div class="field-row" id="deductFields">
            <div class="field">
              <label>Leave credit type</label>
              <select name="leave_type" disabled><option>Service Credit</option></select>
              <div class="hint">The only type this tool writes.</div>
            </div>
            <div class="field">
              <label>Days to deduct</label>
              <input name="used" type="number" step="0.5" min="0.5" value="1" required>
            </div>
          </div>
          <div class="info-box" id="wopBox" style="margin-bottom:0.75rem">
            <label style="display:flex;gap:8px;align-items:flex-start;margin:0;font-weight:600">
              <input name="without_pay" type="checkbox" style="margin-top:2px">
              <span>Without pay (WOP)
                <span style="display:block;font-weight:400;color:var(--text-muted);margin-top:2px">
                  Leave taken without pay consumes no credit and is routed to the HRIS
                  <b>wo_pay</b> column instead of <b>used</b>, protecting the balance.
                </span>
              </span>
            </label>
          </div>
          {% if submit_enabled %}
          <div class="warn-box" style="margin-bottom:0.75rem">
            <b>Dual-step safeguard.</b> Submitting writes directly to {{ api_host }}.
            You will be asked to confirm the description, dates and day count first.
          </div>
          {% endif %}
          <button type="submit" id="addSubmitBtn" class="primary" style="width:100%;justify-content:center">
            {{ 'Submit to HRIS' if submit_enabled else 'Preview payload (dry run)' }}
          </button>
        </form>
        <div id="addResult"></div>
      </div>
    </div>
  </div>
</section>

<!-- ===================== Without-pay audit tab ===================== -->
<section id="tab-wop" role="tabpanel" hidden>
  <div class="card">
    <h4>Without-pay audit</h4>
    <p class="sub">Finds records filed under <b>used</b> that describe leave taken
    without pay. Days in <b>used</b> are deducted from an employee's balance; days in
    <b>wo_pay</b> are not — so a misfiled row costs the employee credit they never spent.</p>
    <div class="info-box" style="margin-bottom:0.85rem">
      This reads a Form 6's without-pay rows and checks each matched employee's ledger.
      Read a transmittal on the <b>Form 6 OCR Batch Upload</b> tab first, then run the check
      here. Nothing changes until you click an individual row's button.
    </div>
    <button type="button" id="wopAudit" class="accent">Check submitted without-pay rows</button>
    <span id="wopStatus" class="stat" style="margin-left:10px"></span>
    <div class="table-scroll" id="wopWrap" style="display:none;margin-top:0.85rem">
      <table id="wopTable">
        <thead><tr>
          <th>Employee</th><th>Description</th><th class="num">Now in "used"</th>
          <th style="width:180px"></th><th>Status</th>
        </tr></thead>
        <tbody></tbody>
      </table>
    </div>
  </div>
</section>

</main>

<footer>
  <div class="wrap foot-grid">
    <span>Zero-cloud storage. Credentials go from this machine straight to the DepEd HRIS endpoint.</span>
    <span>Port 5057 (bound to 127.0.0.1)</span>
    <span>CS Form 6 compliance</span>
  </div>
</footer>

<script>
// ----- Tabs -----

document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => showTab(tab.dataset.tab));
});

function showTab(name) {
  document.querySelectorAll('.tab').forEach(t => {
    t.setAttribute('aria-selected', t.dataset.tab === name ? 'true' : 'false');
  });
  ['batch', 'vsc', 'employee', 'wop'].forEach(n => {
    document.getElementById('tab-' + n).hidden = (n !== name);
  });
}

// Report whether OCR is actually usable up front, rather than letting someone
// pick a file and only then discover Tesseract is missing.
(async () => {
  const chip = document.getElementById('ocrEngineChip');
  try {
    const data = await (await fetch('/api/health')).json();
    chip.innerHTML = `OCR engine: <b>${data.label}</b>`;
    if (!data.ocr) {
      chip.style.color = 'var(--err-text)';
      chip.title = 'Install Tesseract to read Form 6 scans. Everything else works without it.';
    }
  } catch (e) {
    chip.innerHTML = 'OCR engine: <b>unknown</b>';
  }
})();

// Numbers in ledger columns are rendered to 3 decimals, the DepEd convention
// for statutory leave computation.
function fmtDays(n) {
  return (Number(n) || 0).toFixed(3);
}

// ----- Helpers -----

function showError(container, msg) {
  const el = typeof container === 'string' ? document.getElementById(container) : container;
  el.innerHTML = `<div class="err-box">${msg}</div>`;
}

function showSuccess(container, msg) {
  const el = typeof container === 'string' ? document.getElementById(container) : container;
  el.innerHTML = `<div class="ok-box">${msg}</div>`;
}

function showSessionExpired() {
  document.getElementById('sessionBanner').style.display = 'block';
}

async function checkedFetch(url, options) {
  const res = await fetch(url, options);
  if (res.status === 401) {
    showSessionExpired();
    throw new Error('Session expired');
  }
  return res;
}

// ----- Search -----

let searchTimeout = null;
let selectedEmpId = null;
let selectedEmpName = '';

document.getElementById('q').addEventListener('input', (e) => {
  clearTimeout(searchTimeout);
  const q = e.target.value.trim();
  const box = document.getElementById('results');
  if (q.length < 2) { box.innerHTML = ''; return; }
  box.innerHTML = '<div class="loading">' + busyText('Searching...') + '</div>';
  searchTimeout = setTimeout(async () => {
    try {
      const res = await checkedFetch('/api/search?q=' + encodeURIComponent(q));
      const data = await res.json();
      if (data.error) { showError(box, 'Search failed: ' + data.error); return; }
      box.innerHTML = '';
      if (!data.results.length) { box.innerHTML = '<div style="color:#888;padding:6px">No matches found.</div>'; return; }
      data.results.forEach(emp => {
        const d = document.createElement('div');
        d.textContent = emp.full_name + ' — ' + (emp.position || '?') + ' @ ' + (emp.school || '?');
        d.onclick = () => selectEmployee(emp);
        box.appendChild(d);
      });
    } catch (err) {
      if (err.message !== 'Session expired') showError(box, 'Search failed — check your connection and try again.');
    }
  }, 300);
});

const EMP_METRICS = ['empEarned', 'empUsed', 'empBalance', 'empWop'];

async function selectEmployee(emp) {
  selectedEmpId = emp.id;
  selectedEmpName = emp.full_name;
  document.getElementById('results').innerHTML = '';
  document.getElementById('q').value = emp.full_name;
  document.getElementById('employeeCard').style.display = 'block';
  document.getElementById('empName').textContent = emp.full_name;
  document.getElementById('empMeta').textContent =
    (emp.position || 'Position unknown') + ' · ' + (emp.school || 'Station unknown');
  document.getElementById('empIdChip').textContent = 'HRIS ID #' + emp.id;
  document.getElementById('empInitials').textContent =
    (emp.full_name || '?').replace(/[^A-Za-z, ]/g, '').split(/[,\\s]+/).filter(Boolean)
      .slice(0, 2).map(w => w[0]).join('').toUpperCase() || '?';
  EMP_METRICS.forEach(id => { document.getElementById(id).textContent = '…'; });
  document.getElementById('balanceError').innerHTML = '';
  document.getElementById('ledgerCount').textContent = '';
  const tbody = document.querySelector('#ledger tbody');
  tbody.innerHTML = '';
  try {
    const res = await checkedFetch('/api/balance/' + emp.id);
    const data = await res.json();
    if (data.error) {
      EMP_METRICS.forEach(id => { document.getElementById(id).textContent = '—'; });
      showError('balanceError', 'Could not load balance: ' + data.error);
      return;
    }
    document.getElementById('empEarned').textContent = fmtDays(data.earned);
    document.getElementById('empUsed').textContent = fmtDays(data.used);
    document.getElementById('empBalance').textContent = fmtDays(data.balance);
    document.getElementById('empWop').textContent = fmtDays(data.wo_pay);
    data.ledger.forEach(r => {
      const tr = document.createElement('tr');
      tr.innerHTML =
        `<td>${(r.date || '').slice(0, 10)}</td>` +
        `<td>${r.description || '<span class="stat">(no description)</span>'}</td>` +
        `<td class="num">${fmtDays(r.earned)}</td>` +
        `<td class="num">${fmtDays(r.used)}</td>` +
        `<td class="num">${fmtDays(r.wo_pay)}</td>`;
      tbody.appendChild(tr);
    });
    document.getElementById('ledgerCount').textContent =
      `Showing ${data.ledger.length} transaction(s) as held by HRIS.`;
  } catch (err) {
    if (err.message !== 'Session expired') {
      EMP_METRICS.forEach(id => { document.getElementById(id).textContent = '—'; });
      showError('balanceError', 'Failed to load balance — check your connection and try again.');
    }
  }
}

// ----- Confirmation modal -----
// Replaces confirm()/alert() on every path that writes to HRIS. Beyond
// looking like the rest of the app, it can lay the values out as a table —
// and it can't be switched off the way a browser's native dialog can, which
// matters when it is the only thing between a typo and a real record.

let _modalEl = null;

function _modal() {
  if (_modalEl) return _modalEl;
  _modalEl = document.createElement('dialog');
  _modalEl.className = 'modal';
  _modalEl.innerHTML =
    '<form method="dialog">' +
      '<div class="modal-body">' +
        '<h3></h3><p class="modal-intro"></p><dl class="modal-rows"></dl>' +
        '<div class="modal-note"></div>' +
      '</div>' +
      '<div class="modal-actions">' +
        '<button type="submit" value="cancel" class="modal-cancel"></button>' +
        '<button type="submit" value="ok" class="primary modal-ok"></button>' +
      '</div>' +
    '</form>';
  document.body.appendChild(_modalEl);
  return _modalEl;
}

function escapeHtml(value) {
  return String(value === null || value === undefined ? '' : value)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// rows: [[label, value], ...]. note: an optional warn box under them.
// Resolves true only on the confirm button — Esc, the backdrop and Cancel
// all resolve false, so a dismissed dialog can never be read as a yes.
function confirmModal({ title, intro = '', rows = [], note = '', confirmText = 'Continue', cancelText = 'Cancel' }) {
  const el = _modal();
  el.querySelector('h3').textContent = title;
  const introEl = el.querySelector('.modal-intro');
  introEl.textContent = intro;
  introEl.style.display = intro ? 'block' : 'none';
  const rowsEl = el.querySelector('.modal-rows');
  rowsEl.innerHTML = rows
    .map(([label, value]) => `<dt>${escapeHtml(label)}</dt><dd>${escapeHtml(value)}</dd>`)
    .join('');
  rowsEl.style.display = rows.length ? 'grid' : 'none';
  const noteEl = el.querySelector('.modal-note');
  noteEl.innerHTML = note ? `<div class="warn-box">${escapeHtml(note)}</div>` : '';
  const cancelBtn = el.querySelector('.modal-cancel');
  const okBtn = el.querySelector('.modal-ok');
  cancelBtn.textContent = cancelText;
  cancelBtn.style.display = cancelText ? 'inline-flex' : 'none';
  okBtn.textContent = confirmText;

  return new Promise(resolve => {
    el.addEventListener('close', () => resolve(el.returnValue === 'ok'), { once: true });
    el.showModal();
    // Cancel takes focus, not the confirming button: the safe option should
    // be the one a stray Enter lands on.
    (cancelText ? cancelBtn : okBtn).focus();
  });
}

// A modal in place of alert() — one button, nothing to decide.
function alertModal(title, intro) {
  return confirmModal({ title, intro, confirmText: 'OK', cancelText: '' });
}

// ----- Button load states -----
// A disabled button that only changes its label still looks like nothing is
// happening on a slow HRIS call; a spinner in the button says which action
// is the one in flight.
function setBtnBusy(btn, label) {
  if (btn.dataset.idleHtml === undefined) btn.dataset.idleHtml = btn.innerHTML;
  btn.disabled = true;
  btn.setAttribute('aria-busy', 'true');
  btn.innerHTML = '<span class="spinner spinner-sm"></span>' + escapeHtml(label);
}

function clearBtnBusy(btn, label) {
  btn.disabled = false;
  btn.removeAttribute('aria-busy');
  if (label !== undefined) { btn.textContent = label; btn.dataset.idleHtml = label; }
  else if (btn.dataset.idleHtml !== undefined) btn.innerHTML = btn.dataset.idleHtml;
}

function busyText(label) {
  return '<span class="spinner spinner-sm"></span>' + escapeHtml(label);
}

// ----- Add record -----

const SUBMIT_ENABLED = {{ 'true' if submit_enabled else 'false' }};

document.getElementById('addForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  if (!selectedEmpId) {
    alertModal('Pick an employee first', 'Search for an employee and select them before adding a record.');
    return;
  }
  const fd = new FormData(e.target);
  const body = Object.fromEntries(fd.entries());
  body.employee_id = selectedEmpId;

  // Client-side validation
  if (!body.description || !body.description.trim()) { showError('addResult', 'Description is required.'); return; }
  if (!body.date_from || !body.date_to) { showError('addResult', 'Both start and end dates are required.'); return; }
  if (body.date_to < body.date_from) { showError('addResult', 'End date cannot be before start date.'); return; }
  const amount = parseFloat(body.used);
  if (isNaN(amount) || amount <= 0) { showError('addResult', 'Days to deduct must be a positive number.'); return; }

  if (SUBMIT_ENABLED) {
    const ok = await confirmModal({
      title: 'Write this record to HRIS?',
      rows: [
        ['Employee', selectedEmpName || ('HRIS ID #' + selectedEmpId)],
        ['Description', body.description || '(none)'],
        ['Dates', body.date_from + ' to ' + body.date_to],
        [body.without_pay ? 'Without pay' : 'Used', body.used + ' day(s)'],
      ],
      note: 'This writes a real record to HRIS. It cannot be undone from this tool.',
      confirmText: 'Write record',
    });
    if (!ok) return;
  }
  const btn = document.getElementById('addSubmitBtn');
  setBtnBusy(btn, 'Submitting...');
  document.getElementById('addResult').innerHTML = '';
  try {
    const res = await checkedFetch('/api/add', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body)
    });
    const data = await res.json();
    if (data.error) {
      showError('addResult', data.error);
    } else if (data.dry_run) {
      document.getElementById('addResult').innerHTML =
        '<div class="ok-box">Preview (dry run) — nothing was sent to HRIS.</div>' +
        '<pre style="margin-top:8px">' + JSON.stringify(data.result, null, 2) + '</pre>';
    } else {
      showSuccess('addResult', '✓ Record submitted successfully.');
    }
  } catch (err) {
    if (err.message !== 'Session expired') showError('addResult', 'Submission failed — check your connection and try again.');
  } finally {
    clearBtnBusy(btn, SUBMIT_ENABLED ? 'Submit to HRIS' : 'Preview payload (dry run)');
  }
});

// ---------------------------------------------------------------------
// Batch upload / review — one workspace per tab
//
// The Form 6 tab (leave, deducted) and the Service Credits Earned tab
// (grants, added) share every step: upload, OCR, match names against HRIS,
// review, submit. What differs — the row's label, which column its days go
// to, the fourth KPI, one filter, the confirmation — comes in as config, so
// the two can't drift apart.
// ---------------------------------------------------------------------

// Drives a status line during a slow step. Returns a small handle so the
// caller can move the bar along and then replace the whole panel with a
// one-line result.
function busyPanel(target, label, total) {
  const determinate = typeof total === 'number';
  target.innerHTML =
    '<div class="busy"><span class="spinner"></span>' +
    '<span class="busy-label"></span><span class="busy-count"></span></div>' +
    '<div class="progress' + (determinate ? '' : ' indeterminate') + '"><div></div></div>';
  const labelEl = target.querySelector('.busy-label');
  const countEl = target.querySelector('.busy-count');
  const barEl = target.querySelector('.progress > div');
  const started = Date.now();

  // With no progress to report (OCR), a ticking elapsed time is what tells
  // the user the app is still working rather than stuck.
  const tick = determinate ? null : setInterval(() => {
    countEl.textContent = Math.round((Date.now() - started) / 1000) + 's';
  }, 1000);

  labelEl.textContent = label;
  return {
    step(done, text) {
      if (text) labelEl.textContent = text;
      if (!determinate) return;
      countEl.textContent = done + ' / ' + total;
      barEl.style.width = (total ? (done / total) * 100 : 0) + '%';
    },
    done(text) {
      if (tick) clearInterval(tick);
      target.textContent = text;
    },
  };
}

// Rows the reader couldn't read at all don't show up as rows, so they are
// listed here — and kept on screen, unlike the status line, which the
// matching step overwrites.
// A special order's number goes right after "Vacation service credits" —
// "Vacation service credits, SO No. 1673 s. 2026, Aral Summer Program, ...".
// Any number put there before is replaced; the rest of the description,
// including anything the user edited, is left as it is.
function withSoNumber(description, so) {
  const prefix = 'Vacation service credits';
  // Commas would split the number from its series ("1673, s. 2026").
  so = (so || '').replace(/,/g, ' ').replace(/\\s+/g, ' ').trim().replace(/^S[OS]\\s*No\\.?\\s*|^No\\.?\\s*/i, '');
  // Wherever it sits: before a comma, before "(16 hrs)", or at the end.
  const cleaned = description.replace(/,\\s*SO No\\. [^,(]*?(?=\\s*\\(|,|$)|^SO No\\. [^,]*,\\s*/, '');
  if (!so) return cleaned;
  if (!cleaned.startsWith(prefix)) return `SO No. ${so}, ${cleaned}`;
  const rest = cleaned.slice(prefix.length).replace(/^,?\\s*/, '');
  return `${prefix}, SO No. ${so}` + (!rest ? '' : rest.startsWith('(') ? ` ${rest}` : `, ${rest}`);
}

function showNotes(box, notes) {
  box.innerHTML = notes.length
    ? '<div class="warn-box" style="margin-top:0.75rem"><b>Not everything on this upload could be read.</b>' +
      '<ul style="margin:6px 0 0 1.1rem;padding:0">' +
      notes.map(n => `<li>${escapeHtml(n)}</li>`).join('') + '</ul></div>'
    : '';
}

function nameGuess(d) {
  return `${d.last_name}, ${d.first_name} ${d.middle_initial || ''}`.trim();
}

// A settled row collapses to one line — with 70-odd rows, keeping a search
// box open on every already-matched one buries the rows that still need work.
function setMatchState(tr, row) {
  const done = tr.querySelector('.match-done');
  const search = tr.querySelector('.match-search');
  if (row.employeeName) {
    tr.querySelector('.match-picked').textContent = row.employeeName;
    tr.querySelector('.match-id').innerHTML =
      `<span>HRIS ID #${row.employeeId}</span>` +
      (row.autoMatched ? '<span class="pill pill-ok">Auto-linked</span>'
                       : '<span class="pill pill-neutral">Picked</span>');
    done.style.display = 'flex';
    search.style.display = 'none';
  } else {
    done.style.display = 'none';
    search.style.display = 'block';
  }
}

// Employee search results, each clickable. Shared by the review rows and the
// single-grant form.
function renderEmployeeOptions(box, results, onPick) {
  box.innerHTML = '';
  if (!results.length) {
    box.innerHTML = '<div style="color:#888">No matches — edit the name and search again</div>';
    return;
  }
  results.forEach(emp => {
    const opt = document.createElement('div');
    opt.textContent = emp.full_name + ' — ' + (emp.position || '?') + ' @ ' + (emp.school || '?');
    opt.onclick = () => { box.innerHTML = ''; onPick(emp); };
    box.appendChild(opt);
  });
}

// The slow half of an upload: one HRIS search per distinct name. Names go up
// in small groups so the bar moves on real progress and matched rows settle
// into the table as they land — and a failure part-way keeps what matched.
const MATCH_CHUNK = 4;

function createReviewWorkspace(p, cfg) {
  const $ = id => document.getElementById(p + id);
  const state = { rows: [], filter: 'all' };

  const rowEls = () => [...$('Table').querySelectorAll('tbody tr')];
  const visibleRowEls = () => rowEls().filter(tr => tr.style.display !== 'none');
  const draftOf = tr => state.rows[tr.dataset.idx].draft;
  const needsAttention = tr => {
    const row = state.rows[tr.dataset.idx];
    return tr.querySelector('.row-check').checked && (!row.employeeId || !!row.draft.parse_warning);
  };

  // Reflect the rows in the header box: ticked when all are, indeterminate
  // while only some are, so it never claims more than it means.
  function syncCheckAll() {
    const all = visibleRowEls().map(tr => tr.querySelector('.row-check'));
    const checked = all.filter(cb => cb.checked).length;
    const box = $('CheckAll');
    box.checked = all.length > 0 && checked === all.length;
    box.indeterminate = checked > 0 && checked < all.length;
  }

  // Filtering hides rows rather than re-rendering them, so every row's edits,
  // tick state and loaded candidates survive switching between views.
  function applyFilter() {
    rowEls().forEach(tr => {
      let show = true;
      if (state.filter === 'attention') show = needsAttention(tr);
      else if (state.filter === 'checked') show = tr.querySelector('.row-check').checked;
      else if (cfg.extraFilter && state.filter === cfg.extraFilter.key) show = cfg.extraFilter.test(draftOf(tr));
      tr.style.display = show ? '' : 'none';
      tr.classList.toggle('needs-attention', needsAttention(tr));
    });
    $('Empty').style.display = visibleRowEls().length === 0 ? 'block' : 'none';
    syncCheckAll();
  }

  function updateSummary() {
    const els = rowEls();
    const checked = els.filter(tr => tr.querySelector('.row-check').checked);
    const counts = { all: els.length, attention: els.filter(needsAttention).length, checked: checked.length };
    const people = new Set(els.map(tr => state.rows[tr.dataset.idx].employeeId).filter(Boolean)).size;
    const box = $('Summary');
    box.innerHTML = '';

    const chips = [
      ['all', 'All rows', counts.all, ''],
      ['checked', 'Checked for submit', counts.checked, ''],
      ['attention', 'Needs attention', counts.attention, 'chip-warn'],
    ];
    if (cfg.extraFilter) {
      chips.push([cfg.extraFilter.key, cfg.extraFilter.label,
                  els.filter(tr => cfg.extraFilter.test(draftOf(tr))).length, '']);
    }
    chips.forEach(([key, label, n, extra]) => {
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'chip ' + extra;
      b.setAttribute('aria-pressed', state.filter === key ? 'true' : 'false');
      b.innerHTML = `${label} <span class="n">${n}</span>`;
      b.onclick = () => { state.filter = key; updateSummary(); };
      box.appendChild(b);
    });

    const auto = state.rows.filter(r => r.autoMatched).length;
    $('Kpis').style.display = els.length ? 'grid' : 'none';
    $('KpiPeople').textContent = people;
    $('KpiPeopleFoot').textContent = `From ${counts.all} rows`;
    $('KpiAuto').textContent = auto;
    $('KpiAutoFoot').textContent =
      counts.all ? `${Math.round(auto / counts.all * 100)}% of rows resolved to one employee`
                 : 'Resolved to one employee';
    $('KpiAttention').textContent = counts.attention;
    $('Kpi4').textContent = cfg.kpi4(els, checked, draftOf);

    const ready = counts.checked - counts.attention;
    $('BarTitle').textContent = counts.attention
      ? `${ready} of ${counts.checked} checked row(s) ready — ${counts.attention} need attention`
      : `${counts.checked} checked row(s) ready for ledger commit`;
    const submitBtn = $('SubmitAll');
    if (!submitBtn.disabled) {
      submitBtn.textContent = `Submit checked rows (${counts.checked})`;
      // At rest the bar shows how much of the batch is ready to commit; during
      // a submission the loop below takes it over to show progress instead.
      $('Progress').firstElementChild.style.width =
        counts.all ? (Math.max(ready, 0) / counts.all * 100) + '%' : '0';
    }
    applyFilter();
  }

  function showMatchOptions(tr, idx, results) {
    renderEmployeeOptions(tr.querySelector('.match-results'), results, emp => {
      const row = state.rows[idx];
      row.employeeId = emp.id;
      row.employeeName = emp.full_name;
      row.autoMatched = false;
      setMatchState(tr, row);
      updateSummary();
    });
  }

  // Fold one chunk's results into the rows they belong to and update just
  // those rows in place — a full re-render would throw away any edit or tick
  // the user has already made further down the table while this runs.
  function applyMatches(matches) {
    let auto = 0;
    rowEls().forEach(tr => {
      const row = state.rows[tr.dataset.idx];
      const found = (matches[nameGuess(row.draft)] || {}).results;
      if (!found) return;
      row.candidates = found;
      if (found.length === 1) {
        row.employeeId = found[0].id;
        row.employeeName = found[0].full_name;
        row.autoMatched = true;
        auto++;
        tr.classList.add('just-matched');
        setTimeout(() => tr.classList.remove('just-matched'), 1300);
      }
      setMatchState(tr, row);
      if (!row.employeeName) showMatchOptions(tr, Number(tr.dataset.idx), found);
    });
    updateSummary();
    return auto;
  }

  async function autoMatch(status) {
    const names = [...new Set(state.rows.map(r => nameGuess(r.draft)))];
    const busy = busyPanel(status, `Matching ${names.length} name(s) against HRIS...`, names.length);
    let checked = 0, auto = 0, failed = 0;
    for (let i = 0; i < names.length; i += MATCH_CHUNK) {
      const chunk = names.slice(i, i + MATCH_CHUNK);
      let matches = {};
      try {
        const res = await checkedFetch('/api/match', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ names: chunk }),
        });
        const data = await res.json();
        if (data.error) { failed += chunk.length; } else { matches = data.matches || {}; }
      } catch (err) {
        if (err.message === 'Session expired') { busy.done(''); return; }
        failed += chunk.length;
      }
      auto += applyMatches(matches);
      checked += chunk.length;
      busy.step(checked, `Matching names against HRIS — ${auto} linked so far...`);
    }
    const left = state.rows.filter(r => r.draft.in_scope && !r.employeeId).length;
    busy.done(
      `Read ${state.rows.length} row(s). Matched ${auto} automatically` +
      (left ? ` — ${left} still need a pick below.` : '. Review before submitting.') +
      (failed ? ` (${failed} name(s) couldn't be looked up — use Match on those rows.)` : '')
    );
    updateSummary();
  }

  // One box per uploaded order: several orders can be read in one go, and
  // each has its own number. It's handwritten, so it's typed, not OCR'd.
  function renderSources(sources) {
    const box = $('Sources');
    if (!sources.length) { box.innerHTML = ''; return; }
    box.innerHTML =
      '<div class="field" style="margin-top:0.75rem;margin-bottom:0"><label>Special order number</label>' +
      '<div class="hint" style="margin:0 0 6px">Handwritten on the order, so type it in &mdash; it&rsquo;s added to the description of that order&rsquo;s rows.</div>' +
      sources.map((src, i) => `
        <div class="field-row" style="align-items:center;margin-bottom:6px">
          <div style="flex:2" class="sub">${escapeHtml(src.name)} · ${src.rows} row${src.rows === 1 ? '' : 's'}</div>
          <div style="flex:1"><input class="so-number" data-source="${i}" placeholder="SO No. — e.g. 1673 s. 2026"></div>
        </div>`).join('') + '</div>';
    box.querySelectorAll('.so-number').forEach(input => {
      input.addEventListener('input', () => {
        const source = Number(input.dataset.source);
        rowEls().forEach(tr => {
          const row = state.rows[tr.dataset.idx];
          if (row.draft.source !== source) return;
          const desc = tr.querySelector('.f-description');
          desc.value = withSoNumber(desc.value, input.value);
          row.draft.description = desc.value;
        });
      });
    });
  }

  function renderTable() {
    const tbody = $('Table').querySelector('tbody');
    tbody.innerHTML = '';
    $('ReviewCard').style.display = state.rows.length ? 'block' : 'none';

    state.rows.forEach((row, idx) => {
      const d = row.draft;
      const tr = document.createElement('tr');
      tr.dataset.idx = idx;
      const guess = nameGuess(d);
      const warn = d.parse_warning ? `<div class="warn2">${escapeHtml(d.parse_warning)}</div>` : '';
      tr.innerHTML = `
        <td><input type="checkbox" class="row-check" ${d.in_scope ? 'checked' : ''}></td>
        <td class="rowno">${escapeHtml(d.row_no || '')}</td>
        <td class="name-cell">
          <div class="scan-name">${escapeHtml(guess)}</div>
          <div class="scan-meta">
            ${d.position_raw ? `<span class="mini">${escapeHtml(d.position_raw)}</span>` : ''}
            ${cfg.rowLabel(d)}
          </div>${warn}
        </td>
        <td class="match-cell">
          <div class="match-done" style="display:none">
            <div class="grow">
              <div class="match-picked"></div>
              <div class="match-id"></div>
            </div>
            <button type="button" class="match-change dense">Change</button>
          </div>
          <div class="match-search">
            <div class="match-row">
              <input class="match-input" value="${escapeHtml(guess)}">
              <button type="button" class="match-find dense">Match</button>
            </div>
            <div class="match-results"></div>
          </div>
        </td>
        <td><input class="f-description" value="${escapeHtml(d.description || '')}" style="min-width:160px"></td>
        <td><input class="f-date-from" type="date" value="${d.date_from || ''}"></td>
        <td><input class="f-date-to" type="date" value="${d.date_to || ''}"></td>
        <td class="num">
          <input class="f-amount" type="number" step="${cfg.amountStep}" value="${cfg.amountOf(d) || ''}"
                 style="width:80px;text-align:right">
          ${cfg.amountPill(d)}
        </td>
        <td class="row-status"></td>
      `;
      tbody.appendChild(tr);

      tr.querySelector('.row-check').addEventListener('change', () => { syncCheckAll(); updateSummary(); });
      setMatchState(tr, row);
      if (!row.employeeName && row.candidates) showMatchOptions(tr, idx, row.candidates);

      tr.querySelector('.match-change').addEventListener('click', () => {
        row.employeeId = null;
        row.employeeName = '';
        setMatchState(tr, row);
        updateSummary();
        tr.querySelector('.match-input').focus();
      });

      tr.querySelector('.match-find').addEventListener('click', async () => {
        const q = tr.querySelector('.match-input').value.trim();
        const resultsBox = tr.querySelector('.match-results');
        if (q.length < 2) return;
        resultsBox.innerHTML = '<div class="loading">' + busyText('Searching...') + '</div>';
        try {
          const res = await checkedFetch('/api/search?q=' + encodeURIComponent(q));
          const data = await res.json();
          resultsBox.innerHTML = '';
          if (data.error) { resultsBox.innerHTML = `<div class="err">${escapeHtml(data.error)}</div>`; return; }
          showMatchOptions(tr, idx, data.results);
        } catch (err) {
          if (err.message !== 'Session expired') resultsBox.innerHTML = '<div class="err">Search failed</div>';
        }
      });
    });
    updateSummary();
  }

  $('Upload').addEventListener('click', async () => {
    const fileInput = $('File');
    const status = $('Status');
    const uploadBtn = $('Upload');
    const files = [...fileInput.files];
    if (!files.length) { status.textContent = 'Choose a file first.'; return; }
    $('Sources').innerHTML = '';
    const reading = 'Reading the scan — dewarping the page and OCR-ing each cell...';
    // One photo per page is the reliable way to shoot a multi-page order, so
    // several files are read in turn into one review table. A file that fails
    // doesn't cost the ones that read; its error becomes a note naming it.
    const busy = files.length > 1 ? busyPanel(status, `Reading ${files[0].name}...`, files.length) : busyPanel(status, reading);
    uploadBtn.disabled = true;
    try {
      const rows = [], notes = [], errors = [], sources = [];
      for (const [i, file] of files.entries()) {
        busy.step(i, `Reading ${file.name}...`);
        const fd = new FormData();
        fd.append('file', file);
        const res = await checkedFetch('/api/ocr?expect=' + cfg.expect, { method: 'POST', body: fd });
        const data = await res.json();
        const label = files.length > 1 ? `${file.name}: ` : '';
        if (data.error) { errors.push(label + data.error); continue; }
        if (data.rows.length) {
          sources.push({ name: file.name, rows: data.rows.length });
          data.rows.forEach(r => { r.source = sources.length - 1; });
        }
        rows.push(...data.rows);
        notes.push(...(data.notes || []).map(n => label + n));
      }
      busy.step(files.length);
      if (!rows.length) { showNotes($('Notes'), []); busy.done('Error: ' + errors.join(' ')); return; }
      state.rows = rows.map(r => ({ draft: r, employeeId: null, employeeName: '' }));
      showNotes($('Notes'), [...errors, ...notes]);
      busy.done('');
      renderTable();
      if (cfg.soNumbers) renderSources(sources);
      await autoMatch(status);
    } catch (err) {
      busy.done(err.message === 'Session expired' ? '' : 'Upload failed: ' + err);
    } finally {
      uploadBtn.disabled = false;
    }
  });

  $('CheckAll').addEventListener('change', (e) => {
    // Only what the user can currently see, so a tick never changes a row
    // hidden behind a filter.
    visibleRowEls().forEach(tr => { tr.querySelector('.row-check').checked = e.target.checked; });
    syncCheckAll();
    updateSummary();
  });

  $('SubmitAll').addEventListener('click', async () => {
    const toSubmit = rowEls().filter(tr => tr.querySelector('.row-check').checked);
    if (!toSubmit.length) {
      alertModal('No rows checked', 'Tick the rows you want written to HRIS first.');
      return;
    }
    const missingMatch = toSubmit.filter(tr => state.rows[tr.dataset.idx].employeeId === null);
    if (missingMatch.length) {
      await alertModal(
        `${missingMatch.length} row(s) have no employee yet`,
        'Click "Match" on each of those rows and pick the matching employee before submitting.'
      );
      return;
    }
    if (SUBMIT_ENABLED) {
      const people = new Set(toSubmit.map(tr => state.rows[tr.dataset.idx].employeeId)).size;
      const ok = await confirmModal({
        title: `Write ${toSubmit.length} record(s) to HRIS?`,
        rows: [['Rows', `${toSubmit.length} across ${people} employee(s)`], ...cfg.confirmRows(toSubmit, draftOf)],
        note: 'These are real records, written one at a time. It cannot be undone from this tool.',
        confirmText: `Write ${toSubmit.length} record(s)`,
      });
      if (!ok) return;
    }

    const submitBtn = $('SubmitAll');
    const progress = $('Progress');
    const bar = progress.firstElementChild;
    const submitStatus = $('SubmitStatus');
    setBtnBusy(submitBtn, `Submitting 1 of ${toSubmit.length}...`);
    progress.classList.remove('indeterminate');
    let done = 0, failed = 0;

    for (const tr of toSubmit) {
      const row = state.rows[tr.dataset.idx];
      submitStatus.innerHTML = busyText(`Submitting ${done + 1} of ${toSubmit.length}...`);
      setBtnBusy(submitBtn, `Submitting ${done + 1} of ${toSubmit.length}...`);
      // Keep the row being written in view, so a long batch stays followable.
      tr.scrollIntoView({ block: 'nearest' });
      const statusCell = tr.querySelector('.row-status');
      statusCell.innerHTML = busyText('Submitting...');
      statusCell.className = 'row-status';
      const body = {
        employee_id: row.employeeId,
        description: tr.querySelector('.f-description').value,
        date_from: tr.querySelector('.f-date-from').value,
        date_to: tr.querySelector('.f-date-to').value,
        ...cfg.amountFields(row.draft, tr.querySelector('.f-amount').value),
      };
      try {
        const res = await checkedFetch('/api/add', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(body),
        });
        const data = await res.json();
        if (data.error) {
          statusCell.textContent = 'Failed: ' + data.error;
          statusCell.className = 'row-status err';
          failed++;
        } else {
          statusCell.textContent = data.dry_run ? 'Previewed (dry run)' : '✓ Submitted';
          statusCell.className = 'row-status ok';
          tr.classList.add('row-done');
          tr.querySelector('.row-check').checked = false;
        }
      } catch (err) {
        statusCell.textContent = err.message === 'Session expired' ? 'Session expired' : 'Failed: ' + err;
        statusCell.className = 'row-status err';
        failed++;
      }
      done++;
      bar.style.width = (done / toSubmit.length * 100) + '%';
    }

    clearBtnBusy(submitBtn, 'Submit checked rows');
    // Submitted rows untick themselves, so what is left checked is exactly
    // what still needs another go.
    submitStatus.textContent = failed
      ? `${done - failed} submitted, ${failed} failed — the failures are still checked.`
      : `All ${done} submitted.`;
    syncCheckAll();
    updateSummary();
  });

  // Exposed for tests and for the WOP audit, which reads the leave rows.
  state.renderTable = renderTable;
  state.updateSummary = updateSummary;
  return state;
}

const sumAmounts = els => els.reduce((sum, tr) => sum + (parseFloat(tr.querySelector('.f-amount').value) || 0), 0);

// Leave: every row deducts — paid leave from "used", leave without pay into
// "wo_pay".
const leaveReview = createReviewWorkspace('form6', {
  expect: 'leave',
  rowLabel: d => {
    const wop = d.action_taken === 'WOP';
    return `<span class="${wop ? 'wop-text' : 'stat'}" style="font-size:0.6875rem">` +
           `${escapeHtml(d.leave_type)}${wop ? ' · without pay' : ' · with pay'}</span>`;
  },
  amountStep: '0.5',
  amountOf: d => d.used,
  amountPill: d => d.action_taken === 'WOP' ? '<div class="pill pill-warn" style="margin-top:4px">WOP</div>' : '',
  amountFields: (d, value) => ({ used: value, without_pay: d.action_taken === 'WOP' }),
  extraFilter: { key: 'wop', label: 'Without pay (WOP)', test: d => d.action_taken === 'WOP' },
  kpi4: (els, checked, draftOf) => els.filter(tr => draftOf(tr).action_taken === 'WOP').length,
  confirmRows: (toSubmit, draftOf) => {
    const wop = toSubmit.filter(tr => draftOf(tr).action_taken === 'WOP').length;
    return [['Under "used"', `${toSubmit.length - wop} row(s)`], ['Under "without pay"', `${wop} row(s)`]];
  },
});

// Service credits: every row adds, into "earned".
const vscReview = createReviewWorkspace('vsc', {
  expect: 'vsc',
  soNumbers: true,
  rowLabel: d => `<span class="stat" style="font-size:0.6875rem">Vacation service credits` +
                 `${d.hours != null ? ` · ${d.hours} hrs` : ''}</span>`,
  amountStep: 'any',
  amountOf: d => d.earned,
  amountPill: () => '<div class="pill pill-ok" style="margin-top:4px">+ Earned</div>',
  amountFields: (d, value) => ({ earned: value }),
  extraFilter: null,
  kpi4: (els, checked) => fmtDays(sumAmounts(checked)),
  confirmRows: toSubmit => [['Added to "earned"', `${fmtDays(sumAmounts(toSubmit))} day(s) in total`]],
});

// ----- Single grant, by hand -----

let vscEmployee = null;
let vscDescriptionTouched = false;

document.getElementById('vscEmpFind').addEventListener('click', async () => {
  const q = document.getElementById('vscEmpQ').value.trim();
  const box = document.getElementById('vscEmpResults');
  if (q.length < 2) return;
  box.innerHTML = '<div class="loading">' + busyText('Searching...') + '</div>';
  try {
    const res = await checkedFetch('/api/search?q=' + encodeURIComponent(q));
    const data = await res.json();
    if (data.error) { box.innerHTML = `<div class="err">${escapeHtml(data.error)}</div>`; return; }
    renderEmployeeOptions(box, data.results, emp => {
      vscEmployee = emp;
      document.getElementById('vscEmpQ').value = emp.full_name;
      document.getElementById('vscEmpChosen').textContent = `Selected: ${emp.full_name} — HRIS ID #${emp.id}`;
    });
  } catch (err) {
    if (err.message !== 'Session expired') box.innerHTML = '<div class="err">Search failed</div>';
  }
});

// VSC days = hours x factor / 8, to the 3 decimals the ledger keeps. The
// figure stays editable: the order may have granted something else.
function recomputeVsc() {
  const hours = parseFloat(document.getElementById('vscHours').value);
  const factor = parseFloat(document.getElementById('vscFactor').value);
  if (!isNaN(hours) && !isNaN(factor)) {
    const days = Math.round(hours * factor / 8 * 1000) / 1000;
    document.getElementById('vscEarned').value = days;
    document.getElementById('vscFormula').textContent =
      `${hours} hrs × ${factor} ÷ 8 = ${days} day(s) — edit it to match the order.`;
  }
  suggestVscDescription();
}

function suggestVscDescription() {
  if (vscDescriptionTouched) return;
  const from = document.getElementById('vscFrom').value;
  const to = document.getElementById('vscTo').value;
  const hours = document.getElementById('vscHours').value;
  if (!from) return;
  const fmt = iso => new Date(iso + 'T00:00:00').toLocaleDateString('en-US', { month: 'long', day: 'numeric' });
  const period = !to || to === from ? `${fmt(from)}, ${from.slice(0, 4)}` : `${fmt(from)} - ${fmt(to)}, ${to.slice(0, 4)}`;
  document.getElementById('vscDescription').value = withSoNumber(
    `Vacation service credits ${period}` + (hours ? ` (${hours} hrs)` : ''),
    document.getElementById('vscSo').value);
}

['vscHours', 'vscFactor'].forEach(id => document.getElementById(id).addEventListener('input', recomputeVsc));
['vscFrom', 'vscTo'].forEach(id => document.getElementById(id).addEventListener('change', suggestVscDescription));
document.getElementById('vscDescription').addEventListener('input', () => { vscDescriptionTouched = true; });
document.getElementById('vscSo').addEventListener('input', () => {
  const desc = document.getElementById('vscDescription');
  if (!vscDescriptionTouched) suggestVscDescription();
  if (vscDescriptionTouched || desc.value) desc.value = withSoNumber(desc.value, document.getElementById('vscSo').value);
});

document.getElementById('vscAddBtn').addEventListener('click', async () => {
  const result = document.getElementById('vscAddResult');
  const body = {
    employee_id: vscEmployee && vscEmployee.id,
    earned: document.getElementById('vscEarned').value,
    date_from: document.getElementById('vscFrom').value,
    date_to: document.getElementById('vscTo').value,
    description: document.getElementById('vscDescription').value,
  };
  if (!vscEmployee) { showError(result, 'Find and select the employee first.'); return; }
  const earned = parseFloat(body.earned);
  if (isNaN(earned) || earned <= 0) { showError(result, 'Credits earned must be a positive number.'); return; }
  if (!body.date_from || !body.date_to) { showError(result, 'Both start and end dates are required.'); return; }
  if (body.date_to < body.date_from) { showError(result, 'End date cannot be before start date.'); return; }
  if (!body.description.trim()) { showError(result, 'Description is required.'); return; }

  if (SUBMIT_ENABLED) {
    const ok = await confirmModal({
      title: 'Add these credits to HRIS?',
      rows: [
        ['Employee', vscEmployee.full_name],
        ['Description', body.description],
        ['Dates', body.date_from + ' to ' + body.date_to],
        ['Added to "earned"', earned + ' day(s)'],
      ],
      note: 'This writes a real Service Credit record. It cannot be undone from this tool.',
      confirmText: 'Add credits',
    });
    if (!ok) return;
  }
  const btn = document.getElementById('vscAddBtn');
  setBtnBusy(btn, 'Submitting...');
  result.innerHTML = '';
  try {
    const res = await checkedFetch('/api/add', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (data.error) showError(result, escapeHtml(data.error));
    else if (data.dry_run) {
      result.innerHTML = '<div class="ok-box">Preview (dry run) — nothing was sent to HRIS.</div>' +
        '<pre style="margin-top:8px">' + escapeHtml(JSON.stringify(data.result, null, 2)) + '</pre>';
    } else showSuccess(result, '✓ Credits added.');
  } catch (err) {
    if (err.message !== 'Session expired') showError(result, 'Submission failed — check your connection and try again.');
  } finally {
    clearBtnBusy(btn);
  }
});

// Find already-submitted without-pay rows whose days landed in "used".
// Read-only: it lists what it finds and repairs nothing on its own.
document.getElementById('wopAudit').addEventListener('click', async () => {
  const status = document.getElementById('wopStatus');
  const table = document.getElementById('wopWrap');
  const tbody = table.querySelector('tbody');
  const rows = leaveReview.rows
    .filter(r => r.draft.action_taken === 'WOP' && r.employeeId)
    .map(r => ({
      employee_id: r.employeeId,
      employee_name: r.employeeName,
      description: r.draft.description || '',
    }));
  if (!rows.length) { status.textContent = 'No matched without-pay rows to check.'; return; }

  const auditBtn = document.getElementById('wopAudit');
  setBtnBusy(auditBtn, 'Checking...');
  const busy = busyPanel(status, `Checking ${rows.length} row(s) against each ledger...`);
  tbody.innerHTML = '';
  let records;
  try {
    const res = await checkedFetch('/api/wop-audit', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ rows }),
    });
    const data = await res.json();
    if (data.error) { busy.done('Check failed: ' + data.error); return; }
    records = data.records || [];
  } catch (err) {
    busy.done(err.message === 'Session expired' ? '' : 'Check failed: ' + err);
    return;
  } finally {
    clearBtnBusy(auditBtn);
  }

  if (!records.length) {
    table.style.display = 'none';
    busy.done('Nothing to fix — no without-pay row is sitting in "used".');
    return;
  }

  busy.done(`${records.length} record(s) recorded as paid leave. Fix them one at a time, checking HRIS after the first.`);
  table.style.display = 'table';
  records.forEach(rec => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${rec.employee_name || rec.employee_id}</td>
      <td>${rec.description}</td>
      <td>${rec.used}</td>
      <td><button type="button" class="wop-fix">Move to without pay</button></td>
      <td class="row-status"></td>
    `;
    tbody.appendChild(tr);
    const btn = tr.querySelector('.wop-fix');
    const cell = tr.querySelector('.row-status');
    btn.addEventListener('click', async () => {
      if (SUBMIT_ENABLED) {
        const ok = await confirmModal({
          title: 'Move these days to without pay?',
          rows: [
            ['Employee', rec.employee_name || ('HRIS ID #' + rec.employee_id)],
            ['Record', '#' + rec.record_id],
            ['Description', rec.description],
            ['Days to move', rec.used + ' day(s), used \u2192 without pay'],
          ],
          note: 'This edits a real HRIS record.',
          confirmText: 'Move days',
        });
        if (!ok) return;
      }
      setBtnBusy(btn, 'Saving...');
      cell.innerHTML = busyText('Saving...');
      cell.className = 'row-status';
      try {
        const res = await checkedFetch('/api/wop-fix', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ record_id: rec.record_id, days: rec.used }),
        });
        const data = await res.json();
        if (data.error) {
          cell.textContent = data.error; cell.className = 'row-status err'; clearBtnBusy(btn);
        } else {
          cell.textContent = data.dry_run ? 'Dry run OK' : 'Moved';
          cell.className = 'row-status ok';
        }
      } catch (err) {
        if (err.message !== 'Session expired') {
          cell.textContent = 'Failed: ' + err; cell.className = 'row-status err'; clearBtnBusy(btn);
        }
      }
    });
  });
});

</script>
{% endif %}
</body>
</html>
"""


def _render(logged_in: bool, error: str | None = None):
    return render_template_string(
        PAGE,
        logged_in=logged_in,
        submit_enabled=SUBMIT_ENABLED,
        error=error,
        version=APP_VERSION,
        local_url=f"{HOST}:{PORT}",
        api_base=BASE_URL,
        api_host=urlparse(BASE_URL).netloc,
        user_label=session.get("user_label", "HRIS user"),
    )


@app.route("/")
def index():
    return _render(_get_client() is not None)


@app.route("/login", methods=["POST"])
def login():
    identifier = request.form.get("identifier", "")
    password = request.form.get("password", "")
    client = HrisClient()
    try:
        client.login(identifier, password)
    except HrisError as e:
        return _render(False, error=str(e))
    import uuid

    sid = uuid.uuid4().hex
    _clients[sid] = client
    session["sid"] = sid
    session["user_label"] = identifier
    return redirect(url_for("index"))


@app.route("/logout")
def logout():
    sid = session.pop("sid", None)
    if sid:
        _clients.pop(sid, None)
    return redirect(url_for("index"))


@app.route("/api/search")
def api_search():
    client = _get_client()
    if not client:
        return jsonify({"error": "not logged in"}), 401
    q = request.args.get("q", "")
    try:
        matches = client.search_employees(q)
    except HrisError as e:
        return jsonify({"error": str(e)}), 502
    return jsonify(
        {
            "results": [
                {
                    "id": m.id,
                    "full_name": m.full_name,
                    "position": m.position,
                    "school": m.school,
                }
                for m in matches
            ]
        }
    )


@app.route("/api/health")
def api_health():
    """Whether the OCR engine is actually available, for the header chip — so a
    missing Tesseract shows up before someone uploads a scan and hits an error."""
    import pytesseract

    try:
        version = str(pytesseract.get_tesseract_version()).split()[0]
        return jsonify({"ocr": True, "label": f"Tesseract {version}"})
    except Exception:  # noqa: BLE001 - any failure here means "not usable"
        return jsonify({"ocr": False, "label": "Tesseract not found"})


@app.route("/api/balance/<int:employee_id>")
def api_balance(employee_id: int):
    client = _get_client()
    if not client:
        return jsonify({"error": "not logged in"}), 401
    try:
        records = client.get_leave_credits(employee_id)
    except HrisError as e:
        return jsonify({"error": str(e)}), 502

    used_total = earned_total = wo_pay_total = 0.0
    ledger = []
    for r in records:
        attrs = r.get("attributes", r)
        lct = attrs.get("leave_credit_type") or {}
        lct_attrs = lct.get("attributes", lct) or {}
        if lct_attrs.get("leave_credit_type") == "Service Credit":
            used_total += attrs.get("used") or 0
            earned_total += attrs.get("earned") or 0
            wo_pay_total += attrs.get("wo_pay") or 0
        date = attrs.get("date_to") or attrs.get("date_from") or attrs.get("text_date") or ""
        ledger.append(
            {
                "date": date,
                "description": attrs.get("description") or "",
                "earned": attrs.get("earned") or 0,
                "used": attrs.get("used") or 0,
                "wo_pay": attrs.get("wo_pay") or 0,
            }
        )
    ledger.sort(key=lambda r: r["date"] or "", reverse=True)
    return jsonify(
        {
            "used": used_total,
            "earned": earned_total,
            "wo_pay": wo_pay_total,
            "balance": earned_total - used_total,
            "ledger": ledger,
        }
    )


@app.route("/api/add", methods=["POST"])
def api_add():
    client = _get_client()
    if not client:
        return jsonify({"error": "not logged in"}), 401
    body = request.get_json(force=True)

    # --- Input validation (Section 7 hardening) ---
    description = (body.get("description") or "").strip()
    if not description:
        return jsonify({"error": "Description is required."}), 400
    date_from = (body.get("date_from") or "").strip()
    date_to = (body.get("date_to") or "").strip()
    if not date_from or not date_to:
        return jsonify({"error": "Both start and end dates are required."}), 400
    if date_to < date_from:
        return jsonify({"error": "End date cannot be before start date."}), 400
    # A record either adds credit (a service credit grant, "earned") or
    # deducts it (leave, "used" / "wo_pay") — never both. The two directions
    # arrive as different fields so a grant can't be misread as a deduction.
    try:
        days = float(body.get("used") or 0)
        earned = float(body.get("earned") or 0)
    except (ValueError, TypeError):
        return jsonify({"error": "Days must be a number."}), 400
    if days > 0 and earned > 0:
        return jsonify({"error": "A record can't both add and deduct credit — send one or the other."}), 400
    if days <= 0 and earned <= 0:
        return jsonify({"error": "Enter a positive number of days to deduct or credits earned."}), 400

    # Leave taken without pay consumes no leave credit, so HRIS keeps it in
    # its own "wo_pay" column. Putting those days in "used" would deduct them
    # from the employee's balance as if they had been paid leave.
    without_pay = bool(body.get("without_pay")) and earned <= 0

    try:
        result = client.create_leave_credit(
            employee_id=int(body["employee_id"]),
            leave_credit_type_id=LEAVE_CREDIT_TYPE_IDS["Service Credit"],
            description=description,
            date_from=date_from or None,
            date_to=date_to or None,
            used=0 if (without_pay or earned > 0) else days,
            wo_pay=days if without_pay else 0,
            # The ledger keeps three decimals ("25.944"); a float carried
            # through JSON can arrive as 25.943999999.
            earned=round(earned, 3),
            dry_run=not SUBMIT_ENABLED,
        )
    except HrisError as e:
        return jsonify({"error": str(e)}), 502
    return jsonify({"dry_run": not SUBMIT_ENABLED, "result": result})


@app.route("/api/wop-audit", methods=["POST"])
def api_wop_audit():
    """Read-only: find records that were filed under "used" but describe
    leave taken without pay.

    Takes the without-pay rows the caller is holding ({employee_id,
    description, date_from}) and looks each one up in that employee's ledger,
    reporting any Service Credit entry that matches and still has its days in
    `used`. Writes nothing — the caller decides what to repair."""
    client = _get_client()
    if not client:
        return jsonify({"error": "not logged in"}), 401
    rows = (request.get_json(silent=True, force=True) or {}).get("rows")
    if not isinstance(rows, list):
        return jsonify({"error": "expected a list of rows"}), 400

    ledgers: dict[int, list] = {}
    found = []
    for row in rows:
        try:
            employee_id = int(row.get("employee_id"))
        except (TypeError, ValueError):
            continue
        description = (row.get("description") or "").strip()
        if employee_id not in ledgers:
            try:
                ledgers[employee_id] = client.get_leave_credits(employee_id)
            except HrisError as e:
                return jsonify({"error": str(e)}), 502

        for record in ledgers[employee_id]:
            attrs = record.get("attributes", record)
            lct = attrs.get("leave_credit_type") or {}
            if (lct.get("attributes", lct) or {}).get("leave_credit_type") != "Service Credit":
                continue
            if (attrs.get("description") or "").strip() != description:
                continue
            used = float(attrs.get("used") or 0)
            if used <= 0:
                # Already sitting in wo_pay — nothing to repair.
                continue
            found.append(
                {
                    "record_id": record.get("id"),
                    "employee_id": employee_id,
                    "employee_name": row.get("employee_name") or "",
                    "description": attrs.get("description") or "",
                    "date_from": attrs.get("date_from") or "",
                    "date_to": attrs.get("date_to") or "",
                    "used": used,
                    "wo_pay": float(attrs.get("wo_pay") or 0),
                }
            )

    # One ledger entry must never be offered for repair twice.
    unique = list({f["record_id"]: f for f in found}.values())
    return jsonify({"records": unique})


@app.route("/api/wop-fix", methods=["POST"])
def api_wop_fix():
    """Move one record's days from `used` to `wo_pay`."""
    client = _get_client()
    if not client:
        return jsonify({"error": "not logged in"}), 401
    body = request.get_json(silent=True, force=True) or {}
    try:
        record_id = int(body["record_id"])
        days = float(body["days"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "record_id and days are required."}), 400
    if days <= 0:
        return jsonify({"error": "days must be a positive number."}), 400

    try:
        result = client.update_leave_credit(
            record_id, used=0, wo_pay=days, dry_run=not SUBMIT_ENABLED
        )
    except HrisError as e:
        return jsonify({"error": str(e)}), 502
    return jsonify({"dry_run": not SUBMIT_ENABLED, "result": result})


@app.route("/api/match", methods=["POST"])
def api_match():
    """Resolve many Form 6 names to HRIS employees in one request.

    A transmittal runs to dozens of rows, and several rows can belong to the
    same person (one per date range), so names are de-duplicated before being
    looked up. The caller decides what to do with each result — this only
    reports candidates, it never picks one."""
    client = _get_client()
    if not client:
        return jsonify({"error": "not logged in"}), 401
    names = request.get_json(silent=True, force=True) or {}
    names = names.get("names")
    if not isinstance(names, list):
        return jsonify({"error": "expected a list of names"}), 400

    out: dict[str, dict] = {}
    for name in dict.fromkeys(n for n in names if isinstance(n, str) and n.strip()):
        try:
            matches = client.search_employees(name)
        except HrisError as e:
            out[name] = {"error": str(e)}
            continue
        out[name] = {
            "results": [
                {"id": m.id, "full_name": m.full_name, "position": m.position, "school": m.school}
                for m in matches
            ]
        }
    return jsonify({"matches": out})


@app.route("/api/ocr", methods=["POST"])
def api_ocr():
    client = _get_client()
    if not client:
        return jsonify({"error": "not logged in"}), 401
    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify({"error": "no file uploaded"}), 400

    suffix = os.path.splitext(file.filename)[1] or ".jpg"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name
        rows, notes = extract_form6_with_notes(tmp_path)
    except Exception as e:  # noqa: BLE001 - surface any OCR failure to the UI, not a 500 traceback
        return jsonify({"error": f"Couldn't read this file: {e}"}), 422
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    # Each tab takes one kind of document: leave transmittals deduct, service
    # credit grants add. A document of the other kind is sent back with where
    # it belongs, rather than its rows landing in a review table built for the
    # opposite direction.
    expect = request.args.get("expect")
    if expect in ("leave", "vsc"):
        wanted = [r for r in rows if r.kind == expect]
        if rows and not wanted:
            if expect == "leave":
                where = "a vacation service credit grant — upload it on the Service Credits Earned tab"
            else:
                where = "a leave transmittal (Form 6) — upload it on the Form 6 tab"
            return jsonify({"error": f"This looks like {where}."}), 422
        if len(wanted) < len(rows):
            notes = notes + [
                f"{len(rows) - len(wanted)} row(s) of the other kind of document were left out "
                "of this tab — upload those pages on the other tab."
            ]
        rows = wanted

    return jsonify(
        {
            # What couldn't be read — a page whose rows are missing entirely
            # isn't visible from the rows themselves.
            "notes": notes,
            "rows": [
                {
                    "row_no": r.row_no,
                    "last_name": r.last_name,
                    "first_name": r.first_name,
                    "middle_initial": r.middle_initial,
                    "position_raw": r.position_raw,
                    "leave_type": r.leave_type,
                    "action_taken": r.action_taken,
                    "date_from": r.date_from,
                    "date_to": r.date_to,
                    "description": r.description,
                    "used": r.used,
                    "parse_warning": r.parse_warning,
                    "in_scope": r.in_scope,
                    "kind": r.kind,
                    "earned": r.earned,
                    "hours": r.hours,
                }
                for r in rows
            ]
        }
    )


def _open_browser():
    webbrowser.open("http://127.0.0.1:5057")


if __name__ == "__main__":
    threading.Timer(1.0, _open_browser).start()
    app.run(host="127.0.0.1", port=5057, debug=False)
