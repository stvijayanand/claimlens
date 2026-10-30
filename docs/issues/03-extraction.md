# Issue 3: Extraction — damage assessment, estimate parsing, static report

- **Spec:** `docs/implementation-spec.md` §7a–§7b, §11 Phase 3
- **Depends on:** #1
- **Parallel with:** #2, #4, #6

## Goal

Crops → damage JSON; estimate file → line JSON; first static end-to-end report (no agent: naive part-name join for display only).

## Tasks

- [ ] `assess.py`: `assess_crop()` (crop + parent photo context, low effort) + `assess_car()`; retry-then-`not_visible` on schema failure; `MIN_ASSESS_CONFIDENCE` forcing rule
- [ ] `estimate.py`: `extract_estimate()` via file handling (no OCR); per-line validation; failures → `unparsed_count` + logged raw text, never guessed
- [ ] Static report dump (JSON): assessments + lines + naive join, proving the read path works pre-agent
- [ ] Choose the one supported printed estimate layout from real samples; record the choice in the issue/file header

## Acceptance criteria

- One car renders a static JSON report with assessments for every crop and lines for the sample estimate
- A malformed line is counted unparsed (not guessed); a low-confidence assessment is `not_visible` (not "undamaged")
- No agent/tools code touched

## Out of scope

- Reconcile agent and `record_flag` (#4, #5), eval harness (#6, #7), UI (#9)
