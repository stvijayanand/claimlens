# ClaimLens issues — order and run guide

Source of truth for scope: `docs/implementation-spec.md` + `docs/design.md`. Each issue file lists its spec sections, tasks, and acceptance criteria.

## Order and parallel groups

| Wave | Issues | Notes |
|---|---|---|
| 1 | #1 Foundations | First; blocks everything |
| 2 | #2 Vision, #3 Extraction, #4 Tools, #6 Eval data | All four in parallel (each needs only #1) |
| 3 | #5 Agent loop | Needs #2 + #3 + #4 |
| 4 | #7 Eval runner, #9 Serve/UI | In parallel (both need #5; #7 also needs #6) |
| 5 | #8 Harden | Needs #7 |
| 6 | #10 Final eval + README | Needs #8 + #9 |

Dependency graph: `1 → {2, 3, 4, 6} → 5 → {7, 9} → 8 → 10` (7 also ← 6, 10 also ← 9).

## How to run implementation (prompt template)

One issue per session/turn. Copy, fill the bracketed parts, and paste:

> Implement `docs/issues/<file>` (Issue <N>: <Title>) in this repo. First read that issue file, then the spec sections it cites in `docs/implementation-spec.md`, plus `docs/design.md`. Stay strictly within the issue's scope and out-of-scope list. Its acceptance criteria are the definition of done: verify each one, run the relevant tests, and finish with a report of files changed, acceptance evidence, and anything deferred. Do not start any other issue.

Example: replace `<file>` with `01-foundations.md`, `<N>` with `1`, `<Title>` with `Foundations — config, client, contracts, DB, seeds`.

## Notes

- Parallel issues (#2/#3/#4/#6, then #7/#9) touch different files by design; if run in overlapping sessions, keep each session to its issue's file list.
- If behind schedule, #8 invokes the binding cut order (spec §14): labor flags off first, then repair-vs-replace; missing/phantom lines always ship.
