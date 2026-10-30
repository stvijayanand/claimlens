# ClaimLens seed data

- `taxonomy.csv`: canonical exterior part names, sided/unsided scope, estimate-style
  aliases, and the `hidden` marker (1 = part that cannot be photographed, e.g.
  behind-bumper internals; lines for these render as "needs explanation").
- `labor_ranges.csv`: reference labor-hour ranges per part + operation, used by
  `check_labor_range`.

Both files are hackathon seeds authored from common estimator vocabulary, not
measured data. Refine ranges and aliases from real estimates during hardening
(Issue #8). `claimlens.duckdb` (created at runtime) is git-ignored.
