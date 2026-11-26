# Pinned SatML leaderboard data

`satml_leaderboard_83ca5cb.json` is a byte-for-byte copy of the file served with the Taylor Geospatial SatML leaderboard.

| Field | Value |
|---|---|
| Page | https://research.taylorgeospatial.org/satml-leaderboard/ (SatML Leaderboard, Taylor Geospatial and Microsoft AI for Good Lab) |
| Data file | https://research.taylorgeospatial.org/satml-leaderboard/leaderboard.json |
| Harness repository | https://github.com/torchgeo/geo-eval-harness |
| Harness commit | `83ca5cb81bc7ff5917fcd9081ca7062bcdd58c49`, committed 2026-11-16T10:41:20Z (read from the `source` block of the file and confirmed against the repository) |
| Data built | 2026-11-16T16:17:44Z (`source.fetched_at` in the file) |
| Copied here | 2026-11-18 |
| SHA-256 of the JSON | `87cd40ee4db8a96dcc0a276fab2543ec9f0aeaae9e2a99d223d4a77de4192761` |
| License of the harness | MIT (the license of the data file itself is not stated on the page) |

Structure: `models`, `datasets`, `methods`, `bands`, and `cells` = `[model index, dataset index, method index, band index, value, standard error]`.
`experiments/table2/from_leaderboard.py` reads this file to produce the Panel A and B rows of Table 2.
