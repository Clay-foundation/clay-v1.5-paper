# Machine-readable Table 2

Table 2 prints two panels. Panel A is the Taylor Geospatial SatML leaderboard: classification and segmentation, read from
`experiments/results/satml_leaderboard_83ca5cb.json` (source, harness commit and license in the `.md` next to it), except
Clay's segmentation cells, which are our runs in the same harness (marked †). Panel B is our own runs on benchmarks peer
models report and the leaderboard does not cover (OlmoEarth v1.3 tasks, GEO-Bench, Copernicus-Bench, the AlphaEarth/TESSERA
suite). The CSV and the scripts keep four block codes, printed as sub-headings: block A (classification) and B
(segmentation) form Panel A; block C (OlmoEarth, GEO-Bench, Copernicus-Bench tasks) and D (AlphaEarth/TESSERA suite) form
Panel B. Everywhere below, "Panel A-D" means these block codes. `table2_cells.csv` has one row per printed value; the table body in
`DRAFT.md` and `latex/sections/02_model.tex` is generated from it. Run from the repository root:

```bash
python3 experiments/table2/from_leaderboard.py          # Panel A and B rows with ranks and leaders, from the pinned JSON
python3 experiments/table2/build_cells_csv.py           # rebuild table2_cells.csv (leaderboard rows + own runs from table2_cells_v1.csv)
python3 experiments/table2/render_table.py --write      # write the table body into DRAFT.md and the LaTeX source
python3 experiments/table2/render_table.py              # check that both files hold exactly the CSV rendering
python3 experiments/table2/summarize.py                 # aggregates (per-panel medians, pending cells)
python3 experiments/table2/check_cells.py               # verify every cell against its source
```

`make tables` runs `summarize.py` and `check_cells.py`. The caption and the footnote are written by hand.

## Columns of the CSV

| Column | Meaning |
|---|---|
| `panel`, `task`, `sensor` | Panel A-D, the row label as printed, and the printed "Sensor and metric" text |
| `model` | `clay`, `other_eo` (best other EO model) or `general` (best general model) |
| `slot` | `single`, or `first` / `second` for a printed pair "a / b" (Panel B: linear / UPerNet; Panel C: probe / fine-tune or kNN / fine-tune; Panel D: 10-shot / Max) |
| `value` | printed value; empty for `?` and `n/a` |
| `source` | `third_party:satml_leaderboard` (linked into the pinned JSON), `rerun_here` (a number in `experiments/results/*.json` reproduces the value), `not_regenerated`, `third_party:<bibkey>`, `pending` (printed `?`), empty for `n/a` |
| `result_file`, `result_key` | JSON file and key. Leaderboard cells use `model_id\|dataset_id\|method_id\|band_id` into the pinned JSON; other cells use the `/`-joined path to a leaf under `experiments/results/` (a trailing `*100` rescales a fraction) |
| `readout`, `head`, `seeds`, `metric` | readout (`cls`, `default`, ...), head (linear probe, UPerNet, LP, FT, kNN, 10-shot, Max), seeds (`unknown` unless stated in the text), metric |
| `notes` | `tag=` model tag, `model=` leaderboard model name, `band=`, `se=`, `rank=`, `marks=`, `labelmarks=`, `pending`, `n/a`, `from=` (old column a collapsed peer value came from) |

## Rules

* Leaderboard rows (`from_leaderboard.py`, docstring has the full rule): best band set per model; Clay v1.5 is the
  better of `tt_clay_v1_5_large` and `tt_clay_v1_5_large_cls`; EO model = leaderboard kind GFM, general model = kind
  Generic; the earlier Clay v1 Base is not offered as a comparator. Panel A is the linear probe on the 15 core
  datasets. Panel B uses the `seg-linear` and `seg-upernet` methods on the 19 datasets in the author's priority order;
  Clay has no segmentation values on the leaderboard yet.
* Rank = 1 + the number of EO models with a higher best value than Clay, out of the EO models with a value in that row.
  Entries sharing one leaderboard name (readout or input variants of one model, or the same checkpoint listed twice)
  count once. This rule gives smaller counts than counting every leaderboard entry (for example 49 instead of 69 on
  `aid`).
* Panels C and D: `other_eo` is the per-slot maximum over the old DOFA-L, OlmoEarth, best-peer, TESSERA and AlphaEarth
  columns, `general` the old DINOv2/v3 column (`DINOv2` when the result file names it, `DINO` when the cell comes from
  a published table). Peer cells of Vessel Detect., Solar Farm Detect., MADOS, m-cashewplant and Nandi are the values
  in the OlmoEarth v1.3 report.
* `?` cells are being computed and are excluded from every aggregate. `check_cells.py` counts them and does not fail
  on them. Row % Best uses the slots that have a Clay value; a row without one prints `?`.
* Summary row: one row, the median of the unrounded row % Best over all rows with a Clay value. `summarize.py` still
  prints the per-panel medians; the text quotes the Panel A one.
* Our own harness runs of models that are not on the leaderboard (`OWN_HARNESS` in `build_cells_csv.py`, results in
  `experiments/results/harness_*.csv`) compete for the Panel A peer cell of their column. A run that beats the
  leaderboard value replaces it (source `own_harness`, printed with a dagger); a run that equals it adds `tie=<tag>`,
  printed next to the leaderboard model's tag. `check_cells.py` recomputes both from the CSVs.
* The summary row also counts the model families behind each comparison column (`family()` in `summarize.py`).

## Current state (2026-11-19)

Rows: A 15, B 3, C 10, D 15 (43). Median % Best: A 98%, B 96%, C 94%, D 86%, all rows 94% (`summarize.py`). No pending
cells. Single-seed Clay fine-tunes: `Vessel Detect. (S1)` and `(L8)` at
48,000 steps, `Solar Farm Detect.` at 8,000 steps, `m-cashewplant` at 100 epochs; other seeds are in the `seeds` column.
`table2_cells_v1.csv` is the cell list of the previous Table 2 (38 rows, columns DOFA-L, OlmoEarth, best peer, TESSERA,
AlphaEarth, DINOv2/v3); it is the input for the own-run rows and is kept for provenance.
