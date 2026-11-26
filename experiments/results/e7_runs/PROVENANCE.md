# E7 per-run outputs (Sen1Floods11, 5 arms x 3 seeds)

Each `<arm>_s<seed>.npz` holds per-chip confusion counts (the input of `suite/E7_finetune/code/analyze_e7.py`); the matching `.json` holds the training log (epochs, best epoch, wall seconds, peak GPU memory, trainable parameters, test and Bolivia scores).

| Files | Origin |
|---|---|
| `clay_finetune_upernet_s0-2`, `dofa_finetune_upernet_s0-2` | outputs of the original 2026-11 runs (the only per-seed outputs that survived); copied unchanged from the original job directory |
| `clay_frozen_linear_s0-2`, `clay_frozen_upernet_s0-2`, `unet_scratch_s0-2` | re-run on 2026-11-17 with `train_e7.py --arm A --seed S --epochs 100 --max-gpu-gb 10` (the original per-seed outputs of these three arms were lost); same recipe, code and data |

`train_s` is wall-clock on a GPU shared with other jobs and is not a controlled timing. The original runs were also on a shared GPU. Regenerate the table with
`python experiments/suite/E7_finetune/code/analyze_e7.py experiments/results/e7_runs experiments/results/e7_finetune_results.json <chips.json from fetch_data.py>`.
