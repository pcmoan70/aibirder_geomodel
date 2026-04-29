# Training-improvement strategies

After a 24-epoch run that stopped on `--patience 10`, the per-epoch trajectory
showed:

| signal       | epochs 13 → 24            | verdict                                  |
|--------------|---------------------------|------------------------------------------|
| `train_loss` | 0.00714 → 0.00690 ↓        | **not** over-fitting                     |
| `val_loss`   | 0.00693 → 0.00678 ↓        | still learning                           |
| `val_map`    | 0.3402 → 0.3492 ↑          | ranking quality still improving          |
| `val_f1_10`  | 0.269 → 0.270 (peak 0.275 ep 19) | threshold metric is noisy              |
| `geoscore`   | peak 0.3857 at ep 14, bouncing 0.37–0.38 after | **stopping signal** went stale before learning did |

Diagnosis: GeoScore is a composite of threshold-dependent terms
(`f1_10`, `list_ratio_10`) that wobble epoch-to-epoch. With cosine LR budgeted
for 100 epochs, by epoch 24 the LR had only fallen 0.0010 → 0.00089 — no real
annealing had happened yet, so the bouncing was amplified.

## Strategies (in order of expected payoff)

1. **Loosen early stopping.**
   `--patience 25` (or 30). Cheapest change. Lets the noisy GeoScore have room
   to recover its peak before the run dies.

2. **Match cosine length to actual run length.**
   `--num_epochs 40` so the cosine schedule actually decays during the run
   instead of parking near the peak LR. Late-epoch annealing reduces the
   threshold-metric noise that's currently driving early stopping.

3. **Low-LR finetune from `checkpoint_best.pt`.**
   After (1)+(2) plateau, resume at ~1/10 the peak LR for a short refinement:
   ```
   .venv/bin/python train.py --data_path /media/pc/HD1/aibirder_model_data/combined.parquet \
     --model_scale 0.75 --no_amp \
     --resume checkpoints/checkpoint_best.pt \
     --lr 1e-4 --num_epochs 20 --patience 15
   ```
   The "train loss still falling" signature at the end of the original run
   says there is signal left at lower LR.

Recommendation: do (1)+(2) first as a fresh run. Add (3) only if mAP plateaus
clearly and there is room left on the threshold metrics.
