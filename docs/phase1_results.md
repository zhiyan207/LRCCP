# Phase 1 results: learning-rate sweep for the learnable DFT

## Setup

The model is DT2DCNN: the input goes through a range DFT (256 points) and a Doppler DFT (128 points, zero frequency centered), then 3 conv layers and 3 fully connected layers, with 12 gesture classes as output. Data is HGR antenna 0. Train / val / test are users 0–5, split by the official CubeLearn sample CSVs (1080 / 360 / 720 samples). Outset is the unseen users 6 and 7 (720 samples). Each config trains for 30 epochs with batch size 32 and Adam, backbone learning rate 3e-4. The two DFT layers have their own `lpp_lr`; `lpp_lr = 0` means the DFT is frozen, i.e. a fixed FFT. Each config runs 3 seeds (0, 1, 2). The checkpoint is picked by best val accuracy, and test and outset are evaluated once, on that checkpoint only. Drift is ‖W − W₀‖_F / ‖W₀‖_F, where W₀ is the DFT matrix, computed on the best checkpoint and averaged over the two DFT layers and 3 seeds.

## Results

Test and outset are accuracy (%), given as mean ± sample std (n = 3, n−1 denominator).

| lpp_lr | drift | test | outset |
|---|---|---|---|
| 0 (fixed) | 0 | 66.9 ± 2.3 | 59.8 ± 4.1 |
| 3e-4 | 4% | 71.5 ± 4.8 | 62.3 ± 2.6 |
| 1e-3 | 7% | 73.7 ± 5.1 | 60.7 ± 4.0 |
| 3e-3 | 14% | 76.3 ± 4.4 | 59.8 ± 4.4 |

## Caveats

1. **Seed 1 is the lowest in every config.** In all four configs, seed 1 has the lowest test and outset of the three seeds. So the std mostly reflects differences between seeds, not run-to-run noise within a config. Looking at each seed separately, test goes up with lpp_lr almost monotonically (seed 2 is 76.7 at both 1e-3 and 3e-3). Training is not deterministic, though: rerunning the same seed gives slightly different drift, so per-seed comparisons are only a rough guide.
2. **The std is as large as the differences.** With 3 seeds per config, neighbouring lpp_lr values differ by 2–3 points on test, less than the 4–5 point std of each config. Fixed FFT vs 3e-3 is about 9 points, roughly two stds: worth noting, but not conclusive. No significance test was done.
3. **Outset does not improve.** Test goes from 66.9 to 76.3, but outset stays around 60–62 in all four configs. The gain from a learnable DFT shows up only on users seen in training (test) and does not carry over to new users (outset). In more detail, outset is highest at 3e-4 (62.3), then drops as lpp_lr goes up (60.7, 59.8), and at 3e-3 it is back at the fixed-FFT level. Up first and then down, with lower outset at higher drift: this points to the DFT layers overfitting to the training users. It is only a trend; n = 3 is not enough to confirm it.

## Butterfly vs dense

Parameters are the sum over both transform layers.

| First layer | Params | test | outset |
|---|---|---|---|
| Fixed FFT | 0 | 66.9 ± 2.3 | 59.8 ± 4.1 |
| Learnable dense (3e-3) | 81920 | 76.3 ± 4.4 | 59.8 ± 4.4 |
| Learnable butterfly (3e-3) | 5888 | 76.0 ± 5.5 | 63.3 ± 1.0 |

**Preliminary results: with about 1/14 of the parameters, the butterfly keeps about 97% of the in-set gain, and out-of-set shows an upward trend; to be confirmed with 5 seeds and a learning-rate sweep.**

### Butterfly drift

For the butterfly, drift concatenates the coefficients of all levels of a layer (8 levels for range, 7 for Doppler) into one vector and computes ‖W − W₀‖_F / ‖W₀‖_F, where W₀ is the FFT coefficients. Every entry of W₀ has modulus 1, so every level has the same ‖W₀‖, and the total drift of a layer is the root mean square of the per-level ratios, not their arithmetic mean. All numbers below are averages over 3 seeds, computed on the best checkpoint.

| Layer | Dense (3e-3) | Butterfly (3e-3) |
|---|---|---|
| range | 14% | 22% |
| Doppler | 14% | 17% |

Drift of each butterfly level (b is the block size of the level):

| Layer | b=2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|---|---|
| range | 0.170 | 0.152 | 0.156 | 0.181 | 0.215 | 0.246 | 0.285 | 0.319 |
| Doppler | 0.140 | 0.134 | 0.149 | 0.166 | 0.193 | 0.204 | 0.198 | — |

Preliminary: the butterfly coefficients move further from their init than the dense matrix does. In the range layer, drift rises with b from b = 8 on, from 0.156 to 0.319 at b = 256. In the Doppler layer it rises up to b = 64 (0.204) and stays there at b = 128 (0.198). The two columns of the first table are not directly comparable: dense drift is measured on the 256×256 matrix, butterfly drift on the butterfly coefficients, and 20% drift in the coefficients is not 20% drift in the matrix they multiply out to. A fair comparison needs the levels multiplied into the equivalent matrix and ‖W − F‖ / ‖F‖ computed on that. This has not been done yet.

## Reproduce

Run from the repo root. Build the cache first: put the raw CubeLearn HGR data (HuggingFace `zhaoymn/cubelearn`) at `~/data/cubelearn/hand_organized/{user}_{gesture}_{sample}.npy`, then run `python experiments/make_cache.py`. It writes the 2880 antenna-0 samples to `~/data/cubelearn/cache/hgr_ant0.npy` (about 7.5 GB, complex64 memmap) and `hgr_ant0_meta.npy`. This takes about 16 minutes on my machine.

```bash
for s in 0 1 2; do python -m lrccp.train --seed $s --out runs/dt_fixed_s$s > runs/dt_fixed_s$s.log 2>&1; done
for s in 0 1 2; do python -m lrccp.train --trainable --lpp_lr 3e-4 --seed $s --out runs/dt_learn_s$s > runs/dt_learn_s$s.log 2>&1; done
for s in 0 1 2; do python -m lrccp.train --trainable --lpp_lr 1e-3 --seed $s --out runs/dt_learn_lpp1e-3_s$s > runs/dt_learn_lpp1e-3_s$s.log 2>&1; done
for s in 0 1 2; do python -m lrccp.train --trainable --lpp_lr 3e-3 --seed $s --out runs/dt_learn_lpp3e-3_s$s > runs/dt_learn_lpp3e-3_s$s.log 2>&1; done
for s in 0 1 2; do python -m lrccp.train --transform butterfly --seed $s --out runs/dt_bfly_fixed_s$s > runs/dt_bfly_fixed_s$s.log 2>&1; done
for s in 0 1 2; do python -m lrccp.train --transform butterfly --trainable --lpp_lr 3e-3 --seed $s --out runs/dt_bfly_learn_lpp3e-3_s$s > runs/dt_bfly_learn_lpp3e-3_s$s.log 2>&1; done

python experiments/weight_drift.py dt_learn
python experiments/weight_drift.py dt_learn_lpp1e-3
python experiments/weight_drift.py dt_learn_lpp3e-3
python experiments/weight_drift.py dt_bfly_learn_lpp3e-3
```

The last two lines of each `.log` are the test and outset accuracy. The first two commands were written afterwards to match the current `train.py`: the fixed-FFT and 3e-4 batches were run before `--lpp_lr` existed, when all parameters shared one learning rate of 3e-4. That is mathematically the same as the current setup, but training is not deterministic, so a rerun will not reproduce the numbers digit for digit.
