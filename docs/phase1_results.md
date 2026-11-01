# Phase 1 结果：可学习 DFT 的学习率扫描

## 实验设置

模型是 DT2DCNN：输入先经过 range DFT（256 点）和 Doppler DFT（128 点，零频居中），再接 3 层卷积和 3 层全连接，输出 12 类手势。数据用 HGR 天线 0，train / val / test 是 user 0–5 按 CubeLearn 官方 sample CSV 划分的（1080 / 360 / 720 个样本），outset 是没见过的 user 6、7（720 个样本）。每个配置训练 30 个 epoch，batch size 32，优化器 Adam，主干学习率 3e-4。两个 DFT 层单独用 `lpp_lr`，`lpp_lr = 0` 表示 DFT 冻结、就是固定 FFT。每个配置跑 3 个 seed（0、1、2），按 val accuracy 选最佳 epoch 的 checkpoint，test 和 outset 都只在这个 checkpoint 上评估一次。drift 是 ‖W − W₀‖_F / ‖W₀‖_F，W₀ 是 DFT 矩阵，在 best checkpoint 上算，对两个 DFT 层和 3 个 seed 取平均。

## 结果

test、outset 是 accuracy（%），写成均值 ± 样本标准差（n = 3，n−1 分母）。

| lpp_lr | drift | test | outset |
|---|---|---|---|
| 0（固定） | 0 | 66.9 ± 2.3 | 59.8 ± 4.1 |
| 3e-4 | 4% | 71.5 ± 4.8 | 62.3 ± 2.6 |
| 1e-3 | 7% | 73.7 ± 5.1 | 60.7 ± 4.0 |
| 3e-3 | 14% | 76.3 ± 4.4 | 59.8 ± 4.4 |

## Caveat

1. **seed 1 在每组都是最低的。** 四个配置里，seed 1 的 test 和 outset 都是三个 seed 中最低的。所以标准差主要反映的是 seed 之间的差异，不是每个配置自身的随机波动。按 seed 分开看，test 随 lpp_lr 基本单调上升（seed 2 在 1e-3 和 3e-3 都是 76.7）。不过训练本身有非确定性：同一个 seed 重跑，drift 也会有小差别，所以按 seed 配对比较只能当参考。
2. **标准差和差值在同一量级。** 每组只有 3 个 seed，相邻两档 lpp_lr 的 test 差 2–3 个点，小于各组 4–5 个点的标准差。固定 FFT 和 3e-3 之间差约 9 个点，大约是两倍标准差，值得注意但还算不上定论。这里没有做显著性检验。
3. **outset 不涨。** test 从 66.9 涨到 76.3，outset 却在四组里一直停在 60–62 左右。可学习 DFT 带来的提升只出现在见过的用户（test）上，没有推广到新用户（outset）。更细地看，outset 在 3e-4 时最高（62.3），之后随 lpp_lr 增大往下走（60.7、59.8），到 3e-3 回到固定 FFT 的水平。先升后降、drift 越大 outset 越低，这指向"DFT 层过拟合到训练用户"这个解释。这只是趋势，n = 3 不足以确认。

## 蝶形 vs 稠密

参数是两个变换层参数之和。

| 第一层 | 参数 | test | outset |
|---|---|---|---|
| 固定 FFT | 0 | 66.9 ± 2.3 | 59.8 ± 4.1 |
| 可学习稠密（3e-3） | 81920 | 76.3 ± 4.4 | 59.8 ± 4.4 |
| 可学习蝶形（3e-3） | 5888 | 76.0 ± 5.5 | 63.3 ± 1.0 |

**初步结果显示蝶形在约 1/14 参数下保住约 97% 的 in-set 增益，out-of-set 有提升趋势；待 5 seed 与学习率扫描确认。**

### 蝶形的 drift

蝶形的 drift 是把一层的所有级（range 8 级，Doppler 7 级）的系数拼成一根向量，算 ‖W − W₀‖_F / ‖W₀‖_F，W₀ 是 FFT 系数。每级 W₀ 的元素模都是 1，各级 ‖W₀‖ 相同，所以一层的总 drift 等于各级比值的均方根，不是算术平均。下面都是 3 个 seed 的平均，在 best checkpoint 上算。

| 层 | 稠密（3e-3） | 蝶形（3e-3） |
|---|---|---|
| range | 14% | 22% |
| Doppler | 14% | 17% |

蝶形每一级的 drift（b 是这一级的块大小）：

| 层 | b=2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|---|---|
| range | 0.170 | 0.152 | 0.156 | 0.181 | 0.215 | 0.246 | 0.285 | 0.319 |
| Doppler | 0.140 | 0.134 | 0.149 | 0.166 | 0.193 | 0.204 | 0.198 | — |

初步看，蝶形系数离 FFT 初值走得比稠密远，而且大体是块越大的级走得越远；range 层最后一级约是第 2 级的两倍。两列不能直接比：稠密的 drift 是在 256×256 矩阵上算的，蝶形的是在蝶形系数上算的，系数动 20% 不等于等效矩阵动 20%。要比，得把各级乘成等效矩阵再算 ‖W − F‖ / ‖F‖，这一步还没做。

## Reproduce

在仓库根目录运行。先生成 cache：把 CubeLearn HGR 原始数据（HuggingFace `zhaoymn/cubelearn`）放在 `~/data/cubelearn/hand_organized/{user}_{gesture}_{sample}.npy`，再运行 `python experiments/make_cache.py`。它会把天线 0 的 2880 个样本写成 `~/data/cubelearn/cache/hgr_ant0.npy`（约 7.5 GB，complex64 memmap）和 `hgr_ant0_meta.npy`，本机约 16 分钟。

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

每个 `.log` 的最后两行是 test 和 outset accuracy。前两条是按现在的 `train.py` 补写的：固定 FFT 和 3e-4 那两批是在加 `--lpp_lr` 之前跑的，当时所有参数共用一个 3e-4。和现在的写法在数学上等价，但训练有非确定性，重跑的数字不会逐位相同。
