#!/bin/bash
cd ~/LRCCP
for s in 0 1 2; do
  python -m lrccp.train --transform butterfly --trainable --init random --lpp_lr 3e-3 --seed $s --out runs/dt_bfly_rand_lpp3e-3_s$s > runs/dt_bfly_rand_lpp3e-3_s$s.log 2>&1
  python -m lrccp.train --trainable --init random --lpp_lr 3e-3 --seed $s --out runs/dt_learn_rand_lpp3e-3_s$s > runs/dt_learn_rand_lpp3e-3_s$s.log 2>&1
done
for s in 0 1 2 3 4; do
  python -m lrccp.train --trainable --lpp_lr 3e-3 --seed $s --out runs/dt_learn_lpp3e-3_v2_s$s > runs/dt_learn_lpp3e-3_v2_s$s.log 2>&1
  python -m lrccp.train --transform butterfly --trainable --lpp_lr 3e-3 --seed $s --out runs/dt_bfly_learn_lpp3e-3_v2_s$s > runs/dt_bfly_learn_lpp3e-3_v2_s$s.log 2>&1
done