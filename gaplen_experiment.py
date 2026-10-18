# -*- coding: utf-8 -*-
"""Blackout-length generalisation experiment on the (revised) Pedestrian
scenario: forced gaps of 120/150/180/210 frames, CPGD (amortized smooth)
vs IMM-EKF, 20 trials, per-trial No-Obs-segment RMSE.

Same seeds as mc_results/v2行人基准 (base_seed 42, accepted order).
"""
import os
import sys
import numpy as np
import scipy.io as sio

sys.stdout.reconfigure(encoding="utf-8")
R = r"C:\my_program\CPGD\mc_results"
BASE = os.path.join(R, "v2行人基准")
OUT = os.path.join(R, "v2_gaplen_ped.mat")

MC = __import__("mc_compare_v8")
from imm_tracker import run_imm_inference
DEV = MC.DEVICE

LENGTHS = [120, 150, 180, 210]
N_TRIALS = 20

base = sio.loadmat(os.path.join(BASE, "results_pedestrian.mat"))
seeds = base["seed"].ravel().astype(int)[:N_TRIALS]
sc_cfg = dict(MC.SCENARIOS["pedestrian"])

print("Loading CPGD ...")
our_model = MC._load_our_model(MC.V8Config(), MC.CPGD_MODEL_PATH)
from mmen_is2 import PartialInfoInfer_P1
import torch
mmen = PartialInfoInfer_P1(MC.TIMESTEPS).to(DEV)
mmen.load_state_dict(torch.load(MC.MMEN_MODEL_PATH, map_location=DEV)); mmen.eval()
diff = MC.DiffModelV8(MC.V8Config(), DEV)
cfg = MC.V8Config()

res = {f"cpgd_noobs_{L}": np.full(N_TRIALS, np.nan) for L in LENGTHS}
res.update({f"imm_noobs_{L}": np.full(N_TRIALS, np.nan) for L in LENGTHS})

for i, seed in enumerate(seeds):
    for L in LENGTHS:
        built = MC.build_one_trial(sc_cfg, 120, int(seed), blackout_len=L)
        if built is None:
            continue
        states, timestamps, phys_obs, valid, dense, bo_s, bo_e = built
        states = np.asarray(states)
        gt = states[:, :2]
        # CPGD smooth
        _, smooth = MC.run_v8_inference(our_model, diff, cfg, mmen, dense, valid)
        err = np.linalg.norm(smooth[:, :2] - gt, axis=1)
        res[f"cpgd_noobs_{L}"][i] = np.sqrt(np.mean(err[bo_s:bo_e] ** 2))
        # IMM
        imm = run_imm_inference(dense, valid, timestamps)
        err = np.linalg.norm(imm[:, :2] - gt, axis=1)
        res[f"imm_noobs_{L}"][i] = np.sqrt(np.mean(err[bo_s:bo_e] ** 2))
    print(f"[{i+1}/{len(seeds)}] seed={seed} done")

out = {"n_trials": np.int32(N_TRIALS), "lengths": np.array(LENGTHS)}
out.update({k: v.astype(np.float64) for k, v in res.items()})
sio.savemat(OUT, out)
print("saved", OUT)

print("\n=== No-Obs RMSE medians (m) ===")
print("len   CPGD   IMM")
for L in LENGTHS:
    print(f"{L:4d}  {np.nanmedian(res[f'cpgd_noobs_{L}']):5.1f}  {np.nanmedian(res[f'imm_noobs_{L}']):5.1f}")
