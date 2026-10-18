# -*- coding: utf-8 -*-
"""Extract full per-segment statistics (median, Q1, Q3, significance vs CPGD)
for blackout(v2) and pedestrian(v2) from the merged mats, in paper-table format.
"""
import scipy.io as sio
import numpy as np
import sys
from scipy.stats import wilcoxon

sys.stdout.reconfigure(encoding="utf-8")
METHODS = [("cpgd_smooth", "CPGD"), ("cpgd", "CPGD-MMEN-KF"), ("dlstm", "NB-DLSTM"),
           ("transformer", "Transformer"), ("csdi", "CSDI"), ("imm", "IMM-EKF"),
           ("T_FoT_GP", "T-FoT-GP"), ("T_FoT_StP", "T-FoT-StP"), ("T_FoT", "T-FoT"),
           ("GPMT", "GPMT")]
SEGS = [("full", None, None), ("valid", "valid_mask", None),
        ("noobs", "valid_mask", "blackout"), ("postm", "valid_mask", "maneuver")]

def seg_mask(base, key, sc):
    vm = base["valid_mask"].ravel().astype(bool)
    if key == "full":
        return np.ones(len(vm), bool)
    if key == "valid":
        return vm
    if key == "noobs":
        return ~vm
    return None  # postm handled separately (not used here)

for sc, d in [("blackout", r"C:\my_program\CPGD\mc_results\v2最终"),
              ("pedestrian", r"C:\my_program\CPGD\mc_results\v2行人最终")]:
    base = sio.loadmat(d + "\\results_%s.mat" % sc)
    vm = base["valid_mask"].ravel().astype(bool)[:1195]
    noobs = ~vm
    n = len(vm)
    print("\n" + "=" * 90)
    print(sc.upper(), f"  n_frames={n}  valid={vm.sum()}  noobs={noobs.sum()}")
    print("=" * 90)
    print(f"{'method':14s} {'Full':>16s} {'Valid':>16s} {'NoObs':>16s} {'PostM*':>16s}")
    ref = {}
    for key, m, _ in SEGS[:3]:
        mask = vm if key == "valid" else (noobs if key == "noobs" else np.ones(n, bool))
        e = base["cpgd_pos_err"][:n][:, :n][:, mask]
        ref[key] = np.sqrt(np.nanmean(e ** 2, axis=1))
    for mk, lbl in METHODS:
        if mk + "_pos_err" not in base:
            print(f"{lbl:14s}  (missing)")
            continue
        cells = []
        for key in ["full", "valid", "noobs"]:
            mask = vm if key == "valid" else (noobs if key == "noobs" else np.ones(n, bool))
            e = base[mk + "_pos_err"][:n][:, :n][:, mask]
            tr = np.sqrt(np.nanmean(e ** 2, axis=1))
            med, q1, q3 = np.median(tr), np.percentile(tr, 25), np.percentile(tr, 75)
            mark = ""
            if mk != "cpgd":
                try:
                    p = wilcoxon(tr, ref[key]).pvalue
                    mark = "$^{\\mathsection}$" if p < 0.001 else ("$^{\\ddagger}$" if p < 0.01 else ("$^{\\dagger}$" if p < 0.05 else ""))
                except ValueError:
                    pass
            cells.append("%.2f [%.2f, %.2f]%s" % (med, q1, q3, mark))
        print(f"{lbl:14s} {cells[0]:>16s} {cells[1]:>16s} {cells[2]:>16s}")
