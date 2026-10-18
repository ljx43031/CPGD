# -*- coding: utf-8 -*-
"""Tail-behaviour statistics from stored per-trial data.

Validates data source against paper medians, then computes per-method:
  per-trial Full RMSE -> median, Q3, P95, divergence rate (>50 m)
  pooled per-frame    -> P95, P99 (matches the CDF figure view)
"""
import scipy.io as sio
import numpy as np

BASE = r"C:\my_program\CPGD\mc_results\新基准"
AUG  = r"C:\my_program\CPGD\mc_results\augment"
PBASE = r"C:\my_program\CPGD\mc_results\行人基准"
PAUG  = r"C:\my_program\CPGD\mc_results\行人augment"

PAPER_MEDIAN = {   # Full-track median, paper tables (for validation)
    ("easy", "cpgd_smooth"): 6.01, ("easy", "cpgd"): 6.03, ("easy", "imm"): 5.16,
    ("medium", "cpgd_smooth"): 10.31, ("medium", "cpgd"): 8.38, ("medium", "imm"): 7.40,
    ("hard", "cpgd_smooth"): 17.56, ("hard", "cpgd"): 17.01, ("hard", "imm"): 27.80,
    ("blackout", "cpgd_smooth"): 11.75, ("blackout", "cpgd"): 9.88, ("blackout", "imm"): 9.32,
    ("pedestrian", "cpgd_smooth"): 6.69, ("pedestrian", "cpgd"): 7.47, ("pedestrian", "imm"): 9.40,
}

def load_err(sc, key):
    """Return (n_trials, n_frames) position-error array for method key."""
    if sc == "pedestrian":
        base, aug = PBASE, PAUG
    else:
        base, aug = BASE, AUG
    bm = sio.loadmat(fr"{base}\results_{sc}.mat")
    if key == "imm":
        return bm["imm_pos_err"], bm["valid_mask"]
    am = sio.loadmat(fr"{aug}\results_{sc}__{'cpgd' if key in ('cpgd','cpgd_smooth') else key}.mat")
    if key == "cpgd_smooth":
        return am["cpgd_smooth_pos_err"], bm["valid_mask"]
    if key == "cpgd":
        return am["cpgd_pos_err"], bm["valid_mask"]
    pfx = {"dlstm": "dlstm", "transformer": "transformer", "csdi": "csdi"}[key]
    return am[f"{pfx}_pos_err"], bm["valid_mask"]

def trial_full_rmse(err, vm):
    """Per-trial RMSE over ALL frames (paper Full-track metric)."""
    e = err[:len(vm)]
    return np.sqrt(np.nanmean(e**2, axis=1))

print(f"{'scen':10s} {'method':12s} {'med':>6s} {'paper':>6s} {'Q3':>6s} {'P95':>7s} {'div%':>5s} {'fp95':>6s} {'fp99':>7s}")
rows = {}
for sc in ["easy", "medium", "hard", "blackout", "pedestrian"]:
    for key in ["cpgd_smooth", "cpgd", "imm", "dlstm", "transformer", "csdi"]:
        err, vm = load_err(sc, key)
        e = err[:len(vm)]
        tr = trial_full_rmse(err, vm)
        med = np.median(tr); q3 = np.percentile(tr, 75); p95 = np.percentile(tr, 95)
        div = 100.0 * np.mean(tr > 50)
        flat = e.ravel(); flat = flat[~np.isnan(flat)]
        fp95, fp99 = np.percentile(flat, 95), np.percentile(flat, 99)
        paper = PAPER_MEDIAN.get((sc, key), "")
        flag = "" if paper == "" else ("OK" if abs(med - paper) < 0.15 else f"**DIFF {paper}**")
        rows[(sc, key)] = (med, q3, p95, div, fp95, fp99)
        print(f"{sc:10s} {key:12s} {med:6.2f} {str(paper):>6s} {q3:6.2f} {p95:7.2f} {div:5.1f} {fp95:6.1f} {fp99:7.1f}  {flag}")
