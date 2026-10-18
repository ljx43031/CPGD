# -*- coding: utf-8 -*-
"""Post-Maneuver and Velocity-Full columns for blackout(v2)/pedestrian(v2)."""
import scipy.io as sio
import numpy as np
import sys
from scipy.stats import wilcoxon

sys.stdout.reconfigure(encoding="utf-8")
METHODS = [("cpgd_smooth", "CPGD"), ("cpgd", "CPGD-MMEN-KF"), ("dlstm", "NB-DLSTM"),
           ("transformer", "Transformer"), ("csdi", "CSDI"), ("imm", "IMM-EKF"),
           ("T_FoT_GP", "T-FoT-GP"), ("T_FoT_StP", "T-FoT-StP"), ("T_FoT", "T-FoT"),
           ("GPMT", "GPMT")]

for sc, d in [("blackout", r"C:\my_program\CPGD\mc_results\v2最终"),
              ("pedestrian", r"C:\my_program\CPGD\mc_results\v2行人最终")]:
    base = sio.loadmat(d + "\\results_%s.mat" % sc)
    n = 1195
    vm = base["valid_mask"].ravel().astype(bool)[:n]
    mw = base["maneuver_windows"]  # shape (n_mc, k, 2)?
    print("\n" + sc.upper(), "maneuver_windows shape:", np.shape(mw))
    # reference = cpgd
    refp = refv = None
    print(f"{'method':14s} {'PostM':>18s} {'VelFull':>18s}")
    for mk, lbl in METHODS:
        if mk + "_pos_err" not in base:
            continue
        e = base[mk + "_pos_err"][:n][:, :n]
        ve = base.get(mk + "_vel_err")
        # post-maneuver: 20 frames after each maneuver window end, valid frames only
        tr_list = []
        wins = np.atleast_3d(mw) if mw.ndim == 3 else mw
        for t in range(min(len(e), np.shape(mw)[0] if mw.ndim >= 2 else 0)):
            pass
        # simpler: build postm mask from maneuver windows of each trial
        mwm = np.zeros((e.shape[0], n), bool)
        mw_arr = np.array(mw)
        if mw_arr.ndim == 3:
            for t in range(min(e.shape[0], mw_arr.shape[0])):
                for w in mw_arr[t]:
                    s0, s1 = int(w[0]), min(int(w[1]) + 20, n)
                    mwm[t, s0:s1] = True
        elif mw_arr.ndim == 2 and mw_arr.size:
            for w in mw_arr:
                s0, s1 = int(w[0]), min(int(w[1]) + 20, n)
                mwm[:, s0:s1] = True
        mwm &= vm[None, :]
        trp = np.sqrt(np.nanmean(e[:, :n][mwm].reshape(e.shape[0], -1) ** 2, axis=1)) if mwm.any() else np.array([])
        trp = np.sqrt(np.nanmean(np.where(mwm, e[:, :n], np.nan) ** 2, axis=1))
        med = np.median(trp); q1 = np.percentile(trp, 25); q3 = np.percentile(trp, 75)
        mark = ""
        if mk != "cpgd_smooth" and trp.size and refp is not None:
            try:
                p = wilcoxon(trp, refp).pvalue
                mark = "$^{\\mathsection}$" if p < 0.001 else ("$^{\\ddagger}$" if p < 0.01 else ("$^{\\dagger}$" if p < 0.05 else ""))
            except ValueError:
                pass
        cellp = "%.2f [%.2f, %.2f]%s" % (med, q1, q3, mark)
        if ve is not None:
            trv = np.sqrt(np.nanmean(ve[:n][:, :n] ** 2, axis=1))
            med = np.median(trv); q1 = np.percentile(trv, 25); q3 = np.percentile(trv, 75)
            mark = ""
            if mk != "cpgd_smooth" and refv is not None:
                try:
                    p = wilcoxon(trv, refv).pvalue
                    mark = "$^{\\mathsection}$" if p < 0.001 else ("$^{\\ddagger}$" if p < 0.01 else ("$^{\\dagger}$" if p < 0.05 else ""))
                except ValueError:
                    pass
            cellv = "%.2f [%.2f, %.2f]%s" % (med, q1, q3, mark)
        else:
            cellv = "N/A"
        if mk == "cpgd_smooth":
            refp, refv = trp, trv
        print(f"{lbl:14s} {cellp:>18s} {cellv:>18s}")
