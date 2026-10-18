# -*- coding: utf-8 -*-
"""Render blackout(v2)+pedestrian(v2) table rows from significance CSVs:
Full/Valid/No-Obs/Post-Mnv medians with markers + Vel Full + P95/Div."""
import csv
import sys
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ORDER = [("cpgd_smooth", "CPGD"), ("cpgd", "CPGD-MMEN-KF"), ("dlstm", "NB-DLSTM"),
         ("transformer", "Transformer"), ("csdi", "CSDI"), ("imm", "IMM-EKF"),
         ("T_FoT_GP", "T-FoT-GP"), ("T_FoT_StP", "T-FoT-StP"), ("T_FoT", "T-FoT"),
         ("GPMT", "GPMT")]

def mark(p):
    if p is None or p != p:
        return ""
    return "$^{\\mathsection}$" if p < 1e-3 else ("$^{\\ddagger}$" if p < 1e-2 else ("$^{\\dagger}$" if p < 0.05 else ""))

def load(d, sc):
    rows = list(csv.DictReader(open(d + "\\significance_stats.csv")))
    return {(r["method"], r["segment"]): r for r in rows if r["scenario"] == sc and r["metric"] == "pos"}

for sc, d in [("blackout", r"C:\my_program\CPGD\mc_results\v2最终"),
              ("pedestrian", r"C:\my_program\CPGD\mc_results\v2行人最终")]:
    D = load(d, sc)
    print("\n### " + sc.upper())
    print("RAW WLS Obs Full:", "%.1f" % float(D[("obs", "Full")]["median"]))
    for mk, lbl in ORDER:
        if (mk, "Full") not in D:
            continue
        cells = []
        for seg in ["Full", "Valid", "No-Obs", "Post-Mnv"]:
            r = D[(mk, seg)]
            med, q1, q3 = float(r["median"]), float(r["q1"]), float(r["q3"])
            ref = D[("cpgd_smooth", seg)] if (mk, seg) in D else None
            cells.append("%.2f [%.2f, %.2f]%s" % (med, q1, q3, mark(float(r["wilcoxon_p"]))))
        # P95/Div from Full
        r = D[(mk, "Full")]
        p95, div = float(r["p95"]), float(r["diverge_frac"]) * 100
        print("%-13s | %s | P95 %.1f  Div %.0f%%" % (lbl, " | ".join(cells), p95, div))
    # velocity full
    print("-- velocity Full --")
    rows = list(csv.DictReader(open(d + "\\significance_stats.csv")))
    for mk, lbl in ORDER:
        r = [x for x in rows if x["scenario"] == sc and x["metric"] == "vel"
             and x["segment"] == "Full" and x["method"] == mk]
        if r:
            r = r[0]
            print("%-13s %s [%.2f, %.2f]%s" % (lbl, "%.2f" % float(r["median"]),
                  float(r["q1"]), float(r["q3"]), mark(float(r["wilcoxon_p"]))))
