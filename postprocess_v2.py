# -*- coding: utf-8 -*-
"""One-shot post-processing after all v2 augment runs finish:
merge -> summary tables -> significance -> paper figures (blackout+pedestrian).
"""
import subprocess, sys, os

PY = sys.executable
R = r"C:\my_program\CPGD\mc_results"
FIGS = r"C:\投稿\协同定位与导航\论文内容\figures"
ENV = dict(os.environ, PYTHONIOENCODING="utf-8")

def run(cmd):
    print("\n>>", " ".join(cmd))
    r = subprocess.run(cmd, cwd=r"C:\my_program\CPGD", env=ENV,
                       capture_output=True, text=True)
    out = (r.stdout or "") + (r.stderr or "")
    print(out[-1500:])
    if r.returncode != 0:
        print("!! FAILED:", r.returncode)
    return r.returncode

# 1. merge blackout (base + all augment mats)
run([PY, "merge_mc.py", "--base_dir", os.path.join(R, "v2基准"),
     "--augment_dir", os.path.join(R, "v2augment"),
     "--out_dir", os.path.join(R, "v2最终"), "--scenarios", "blackout"])
# 2. merge pedestrian
run([PY, "merge_mc.py", "--base_dir", os.path.join(R, "v2行人基准"),
     "--augment_dir", os.path.join(R, "v2行人augment"),
     "--out_dir", os.path.join(R, "v2行人最终"), "--scenarios", "pedestrian"])
# 3. significance vs cpgd
run([PY, "significance_analysis.py", "--reference", "cpgd",
     "--results_dir", os.path.join(R, "v2最终"), "--out_dir", os.path.join(R, "v2最终")])
run([PY, "significance_analysis.py", "--reference", "cpgd",
     "--results_dir", os.path.join(R, "v2行人最终"), "--out_dir", os.path.join(R, "v2行人最终")])
# 4. figures
run([PY, "plot_paper_figures.py", "--results_dir", os.path.join(R, "v2最终"),
     "--out_dir", FIGS, "--scenarios", "blackout"])
run([PY, "plot_paper_figures.py", "--results_dir", os.path.join(R, "v2行人最终"),
     "--out_dir", FIGS, "--scenarios", "pedestrian"])
print("\nALL DONE")
