"""
significance_analysis.py
========================
Post-processing of the Monte-Carlo result .mat files produced by mc_compare_v8.py
to report dispersion and statistical significance -- directly addressing the
reviewers' objection (R3, R4, comments): "means over 50 trials but no standard
deviations, confidence intervals, or significance tests; the 'consistently
outperforms' claim cannot be assessed."

For every scenario and every evaluation segment (Full / Valid / No-Obs /
Post-Maneuver, matching Tables III-V), and for both position and velocity, it:

  * reconstructs per-trial RMSE from the stored per-frame errors,
  * reports mean +/- std and a 95% confidence interval across trials,
  * runs a paired Wilcoxon signed-rank test (and a paired t-test) of each
    method against CPGD on the matched trials [Wilcoxon 1945, ref 8],
  * reports CPGD's reduction relative to the BEST-per-scenario baseline
    (fixing R3's headline-number issue), stating which baseline that is.

It is pure post-processing: it re-uses the per-trial data already saved, so it
adds NO simulation cost. NaN-padded trials (e.g. causal mode run on a subset)
are handled by dropping them per-method, with the effective n reported.

Usage
-----
  python significance_analysis.py                         # all scenarios in ./mc_results
  python significance_analysis.py --scenarios easy,hard
  python significance_analysis.py --results_dir ./mc_results --out_dir ./stats_out
  python significance_analysis.py --post_mnv_frames 20 --alpha 0.05
"""

import env_config  # noqa
import os
import argparse
import numpy as np
import scipy.io as sio
from scipy import stats

# Reference estimator against which every other method is paired-tested.
# 默认以 cpgd(=CPGD-MMEN-KF) 为参照。可用 --reference 切换到 cpgd_smooth(=CPGD)。
REFERENCE = 'cpgd'

# Preferred display order / friendly labels. Unknown methods are appended.
# 字段前缀与 mc_compare_v8.py 对齐:
#   cpgd_smooth(扩散+Hann-OLA,无KF) = 论文 "CPGD";  cpgd(再过 MMEN-KF) = "CPGD-MMEN-KF"
#   cpgd_causal = 严格因果零滞后模式
METHOD_LABELS = {
    'obs':            'Raw WLS Obs.',
    'cpgd':           'CPGD-MMEN-KF',
    'cpgd_smooth':    'CPGD',
    'cpgd_causal':    'CPGD (Causal)',
    # ── 外部基线 ──
    'dlstm':          'NB-DLSTM [Gao2019]',
    'transformer':    'Transformer [Giuliari2020]',
    'transformer_causal': 'Transformer (Causal)',
    'csdi':           'CSDI [Tashiro2021]',
    'imm':            'IMM-EKF',
    'T_FoT_GP':       'T-FoT-GP',
    'T_FoT_StP':      'T-FoT-StP',
    'T_FoT':          'T-FoT',
    'GPMT':           'GPMT',
}
DISPLAY_ORDER = ['obs',
                 # 我们的方法(CPGD / +KF / 因果)
                 'cpgd', 'cpgd_smooth', 'cpgd_causal',
                 # 外部基线
                 'dlstm', 'transformer', 'transformer_causal', 'csdi',
                 'imm', 'T_FoT_GP', 'T_FoT_StP', 'T_FoT', 'GPMT']

# Methods that are NOT candidate baselines for the "best baseline" comparison.
# CPGD 的所有变体(含 smooth/causal)与原始观测排除;transformer_causal 同理
# (其 smooth 版 'transformer' 已计入,避免同一模型重复计数)。
NON_BASELINE = {'cpgd', 'cpgd_smooth', 'cpgd_causal',
                'obs', 'transformer_causal'}

# 尾部/发散阈值(m):一个 trial 的分段 RMSE 超过它即计为"发散"。用于量化盲区
# 尾部(极端 trial)——稳定性问题,与典型精度(中位数)区分。由 --diverge_thresh 覆盖。
DIVERGE_THRESH = 50.0


# ─────────────────────────────────────────────────────────────────────────────
# Method / metric discovery
# ─────────────────────────────────────────────────────────────────────────────
def discover_methods(mat, metric='pos'):
    """Return list of method prefixes that have a per-frame '<prefix>_<metric>_err'
    field in the .mat, ordered by DISPLAY_ORDER then alphabetically."""
    suffix = f'_{metric}_err'
    found = []
    for k in mat:
        if k.startswith('__'):
            continue
        if k.endswith(suffix):
            found.append(k[:-len(suffix)])
    # obs has no velocity; keep only what exists
    ordered = [m for m in DISPLAY_ORDER if m in found]
    ordered += sorted(m for m in found if m not in DISPLAY_ORDER)
    return ordered


# ─────────────────────────────────────────────────────────────────────────────
# Per-trial segment masks (reconstruct Tables III-V segments)
# ─────────────────────────────────────────────────────────────────────────────
def build_segment_masks(mat, trial, N, post_mnv_frames):
    """Return dict seg_name -> boolean frame mask (length N) for one trial.

    Segments (matching the manuscript):
      Full        : all frames
      Valid       : frames with a sensor fix (valid_mask == 1)
      No-Obs      : blackout frames (valid_mask == 0)
      Post-Mnv    : the `post_mnv_frames` frames following each detected
                    maneuver (turn-rate mode switch) peak.
    """
    valid = mat['valid_mask'][trial, :N].astype(bool)
    seg = {
        'Full':   np.ones(N, dtype=bool),
        'Valid':  valid,
        'No-Obs': ~valid,
    }
    # Post-Maneuver: use stored maneuver_windows (object array of (k,3) [s,peak,e])
    post = np.zeros(N, dtype=bool)
    mw = mat.get('maneuver_windows', None)
    if mw is not None:
        try:
            wins = mw[0, trial] if mw.ndim == 2 else mw[trial]
        except Exception:
            wins = None
        if wins is not None and np.size(wins) > 0:
            wins = np.atleast_2d(wins)
            for row in wins:
                peak = int(row[1])
                lo, hi = peak, min(N, peak + post_mnv_frames)
                if 0 <= lo < N:
                    post[lo:hi] = True
    seg['Post-Mnv'] = post
    return seg


def per_trial_rmse(err_frames, mask):
    """RMSE over the masked frames of one trial. err_frames is per-frame L2 error.
    Returns NaN if the trial is NaN-padded (absent) or the mask is empty."""
    e = err_frames[mask]
    e = e[np.isfinite(e)]
    if e.size == 0:
        return np.nan
    return float(np.sqrt(np.mean(e ** 2)))


def collect_segment_rmse(mat, method, metric, seg_name, N, post_mnv_frames):
    """Vector of per-trial RMSE for (method, metric, segment). Length = n_mc,
    with NaN for trials where the method has no data (e.g. causal subset)."""
    field = f'{method}_{metric}_err'
    if field not in mat:
        return None
    errs = mat[field]                    # (n_mc, N_max)
    n_mc = errs.shape[0]
    out = np.full(n_mc, np.nan, dtype=float)
    for t in range(n_mc):
        seg = build_segment_masks(mat, t, N, post_mnv_frames)
        m = seg.get(seg_name)
        if m is None:
            continue
        out[t] = per_trial_rmse(errs[t, :N], m)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────────────────────────────────────
def summarise(vec):
    """mean, std, n, 95% CI half-width (t-based), plus robust median / IQR /
    MAD over finite entries. Median & IQR are the preferred summary for the
    blackout (No-Obs) segment where a few diverging trials blow up the mean."""
    v = np.asarray(vec, dtype=float)
    v = v[np.isfinite(v)]
    n = v.size
    if n == 0:
        return dict(mean=np.nan, std=np.nan, n=0, ci=np.nan,
                    median=np.nan, q1=np.nan, q3=np.nan, iqr=np.nan, mad=np.nan)
    mean = float(np.mean(v))
    std = float(np.std(v, ddof=1)) if n > 1 else 0.0
    if n > 1:
        tcrit = stats.t.ppf(0.975, n - 1)
        ci = float(tcrit * std / np.sqrt(n))
    else:
        ci = np.nan
    median = float(np.median(v))
    q1, q3 = (float(np.percentile(v, 25)), float(np.percentile(v, 75))) if n > 1 else (median, median)
    iqr = q3 - q1
    mad = float(np.median(np.abs(v - median)))   # median absolute deviation
    # 尾部指标:p90/p95/max 刻画最坏情况,diverge_frac 量化发散 trial 比例
    p90 = float(np.percentile(v, 90)) if n > 1 else median
    p95 = float(np.percentile(v, 95)) if n > 1 else median
    vmax = float(np.max(v))
    diverge_frac = float(np.mean(v > DIVERGE_THRESH))
    return dict(mean=mean, std=std, n=n, ci=ci,
                median=median, q1=q1, q3=q3, iqr=iqr, mad=mad,
                p90=p90, p95=p95, max=vmax, diverge_frac=diverge_frac)


def paired_tests(ref_vec, other_vec):
    """Paired Wilcoxon signed-rank + paired t-test on trials where BOTH methods
    have finite RMSE. Returns dict with p-values, effective n, and mean diff
    (other - ref); positive => other is worse (higher error) than the reference."""
    a = np.asarray(ref_vec, dtype=float)
    b = np.asarray(other_vec, dtype=float)
    both = np.isfinite(a) & np.isfinite(b)
    a, b = a[both], b[both]
    n = a.size
    res = dict(n=n, mean_diff=np.nan, median_diff=np.nan,
               wilcoxon_p=np.nan, ttest_p=np.nan)
    if n < 1:
        return res
    res['mean_diff'] = float(np.mean(b - a))
    # median of the paired per-trial differences: the robust effect size that
    # matches what the Wilcoxon signed-rank test is actually assessing.
    res['median_diff'] = float(np.median(b - a))
    if n >= 2 and np.any((b - a) != 0):
        try:
            res['wilcoxon_p'] = float(stats.wilcoxon(a, b, zero_method='wilcox').pvalue)
        except Exception:
            res['wilcoxon_p'] = np.nan
        try:
            res['ttest_p'] = float(stats.ttest_rel(a, b).pvalue)
        except Exception:
            res['ttest_p'] = np.nan
    return res


def best_baseline_improvement(seg_data, ref='cpgd'):
    """Given {method: rmse_vec}, find the baseline (excluding CPGD variants/obs)
    with the lowest mean RMSE and report CPGD's relative reduction against it."""
    ref_mean = np.nanmean(seg_data[ref]) if ref in seg_data else np.nan
    best_name, best_mean = None, np.inf
    for m, vec in seg_data.items():
        if m in NON_BASELINE:
            continue
        mu = np.nanmean(vec)
        if np.isfinite(mu) and mu < best_mean:
            best_mean, best_name = mu, m
    if best_name is None or not np.isfinite(ref_mean):
        return None
    reduction = (best_mean - ref_mean) / best_mean * 100.0
    return dict(best=best_name, best_mean=best_mean, ref_mean=ref_mean,
                reduction_pct=reduction)


def stars(p):
    if not np.isfinite(p):
        return '  - '
    if p < 0.001:
        return '***'
    if p < 0.01:
        return ' **'
    if p < 0.05:
        return '  *'
    return ' ns'


# ─────────────────────────────────────────────────────────────────────────────
# Per-scenario analysis
# ─────────────────────────────────────────────────────────────────────────────
SEGMENTS = ['Full', 'Valid', 'No-Obs', 'Post-Mnv']


def analyse_scenario(mat, scenario, metric, post_mnv_frames, csv_rows):
    """Compute + print stats for one scenario/metric. Appends rows to csv_rows."""
    methods = discover_methods(mat, metric)
    if REFERENCE not in methods:
        print(f"  [skip] no '{REFERENCE}' {metric} data in {scenario}")
        return
    N = int(mat['N_max'].ravel()[0])
    label = str(mat['scenario_label'].ravel()[0]) if 'scenario_label' in mat else scenario

    unit = 'm' if metric == 'pos' else 'm/s'
    print(f"\n{'='*94}")
    print(f"  SCENARIO: {label}   |   metric: {metric.upper()} RMSE ({unit})")
    print(f"{'='*94}")

    n_mc = int(mat['n_mc'].ravel()[0]) if 'n_mc' in mat else mat['N_max'].size
    if n_mc < 6:
        print(f"  [WARN] n_mc={n_mc}: paired Wilcoxon cannot reach p<0.05 "
              f"(needs >=6 trials; min p at n=5 is 0.0625). Run the full 50-trial "
              f"MC before quoting Wilcoxon; t-test p-values are shown meanwhile.")

    for seg in SEGMENTS:
        # gather per-trial rmse for every method
        seg_data = {}
        for m in methods:
            vec = collect_segment_rmse(mat, m, metric, seg, N, post_mnv_frames)
            if vec is not None:
                seg_data[m] = vec
        if REFERENCE not in seg_data:
            continue

        # No-Obs (盲区) 段的均值被少数发散 trial 主导，中位数[IQR]更能反映典型精度。
        robust_seg = (seg == 'No-Obs')
        tag = '  <- median[IQR] is the headline here' if robust_seg else ''
        print(f"\n  --- Segment: {seg} ---{tag}")
        print(f"  {'Method':<24}{'mean':>9}{'std':>8}{'median':>9}"
              f"{'IQR':>8}{'n':>4}   {'Δmean':>9}{'Δmed':>9}"
              f"{'Wilcox':>8}{'t':>7}")
        print(f"  {'-'*97}")

        ref_vec = seg_data[REFERENCE]
        seg_summ = {}   # method -> summarise() dict, reused for the tail sub-table
        for m in methods:
            if m not in seg_data:
                continue
            s = summarise(seg_data[m]); seg_summ[m] = s
            lbl = METHOD_LABELS.get(m, m)
            if m == REFERENCE:
                line = (f"  {lbl:<24}{s['mean']:>9.2f}{s['std']:>8.2f}"
                        f"{s['median']:>9.2f}{s['iqr']:>8.2f}{s['n']:>4}"
                        f"   {'(ref)':>9}{'(ref)':>9}{'':>8}{'':>7}")
                pt = dict(n=s['n'], mean_diff=0.0, median_diff=0.0,
                          wilcoxon_p=np.nan, ttest_p=np.nan)
            else:
                pt = paired_tests(ref_vec, seg_data[m])
                line = (f"  {lbl:<24}{s['mean']:>9.2f}{s['std']:>8.2f}"
                        f"{s['median']:>9.2f}{s['iqr']:>8.2f}{s['n']:>4}"
                        f"   {pt['mean_diff']:>+9.2f}{pt['median_diff']:>+9.2f}"
                        f"{stars(pt['wilcoxon_p'])}{stars(pt['ttest_p'])}")
            print(line)

            # CSV row (mean/std for worst-case robustness + median/IQR/MAD for typical)
            csv_rows.append(dict(
                scenario=scenario, metric=metric, segment=seg, method=m,
                label=METHOD_LABELS.get(m, m),
                mean=s['mean'], std=s['std'], ci95=s['ci'], n=s['n'],
                median=s['median'], q1=s['q1'], q3=s['q3'], iqr=s['iqr'], mad=s['mad'],
                p90=s['p90'], p95=s['p95'], max=s['max'], diverge_frac=s['diverge_frac'],
                mean_diff_vs_ref=pt['mean_diff'], median_diff_vs_ref=pt['median_diff'],
                wilcoxon_p=pt['wilcoxon_p'], ttest_p=pt['ttest_p']))

        # 尾部/发散子表(仅盲区段,那里稳定性问题最突出):量化"最坏情况"与发散率,
        # 与上表的"典型精度"(中位数)互补,凸显各方法盲区尾部差异。
        if robust_seg:
            print(f"  {'.'*97}")
            print(f"  [Tail/robustness] {'Method':<22}{'p90':>9}{'p95':>9}"
                  f"{'max':>10}{'>%.0fm'%DIVERGE_THRESH:>9}")
            for m in methods:
                if m not in seg_summ:
                    continue
                s = seg_summ[m]
                print(f"  {'':<18}{METHOD_LABELS.get(m, m):<22}"
                      f"{s['p90']:>9.2f}{s['p95']:>9.2f}{s['max']:>10.2f}"
                      f"{100*s['diverge_frac']:>8.0f}%")

        # best-baseline improvement (position only is the headline metric)
        imp = best_baseline_improvement(seg_data, ref=REFERENCE)
        if imp is not None:
            print(f"  {'-'*90}")
            print(f"  >> CPGD vs best baseline ({METHOD_LABELS.get(imp['best'], imp['best'])}"
                  f", {imp['best_mean']:.2f}{unit}): "
                  f"{imp['reduction_pct']:+.1f}% "
                  f"({'reduction' if imp['reduction_pct'] >= 0 else 'INCREASE'})")


# ─────────────────────────────────────────────────────────────────────────────
# Output writers
# ─────────────────────────────────────────────────────────────────────────────
def write_csv(csv_rows, path):
    import csv as _csv
    if not csv_rows:
        return
    fields = ['scenario', 'metric', 'segment', 'method', 'label', 'mean', 'std',
              'ci95', 'n', 'median', 'q1', 'q3', 'iqr', 'mad',
              'p90', 'p95', 'max', 'diverge_frac',
              'mean_diff_vs_ref', 'median_diff_vs_ref', 'wilcoxon_p', 'ttest_p']
    with open(path, 'w', newline='') as f:
        w = _csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in csv_rows:
            w.writerow(r)
    print(f"\n[Saved] per-cell statistics CSV -> {path}")


def write_latex(csv_rows, path, metric='pos', segment='Full', stat='median'):
    """Emit a paper-ready table for one metric/segment, columns = scenarios.

    stat='median' (default, recommended): 每格 "median [q1, q3]" + 相对参照的
        配对 Wilcoxon 显著性星。每场景最优(最低中位数)加粗。稳健,不受盲区/
        机动段少数发散 trial 影响 —— 直接回应审稿人"只报均值无离散度"。
    stat='mean':   每格 "mean ± std" + 星(旧版,均值口径,做鲁棒性/最坏情况补充)。
    """
    rows = [r for r in csv_rows if r['metric'] == metric and r['segment'] == segment]
    if not rows:
        return
    scenarios = sorted({r['scenario'] for r in rows})
    methods, seen = [], set()
    for m in DISPLAY_ORDER:
        if any(r['method'] == m for r in rows) and m not in seen:
            methods.append(m); seen.add(m)
    for r in rows:
        if r['method'] not in seen:
            methods.append(r['method']); seen.add(r['method'])

    idx = {(r['scenario'], r['method']): r for r in rows}
    unit = 'm' if metric == 'pos' else 'm/s'
    star_tex = {'  *': '$^{*}$', ' **': '$^{**}$', '***': '$^{***}$',
                ' ns': '', '  - ': ''}
    val_key = 'median' if stat == 'median' else 'mean'

    # 每场景最优值(最低中位数/均值)用于加粗;只在候选方法(排除参照自身也参与)里找
    best_per_sc = {}
    for sc in scenarios:
        vals = [(m, idx[(sc, m)][val_key]) for m in methods
                if (sc, m) in idx and np.isfinite(idx[(sc, m)][val_key])]
        if vals:
            best_per_sc[sc] = min(v for _, v in vals)

    lines = []
    lines.append('% Auto-generated by significance_analysis.py')
    lines.append('\\begin{table}[t]\\centering')
    ref_lbl = METHOD_LABELS.get(REFERENCE, REFERENCE)
    if stat == 'median':
        cap_stat = ('median [Q1, Q3] over MC trials (robust to the few diverging '
                    'trials that inflate the mean in blackout/post-maneuver segments)')
    else:
        cap_stat = 'mean $\\pm$ std over MC trials'
    lines.append(f'\\caption{{{segment} {metric.upper()} RMSE ({unit}): {cap_stat}. '
                 f'Best per column in \\textbf{{bold}}. '
                 f'$^{{*}}p<0.05$, $^{{**}}p<0.01$, $^{{***}}p<0.001$ '
                 f'(paired Wilcoxon vs.\\ {ref_lbl}).}}')
    col = 'l' + 'c' * len(scenarios)
    lines.append(f'\\begin{{tabular}}{{{col}}}\\hline')
    lines.append('Method & ' + ' & '.join(scenarios) + ' \\\\ \\hline')
    for m in methods:
        cells = []
        for sc in scenarios:
            r = idx.get((sc, m))
            if r is None or not np.isfinite(r.get(val_key, np.nan)):
                cells.append('--'); continue
            st = star_tex.get(stars(r['wilcoxon_p']), '') if m != REFERENCE else ''
            if stat == 'median':
                body = f"{r['median']:.2f} [{r['q1']:.2f}, {r['q3']:.2f}]"
            else:
                body = f"{r['mean']:.2f}$\\pm${r['std']:.2f}"
            # 最优加粗(星放粗体外)
            if sc in best_per_sc and abs(r[val_key] - best_per_sc[sc]) < 1e-9:
                body = f"\\textbf{{{body}}}"
            cells.append(body + st)
        lines.append(f"{METHOD_LABELS.get(m, m)} & " + ' & '.join(cells) + ' \\\\')
    lines.append('\\hline\\end{tabular}\\end{table}')
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"[Saved] LaTeX table ({segment}/{metric}, {stat}) -> {path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    global REFERENCE, DIVERGE_THRESH
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_dir', type=str, default='./mc_results')
    ap.add_argument('--out_dir', type=str, default='./mc_results')
    ap.add_argument('--scenarios', type=str, default='easy,medium,hard,blackout')
    ap.add_argument('--metrics', type=str, default='pos,vel')
    ap.add_argument('--post_mnv_frames', type=int, default=20)
    ap.add_argument('--alpha', type=float, default=0.05)
    ap.add_argument('--reference', type=str, default=REFERENCE,
                    help="参照方法(字段前缀),所有方法都与它做配对检验。"
                         "默认 'cpgd'(=CPGD-MMEN-KF);可选 'cpgd_smooth'(=CPGD)。")
    ap.add_argument('--diverge_thresh', type=float, default=DIVERGE_THRESH,
                    help="发散阈值(m):分段 RMSE 超过它的 trial 计为发散,用于尾部/发散率统计。")
    args = ap.parse_args()

    # 以命令行指定的参照方法/阈值覆盖模块级默认值(其余函数统一读取这些全局)
    REFERENCE = args.reference
    DIVERGE_THRESH = args.diverge_thresh

    os.makedirs(args.out_dir, exist_ok=True)
    scenarios = [s.strip() for s in args.scenarios.split(',')]
    metrics = [s.strip() for s in args.metrics.split(',')]

    csv_rows = []
    for sc in scenarios:
        path = os.path.join(args.results_dir, f'results_{sc}.mat')
        if not os.path.exists(path):
            print(f"  [skip] {path} not found")
            continue
        mat = sio.loadmat(path)
        for metric in metrics:
            analyse_scenario(mat, sc, metric, args.post_mnv_frames, csv_rows)

    write_csv(csv_rows, os.path.join(args.out_dir, 'significance_stats.csv'))
    # 主表:中位数[IQR](推荐入稿口径)。四段 × pos 各出一张,vel 出 Valid/Full。
    for seg in SEGMENTS:
        write_latex(csv_rows, os.path.join(args.out_dir, f'table_{seg.lower().replace("-","")}_pos_median.tex'),
                    metric='pos', segment=seg, stat='median')
    write_latex(csv_rows, os.path.join(args.out_dir, 'table_valid_vel_median.tex'),
                metric='vel', segment='Valid', stat='median')
    # 补充:均值±std 版(鲁棒性/最坏情况参照),保留 Full/No-Obs 两段位置
    write_latex(csv_rows, os.path.join(args.out_dir, 'table_full_pos_mean.tex'),
                metric='pos', segment='Full', stat='mean')
    write_latex(csv_rows, os.path.join(args.out_dir, 'table_noobs_pos_mean.tex'),
                metric='pos', segment='No-Obs', stat='mean')

    ref_lbl = METHOD_LABELS.get(REFERENCE, REFERENCE)
    print("\n[Done] Significance analysis complete.")
    print(f"  Reference method = '{REFERENCE}' ({ref_lbl}).")
    print(f"  Legend: mean_diff is (method - {REFERENCE}); NEGATIVE => method has")
    print(f"          LOWER error (BETTER than {REFERENCE}). Stars = paired Wilcoxon significance.")
    print("  命名: cpgd_smooth=CPGD(无KF), cpgd=CPGD-MMEN-KF, cpgd_causal=严格因果模式。")


if __name__ == '__main__':
    main()
