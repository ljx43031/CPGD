"""
make_result_tables.py
=====================
行=方法,列= Position RMSE 的 Full/Valid/No-Obs/Post-Mnv 四段 + Velocity RMSE。
每格 median [Q1,Q3](稳健口径),显著性星为该方法 vs 参照(默认 cpgd_smooth=CPGD)的
配对 Wilcoxon,每列最优(最低中位数)加粗。

复用 significance_analysis 的重建逻辑(per-trial RMSE → 分段 → 中位数/Wilcoxon),
保证与显著性表数字一致。

用法:
  python make_result_tables.py --results_dir ./mc_results/最终 --out_dir ./mc_results/最终 \
      --reference cpgd_smooth --scenarios easy,medium,hard
  # blackout 想单独出也可以加进 --scenarios
"""

import os
import argparse
import numpy as np
import scipy.io as sio

import significance_analysis as SA

# 论文表里的方法显示顺序(字段前缀)。
ROW_ORDER = ['cpgd_smooth', 'cpgd', 'cpgd_causal', 'dlstm', 'transformer', 'csdi', 'imm',
             'T_FoT_GP', 'T_FoT_StP', 'T_FoT', 'GPMT']
# 论文表列(位置四段 + 速度Full)
POS_SEGMENTS = ['Full', 'Valid', 'No-Obs', 'Post-Mnv']

STAR = {'  *': '$^{\\dagger}$', ' **': '$^{\\ddagger}$', '***': '$^{\\mathsection}$', ' ns': '', '  - ': ''}


def cell_median(mat, method, metric, seg, N, pmf):
    vec = SA.collect_segment_rmse(mat, method, metric, seg, N, pmf)
    if vec is None:
        return None
    s = SA.summarise(vec)
    return s


def fmt_cell(s, star=''):
    if s is None or not np.isfinite(s['median']):
        return '--'
    return f"{s['median']:.2f} [{s['q1']:.2f}, {s['q3']:.2f}]{star}"


def build_scene_table(mat, scenario, reference, pmf):
    N = int(mat['N_max'].ravel()[0])
    n_mc = int(mat['n_mc'].ravel()[0]) if 'n_mc' in mat else int(mat['seed'].ravel().size)
    present = {k[:-8] for k in mat if k.endswith('_pos_err')}
    methods = [m for m in ROW_ORDER if m in present]

    # 预先算每格 summarise + 每列最优中位数
    # 列: pos×4段 + vel(Full)
    col_keys = [('pos', seg) for seg in POS_SEGMENTS] + [('vel', 'Full')]
    data = {}   # (method, metric, seg) -> summarise dict
    for m in methods:
        for metric, seg in col_keys:
            data[(m, metric, seg)] = cell_median(mat, m, metric, seg, N, pmf)

    best = {}
    for metric, seg in col_keys:
        vals = [(m, data[(m, metric, seg)]['median']) for m in methods
                if data[(m, metric, seg)] is not None
                and np.isfinite(data[(m, metric, seg)]['median'])
                # cpgd_smooth 的速度不可用(填 N/A),不参与速度列最优比较
                and not (metric == 'vel' and m == 'cpgd_smooth')]
        if vals:
            best[(metric, seg)] = min(v for _, v in vals)

    # 参照向量(算 Wilcoxon 用),按列缓存
    ref_vecs = {}
    for metric, seg in col_keys:
        ref_vecs[(metric, seg)] = SA.collect_segment_rmse(mat, reference, metric, seg, N, pmf)

    lines = []
    lines.append('\\begin{table}[htbp]\\centering')
    lines.append(f'\\caption{{{scenario.capitalize()} Scenario: Position and Velocity '
                 f'RMSE, median [Q1, Q3] over {n_mc} MC Trials. Best per column in \\textbf{{bold}}; '
                 f'$^{{\\dagger}}p{{<}}0.05$, $^{{\\ddagger}}p{{<}}0.01$, $^{{\\mathsection}}p{{<}}0.001$ '
                 f'(paired Wilcoxon vs.\\ {SA.METHOD_LABELS.get(reference, reference)}).}}')
    lines.append(f'\\label{{tab:{scenario}_results}}')
    lines.append('\\setlength{\\tabcolsep}{4pt}')
    lines.append('\\centering\\small\\begin{tabular}{lccccc}')
    lines.append('\\toprule')
    lines.append('\\multirow{2}{*}{Method} & \\multicolumn{4}{c}{Position RMSE (m)} '
                 '& Vel. (m/s) \\\\')
    lines.append('\\cmidrule(lr){2-5}\\cmidrule(lr){6-6}')
    lines.append('& Full & Valid & No-Obs & Post-Mnv & Full \\\\')
    lines.append('\\midrule')

    for m in methods:
        label = SA.METHOD_LABELS.get(m, m)
        # 主方法(参照)加粗方法名
        name = f'\\textbf{{{label}}}' if m == reference else label
        cells = []
        for metric, seg in col_keys:
            s = data[(m, metric, seg)]
            # CPGD(纯扩散平滑,无KF)的原始速度未经滤波、不可用 -> N/A;
            # 校准速度由 CPGD-MMEN-KF 提供。
            if metric == 'vel' and m == 'cpgd_smooth':
                cells.append('N/A'); continue
            # 速度对 T-FoT 系无输出 -> N/A
            if s is None:
                cells.append('N/A' if metric == 'vel' else '--'); continue
            star = ''
            if m != reference and ref_vecs[(metric, seg)] is not None:
                vec = SA.collect_segment_rmse(mat, m, metric, seg, N, pmf)
                pt = SA.paired_tests(ref_vecs[(metric, seg)], vec)
                star = STAR.get(SA.stars(pt['wilcoxon_p']), '')
            body = fmt_cell(s)
            if (metric, seg) in best and np.isfinite(s['median']) \
                    and abs(s['median'] - best[(metric, seg)]) < 1e-9:
                body = fmt_cell(s).replace(f"{s['median']:.2f}",
                                           f"\\textbf{{{s['median']:.2f}}}", 1)
            cells.append(body + star)
        lines.append(f'{name} & ' + ' & '.join(cells) + ' \\\\')

    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    lines.append('\\end{table}')
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_dir', type=str, default='./mc_results/最终')
    ap.add_argument('--out_dir', type=str, default='./mc_results/最终')
    ap.add_argument('--scenarios', type=str, default='easy,medium,hard')
    ap.add_argument('--reference', type=str, default='cpgd_smooth',
                    help='显著性参照(cpgd_smooth=论文CPGD;cpgd=CPGD-MMEN-KF)')
    ap.add_argument('--post_mnv_frames', type=int, default=20)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for sc in [s.strip() for s in args.scenarios.split(',')]:
        path = os.path.join(args.results_dir, f'results_{sc}.mat')
        if not os.path.exists(path):
            print(f"  [skip] {path} 不存在"); continue
        mat = sio.loadmat(path)
        tex = build_scene_table(mat, sc, args.reference, args.post_mnv_frames)
        out = os.path.join(args.out_dir, f'table_{sc}_results.tex')
        with open(out, 'w', encoding='utf-8') as f:
            f.write(tex + '\n')
        print(f"[Saved] {out}")


if __name__ == '__main__':
    main()
