"""
benchmark_speed.py
==================
测各方法的推理 wall-clock 时间,生成论文推理速度表(paper.tex 的 tab:timing)。
对应表格:每方法 Full-seq(s) / Per-step(ms) / 相对 CPGD 倍数。

现有 mc_compare_v8.py 不计时,故单独 benchmark:在同一批(默认 medium)trial 上,
每方法先 warmup 再重复计时取均值±std。CPGD 的 DDIM 每 S=16 帧触发一次(摊销),
计时如实反映这一摊销成本。

覆盖方法(有 checkpoint 才计):
  CPGD(cpgd_smooth 路径, 无KF) / CPGD-MMEN-KF(cpgd, 含KF) / CPGD-Causal
  NB-DLSTM / Transformer / CSDI / IMM-EKF / T-FoT-GP/StP/T-FoT/GPMT

用法:
  python benchmark_speed.py --scenario medium --n_trials 5 --repeats 3
  python benchmark_speed.py --methods cpgd,cpgd_kf,dlstm,transformer,csdi,imm,T-FoT-GP
"""

import env_config  # noqa
import os
import time
import argparse
import numpy as np
import torch

import mc_compare_v8 as MC
import baselines_dl as BDL
from imm_tracker import run_imm_inference

DEVICE = MC.DEVICE
DT = MC.DT


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_method(fn, repeats):
    """跑 repeats 次,返回每次 full-seq 秒的列表(不含第一次 warmup)。"""
    fn()  # warmup
    _sync()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        _sync()
        ts.append(time.perf_counter() - t0)
    return ts


def build_method_runners(models, dense, valid, ts_arr):
    """返回 {label: (thunk, kind)}。thunk 无参,跑一整条序列一次。"""
    v8, diff, cfg, mmen, dl, our = models
    R = {}
    # 我方主方法:primary(=v8微调)占 cpgd_* 字段
    prim = our.get(MC.PRIMARY_METHOD) or (next(iter(our.values())) if our else None)
    if prim is not None:
        m = prim['model']
        # CPGD(smooth,无KF)与 CPGD-MMEN-KF(含KF)共用 run_v8_inference,
        # 该函数同时算 smooth 和 kf;KF pass 每帧 <0.2ms,单独拆分意义不大,
        # 故 CPGD 计 smooth-only 成本(去掉 KF),CPGD-MMEN-KF 计完整成本。
        R['CPGD-MMEN-KF'] = (lambda mm=m: MC.run_v8_inference(mm, diff, cfg, mmen, dense, valid), 'ours')
        R['CPGD'] = (lambda mm=m: MC.run_v8_inference(mm, diff, cfg, None, dense, valid), 'ours')
        R['CPGD (Causal)'] = (lambda mm=m: MC.run_v8_inference_causal(
            mm, diff, cfg, dense, valid, ddim_steps=MC.CAUSAL_DDIM_STEPS), 'ours')
    # DL 基线
    if 'dlstm' in dl:
        R['NB-DLSTM [Gao2019]'] = (lambda: BDL.run_dlstm_inference(dl['dlstm'], dense, valid, DEVICE), 'dl')
    if 'transformer' in dl:
        R['Transformer [Giuliari2020]'] = (lambda: BDL.run_transformer_inference(
            dl['transformer'], dense, valid, DEVICE, window=MC.TIMESTEPS, stride=MC.STRIDE, mode='smooth'), 'dl')
    if 'csdi' in dl:
        mc, dc = dl['csdi']
        R['CSDI [Tashiro2021]'] = (lambda: BDL.run_csdi_inference(
            mc, dc, dense, valid, DEVICE, window=MC.TIMESTEPS, stride=MC.STRIDE), 'dl')
    # IMM-EKF(CPU)
    R['IMM-EKF'] = (lambda: run_imm_inference(dense, valid, ts_arr), 'classic')
    # T-FoT 系(CPU;GP/StP 慢)
    for nm in ('T-FoT-GP', 'T-FoT-StP', 'T-FoT', 'GPMT'):
        R[nm] = (lambda n=nm: MC.run_tfot_all(dense, valid, ts_arr, tfot_filter={n}), 'classic')
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', type=str, default='medium')
    ap.add_argument('--n_trials', type=int, default=5, help='用几条 trial 求均值')
    ap.add_argument('--repeats', type=int, default=3, help='每 trial 每方法重复计时次数')
    ap.add_argument('--duration', type=float, default=120.0)
    ap.add_argument('--base_seed', type=int, default=42)
    ap.add_argument('--methods', type=str, default=None,
                    help='逗号分隔只测子集(标签匹配子串);默认全部可用方法')
    ap.add_argument('--ref', type=str, default='CPGD', help='相对速度的参照方法标签')
    ap.add_argument('--out', type=str, default='./mc_results/最终/table_timing.tex')
    args = ap.parse_args()

    print("加载模型...")
    models = MC.load_models()  # (v8, diff, cfg, mmen, dl, our)
    sc_cfg = MC.SCENARIOS[args.scenario]

    # 收集 n_trials 条有效 trial
    trials = []
    seed = args.base_seed
    while len(trials) < args.n_trials:
        np.random.seed(seed); torch.manual_seed(seed)
        try:
            states, tstamps, phys, valid_mask, dense_obs, bo_s, bo_e = \
                MC.build_one_trial(sc_cfg, args.duration, seed,
                                   blackout_len=sc_cfg.get('blackout_len', 0))
            if valid_mask.mean() >= sc_cfg.get('min_coverage', 0.75) and valid_mask.sum() >= MC.TIMESTEPS:
                trials.append((dense_obs, valid_mask, tstamps))
        except Exception as e:
            print(f"  seed {seed} err: {e}")
        seed += 1
    T = len(trials[0][1])
    print(f"用 {len(trials)} 条 trial(每条 ~{T} 帧), 每方法 repeats={args.repeats}")

    # 每方法在每条 trial 上计时,汇总所有 (trial×repeat) 的 full-seq 秒
    per_method = {}  # label -> list of full-seq seconds
    for dense, valid, tarr in trials:
        R = build_method_runners(models, dense, valid, tarr)
        for label, (thunk, kind) in R.items():
            if args.methods and not any(s.strip() in label for s in args.methods.split(',')):
                continue
            try:
                secs = time_method(thunk, args.repeats)
            except Exception as e:
                print(f"  [warn] {label} 计时失败: {e}")
                continue
            per_method.setdefault(label, []).extend(secs)

    # 汇总 + 相对参照
    rows = []
    ref_mean = None
    for label, secs in per_method.items():
        arr = np.array(secs)
        full_mean, full_std = arr.mean(), arr.std()
        per_step_mean = full_mean / T * 1000.0
        per_step_std = full_std / T * 1000.0
        rows.append([label, full_mean, full_std, per_step_mean, per_step_std])
        if args.ref in label and ref_mean is None:
            ref_mean = full_mean
    if ref_mean is None and rows:
        ref_mean = rows[0][1]

    def rel(x):
        if abs(x - ref_mean) < 1e-9:
            return 'reference'
        r = x / ref_mean
        return f'${r:.2f}\\times$ slower' if r > 1 else f'${1/r:.2f}\\times$ faster'

    print(f"\n{'Method':<28}{'Full-seq(s)':>16}{'Per-step(ms)':>16}{'rel':>18}")
    for label, fm, fs, pm, ps in rows:
        print(f"{label:<28}{fm:>9.2f}±{fs:<5.2f}{pm:>10.2f}±{ps:<4.2f}{rel(fm):>18}")

    # LaTeX
    lines = ['% Auto-generated by benchmark_speed.py',
             '\\begin{table}[htbp]\\centering',
             f'\\caption{{Inference Time for a {T}-Step Sequence '
             f'({args.scenario} scenario, {DEVICE.type.upper()})}}',
             '\\label{tab:timing}',
             '\\setlength{\\tabcolsep}{4pt}',
             '\\begin{tabular}{lrrc}', '\\toprule',
             'Method & Full-seq (s) & Per-step (ms) & Speed rel.\\ ' + args.ref + ' \\\\',
             '\\midrule']
    for label, fm, fs, pm, ps in rows:
        star = '\\textbf{' + label + '}' if args.ref in label and 'reference' == rel(fm) else label
        lines.append(f'{star} & ${fm:.2f}\\pm{fs:.2f}$ & ${pm:.2f}\\pm{ps:.2f}$ & {rel(fm)} \\\\')
    lines += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"\n[Saved] LaTeX 速度表 -> {args.out}")


if __name__ == '__main__':
    main()
