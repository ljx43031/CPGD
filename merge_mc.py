"""
merge_mc.py
===========
把「基准 mat」+ 若干「augment mat」(augment_mc.py 产出的独立方法结果)按 seed
对齐合并成一个最终对比 mat,供 significance_analysis.py 使用。

用途:并行工作流 —— 多个控制台各跑 augment_mc.py 的不同 --methods(输出到独立
文件),最后用本脚本一次性合并。基准 mat 提供全部原方法字段 + 元信息;每个
augment mat 只带自己算的新方法字段。

对齐保证:合并前逐场景断言各 mat 的 seed 序列完全一致(顺序也一致),不一致即
中止。字段冲突时保留基准的(除非 --overwrite)。

用法:
  # 合并「完整对比」基准 + augment 目录下所有该场景的 augment mat
  python merge_mc.py --base_dir ./mc_results/完整对比 \
      --augment_dir ./mc_results/augment --out_dir ./mc_results/最终

  # 显式指定要并入的 augment 文件
  python merge_mc.py --base_dir ./mc_results/完整对比 \
      --augment ./mc_results/augment/results_medium__imm.mat \
                ./mc_results/因果/results_medium.mat \
      --scenarios medium --out_dir ./mc_results/最终

注:「因果」文件夹的 mat 是完整 mat(含全部原方法+因果),也可直接作为 augment
输入并入——本脚本只挑它里面基准没有的方法字段合并进来。
"""

import env_config  # noqa
import os
import glob
import argparse
import numpy as np
import scipy.io as sio


META_KEYS = {'n_mc', 'N_max', 'dt', 'scenario_name', 'scenario_label',
             'seed', 'augment_methods', 'cpgd_causal_n', 'maneuver_windows',
             'blackout_start', 'blackout_end'}


def _method_fields(mat):
    """返回 mat 里所有 <prefix>_(pos|vel|pos_err|vel_err) 字段名集合。"""
    keys = set()
    for k in mat:
        if k.startswith('__'):
            continue
        for suf in ('_pos', '_vel', '_pos_err', '_vel_err'):
            if k.endswith(suf):
                keys.add(k)
    return keys


def _assert_seed_aligned(base_seed, other_seed, name):
    b = np.asarray(base_seed).ravel().astype(int)
    o = np.asarray(other_seed).ravel().astype(int)
    if b.shape != o.shape or not np.array_equal(b, o):
        raise SystemExit(
            f"[error] {name} 的 seed 与基准不一致,无法对齐合并。\n"
            f"  base seeds[:8]={list(b[:8])}\n  {name} seeds[:8]={list(o[:8])}")


def merge_scene(base_path, augment_paths, out_path, overwrite=False):
    if not os.path.exists(base_path):
        print(f"  [skip] 基准 {base_path} 不存在"); return
    base = {k: v for k, v in sio.loadmat(base_path).items() if not k.startswith('__')}
    if 'seed' not in base:
        raise SystemExit(f"[error] 基准 {base_path} 无 seed 字段,无法对齐。")
    base_seed = base['seed']
    base_methods = _method_fields(base)
    merged = dict(base)
    added_log = []

    for ap_ in augment_paths:
        if not os.path.exists(ap_):
            print(f"    [skip] {ap_} 不存在"); continue
        a = {k: v for k, v in sio.loadmat(ap_).items() if not k.startswith('__')}
        if 'seed' not in a:
            print(f"    [warn] {os.path.basename(ap_)} 无 seed,跳过对齐校验并跳过"); continue
        _assert_seed_aligned(base_seed, a['seed'], os.path.basename(ap_))
        a_methods = _method_fields(a)
        added = []
        for k in sorted(a_methods):
            if k in merged and not overwrite:
                continue                       # 冲突默认保留基准
            merged[k] = a[k]
            added.append(k)
        # 因果类 mat 带 cpgd_causal_n 元信息:取各来源最大值(基准常为 0,因果源为 10)
        if 'cpgd_causal_n' in a:
            av = int(np.asarray(a['cpgd_causal_n']).ravel()[0])
            cur = int(np.asarray(merged['cpgd_causal_n']).ravel()[0]) if 'cpgd_causal_n' in merged else 0
            merged['cpgd_causal_n'] = np.int32(max(av, cur))
        if added:
            prefixes = sorted({k.rsplit('_', 1)[0].replace('_pos', '').replace('_vel', '')
                               for k in added})
            added_log.append((os.path.basename(ap_), len(added)))
            print(f"    + {os.path.basename(ap_)}: 并入 {len(added)} 字段")

    methods_final = sorted({k[:-8] for k in _method_fields(merged) if k.endswith('_pos_err')})
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    sio.savemat(out_path, merged, do_compression=True)
    print(f"  Saved → {out_path}")
    print(f"    最终方法({len(methods_final)}): {methods_final}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base_dir', type=str, default='./mc_results/完整对比')
    ap.add_argument('--augment_dir', type=str, default='./mc_results/augment',
                    help='自动收集该目录下 results_<scene>__*.mat 作为 augment 输入')
    ap.add_argument('--augment', type=str, nargs='*', default=None,
                    help='显式指定 augment mat 文件列表(覆盖 --augment_dir 自动收集)')
    ap.add_argument('--out_dir', type=str, default='./mc_results/最终')
    ap.add_argument('--scenarios', type=str, default='easy,medium,hard,blackout')
    ap.add_argument('--overwrite', action='store_true',
                    help='字段冲突时用 augment 覆盖基准(默认保留基准)')
    args = ap.parse_args()

    for sc in [s.strip() for s in args.scenarios.split(',')]:
        base = os.path.join(args.base_dir, f'results_{sc}.mat')
        if args.augment is not None:
            # 显式列表里挑文件名含该场景的
            aug = [p for p in args.augment if f'results_{sc}' in os.path.basename(p)]
        else:
            aug = sorted(glob.glob(os.path.join(args.augment_dir, f'results_{sc}__*.mat')))
        out = os.path.join(args.out_dir, f'results_{sc}.mat')
        print(f"\n{'='*66}\n  Merge {sc}  (基准 + {len(aug)} 个 augment)\n{'='*66}")
        merge_scene(base, aug, out, overwrite=args.overwrite)

    print(f"\n[Done] 合并完成 → {args.out_dir}/")
    print("  跑 significance_analysis.py --results_dir 指向该目录即可。")


if __name__ == '__main__':
    main()
