"""
augment_mc.py
=============
把"之前 50 次 MC 里缺的方法"补算并插回现有 results_<scene>.mat,与原有方法
逐 trial 对齐(同一 seed = 同一条轨迹/观测),无需重跑昂贵的扩散/DL 方法。

原理:mc_compare_v8 的 trial 生成完全由 seed 决定(build_one_trial 在任何方法
推理之前),且每条 trial 的 seed 已存在 mat['seed']。本脚本:
  1) 读现有 mat,取出它接受的那批 seed(逐 trial);
  2) 用同样的 np/torch 随机种子 + 同样的 build_one_trial 重放,得到与原来
     逐帧一致的 states/dense_obs/valid_mask;
  3) 用 obs_pos_err 做逐 trial 硬校验(重放误差必须与 mat 存的吻合),不吻合
     即中止,绝不错位;
  4) 只对新方法(IMM-EKF / T-FoT-GP / T-FoT-StP / GPMT)算 pos/vel/pos_err/
     vel_err,按 N_max 截断,写回 mat 新字段,旧字段一字不动。

新方法(默认全部):
  imm            -> IMM-EKF(3×CT-EKF 交互多模型),回应 R5
  T-FoT-GP/StP/GPMT -> 原文三个传统基线(之前只跑了快的 T-FoT)

用法:
  python augment_mc.py                              # 全场景全新方法
  python augment_mc.py --scenarios blackout,medium
  python augment_mc.py --methods imm                # 只补 IMM
  python augment_mc.py --methods T-FoT-GP,T-FoT-StP,GPMT   # 只补慢 T-FoT
"""

import env_config  # noqa
import os
import argparse
import numpy as np
import scipy.io as sio

# 复用 mc_compare_v8 的数据生成/传统基线,保证与原 MC 完全同源
import mc_compare_v8 as MC
from imm_tracker import run_imm_inference

DT = MC.DT
TIMESTEPS = MC.TIMESTEPS

# 可补的新方法 -> mat 字段前缀('-'/' ' 转 '_' 与 mc_compare 存法一致)
TFOT_SET = {'T-FoT-GP', 'T-FoT-StP', 'GPMT'}
ALL_METHODS = ['imm'] + sorted(TFOT_SET)


def _safe(nm):
    return nm.replace('-', '_').replace(' ', '_')


def replay_trial(sc_cfg, duration, seed):
    """用给定 seed 重放一条 trial,返回与 mc_compare 一致的数据,或 None(被拒)。"""
    np.random.seed(seed)
    try:
        import torch; torch.manual_seed(seed)
    except Exception:
        pass
    blen = sc_cfg.get('blackout_len', 0)
    min_cov = sc_cfg.get('min_coverage', 0.75)
    states, timestamps, phys_obs, valid_mask, dense_obs, bo_s, bo_e = \
        MC.build_one_trial(sc_cfg, duration, seed, blackout_len=blen)

    # 与 run_one_trial 相同的接受判据
    if blen > 0 and bo_s >= 0:
        nat = valid_mask.copy().astype(int); nat[bo_s:bo_e] = 1
        natural_coverage = nat.mean()
    else:
        natural_coverage = valid_mask.mean()
    if natural_coverage < min_cov:
        return None
    if valid_mask.sum() < TIMESTEPS:
        return None
    return dict(states=states, timestamps=timestamps, valid_mask=valid_mask,
                dense_obs=dense_obs)


def compute_methods(trial, methods):
    """对一条 trial 算指定新方法,返回 {prefix: (N,4 或 N,2)}。"""
    states = trial['states']; ts = trial['timestamps']
    valid = trial['valid_mask']; dense = trial['dense_obs']
    out = {}
    if 'imm' in methods:
        out['imm'] = run_imm_inference(dense, valid, ts)          # (N,4)
    tfot_want = {m for m in methods if m in TFOT_SET}
    if tfot_want:
        res = MC.run_tfot_all(dense, valid, ts, tfot_filter=tfot_want)  # {nm:(N,2)}
        for nm, arr in res.items():
            out[nm] = arr
    return out


def augment_scene(base_path, out_path, sc_name, duration, methods, tol=1e-3):
    """从 base_path 取 seed/校验基准,只算 methods,写到独立的 out_path
    (仅含新方法字段 + seed/n_mc/N_max 元信息,供 merge_mc.py 合并)。base 文件不动。"""
    if not os.path.exists(base_path):
        print(f"  [skip] 基准 {base_path} 不存在"); return
    mat = sio.loadmat(base_path)
    sc_cfg = MC.SCENARIOS[sc_name]
    n_mc = int(mat['n_mc'].ravel()[0])
    N_max = int(mat['N_max'].ravel()[0])
    seeds = mat['seed'].ravel().astype(int)
    obs_err_ref = mat.get('obs_pos_err', None)
    print(f"\n{'='*66}\n  Augment {sc_name}: n_mc={n_mc} N_max={N_max} "
          f"methods={methods}\n  base={base_path}\n  out ={out_path}\n{'='*66}")

    # 预分配新字段(全 0),逐 trial 填
    new = {}
    for m in methods:
        pfx = 'imm' if m == 'imm' else _safe(m)
        new[f'{pfx}_pos'] = np.zeros((n_mc, N_max, 2), np.float32)
        new[f'{pfx}_pos_err'] = np.zeros((n_mc, N_max), np.float32)
        if m == 'imm':                        # 只有 IMM 产速度;T-FoT 系仅位置
            new[f'{pfx}_vel'] = np.zeros((n_mc, N_max, 2), np.float32)
            new[f'{pfx}_vel_err'] = np.zeros((n_mc, N_max), np.float32)

    from tqdm import tqdm
    n_verify_fail = 0
    for i in tqdm(range(n_mc), desc=sc_name):
        trial = replay_trial(sc_cfg, duration, int(seeds[i]))
        if trial is None:
            raise SystemExit(
                f"[error] seed={seeds[i]} 重放被拒(coverage),与原 MC 不一致——"
                f"检查 duration/场景配置是否与原跑一致。")
        N = min(len(trial['states']), N_max)
        gt = trial['states'][:N]
        # 硬校验:重放观测误差须与 mat 存的一致(逐 trial),否则错位
        if obs_err_ref is not None:
            rep_obs_err = np.linalg.norm(trial['dense_obs'][:N] - gt[:, :2], axis=1)
            ref = obs_err_ref[i, :N]
            d = np.nanmax(np.abs(rep_obs_err - ref)) if N else 0.0
            if d > tol:
                n_verify_fail += 1
                if n_verify_fail <= 3:
                    print(f"  [warn] trial {i} seed={seeds[i]} obs_err 偏差 {d:.3g} > {tol}")
        res = compute_methods(trial, methods)
        gt_pos = gt[:, :2]; gt_vel = gt[:, 2:]
        for m in methods:
            pfx = 'imm' if m == 'imm' else _safe(m)
            arr = res[m][:N]
            new[f'{pfx}_pos'][i, :N] = arr[:, :2]
            new[f'{pfx}_pos_err'][i, :N] = np.linalg.norm(arr[:, :2] - gt_pos, axis=1)
            if m == 'imm':
                new[f'{pfx}_vel'][i, :N] = arr[:, 2:]
                new[f'{pfx}_vel_err'][i, :N] = np.linalg.norm(arr[:, 2:] - gt_vel, axis=1)

    if n_verify_fail:
        print(f"  [warn] {n_verify_fail}/{n_mc} trial 校验超差(仍写入,请核对 duration)。")
    else:
        print(f"  [ok] 全部 {n_mc} trial 逐条对齐校验通过。")

    # 只写独立输出:新方法字段 + 对齐用元信息(seed/n_mc/N_max)。基准文件不动。
    new['seed'] = seeds.astype(np.int32)
    new['n_mc'] = np.int32(n_mc)
    new['N_max'] = np.int32(N_max)
    new['scenario_name'] = sc_name
    new['augment_methods'] = ','.join(methods)
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    sio.savemat(out_path, new, do_compression=True)
    n_fields = len([k for k in new if k.endswith('_err')])
    print(f"  Saved → {out_path}  ({n_fields} 个新方法, 每方法 pos/vel/err 字段)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base_dir', type=str, default='./mc_results/完整对比',
                    help='基准 mat 所在目录(提供 seed 与逐 trial 校验基准,不会被修改)')
    ap.add_argument('--out_dir', type=str, default='./mc_results/augment',
                    help='独立输出目录:每场景一个 results_<scene>__<tag>.mat')
    ap.add_argument('--tag', type=str, default=None,
                    help="输出文件名后缀,区分不同方法批次(默认取 methods 拼接)。"
                         "并行多控制台时给不同 tag,避免互相覆盖。")
    ap.add_argument('--scenarios', type=str, default='easy,medium,hard,blackout')
    ap.add_argument('--methods', type=str, default=','.join(ALL_METHODS),
                    help=f"逗号分隔,可选: {ALL_METHODS}")
    ap.add_argument('--duration', type=float, default=120.0,
                    help='必须与原 MC 跑时的 --duration 一致(默认 120)')
    ap.add_argument('--tol', type=float, default=1e-3,
                    help='obs_pos_err 逐 trial 校验容差(m)')
    args = ap.parse_args()

    methods = [m.strip() for m in args.methods.split(',') if m.strip()]
    bad = [m for m in methods if m not in ALL_METHODS]
    if bad:
        raise SystemExit(f"[error] 未知方法 {bad},可选 {ALL_METHODS}")

    tag = args.tag or '_'.join(_safe(m) for m in methods)
    for sc in [s.strip() for s in args.scenarios.split(',')]:
        if sc not in MC.SCENARIOS:
            print(f"  [skip] 未知场景 {sc}"); continue
        base = os.path.join(args.base_dir, f'results_{sc}.mat')
        out = os.path.join(args.out_dir, f'results_{sc}__{tag}.mat')
        augment_scene(base, out, sc, args.duration, methods, tol=args.tol)

    print(f"\n[Done] augment 完成 → {args.out_dir}/  (tag={tag})")
    print("  用 merge_mc.py 把基准 + 各 augment mat 合并成最终对比 mat。")


if __name__ == '__main__':
    main()
