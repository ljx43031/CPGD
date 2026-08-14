"""
augment_mc.py
=============
按 seed 逐 trial 重放"新基准"的轨迹,只计算指定的方法,写到独立的 augment mat,
供 merge_mc.py 合并。用于**分方法并行**跑:基准只跑一个最快的方法(如 IMM)建立
seed/轨迹,其余所有方法(含 CPGD / DL / CSDI / 因果 / 慢 T-FoT)各开一个控制台
并行 augment。

原理:mc_compare_v8 的 trial 生成完全由 seed 决定(build_one_trial 在任何方法
推理之前),且每条 trial 的 seed 存在基准 mat['seed']。本脚本:
  1) 读基准 mat,取其接受的 seed 序列(逐 trial)与 N_max;
  2) 用同 seed + 同 build_one_trial 重放,得到与基准逐帧一致的 states/dense_obs/valid_mask;
  3) 用 obs_pos_err 逐 trial 硬校验(重放误差须与基准存的吻合),不吻合即警告;
  4) 只对请求的方法算 pos/vel/pos_err/vel_err(截断到 N_max),写回独立 mat 新字段。

⚠️ 基准必须与本次运行**同一轨迹代码 + 同 --duration + 同 --base_seed**,否则 seed
   重放出的轨迹不一致,校验会失败。

可算方法(--methods,逗号分隔):
  imm                              IMM-EKF(CPU,快)             -> imm_*
  cpgd                             CPGD smooth+MMEN-KF(GPU)     -> cpgd_* / cpgd_smooth_*
  cpgd_causal                      CPGD 严格因果零滞后(GPU,慢)  -> cpgd_causal_*
  dlstm                            NB-DLSTM(GPU,快)             -> dlstm_*
  transformer                      Transformer smooth(GPU)      -> transformer_*
  transformer_causal               Transformer 因果(GPU)        -> transformer_causal_*
  csdi                             CSDI 扩散插补(GPU,慢)        -> csdi_*
  T-FoT / T-FoT-GP / T-FoT-StP / GPMT   传统基线(CPU,GP/StP慢)  -> T_FoT_* / GPMT_*

用法(分方法并行,各控制台一条):
  # 基准(最快 IMM 建立 seed/轨迹)
  python mc_compare_v8.py --n_mc 50 --scenarios easy,medium,hard,blackout \
      --only_ours --out_dir ./mc_results/新基准          # 只跑 CPGD? 见下方"最轻基准"说明
  # —— 推荐用 make_baseline 模式(见文件末)只跑 IMM 建立 seed ——

  python augment_mc.py --base_dir ./mc_results/新基准 --methods cpgd
  python augment_mc.py --base_dir ./mc_results/新基准 --methods cpgd_causal
  python augment_mc.py --base_dir ./mc_results/新基准 --methods csdi
  python augment_mc.py --base_dir ./mc_results/新基准 --methods dlstm,transformer,transformer_causal
  python augment_mc.py --base_dir ./mc_results/新基准 --methods T-FoT,T-FoT-GP,T-FoT-StP,GPMT
"""

import env_config  # noqa
import os
import argparse
import numpy as np
import scipy.io as sio

import mc_compare_v8 as MC
from imm_tracker import run_imm_inference
import baselines_dl as BDL

DT = MC.DT
TIMESTEPS = MC.TIMESTEPS
STRIDE = MC.STRIDE
DEVICE = MC.DEVICE

# 方法分类
TFOT_SET   = {'T-FoT', 'T-FoT-GP', 'T-FoT-StP', 'GPMT'}
OUR_SET    = {'cpgd', 'cpgd_causal'}
DL_SET     = {'dlstm', 'transformer', 'transformer_causal', 'csdi'}
ALL_METHODS = ['imm', 'cpgd', 'cpgd_causal', 'dlstm', 'transformer',
               'transformer_causal', 'csdi'] + sorted(TFOT_SET)


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


def load_needed_models(methods):
    """按请求方法加载所需模型。返回 (models, diff, cfg, mmen, dl)。CPU方法无需加载。"""
    need_our = any(m in OUR_SET for m in methods)
    need_dl  = {m for m in methods if m in DL_SET}
    cfg = diff = mmen = None
    our_model = None
    dl = {}
    if need_our or need_dl:
        cfg = MC.V8Config(); diff = MC.DiffModelV8(cfg, DEVICE)
    if need_our:
        print(f"Loading CPGD from {MC.CPGD_MODEL_PATH} ...")
        our_model = MC._load_our_model(cfg, MC.CPGD_MODEL_PATH)
        print("Loading MMEN ...")
        from mmen_is2 import PartialInfoInfer_P1
        import torch
        mmen = PartialInfoInfer_P1(TIMESTEPS).to(DEVICE)
        try:
            mmen.load_state_dict(torch.load(MC.MMEN_MODEL_PATH, map_location=DEVICE)); mmen.eval()
        except Exception as e:
            print(f"  MMEN load FAILED: {e}"); mmen = None
    if need_dl:
        # transformer_causal 复用 transformer 权重
        dl_filter = set()
        for m in need_dl:
            dl_filter.add('transformer' if m == 'transformer_causal' else m)
        dl = MC.load_dl_baselines(dl_filter=dl_filter)
    return our_model, diff, cfg, mmen, dl


def compute_methods(trial, methods, models):
    """对一条 trial 算指定方法,返回 {字段前缀: (N,4) 或 (N,2)}。"""
    states = trial['states']; ts = trial['timestamps']
    valid = trial['valid_mask']; dense = trial['dense_obs']
    our_model, diff, cfg, mmen, dl = models
    out = {}

    # ── IMM-EKF ──
    if 'imm' in methods:
        out['imm'] = run_imm_inference(dense, valid, ts)               # (N,4)

    # ── CPGD (smooth + full) ──
    if 'cpgd' in methods and our_model is not None:
        kf, smooth = MC.run_v8_inference(our_model, diff, cfg, mmen, dense, valid)
        out['cpgd'] = kf                                               # (N,4) full+KF
        out['cpgd_smooth'] = smooth                                    # (N,4) smooth-only
    if 'cpgd_causal' in methods and our_model is not None:
        out['cpgd_causal'] = MC.run_v8_inference_causal(
            our_model, diff, cfg, dense, valid, ddim_steps=MC.CAUSAL_DDIM_STEPS)

    # ── DL 基线 ──
    if 'dlstm' in methods and 'dlstm' in dl:
        out['dlstm'] = BDL.run_dlstm_inference(dl['dlstm'], dense, valid, DEVICE)
    if 'transformer' in methods and 'transformer' in dl:
        out['transformer'] = BDL.run_transformer_inference(
            dl['transformer'], dense, valid, DEVICE, window=TIMESTEPS, stride=STRIDE, mode='smooth')
    if 'transformer_causal' in methods and 'transformer' in dl:
        out['transformer_causal'] = BDL.run_transformer_inference(
            dl['transformer'], dense, valid, DEVICE, mode='causal')
    if 'csdi' in methods and 'csdi' in dl:
        m_csdi, diff_csdi = dl['csdi']
        out['csdi'] = BDL.run_csdi_inference(
            m_csdi, diff_csdi, dense, valid, DEVICE, window=TIMESTEPS, stride=STRIDE)

    # ── T-FoT 系 ──
    tfot_want = {m for m in methods if m in TFOT_SET}
    if tfot_want:
        res = MC.run_tfot_all(dense, valid, ts, tfot_filter=tfot_want)  # {nm:(N,2)}
        for nm, arr in res.items():
            out[nm] = arr
    return out


def augment_scene(base_path, out_path, sc_name, duration, methods, models, tol=1e-3):
    """从 base_path 取 seed/校验基准,只算 methods,写独立 out_path。base 不动。
    因果方法只在前 cpgd_causal_n 个 trial 上算(与基准一致);其余 trial 填 NaN。"""
    if not os.path.exists(base_path):
        print(f"  [skip] 基准 {base_path} 不存在"); return
    mat = sio.loadmat(base_path)
    sc_cfg = MC.SCENARIOS[sc_name]
    n_mc = int(mat['n_mc'].ravel()[0])
    N_max = int(mat['N_max'].ravel()[0])
    seeds = mat['seed'].ravel().astype(int)
    obs_err_ref = mat.get('obs_pos_err', None)
    n_causal = int(np.asarray(mat['cpgd_causal_n']).ravel()[0]) if 'cpgd_causal_n' in mat else n_mc
    causal_methods = {'cpgd_causal', 'transformer_causal'}
    has_causal = any(m in causal_methods for m in methods)

    print(f"\n{'='*66}\n  Augment {sc_name}: n_mc={n_mc} N_max={N_max} "
          f"methods={methods}\n  base={base_path}\n  out ={out_path}")
    if has_causal:
        print(f"  因果方法仅前 {n_causal} 个 trial(与基准 cpgd_causal_n 对齐)")
    print('='*66)

    # 每个方法的输出前缀(cpgd 会额外产出 cpgd_smooth)
    def prefixes_of(m):
        return ['cpgd', 'cpgd_smooth'] if m == 'cpgd' else [_safe(m)]
    # 是否有速度(T-FoT 系仅位置)
    def has_vel(pfx):
        return pfx not in {'T_FoT', 'T_FoT_GP', 'T_FoT_StP', 'GPMT'}

    all_pfx = []
    for m in methods:
        all_pfx += prefixes_of(m)
    # 因果字段初值 NaN(仅前 n_causal 填),其余 0
    new = {}
    for pfx in all_pfx:
        is_causal = 'causal' in pfx
        pad = np.nan if is_causal else 0.0
        new[f'{pfx}_pos'] = np.full((n_mc, N_max, 2), pad, np.float32)
        new[f'{pfx}_pos_err'] = np.full((n_mc, N_max), pad, np.float32)
        if has_vel(pfx):
            new[f'{pfx}_vel'] = np.full((n_mc, N_max, 2), pad, np.float32)
            new[f'{pfx}_vel_err'] = np.full((n_mc, N_max), pad, np.float32)

    from tqdm import tqdm
    n_verify_fail = 0
    for i in tqdm(range(n_mc), desc=sc_name):
        trial = replay_trial(sc_cfg, duration, int(seeds[i]))
        if trial is None:
            raise SystemExit(
                f"[error] seed={seeds[i]} 重放被拒——与基准不一致,检查轨迹代码/duration/base_seed。")
        N = min(len(trial['states']), N_max)
        gt = trial['states'][:N]; gt_pos = gt[:, :2]; gt_vel = gt[:, 2:]
        # 硬校验:重放观测误差须与基准一致
        if obs_err_ref is not None:
            rep = np.linalg.norm(trial['dense_obs'][:N] - gt_pos, axis=1)
            d = np.nanmax(np.abs(rep - obs_err_ref[i, :N])) if N else 0.0
            if d > tol:
                n_verify_fail += 1
                if n_verify_fail <= 3:
                    print(f"  [warn] trial {i} seed={seeds[i]} obs_err 偏差 {d:.3g} > {tol}")

        # 因果方法:超过 n_causal 的 trial 跳过(保持 NaN)
        run_methods = list(methods)
        if i >= n_causal:
            run_methods = [m for m in methods if m not in causal_methods]
        if not run_methods:
            continue
        res = compute_methods(trial, run_methods, models)

        for m in run_methods:
            for pfx in prefixes_of(m):
                arr = res[('cpgd_smooth' if pfx == 'cpgd_smooth' else m)][:N]
                new[f'{pfx}_pos'][i, :N] = arr[:, :2]
                new[f'{pfx}_pos_err'][i, :N] = np.linalg.norm(arr[:, :2] - gt_pos, axis=1)
                if has_vel(pfx):
                    new[f'{pfx}_vel'][i, :N] = arr[:, 2:]
                    new[f'{pfx}_vel_err'][i, :N] = np.linalg.norm(arr[:, 2:] - gt_vel, axis=1)

    if n_verify_fail:
        print(f"  [warn] {n_verify_fail}/{n_mc} trial 校验超差(仍写入,请核对 duration/轨迹代码)。")
    else:
        print(f"  [ok] 全部 {n_mc} trial 逐条对齐校验通过。")

    new['seed'] = seeds.astype(np.int32)
    new['n_mc'] = np.int32(n_mc)
    new['N_max'] = np.int32(N_max)
    new['scenario_name'] = sc_name
    new['augment_methods'] = ','.join(methods)
    if has_causal:
        new['cpgd_causal_n'] = np.int32(n_causal)
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    sio.savemat(out_path, new, do_compression=True)
    n_fields = len([k for k in new if k.endswith('_pos_err')])
    print(f"  Saved → {out_path}  ({n_fields} 个方法字段组)")


def make_base(sc_name, out_path, n_mc, duration, base_seed, causal_n, imm=True):
    """生成最轻基准 mat:只跑轨迹生成 + IMM(可选),但写全下游需要的字段:
       seed / n_mc / N_max / cpgd_causal_n / states / timestamps / valid_mask /
       dense_obs / phys_obs / obs_pos_err / blackout_start/end / maneuver_windows。
    seed 接受判据与 run_one_trial 完全一致,故与后续 augment 的重放逐 trial 对齐。"""
    sc_cfg = MC.SCENARIOS[sc_name]
    blen = sc_cfg.get('blackout_len', 0)
    trials, seed, seeds = [], base_seed, []
    from tqdm import tqdm
    pbar = tqdm(total=n_mc, desc=f'base:{sc_name}')
    while len(trials) < n_mc:
        tr = replay_trial(sc_cfg, duration, seed)
        if tr is not None:
            # 记录盲区索引(replay_trial 没返回,这里重取一次 build_one_trial 的 bo)
            np.random.seed(seed)
            try:
                import torch; torch.manual_seed(seed)
            except Exception:
                pass
            _, _, phys_obs, _, _, bo_s, bo_e = MC.build_one_trial(sc_cfg, duration, seed, blackout_len=blen)
            tr['phys_obs'] = phys_obs; tr['bo_s'] = bo_s; tr['bo_e'] = bo_e
            tr['seed'] = seed
            trials.append(tr); seeds.append(seed); pbar.update(1)
        seed += 1
    pbar.close()

    N_max = min(len(t['states']) for t in trials)
    imm_res = None
    if imm:
        imm_res = [run_imm_inference(t['dense_obs'], t['valid_mask'], t['timestamps']) for t in trials]

    def stack(key_fn, shape1, pad=0.0):
        out = np.full((n_mc, N_max) + shape1, pad, np.float32)
        for i, t in enumerate(trials):
            a = key_fn(t); Ni = min(len(a), N_max)
            out[i, :Ni] = a[:Ni]
        return out

    mat = {}
    mat['states']     = stack(lambda t: t['states'], (4,))
    mat['timestamps'] = np.full((n_mc, N_max), 0.0, np.float32)
    for i, t in enumerate(trials):
        Ni = min(len(t['timestamps']), N_max); mat['timestamps'][i, :Ni] = t['timestamps'][:Ni]
    mat['valid_mask'] = stack(lambda t: t['valid_mask'].astype(np.float32), (), 0.0)
    mat['dense_obs']  = stack(lambda t: t['dense_obs'], (2,))
    mat['phys_obs']   = stack(lambda t: np.where(t['valid_mask'][:, None], t['phys_obs'], np.nan), (2,))
    # obs_pos_err 供 augment 逐 trial 硬校验
    mat['obs_pos_err'] = stack(
        lambda t: np.linalg.norm(t['dense_obs'] - t['states'][:, :2], axis=1), ())
    if imm_res is not None:
        mat['imm_pos'] = np.zeros((n_mc, N_max, 2), np.float32)
        mat['imm_vel'] = np.zeros((n_mc, N_max, 2), np.float32)
        mat['imm_pos_err'] = np.zeros((n_mc, N_max), np.float32)
        mat['imm_vel_err'] = np.zeros((n_mc, N_max), np.float32)
        for i, (t, r) in enumerate(zip(trials, imm_res)):
            Ni = min(len(r), N_max); gt = t['states'][:Ni]
            mat['imm_pos'][i, :Ni] = r[:Ni, :2]
            mat['imm_vel'][i, :Ni] = r[:Ni, 2:]
            mat['imm_pos_err'][i, :Ni] = np.linalg.norm(r[:Ni, :2] - gt[:, :2], axis=1)
            mat['imm_vel_err'][i, :Ni] = np.linalg.norm(r[:Ni, 2:] - gt[:, 2:], axis=1)
    mat['seed'] = np.array(seeds, np.int32)
    mat['n_mc'] = np.int32(n_mc)
    mat['N_max'] = np.int32(N_max)
    mat['cpgd_causal_n'] = np.int32(min(causal_n, n_mc))
    mat['scenario_name'] = sc_name
    mat['blackout_start'] = np.array([t['bo_s'] for t in trials], np.int32)
    mat['blackout_end']   = np.array([t['bo_e'] for t in trials], np.int32)
    # 机动切换窗口(供 Post-Mnv 段分析),与 run_one_trial 同源,存为 object cell 数组
    mw_cell = np.empty(n_mc, dtype=object)
    for i, t in enumerate(trials):
        mw = MC.detect_maneuver_windows(t['states'], t['timestamps'])
        mw_cell[i] = (np.array(mw, dtype=np.int32) if mw
                      else np.zeros((0, 3), dtype=np.int32))
    mat['maneuver_windows'] = mw_cell
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    sio.savemat(out_path, mat, do_compression=True)
    print(f"  [base] Saved → {out_path}  (n_mc={n_mc}, N_max={N_max}, seeds[:5]={seeds[:5]})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--make_base', action='store_true',
                    help='生成最轻基准(轨迹+IMM),建立 seed 序列供后续 augment 对齐')
    ap.add_argument('--n_mc', type=int, default=50, help='make_base 时的 MC 次数')
    ap.add_argument('--base_seed', type=int, default=42,
                    help='make_base 起始 seed;augment 无需设(从基准读 seed)')
    ap.add_argument('--causal_n', type=int, default=50,
                    help='make_base 时写入 cpgd_causal_n:因果方法要算的前 N 个 trial')
    ap.add_argument('--base_dir', type=str, default='./mc_results/新基准',
                    help='基准 mat 目录(提供 seed 与逐 trial 校验基准,不修改)')
    ap.add_argument('--out_dir', type=str, default='./mc_results/augment',
                    help='独立输出目录:每场景一个 results_<scene>__<tag>.mat')
    ap.add_argument('--tag', type=str, default=None,
                    help='输出文件名后缀(默认取 methods 拼接)。并行时不同方法自动不同。')
    ap.add_argument('--scenarios', type=str, default='easy,medium,hard,blackout')
    ap.add_argument('--methods', type=str, default='imm',
                    help=f'逗号分隔,可选: {ALL_METHODS}')
    ap.add_argument('--duration', type=float, default=120.0,
                    help='必须与基准跑时的 --duration 一致(默认 120)')
    ap.add_argument('--tol', type=float, default=1e-3, help='obs_pos_err 校验容差(m)')
    args = ap.parse_args()

    scenarios = [s.strip() for s in args.scenarios.split(',') if s.strip()]

    # ── 模式一:生成最轻基准 ──
    if args.make_base:
        for sc in scenarios:
            if sc not in MC.SCENARIOS:
                print(f"  [skip] 未知场景 {sc}"); continue
            out = os.path.join(args.base_dir, f'results_{sc}.mat')
            make_base(sc, out, args.n_mc, args.duration, args.base_seed, args.causal_n)
        print(f"\n[Done] 基准生成 → {args.base_dir}/  (含 IMM + 全轨迹数据 + seed 对齐信息)")
        print("  接着分方法并行:python augment_mc.py --methods cpgd / csdi / ...")
        return

    # ── 模式二:augment 指定方法 ──
    methods = [m.strip() for m in args.methods.split(',') if m.strip()]
    bad = [m for m in methods if m not in ALL_METHODS]
    if bad:
        raise SystemExit(f"[error] 未知方法 {bad},可选 {ALL_METHODS}")

    # 只加载本批方法所需模型(一次),供所有场景复用
    models = load_needed_models(methods)

    tag = args.tag or '_'.join(_safe(m) for m in methods)
    for sc in scenarios:
        if sc not in MC.SCENARIOS:
            print(f"  [skip] 未知场景 {sc}"); continue
        base = os.path.join(args.base_dir, f'results_{sc}.mat')
        out = os.path.join(args.out_dir, f'results_{sc}__{tag}.mat')
        augment_scene(base, out, sc, args.duration, methods, models, tol=args.tol)

    print(f"\n[Done] augment 完成 → {args.out_dir}/  (tag={tag})")
    print("  用 merge_mc.py 合并基准 + 各 augment mat。")


if __name__ == '__main__':
    main()
