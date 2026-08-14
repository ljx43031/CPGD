"""
plot_paper_figures.py
=====================
生成论文最终的每场景一张 2x2 大图。
子图布局:
  (a) 左上: XY 轨迹 —— 仅 真值 + CPGD + CPGD-MMEN-KF(干净,不堆全部算法)
  (b) 右上: 位置误差时序(50 trial 中位数)—— 精选代表方法,盲区带高亮
  (c) 左下: 速度时序(代表 trial)—— CPGD-MMEN-KF vs 真值(CPGD 无KF速度不可用)
  (d) 右下: 位置误差经验 CDF(全序列汇总)—— 精选代表方法

精选代表方法(误差子图):CPGD / CPGD-MMEN-KF + NB-DLSTM + CSDI + IMM-EKF + 最强T-FoT。
每场景输出 figures/<scene>_result.png / .pdf。

用法:
  python plot_paper_figures.py --results_dir ./mc_results/最终 \
      --out_dir "E:/_广州航海学院/_研究/投稿/协同定位与导航/论文内容/figures"
"""

import os
import argparse
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 轨迹子图:只画我们的方法 vs 真值
TRAJ_METHODS = [
    ('states',      'Ground Truth',   dict(color='k',          lw=2.4, ls='-',  zorder=5)),
    ('cpgd_smooth', 'CPGD',           dict(color='tab:blue',   lw=1.6, ls='-')),
    ('cpgd',        'CPGD-MMEN-KF',   dict(color='tab:red',    lw=1.6, ls='--')),
]

# 误差/CDF 子图:精选代表方法(字段前缀 -> 标签, 颜色)
ERR_METHODS = [
    ('cpgd_smooth', 'CPGD',                 'tab:blue'),
    ('cpgd',        'CPGD-MMEN-KF',         'tab:red'),
    ('dlstm',       'NB-DLSTM [Gao2019]',   'tab:green'),
    ('csdi',        'CSDI [Tashiro2021]',   'tab:purple'),
    ('imm',         'IMM-EKF',              'tab:orange'),
    ('T_FoT',       'T-FoT',                'tab:brown'),
    ('obs',         'Raw WLS Obs.',         '0.6'),
]

FS = 14  # base font size (放大图例/标签, 回应 R4/R5 图例太小)


def per_trial_median_over_time(err):
    """err (n_mc, N) -> 每帧跨 trial 的中位数 (N,)。忽略 NaN。"""
    return np.nanmedian(err, axis=0)


def pick_rep_trial(pos_err):
    """选 CPGD 位置 RMSE 最接近中位数的 trial 作代表。"""
    n = pos_err.shape[0]
    rmse = np.array([np.sqrt(np.nanmean(pos_err[i]**2)) for i in range(n)])
    return int(np.argsort(rmse)[len(rmse)//2])


def plot_scene(mat, scene, out_dir):
    N = int(mat['N_max'].ravel()[0])
    n_mc = int(mat['n_mc'].ravel()[0]) if 'n_mc' in mat else int(mat['seed'].ravel().size)
    rep = pick_rep_trial(mat['cpgd_pos_err'])
    t = mat['timestamps'][rep, :N]
    gt = mat['states'][rep, :N, :2]
    valid = mat['valid_mask'][rep, :N].astype(bool)
    present = {k[:-8] for k in mat if k.endswith('_pos_err')}

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax_xy, ax_err, ax_vel, ax_cdf = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    # ── (a) XY 轨迹: 淡灰观测散点(背景) + 真值 + CPGD + CPGD-MMEN-KF ──
    # 先铺观测散点作背景,直观显示噪声幅度,凸显航迹估计对噪声的压制。
    if 'dense_obs' in mat:
        obs_xy = mat['dense_obs'][rep, :N, :2]
        obs_bg = obs_xy[valid] if valid.any() else obs_xy
        ax_xy.scatter(obs_bg[:, 0], obs_bg[:, 1], s=6, c='0.75',
                      alpha=0.45, linewidths=0, zorder=0, label='Observations')
    for key, lbl, style in TRAJ_METHODS:
        if key == 'states':
            xy = gt
        else:
            fld = f'{key}_pos'
            if fld not in mat:
                continue
            xy = mat[fld][rep, :N, :2]
        st = {**style}; st.setdefault('zorder', 3)
        ax_xy.plot(xy[:, 0], xy[:, 1], label=lbl, **st)
    # 盲区段真值高亮
    if not valid.all():
        bo = ~valid
        ax_xy.plot(gt[bo, 0], gt[bo, 1], 'o', ms=2.5, color='orange',
                   alpha=0.5, zorder=1, label='No-Obs (GT)')
    ax_xy.set_xlabel('X [m]', fontsize=FS); ax_xy.set_ylabel('Y [m]', fontsize=FS)
    ax_xy.set_title(f'({chr(97)}) {scene.capitalize()}: trajectory (representative trial)',
                    fontsize=FS)
    ax_xy.legend(fontsize=FS-1, loc='best'); ax_xy.grid(alpha=0.3); ax_xy.axis('equal')

    # ── (b) 位置误差时序(50 trial 中位数) ──
    for key, lbl, col in ERR_METHODS:
        fld = f'{key}_pos_err'
        if fld not in mat:
            continue
        med = per_trial_median_over_time(mat[fld][:, :N])
        lw = 2.0 if key in ('cpgd_smooth', 'cpgd') else 1.2
        ax_err.plot(t, med, label=lbl, color=col, lw=lw)
    if not valid.all():
        _shade_blackout(ax_err, t, valid)
    ax_err.set_xlabel('Time [s]', fontsize=FS)
    ax_err.set_ylabel('Median position error [m]', fontsize=FS)
    ax_err.set_title(f'(b) {scene.capitalize()}: position error over time ({n_mc} MC)',
                     fontsize=FS)
    # y 轴上限裁到代表方法中位数曲线峰值的 1.25 倍,避免个别方法末端发散尖峰压扁主体
    ymax = 0.0
    for key in ('cpgd_smooth', 'cpgd', 'dlstm', 'imm', 'T_FoT'):
        fld = f'{key}_pos_err'
        if fld in mat:
            ymax = max(ymax, np.nanmax(per_trial_median_over_time(mat[fld][:, :N])))
    if ymax > 0:
        ax_err.set_ylim(0, ymax * 1.25)
    ax_err.legend(fontsize=FS-2, loc='best', ncol=2); ax_err.grid(alpha=0.3)

    # ── (c) 速度时序: CPGD-MMEN-KF vs 真值(代表 trial) ──
    gt_spd = np.linalg.norm(mat['states'][rep, :N, 2:], axis=1)
    ax_vel.plot(t, gt_spd, 'k-', lw=2.0, label='Ground Truth')
    if 'cpgd_vel' in mat:
        kf_spd = np.linalg.norm(mat['cpgd_vel'][rep, :N], axis=1)
        ax_vel.plot(t, kf_spd, color='tab:red', lw=1.5, ls='--', label='CPGD-MMEN-KF')
    if 'imm_vel' in mat:
        imm_spd = np.linalg.norm(mat['imm_vel'][rep, :N], axis=1)
        ax_vel.plot(t, imm_spd, color='tab:orange', lw=1.0, alpha=0.8, label='IMM-EKF')
    if not valid.all():
        _shade_blackout(ax_vel, t, valid)
    ax_vel.set_xlabel('Time [s]', fontsize=FS); ax_vel.set_ylabel('Speed [m/s]', fontsize=FS)
    ax_vel.set_title(f'(c) {scene.capitalize()}: speed estimate (representative trial)',
                     fontsize=FS)
    ax_vel.legend(fontsize=FS-1, loc='best'); ax_vel.grid(alpha=0.3)

    # ── (d) 位置误差 CDF(全序列全 trial 汇总) ──
    for key, lbl, col in ERR_METHODS:
        fld = f'{key}_pos_err'
        if fld not in mat:
            continue
        e = mat[fld][:, :N].ravel()
        e = e[np.isfinite(e)]
        if e.size == 0:
            continue
        xs = np.sort(e); ys = np.arange(1, xs.size + 1) / xs.size
        lw = 2.0 if key in ('cpgd_smooth', 'cpgd') else 1.2
        ax_cdf.plot(xs, ys, label=lbl, color=col, lw=lw)
    ax_cdf.set_xlabel('Position error [m]', fontsize=FS)
    ax_cdf.set_ylabel('Empirical CDF', fontsize=FS)
    ax_cdf.set_title(f'(d) {scene.capitalize()}: error CDF (all frames, {n_mc} MC)', fontsize=FS)
    # x 轴裁到 95 分位,避免重尾拉伸看不清主体
    all_e = mat['obs_pos_err'][:, :N].ravel() if 'obs_pos_err' in mat else np.array([50.0])
    all_e = all_e[np.isfinite(all_e)]
    ax_cdf.set_xlim(0, np.percentile(all_e, 95) if all_e.size else 50)
    ax_cdf.legend(fontsize=FS-2, loc='lower right', ncol=2); ax_cdf.grid(alpha=0.3)

    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(out_dir, f'{scene}_result.{ext}'), dpi=150,
                    bbox_inches='tight')
    plt.close(fig)
    print(f'[Saved] {scene}_result.png / .pdf  (rep trial {rep})')


def _shade_blackout(ax, t, valid):
    """在无观测帧区间画竖向阴影带。"""
    bo = ~valid
    if not bo.any():
        return
    idx = np.where(bo)[0]
    # 连续段分组
    splits = np.where(np.diff(idx) > 1)[0]
    groups = np.split(idx, splits + 1)
    first = True
    for g in groups:
        if g.size == 0:
            continue
        ax.axvspan(t[g[0]], t[g[-1]], color='0.85', alpha=0.6, zorder=0,
                   label='No-Obs' if first else None)
        first = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_dir', type=str, default='./mc_results/最终')
    ap.add_argument('--out_dir', type=str, default='./figures_paper')
    ap.add_argument('--scenarios', type=str, default='easy,medium,hard,blackout')
    args = ap.parse_args()

    for sc in [s.strip() for s in args.scenarios.split(',')]:
        path = os.path.join(args.results_dir, f'results_{sc}.mat')
        if not os.path.exists(path):
            print(f'[skip] {path} 不存在'); continue
        plot_scene(sio.loadmat(path), sc, args.out_dir)
    print(f'\n[Done] 图输出到 {args.out_dir}/')


if __name__ == '__main__':
    main()

