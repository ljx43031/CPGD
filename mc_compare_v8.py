"""
mc_compare_v8.py
================
蒙特卡洛批量对比实验脚本（基于 phys_pipeline_compare_v8.py）

实验设计：
  [场景 A] Easy   : n_drones=7, n_obs=1, area=700x700,  speed=[3,10]
  [场景 B] Medium : n_drones=6, n_obs=3, area=1000x1000, speed=[5,15]
  [场景 C] Hard   : n_drones=5, n_obs=5, area=1200x1200, speed=[8,20]

  每种场景做 N_MC=50 次蒙特卡洛，记录：
    - 真实航迹 (states)
    - 原始观测 (phys_obs, valid_mask)
    - CPGD、NB-DLSTM、Transformer、CSDI、IMM-EKF、T-FoT系 的位置+速度预测
    - 每一次运行的位置误差时序

  [场景 D] Blackout : Medium 场景 + 强制插入 blackout_len=50帧盲区
    同样做 50 次蒙特卡洛

  额外分析：
    - 自动检测每条航迹中"机动切换剧烈"时刻（转弯率变化大），提取窗口做专项分析
    - 在盲区段内分析各方法的预测精度与稳定性

输出：
  results_easy.mat
  results_medium.mat
  results_hard.mat
  results_blackout.mat
  每个 mat 包含所有 50 次 trial 的完整数据，格式见文件末尾说明。

用法：
  python mc_compare_v8.py [--n_mc 50] [--skip_v1] [--skip_tfot] [--duration 120]
"""

import env_config  # noqa
import numpy as np
import torch
import argparse
import time
import os
import scipy.io as sio
from collections import defaultdict
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve
from tqdm import tqdm

# ── 复用 phys_pipeline_compare_v8 中的核心组件 ─────────────────────────────
from scene_configuration import SceneConfigurator
from trajectory_generation import TrajectoryGenerator
from detection_process import FusedDetectionProcess

from train_prior_denoiser_v8 import (
    TrainConfig as V8Config, DualTrackModel,
    GaussianDiffusion as DiffModelV8,
    anchor_normalize, anchor_denormalize, hann_window)
from mmen_is2 import PartialInfoInfer_P1
from kalman_tracker import KalmanTracker

# ── Learning-based baselines (NB-DLSTM / Transformer / CSDI diffusion imputation) ─
import baselines_dl as BDL

# ─────────────────────────────────────────────────────────────────────────────
DEVICE         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TIMESTEPS      = 32
STRIDE         = 16
DT             = 0.1
Q_SCALE        = 10.0
ANCHOR_SCALE   = 1000.0
VEL_SCALE      = 50.0
DDIM_STEPS     = 25
BLACKOUT_THRESH= 8

CPGD_MODEL_PATH = './models_prior_denoiser_v8_ft/final.pth'
MMEN_MODEL_PATH = './models_mmen_diffusion_is2/PII_P1_is2.pkl'

# ── CPGD 方法（字段前缀 'cpgd'）───────────────────────────────────────────────
# CPGD 是双轨 CP-GRU + RD-Denoiser 架构（DualTrackModel），残差空间去噪，
# 尾接 MMEN-KF。run_cpgd_inference 同时产出 smooth(无KF) 与 full(+KF) 两条轨迹。
OUR_METHODS = {
    'cpgd': dict(ckpt=CPGD_MODEL_PATH, label='CPGD'),
}
PRIMARY_METHOD = 'cpgd'

# ── Learning-based baseline checkpoints (trained via train_baselines_dl.py) ────
CSDI_MODEL_PATH        = './models_baselines_dl/csdi_final.pth'
DLSTM_MODEL_PATH       = './models_baselines_dl/dlstm_final.pth'
TRANSFORMER_MODEL_PATH = './models_baselines_dl/transformer_final.pth'

# ── Causal (zero-lag) CPGD inference settings ─────────────────────────────────
# Causal mode fires the denoiser every frame on window [k-W+1, k] and reads the
# window's BOUNDARY column (frame k) -> uses only past+current observations.
# It is ~W/STRIDE x more expensive than smooth mode, so it is run on a subset of
# trials by default (--causal_trials) and can use fewer DDIM steps (--causal_ddim).
CAUSAL_DDIM_STEPS = 15

TFOT_POLY_ORDER = 2
TFOT_NU         = 5.0

# 蒙特卡洛次数
N_MC = 50

# ─────────────────────────────────────────────────────────────────────────────
# 场景定义
# ─────────────────────────────────────────────────────────────────────────────
SCENARIOS = {
    'easy': dict(
        n_drones=7, n_obstacles=1,
        area_size=[700, 700, 200],
        target_speed=[3, 10],
        drone_speed=[10, 20],
        label='Easy (7UAV, 1obs, 700m)',
        min_coverage=0.85,
    ),
    'medium': dict(
        n_drones=6, n_obstacles=3,
        area_size=[1000, 1000, 200],
        target_speed=[5, 15],
        drone_speed=[10, 20],
        label='Medium (6UAV, 3obs, 1000m)',
        min_coverage=0.75,
    ),
    'hard': dict(
        n_drones=5, n_obstacles=5,
        area_size=[1200, 1200, 200],
        target_speed=[8, 20],
        drone_speed=[10, 20],
        label='Hard (5UAV, 5obs, 1200m)',
        min_coverage=0.65,
    ),
    'blackout': dict(
        n_drones=5, n_obstacles=5,
        area_size=[1200, 1200, 200],
        target_speed=[8, 20],
        drone_speed=[10, 20],
        label='Blackout (hard + 50-frame forced gap)',
        blackout_len=50,
        min_coverage=0.65,   # checked before blackout injection
    ),
    # 补充场景:行人/车非运动学目标(随机急停 + 锐角转向),medium 难度基底。
    # 刻画 CT 模型覆盖不到的真实地面目标行为——物理正当的"更难场景"。
    # 速度 1-5 m/s 标定为真实行人步行-慢跑区间(训练覆盖 2-25 m/s,1-2 m/s 为轻度外推)。
    'pedestrian': dict(
        n_drones=6, n_obstacles=3,
        area_size=[1000, 1000, 200],
        target_speed=[1, 5],
        drone_speed=[10, 20],
        label='Pedestrian (non-kinematic: random stops + sharp turns, 1-5 m/s)',
        nonkinematic=True,
        min_coverage=0.75,
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# T-FoT 工具函数（直接从 phys_pipeline_compare_v8 移植）
# ─────────────────────────────────────────────────────────────────────────────
def _rbf_kernel(t1, t2, var, ls):
    t1 = np.asarray(t1, dtype=float).reshape(-1, 1)
    t2 = np.asarray(t2, dtype=float).reshape(1, -1)
    return var * np.exp(-0.5 * (t1 - t2) ** 2 / ls ** 2)

def _learn_hyperparams(t, e, iv, ils,
                       vb=(1e-3, 1e8), lb=(0.05, 10.0)):
    t, e = np.asarray(t, dtype=float), np.asarray(e, dtype=float)
    n = len(t)
    if n < 2:
        return iv, ils
    def nll(lp):
        v, l = np.exp(lp[0]), np.exp(lp[1])
        K = _rbf_kernel(t, t, v, l) + 1e-5 * np.eye(n)
        try:
            cf = cho_factor(K); a = cho_solve(cf, e)
            _, ld = np.linalg.slogdet(K)
            return float(0.5 * (e @ a + ld + n * np.log(2 * np.pi))) if np.isfinite(ld) else 1e10
        except Exception:
            return 1e10
    x0  = np.log([iv, ils])
    bnd = [(np.log(vb[0]), np.log(vb[1])), (np.log(lb[0]), np.log(lb[1]))]
    try:
        r = minimize(nll, x0, method='L-BFGS-B', bounds=bnd, options={'maxiter': 50})
        if r.success or r.fun < nll(x0):
            return float(np.exp(r.x[0])), float(np.exp(r.x[1]))
    except Exception:
        pass
    return iv, ils

def _poly_fit(t, y, order=2):
    V = np.vander(np.asarray(t, dtype=float), N=order + 1, increasing=True)
    C, *_ = np.linalg.lstsq(V, np.asarray(y, dtype=float), rcond=None)
    return C

def _poly_eval(t, C):
    V = np.vander(np.asarray(t, dtype=float), N=len(C), increasing=True)
    return V @ C

class _TFoTBase:
    def __init__(self, d, poly_order, state_dim=2):
        self.d, self.poly_order, self.state_dim = d, poly_order, state_dim
        self._t, self._y = [], []
    def reset(self, noise_var=10.0, noise_ls=0.5):
        self._t, self._y = [], []
        self.noise_var, self.noise_ls = noise_var, noise_ls
        self._eta_var, self._eta_ls  = max(noise_var, 1.0), 0.3
    def _push(self, t_k, y_k):
        self._t.append(float(t_k)); self._y.append(np.asarray(y_k, dtype=float))
        if len(self._t) > self.d:
            self._t = self._t[-self.d:]; self._y = self._y[-self.d:]
        return np.array(self._t), np.array(self._y)
    def _normalise_t(self, t_arr):
        mn, mx = t_arr[0], t_arr[-1]; sp = mx - mn
        return (np.zeros_like(t_arr), mn, 1.0) if sp < 1e-9 else ((t_arr - mn) / sp, mn, sp)

class TFoTBaseline(_TFoTBase):
    def update(self, t_k, y_k):
        t_win, Y_win = self._push(t_k, y_k)
        return np.array([float(_poly_eval([float(t_k)], _poly_fit(t_win, Y_win[:, c], self.poly_order))[0])
                         for c in range(self.state_dim)])

class TFoTGP(_TFoTBase):
    def update(self, t_k, y_k):
        t_win, Y_win = self._push(t_k, y_k); n = len(t_win)
        t_norm, t_min, t_span = self._normalise_t(t_win)
        ts_norm = (float(t_k) - t_min) / t_span if t_span > 1e-9 else 1.0
        x_hat = np.zeros(self.state_dim)
        for c in range(self.state_dim):
            y_w = Y_win[:, c]; C = _poly_fit(t_win, y_w, self.poly_order); e = y_w - _poly_eval(t_win, C)
            if n >= 3:
                try: self._eta_var, self._eta_ls = _learn_hyperparams(t_norm, e, self._eta_var, self._eta_ls)
                except Exception: pass
            jit = max(1e-4 * self._eta_var, 1e-4)
            K_e = _rbf_kernel(t_norm, t_norm, self._eta_var, self._eta_ls) + (self.noise_var + jit) * np.eye(n)
            ke_s = _rbf_kernel([ts_norm], t_norm, self._eta_var, self._eta_ls).ravel()
            try:
                cf = cho_factor(K_e); mu = float(ke_s @ cho_solve(cf, e))
            except Exception:
                mu = float(np.mean(e)) if n > 0 else 0.0
            x_hat[c] = float(_poly_eval([float(t_k)], C)[0]) + mu
        return x_hat

class TFoTStP(_TFoTBase):
    def __init__(self, d, poly_order, nu=5.0, state_dim=2):
        super().__init__(d, poly_order, state_dim); self.nu = nu
    def update(self, t_k, y_k):
        t_win, Y_win = self._push(t_k, y_k); n = len(t_win)
        t_norm, t_min, t_span = self._normalise_t(t_win)
        ts_norm = (float(t_k) - t_min) / t_span if t_span > 1e-9 else 1.0
        x_hat = np.zeros(self.state_dim)
        for c in range(self.state_dim):
            y_w = Y_win[:, c]; C = _poly_fit(t_win, y_w, self.poly_order); e = y_w - _poly_eval(t_win, C)
            if n >= 3:
                try: self._eta_var, self._eta_ls = _learn_hyperparams(t_norm, e, self._eta_var, self._eta_ls)
                except Exception: pass
            jit = max(1e-4 * self._eta_var, 1e-4)
            K_e = _rbf_kernel(t_norm, t_norm, self._eta_var, self._eta_ls) + (self.noise_var + jit) * np.eye(n)
            ke_s = _rbf_kernel([ts_norm], t_norm, self._eta_var, self._eta_ls).ravel()
            try:
                cf = cho_factor(K_e); a = cho_solve(cf, e)
                beta = float(e @ a)
                nu_sc = (self.nu + n) / (self.nu + beta) if (self.nu + beta) > 1e-9 else 1.0
                mu = nu_sc * float(ke_s @ a)
            except Exception:
                mu = float(np.mean(e)) if n > 0 else 0.0
            x_hat[c] = float(_poly_eval([float(t_k)], C)[0]) + mu
        return x_hat

class GPMTTracker(_TFoTBase):
    def __init__(self, d, state_dim=2): super().__init__(d, poly_order=1, state_dim=state_dim)
    def update(self, t_k, y_k):
        t_win, Y_win = self._push(t_k, y_k); n = len(t_win)
        t_norm, t_min, t_span = self._normalise_t(t_win)
        ts_norm = (float(t_k) - t_min) / t_span if t_span > 1e-9 else 1.0
        x_hat = np.zeros(self.state_dim)
        for c in range(self.state_dim):
            y_w = Y_win[:, c]
            if n >= 3:
                try: self._eta_var, self._eta_ls = _learn_hyperparams(t_norm, y_w, self._eta_var, self._eta_ls)
                except Exception: pass
            jit = max(1e-4 * self._eta_var, 1e-4)
            K_e = _rbf_kernel(t_norm, t_norm, self._eta_var, self._eta_ls) + (self.noise_var + jit) * np.eye(n)
            ke_s = _rbf_kernel([ts_norm], t_norm, self._eta_var, self._eta_ls).ravel()
            try:
                cf = cho_factor(K_e); x_hat[c] = float(ke_s @ cho_solve(cf, y_w))
            except Exception:
                x_hat[c] = float(y_w[-1])
        return x_hat

# ─────────────────────────────────────────────────────────────────────────────
# KF 辅助
# ─────────────────────────────────────────────────────────────────────────────
class StepwiseKalmanTracker(KalmanTracker):
    def predict_step(self, F_matrix, Q_scale=10.0):
        if not self.is_initialized: return
        dt = DT; q = self.process_noise_std ** 2 * Q_scale
        Q = np.array([[dt**4/4, 0, dt**3/2, 0],
                      [0, dt**4/4, 0, dt**3/2],
                      [dt**3/2, 0, dt**2, 0],
                      [0, dt**3/2, 0, dt**2]]) * q
        self.state = F_matrix @ self.state
        self.covariance = F_matrix @ self.covariance @ F_matrix.T + Q

def _detect_blackout_segments(valid_mask, thresh=BLACKOUT_THRESH):
    segs, N, i = [], len(valid_mask), 0
    while i < N:
        if not valid_mask[i]:
            j = i
            while j < N and not valid_mask[j]: j += 1
            if j - i >= thresh: segs.append((i, j - 1))
            i = j
        else: i += 1
    return segs

def _in_blackout(t, segs):
    return any(s <= t <= e for s, e in segs)

# ─────────────────────────────────────────────────────────────────────────────
# 模型加载
# ─────────────────────────────────────────────────────────────────────────────
def _load_our_model(cfg, ckpt_path):
    """Load one DualTrackModel variant. denoiser<-ema (推理用 EMA 权重), prior<-model.
    Returns the eval model, or None if the checkpoint is missing / fails."""
    if not os.path.exists(ckpt_path):
        print(f"  [skip] checkpoint not found: {ckpt_path}")
        return None
    model = DualTrackModel(cfg).to(DEVICE)
    try:
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        if 'ema' in ckpt:
            model.denoiser.load_state_dict(ckpt['ema'])
            prior_sd = {k[len('prior.'):]: v for k, v in ckpt['model'].items() if k.startswith('prior.')}
            model.prior.load_state_dict(prior_sd)
        else:
            model.load_state_dict(ckpt.get('model', ckpt), strict=False)
        model.eval()
        return model
    except Exception as e:
        print(f"  load FAILED ({ckpt_path}): {e}")
        return None


def load_models(load_dl=True, dl_filter=None):
    """dl_filter: 可选方法名集合,仅加载其中的 DL 基线(dlstm/transformer/csdi),None=全部。"""
    cfg = V8Config(); diff = DiffModelV8(cfg, DEVICE)
    # ── CPGD 方法 ──────────────────────────────────────────────────────────────
    our = {}
    for name, spec in OUR_METHODS.items():
        print(f"Loading {spec['label']} from {spec['ckpt']} on {DEVICE}...")
        m = _load_our_model(cfg, spec['ckpt'])
        if m is not None:
            our[name] = dict(model=m, label=spec['label'])
            print(f"  {spec['label']} loaded.")
    # 主方法句柄（供 causal 与历史 cpgd_* 字段；缺失则为 None）
    model = our[PRIMARY_METHOD]['model'] if PRIMARY_METHOD in our else None

    print(f"Loading MMEN from {MMEN_MODEL_PATH}...")
    mmen = PartialInfoInfer_P1(TIMESTEPS).to(DEVICE)
    try:
        mmen.load_state_dict(torch.load(MMEN_MODEL_PATH, map_location=DEVICE)); mmen.eval()
        print("  MMEN loaded.")
    except Exception as e:
        print(f"  MMEN load FAILED: {e}"); mmen = None

    # ── Learning-based baselines (optional; gracefully skipped if missing) ─────
    dl = load_dl_baselines(dl_filter=dl_filter) if load_dl else {}

    return model, diff, cfg, mmen, dl, our


def load_dl_baselines(dl_filter=None):
    """Load CSDI/DLSTM/Transformer baseline checkpoints if present.
    Returns a dict of ready-to-run entries; missing checkpoints are silently
    skipped. dl_filter: 可选方法名集合,仅加载其中的基线(None=全部)。"""
    dl = {}
    def _want(kind):
        return dl_filter is None or kind in dl_filter

    if _want('csdi') and os.path.exists(CSDI_MODEL_PATH):
        try:
            m = BDL.CSDIDenoiser().to(DEVICE)
            ck = torch.load(CSDI_MODEL_PATH, map_location=DEVICE, weights_only=False)
            m.load_state_dict(ck['model']); m.eval()
            diff = BDL.CSDIDiffusion(num_steps=50, device=DEVICE)   # CSDI: quadratic, T=50
            dl['csdi'] = (m, diff)
            print(f"  CSDI baseline loaded from {CSDI_MODEL_PATH}")
        except Exception as e:
            print(f"  CSDI load FAILED: {e}")
    elif _want('csdi'):
        print(f"  CSDI baseline not found ({CSDI_MODEL_PATH}), skipping.")

    # Faithful NB-DLSTM (Gao 2019) — literature recurrent baseline, no pipeline
    if _want('dlstm') and os.path.exists(DLSTM_MODEL_PATH):
        try:
            m = BDL.FaithfulDLSTM().to(DEVICE)
            ck = torch.load(DLSTM_MODEL_PATH, map_location=DEVICE, weights_only=False)
            m.load_state_dict(ck['model']); m.eval()
            dl['dlstm'] = m
            print(f"  DLSTM (faithful Gao2019) loaded from {DLSTM_MODEL_PATH}")
        except Exception as e:
            print(f"  DLSTM load FAILED: {e}")
    elif _want('dlstm'):
        print(f"  DLSTM baseline not found ({DLSTM_MODEL_PATH}), skipping.")

    # Transformer denoiser (Giuliari 2020, task-adapted)
    if _want('transformer') and os.path.exists(TRANSFORMER_MODEL_PATH):
        try:
            m = BDL.TransformerDenoiser().to(DEVICE)
            ck = torch.load(TRANSFORMER_MODEL_PATH, map_location=DEVICE, weights_only=False)
            m.load_state_dict(ck['model']); m.eval()
            dl['transformer'] = m
            print(f"  Transformer (Giuliari2020) loaded from {TRANSFORMER_MODEL_PATH}")
        except Exception as e:
            print(f"  Transformer load FAILED: {e}")
    elif _want('transformer'):
        print(f"  Transformer baseline not found ({TRANSFORMER_MODEL_PATH}), skipping.")

    return dl

# ─────────────────────────────────────────────────────────────────────────────
# MMEN 预测
# ─────────────────────────────────────────────────────────────────────────────
def run_mmen_predict(mmen, seg):
    if mmen is None: return np.eye(4)
    inp = torch.from_numpy(seg).float().unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        raw = mmen(inp).squeeze(0).cpu().numpy()
    F = raw.T; F[:, 0] = [1, 0, 0, 0]; F[:, 1] = [0, 1, 0, 0]
    return F

# ─────────────────────────────────────────────────────────────────────────────
# 场景观测生成
# ─────────────────────────────────────────────────────────────────────────────
def build_one_trial(sc, duration, seed, blackout_len=0):
    cfg_s = SceneConfigurator(
        area_size=tuple(sc['area_size']),
        n_drones=sc['n_drones'], n_targets=1,
        n_obstacles=sc['n_obstacles'])
    tgen = TrajectoryGenerator(dt=DT, duration=duration)
    scene = cfg_s.generate_scene()

    drone_trajs = {
        dc.agent_id: tgen.generate_drone_trajectory(
            start_position=dc.initial_state_2d[:2], altitude=dc.altitude,
            speed_range=sc['drone_speed'], area_size=sc['area_size'],
            obstacles=scene['obstacles'])
        for dc in scene['drones']}

    tgt_id, = [tc.agent_id for tc in scene['targets']]
    tgt_traj = tgen.generate_target_trajectory(
        start_position=scene['targets'][0].initial_state_2d[:2],
        speed_range=sc['target_speed'],
        area_size=sc['area_size'], obstacles=scene['obstacles'])

    # 可选:注入行人/车非运动学行为(随机急停 + 锐角转向)。位置由速度重新积分,
    # pos/vel 严格一致,并同步更新 positions_3d(检测过程读它)。
    if sc.get('nonkinematic', False):
        from generate_trajectory import apply_nonkinematic
        new_states = apply_nonkinematic(tgt_traj.states_2d, tgen.dt)
        tgt_traj.states_2d = new_states
        tgt_traj.positions_3d[:, :2] = new_states[:, :2]

    target_trajs = {tgt_id: tgt_traj}

    states = tgt_traj.states_2d
    timestamps = tgt_traj.timestamps
    N = min(len(states), len(tgen.timestamps))
    states, timestamps = states[:N], timestamps[:N]

    det_proc = FusedDetectionProcess(tx_power=20.0, sensitivity=-95.0, frequency=2.4e9)
    dets = det_proc.generate_detections(
        drone_trajectories=drone_trajs, target_trajectories=target_trajs,
        obstacles=scene['obstacles'], timestamps=timestamps)

    # 还原观测
    step_to_dets = defaultdict(list)
    for det in dets.get(tgt_id, []):
        step_to_dets[int(round(det.timestamp / DT))].append(det)

    phys_obs  = np.full((N, 2), np.nan)
    valid_mask = np.zeros(N, dtype=bool)
    for k in range(N):
        ds = step_to_dets.get(k, [])
        if ds:
            pos, w = [], []
            for d in ds:
                dx, dy, _ = d.drone_position
                dh = d.distance * np.cos(d.elevation)
                pos.append([dx + dh * np.cos(d.azimuth), dy + dh * np.sin(d.azimuth)])
                w.append(10 ** (d.RSSI / 10.0))
            w = np.array(w); w /= w.sum()
            phys_obs[k] = (w[:, None] * np.array(pos)).sum(0)
            valid_mask[k] = True

    # 强制插入盲区
    blackout_start = blackout_end = -1
    if blackout_len > 0:
        blackout_start = N // 2 - blackout_len // 2
        blackout_end   = blackout_start + blackout_len
        valid_mask[blackout_start:blackout_end] = False
        phys_obs[blackout_start:blackout_end]   = np.nan

    # 前向填充
    dense_obs = phys_obs.copy()
    last = None
    for i in range(N):
        if valid_mask[i]: last = dense_obs[i].copy()
        elif last is not None: dense_obs[i] = last
    first = None
    for i in range(N - 1, -1, -1):
        if valid_mask[i]: first = dense_obs[i].copy()
        elif np.isnan(dense_obs[i, 0]) and first is not None: dense_obs[i] = first
    dense_obs = np.nan_to_num(dense_obs, nan=0.0)

    return states, timestamps, phys_obs, valid_mask, dense_obs, blackout_start, blackout_end

# ─────────────────────────────────────────────────────────────────────────────
# CPGD 推理
# ─────────────────────────────────────────────────────────────────────────────
def run_v8_inference(v8_model, diff_v8, v8_cfg, mmen, dense_obs, valid_mask):
    total_len = len(dense_obs); W = TIMESTEPS

    dense_obs_v8 = dense_obs.copy(); dense_obs_v8[~valid_mask] = 0.0
    obs_np = np.column_stack([dense_obs_v8, valid_mask.astype(np.float32)])
    obs_t  = torch.from_numpy(obs_np).float().unsqueeze(0).to(DEVICE)

    fv = int(np.argmax(valid_mask)) if valid_mask.any() else 0
    anchor = obs_t[:, fv:fv+1, :2]
    hann_w = hann_window(W, DEVICE)

    blackout_segs = _detect_blackout_segments(valid_mask)
    global_pred   = torch.zeros((1, total_len, 4), device=DEVICE)
    global_weight = torch.zeros((1, total_len, 1), device=DEVICE)
    h_val         = None
    rolling_pt    = torch.empty((1, 0, 2), device=DEVICE)

    if v8_model is None:
        smooth = np.column_stack([dense_obs, np.zeros((total_len, 2))])
    else:
        with torch.no_grad():
            for cs in range(0, total_len, STRIDE):
                ce   = min(cs + STRIDE, total_len)
                obs_c = obs_t[:, cs:ce, :]
                vc    = obs_c[:, :, 2:3]
                on_c  = torch.cat([anchor_normalize(obs_c[:, :, :2], anchor, ANCHOR_SCALE) * vc, vc], dim=-1)
                pt, pf, h_val = v8_model.prior(on_c, h0=h_val)
                rolling_pt = torch.cat([rolling_pt, pt], dim=1)
                if rolling_pt.shape[1] > W: rolling_pt = rolling_pt[:, -W:, :]
                if ce >= W:
                    ws = ce - W; pw_norm = rolling_pt
                    pw_abs = anchor_denormalize(pw_norm, anchor, ANCHOR_SCALE)
                    pf_anc = pf[:, -1, :]
                    ow_abs = obs_t[:, ws:ce, :2]; vw = obs_t[:, ws:ce, 2:3]
                    innov  = torch.norm(ow_abs - pw_abs, dim=-1, keepdim=True) * vw
                    vsm    = vw.sum(dim=1, keepdim=True).clamp(min=1e-5)
                    scw    = torch.clamp(innov.sum(dim=1, keepdim=True) / vsm, min=5.0)
                    orn    = (ow_abs - pw_abs) / scw * vw
                    cond   = torch.cat([orn.permute(0,2,1), vw.permute(0,2,1)], dim=1)
                    x0p    = diff_v8.sample_ddim(v8_model.denoiser, cond, pf_anc, num_steps=DDIM_STEPS)
                    vr     = vw.mean().item()
                    cm     = (vw[:, -BLACKOUT_THRESH:, :] == 0).all().item()
                    tf     = 0.0 if cm else float(np.clip((vr - 0.3) / 0.3, 0.0, 1.0))
                    pp_abs = x0p[0, :2, :].T * scw[0, 0, 0] * tf + pw_abs[0]
                    pv_abs = x0p[0, 2:, :].T * VEL_SCALE
                    seg    = torch.cat([pp_abs, pv_abs], dim=-1)
                    global_pred[0, ws:ce, :]   += seg * hann_w.view(W, 1)
                    global_weight[0, ws:ce, :] += hann_w.view(W, 1)

        smooth = (global_pred / global_weight.clamp(min=1e-6))[0].cpu().numpy()
        cov = (global_weight[0, :, 0] > 1e-7).cpu().numpy()
        if not cov.all() and cov.any():
            fc = int(np.argmax(cov)); lc = int(len(cov) - 1 - np.argmax(cov[::-1]))
            smooth[:fc] = smooth[fc]; smooth[lc+1:] = smooth[lc]

    kf = run_mmen_kf(smooth, dense_obs, valid_mask, mmen)
    return kf, smooth  # (N,4), (N,4)


# ─────────────────────────────────────────────────────────────────────────────
# MMEN-KF 尾（CPGD smooth 之后的速度正则化 / Kalman 后处理）
# ─────────────────────────────────────────────────────────────────────────────
def run_mmen_kf(smooth, dense_obs, valid_mask, mmen):
    """给定任意 smooth 轨迹 (N,4)，跑 MMEN 预测 + 逐帧 Kalman，返回 kf (N,4)。"""
    total_len = len(smooth)
    kf = []
    tracker = StepwiseKalmanTracker(process_noise_std=0.5, initial_velocity_std=5.0)
    last_F   = run_mmen_predict(mmen, smooth[0:TIMESTEPS].copy())
    tracker.initialize(smooth[0, :2], np.eye(2) * 100.0, timestamp=0.0)
    tracker.state[2:] = smooth[0, 2:]
    kf.append(tracker.get_state()[0])
    consec_lost = 0; freeze = False
    for t in range(1, total_len):
        consec_lost = (consec_lost + 1) if not valid_mask[t] else 0
        blind = consec_lost >= BLACKOUT_THRESH
        if blind: freeze = True
        if not blind and freeze:
            ws = max(0, t - TIMESTEPS + 1)
            if valid_mask[ws:t+1].mean() >= 0.6:
                last_F = run_mmen_predict(mmen, smooth[ws:ws+TIMESTEPS].copy()); freeze = False
        if not blind and not freeze and t >= TIMESTEPS - 1:
            ws = max(0, t - TIMESTEPS + 1)
            if ws + TIMESTEPS <= total_len:
                last_F = run_mmen_predict(mmen, smooth[ws:ws+TIMESTEPS].copy())
        tracker.predict_step(last_F, Q_scale=Q_SCALE)
        if not blind and valid_mask[t]:
            tracker.update(dense_obs[t], np.eye(2) * (1.0 if t >= TIMESTEPS else 50.0))
        kf.append(tracker.get_state()[0])
    return np.array(kf)


# ─────────────────────────────────────────────────────────────────────────────
# CPGD 严格因果（零滞后）推理
#
# 与 run_v8_inference 的平滑（Hann-OLA）模式对照：证明审稿人质疑的"未来数据
# 依赖"来自两处——(1) 每 STRIDE 帧才触发去噪的摊销步长（纯算力问题），以及
# (2) Hann-OLA 把每帧定稿值加权到窗口中心（算法层面的固定滞后）。本函数消除两者：
#   * 每来一帧即在窗口 [k-W+1, k] 上触发一次去噪（去掉摊销 → 需更高算力）；
#   * 只读窗口最后一列（第 k 帧，即边界）作为该帧估计（去掉 OLA 的中心加权 →
#     严格只用过去+当前观测，零滞后）。
# CP-GRU 先验本身即严格因果。因果模式因此是一个真正的实时学习型滤波器。
# 代价：读边界处去噪器上下文最少，精度必然低于读中心的平滑模式——这正是
# filtering vs smoothing 的本质差异，如实报告而非掩盖。
# ─────────────────────────────────────────────────────────────────────────────
def run_v8_inference_causal(v8_model, diff_v8, v8_cfg, dense_obs, valid_mask,
                            ddim_steps=CAUSAL_DDIM_STEPS):
    total_len = len(dense_obs); W = TIMESTEPS
    if v8_model is None:
        return np.column_stack([dense_obs, np.zeros((total_len, 2))])

    dense_obs_v8 = dense_obs.copy(); dense_obs_v8[~valid_mask] = 0.0
    obs_np = np.column_stack([dense_obs_v8, valid_mask.astype(np.float32)])
    obs_t  = torch.from_numpy(obs_np).float().unsqueeze(0).to(DEVICE)

    fv = int(np.argmax(valid_mask)) if valid_mask.any() else 0
    anchor = obs_t[:, fv:fv+1, :2]

    out = np.zeros((total_len, 4), dtype=np.float32)

    with torch.no_grad():
        # 1) 严格因果地把整条序列的 GRU 先验一次性算出（GRU 顺序推进，天然因果）
        vc_full = obs_t[:, :, 2:3]
        on_full = torch.cat(
            [anchor_normalize(obs_t[:, :, :2], anchor, ANCHOR_SCALE) * vc_full, vc_full],
            dim=-1)
        prior_norm_full, prior_feat_full, _ = v8_model.prior(on_full, h0=None)
        prior_abs_full = anchor_denormalize(prior_norm_full, anchor, ANCHOR_SCALE)  # (1,N,2)

        # 2) 逐帧：窗口 [k-W+1, k]，去噪后只取边界列（第 k 帧）
        for k in range(total_len):
            ws = k - W + 1
            if ws < 0:
                # 前 W-1 帧凑不满窗口：退化为纯先验（仍严格因果）
                out[k, :2] = prior_abs_full[0, k, :].cpu().numpy()
                continue
            we = k + 1
            pw_abs = prior_abs_full[:, ws:we, :]                      # (1,W,2)
            pf_anc = prior_feat_full[:, k, :]                         # (1,df)  当前帧特征
            ow_abs = obs_t[:, ws:we, :2]; vw = obs_t[:, ws:we, 2:3]
            innov  = torch.norm(ow_abs - pw_abs, dim=-1, keepdim=True) * vw
            vsm    = vw.sum(dim=1, keepdim=True).clamp(min=1e-5)
            scw    = torch.clamp(innov.sum(dim=1, keepdim=True) / vsm, min=5.0)
            orn    = (ow_abs - pw_abs) / scw * vw
            cond   = torch.cat([orn.permute(0, 2, 1), vw.permute(0, 2, 1)], dim=1)
            x0p    = diff_v8.sample_ddim(v8_model.denoiser, cond, pf_anc, num_steps=ddim_steps)

            # 盲区门控（与平滑部署一致）：窗口尾部 BLACKOUT_THRESH 帧全无效 → 退回纯先验
            cm = (vw[:, -BLACKOUT_THRESH:, :] == 0).all().item()
            vr = vw.mean().item()
            tf = 0.0 if cm else float(np.clip((vr - 0.3) / 0.3, 0.0, 1.0))

            # 只取边界列（窗口最后一帧 = 第 k 帧），严格零滞后
            pos_k = x0p[0, :2, -1] * scw[0, 0, 0] * tf + pw_abs[0, -1, :]
            vel_k = x0p[0, 2:, -1] * VEL_SCALE
            out[k, :2] = pos_k.cpu().numpy()
            out[k, 2:] = vel_k.cpu().numpy()

    # 前导未覆盖帧回填最近有效估计
    return out

# ─────────────────────────────────────────────────────────────────────────────
# T-FoT 推理
# ─────────────────────────────────────────────────────────────────────────────
def run_tfot_all(dense_obs, valid_mask, timestamps, tfot_filter=None):
    """tfot_filter: 可选方法名集合(T-FoT-GP / T-FoT-StP / T-FoT / GPMT),
    仅跑其中的方法(None=全部)。T-FoT-GP/StP 每帧学超参很慢,只需 T-FoT 时
    传 {'T-FoT'} 可大幅提速。"""
    N = len(dense_obs)
    obs_v = dense_obs[valid_mask]
    if len(obs_v) >= 4:
        sd = np.diff(obs_v, n=2, axis=0)
        ns = float(np.median(np.abs(sd))) * 1.4826 / np.sqrt(6)
    else: ns = 5.0
    nv = max(ns ** 2, 0.1); nl = DT * 5

    tks = {'T-FoT-GP': TFoTGP(d=20, poly_order=TFOT_POLY_ORDER),
           'T-FoT-StP': TFoTStP(d=20, poly_order=TFOT_POLY_ORDER, nu=TFOT_NU),
           'T-FoT':     TFoTBaseline(d=20, poly_order=TFOT_POLY_ORDER),
           'GPMT':      GPMTTracker(d=64)}
    if tfot_filter is not None:
        tks = {nm: tr for nm, tr in tks.items() if nm in tfot_filter}
    for tr in tks.values(): tr.reset(noise_var=nv, noise_ls=nl)

    res = {nm: np.zeros((N, 2)) for nm in tks}
    for k in range(N):
        tk = float(timestamps[k]); yk = dense_obs[k]
        for nm, tr in tks.items():
            res[nm][k] = tr.update(tk, yk)
    return res  # each (N,2) positions only

# ─────────────────────────────────────────────────────────────────────────────
# 机动切换检测
# ─────────────────────────────────────────────────────────────────────────────
def detect_maneuver_windows(states, timestamps, half_win=20, top_k=3):
    """
    返回 top_k 个机动最剧烈的时刻（转向角变化率最大处）的窗口索引。
    states: (N,4) [x,y,vx,vy]
    返回: [(start_idx, peak_idx, end_idx), ...]
    """
    N = len(states)
    vx, vy = states[:, 2], states[:, 3]
    heading = np.arctan2(vy, vx)
    dh = np.abs(np.diff(heading))
    dh = np.concatenate([[0], dh])  # (N,)
    # 平滑一下
    kernel = np.ones(5) / 5
    dh_sm  = np.convolve(dh, kernel, mode='same')
    # 找局部极大值
    peaks = []
    for i in range(half_win, N - half_win):
        if dh_sm[i] == dh_sm[i-half_win:i+half_win].max() and dh_sm[i] > 0.05:
            peaks.append((dh_sm[i], i))
    peaks.sort(key=lambda x: -x[0])
    results = []
    used = set()
    for _, pi in peaks:
        if any(abs(pi - u) < half_win for u in used): continue
        s = max(0, pi - half_win); e = min(N - 1, pi + half_win)
        results.append((s, pi, e)); used.add(pi)
        if len(results) >= top_k: break
    return results

# ─────────────────────────────────────────────────────────────────────────────
# 误差计算
# ─────────────────────────────────────────────────────────────────────────────
def pos_err(pred_pos, gt_pos):
    return np.linalg.norm(pred_pos - gt_pos, axis=-1)

def vel_err(pred_vel, gt_vel):
    return np.linalg.norm(pred_vel - gt_vel, axis=-1)

# ─────────────────────────────────────────────────────────────────────────────
# 单次 trial 运行
# ─────────────────────────────────────────────────────────────────────────────
def run_one_trial(sc_name, sc_cfg, v8_model, diff_v8, v8_cfg, mmen,
                  duration, seed, run_tfot=True, dl=None,
                  do_causal=False, causal_ddim=CAUSAL_DDIM_STEPS, our=None,
                  tfot_filter=None):
    blen = sc_cfg.get('blackout_len', 0)
    min_cov = sc_cfg.get('min_coverage', 0.75)

    states, timestamps, phys_obs, valid_mask, dense_obs, bo_s, bo_e = \
        build_one_trial(sc_cfg, duration, seed, blackout_len=blen)
    N = len(states)

    # Coverage check on natural (pre-blackout) valid_mask:
    # blackout_len > 0 means valid_mask already has blackout applied inside
    # build_one_trial, so we check the raw physical coverage before that.
    # Re-derive natural coverage: count frames that were valid before blackout.
    if blen > 0 and bo_s >= 0:
        # bo_s/bo_e are 0-based Python indices of the injected blackout
        natural_valid = valid_mask.copy().astype(int)
        natural_valid[bo_s:bo_e] = 1           # undo blackout to check physical coverage
        natural_coverage = natural_valid.mean()
    else:
        natural_coverage = valid_mask.mean()

    if natural_coverage < min_cov:
        return None  # too sparse, caller will retry with a different seed

    if valid_mask.sum() < TIMESTEPS:
        return None  # absolute minimum guard

    # CPGD 推理：smooth(无KF) + full(+MMEN-KF)。写入历史字段 cpgd_* / cpgd_smooth_*。
    if not our:
        our = {PRIMARY_METHOD: dict(model=v8_model, label='CPGD')}
    our_out = {}   # name -> dict(kf=(N,4), smooth=(N,4))
    for name, spec in our.items():
        kf_m, smooth_m = run_v8_inference(
            spec['model'], diff_v8, v8_cfg, mmen, dense_obs, valid_mask)
        our_out[name] = dict(kf=kf_m, smooth=smooth_m)

    primary = PRIMARY_METHOD if PRIMARY_METHOD in our_out else next(iter(our_out))
    kf_v8     = our_out[primary]['kf']
    smooth_v8 = our_out[primary]['smooth']

    # CPGD（严格因果 / 零滞后模式）—— 可选，逐帧触发去噪只读窗口边界
    causal_v8 = None
    if do_causal:
        causal_v8 = run_v8_inference_causal(
            our.get(primary, {}).get('model', v8_model),
            diff_v8, v8_cfg, dense_obs, valid_mask, ddim_steps=causal_ddim)

    # 深度学习基线（NB-DLSTM 递归跟踪器 + Transformer + CSDI 扩散插补）
    dl_res = {}
    if dl:
        if 'dlstm' in dl:
            dl_res['dlstm'] = BDL.run_dlstm_inference(dl['dlstm'], dense_obs, valid_mask, DEVICE)
        if 'transformer' in dl:
            # smooth mode = sliding-window Hann-OLA, matched to CPGD-Smooth/CSDI
            dl_res['transformer'] = BDL.run_transformer_inference(
                dl['transformer'], dense_obs, valid_mask, DEVICE,
                window=TIMESTEPS, stride=STRIDE, mode='smooth')
            # strict-causal mode (only when causal trials are being computed) so it
            # sits in the causal table alongside CPGD-Causal / DLSTM
            if do_causal:
                dl_res['transformer_causal'] = BDL.run_transformer_inference(
                    dl['transformer'], dense_obs, valid_mask, DEVICE, mode='causal')
        if 'csdi' in dl:
            m_csdi, diff_csdi = dl['csdi']
            dl_res['csdi'] = BDL.run_csdi_inference(
                m_csdi, diff_csdi, dense_obs, valid_mask, DEVICE,
                window=TIMESTEPS, stride=STRIDE)

    # T-FoT
    tfot_res = {}
    if run_tfot:
        tfot_res = run_tfot_all(dense_obs, valid_mask, timestamps, tfot_filter=tfot_filter)

    # 机动切换窗口
    maneuver_wins = detect_maneuver_windows(states, timestamps)

    # 误差序列
    gt_pos = states[:, :2]; gt_vel = states[:, 2:]
    obs_pos_err = np.linalg.norm(dense_obs - gt_pos, axis=1)

    cpgd_pos_err    = pos_err(kf_v8[:N, :2], gt_pos)
    cpgd_vel_err    = vel_err(kf_v8[:N, 2:], gt_vel)
    cpgd_smooth_pos_err = pos_err(smooth_v8[:N, :2], gt_pos)
    cpgd_smooth_vel_err = vel_err(smooth_v8[:N, 2:], gt_vel)

    tfot_pos_errs = {}
    for nm, res in tfot_res.items():
        tfot_pos_errs[nm] = pos_err(res[:N], gt_pos)

    # 深度学习基线误差
    dl_pos_errs, dl_vel_errs = {}, {}
    for kind, res in dl_res.items():
        dl_pos_errs[kind] = pos_err(res[:N, :2], gt_pos)
        dl_vel_errs[kind] = vel_err(res[:N, 2:], gt_vel)

    result = dict(
        # 原始数据
        states       = states.astype(np.float32),       # (N,4)
        timestamps   = timestamps.astype(np.float32),   # (N,)
        phys_obs     = np.where(valid_mask[:, None], phys_obs, np.nan).astype(np.float32),  # (N,2)
        valid_mask   = valid_mask.astype(np.uint8),     # (N,)
        dense_obs    = dense_obs.astype(np.float32),    # (N,2)
        # 预测结果 — V8 Smooth（扩散模型平滑，不含KF）
        cpgd_smooth_pos = smooth_v8[:N, :2].astype(np.float32),
        cpgd_smooth_vel = smooth_v8[:N, 2:].astype(np.float32),
        # 预测结果 — CPGD（V8 Smooth + MMEN-KF）
        cpgd_pos     = kf_v8[:N, :2].astype(np.float32),
        cpgd_vel     = kf_v8[:N, 2:].astype(np.float32),
        # 误差
        obs_pos_err         = obs_pos_err.astype(np.float32),
        cpgd_smooth_pos_err = cpgd_smooth_pos_err.astype(np.float32),
        cpgd_smooth_vel_err = cpgd_smooth_vel_err.astype(np.float32),
        cpgd_pos_err        = cpgd_pos_err.astype(np.float32),
        cpgd_vel_err        = cpgd_vel_err.astype(np.float32),
        # 盲区信息
        blackout_start = np.int32(bo_s),
        blackout_end   = np.int32(bo_e),
        # 机动切换窗口 (top3 x 3: [start, peak, end])
        maneuver_windows = np.array(maneuver_wins, dtype=np.int32) if maneuver_wins
                           else np.zeros((0, 3), dtype=np.int32),
    )

    # CPGD 因果模式结果（如启用）
    if causal_v8 is not None:
        result['cpgd_causal_pos']     = causal_v8[:N, :2].astype(np.float32)
        result['cpgd_causal_vel']     = causal_v8[:N, 2:].astype(np.float32)
        result['cpgd_causal_pos_err'] = pos_err(causal_v8[:N, :2], gt_pos).astype(np.float32)
        result['cpgd_causal_vel_err'] = vel_err(causal_v8[:N, 2:], gt_vel).astype(np.float32)

    # 深度学习基线结果
    for kind, res in dl_res.items():
        result[f'{kind}_pos']     = res[:N, :2].astype(np.float32)
        result[f'{kind}_vel']     = res[:N, 2:].astype(np.float32)
        result[f'{kind}_pos_err'] = dl_pos_errs[kind].astype(np.float32)
        result[f'{kind}_vel_err'] = dl_vel_errs[kind].astype(np.float32)

    # T-FoT 结果
    for nm, res in tfot_res.items():
        safe = nm.replace('-', '_').replace(' ', '_')
        result[f'{safe}_pos']     = res[:N].astype(np.float32)
        result[f'{safe}_pos_err'] = tfot_pos_errs[nm].astype(np.float32)

    return result

# ─────────────────────────────────────────────────────────────────────────────
# 蒙特卡洛循环
# ─────────────────────────────────────────────────────────────────────────────
def run_monte_carlo(sc_name, sc_cfg, v8_model, diff_v8, v8_cfg, mmen,
                    n_mc, duration, base_seed, run_tfot, out_path,
                    dl=None, causal_trials=0, causal_ddim=CAUSAL_DDIM_STEPS, our=None,
                    tfot_filter=None):
    print(f"\n{'='*60}")
    print(f"  Scenario: {sc_cfg['label']}")
    print(f"  MC runs : {n_mc}")
    print(f"  Min coverage: {sc_cfg.get('min_coverage', 0.75):.0%}")
    if our:
        _names = ', '.join('{}({})'.format(k, v['label']) for k, v in our.items())
        print(f"  Our methods : {_names}")
    if dl:
        print(f"  DL baselines: {', '.join(sorted(dl.keys()))}")
    if causal_trials > 0:
        print(f"  Causal mode : first {causal_trials} trials (DDIM={causal_ddim})")
    print(f"{'='*60}")

    all_trials = []
    valid_count = 0
    seed = base_seed

    pbar = tqdm(total=n_mc, desc=sc_name)
    while valid_count < n_mc:
        try:
            np.random.seed(seed); torch.manual_seed(seed)
            do_causal = valid_count < causal_trials
            res = run_one_trial(sc_name, sc_cfg, v8_model, diff_v8, v8_cfg, mmen,
                                duration, seed, run_tfot=run_tfot, dl=dl,
                                do_causal=do_causal, causal_ddim=causal_ddim, our=our,
                                tfot_filter=tfot_filter)
            if res is not None:
                res['seed'] = np.int32(seed)
                all_trials.append(res)
                valid_count += 1
                pbar.update(1)
        except Exception as ex:
            print(f"\n  [seed={seed}] Error: {ex}")
        seed += 1
    pbar.close()

    # 打包成 mat 格式：每个字段是 (n_mc, ...) 的 numpy 数组
    # 截断到所有 trial 的最短帧数，避免用零填充引入虚假的末尾零点
    N_max = min(len(t['states']) for t in all_trials)

    # 键可能因 trial 而异（因果模式仅前若干 trial 有 cpgd_causal_*）——取并集，
    # 缺失键的 trial 填 NaN 便于下游按需过滤，绝不用 0 冒充真实估计。
    def stack_field(key, pad_val=0.0):
        present = [(i, t[key]) for i, t in enumerate(all_trials) if key in t]
        ref = present[0][1]
        if ref.ndim == 0:
            out = np.full(n_mc, np.nan, dtype=np.float32)
            for i, a in present:
                out[i] = a.item()
            return out
        shape1 = ref.shape[1:] if ref.ndim > 1 else ()
        out = np.full((n_mc, N_max) + shape1, pad_val, dtype=np.float32)
        for i, a in present:
            Ni = min(a.shape[0], N_max) if a.ndim >= 1 else 1
            out[i, :Ni] = a[:Ni]
        return out

    mat = {}
    scalar_keys = {'seed', 'blackout_start', 'blackout_end'}
    all_keys = set().union(*[set(t.keys()) for t in all_trials])
    vec_keys = all_keys - scalar_keys - {'maneuver_windows'}
    # 因果键缺失处填 NaN（非 0），供显著性分析脚本按 trial 过滤
    # （cpgd_causal_* 以及 transformer_causal_* 只在前 causal_trials 个 trial 上计算）
    causal_keys = {k for k in vec_keys if 'causal' in k}

    for k in vec_keys:
        mat[k] = stack_field(k, pad_val=(np.nan if k in causal_keys else 0.0))
    for k in scalar_keys:
        mat[k] = np.array([t.get(k, np.int32(-1)) for t in all_trials])
    mat['cpgd_causal_n'] = np.int32(min(causal_trials, n_mc))

    # maneuver_windows: object array of (ki, 3) arrays
    mw_cell = np.empty(n_mc, dtype=object)
    for i, t in enumerate(all_trials):
        mw_cell[i] = t['maneuver_windows']
    mat['maneuver_windows'] = mw_cell

    # 元信息
    mat['scenario_name']    = sc_name
    mat['scenario_label']   = sc_cfg['label']
    mat['n_mc']             = np.int32(n_mc)
    mat['dt']               = np.float32(DT)
    mat['N_max']            = np.int32(N_max)

    sio.savemat(out_path, mat, do_compression=True)
    print(f"  Saved → {out_path}  ({n_mc} trials, N_max={N_max})")
    return mat

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--n_mc',      type=int,   default=N_MC)
    p.add_argument('--duration',  type=float, default=120.0)
    p.add_argument('--base_seed', type=int,   default=42)
    p.add_argument('--out_dir',   type=str,   default='./mc_results')
    p.add_argument('--skip_tfot', action='store_true')
    p.add_argument('--skip_dl', action='store_true',
                   help='skip learning-based baselines (CSDI/DLSTM/Transformer)')
    p.add_argument('--only_ours', action='store_true',
                   help='只跑 CPGD，跳过所有 T-FoT 与 DL 基线'
                        '（等价于 --skip_tfot --skip_dl，且不加载 DL 权重）')
    p.add_argument('--dl_methods', type=str, default=None,
                   help='逗号分隔,仅跑 DL 基线的子集(dlstm,transformer,csdi)。默认全部。')
    p.add_argument('--tfot_methods', type=str, default=None,
                   help='逗号分隔,仅跑 T-FoT 系列的子集(T-FoT-GP,T-FoT-StP,T-FoT,GPMT)。'
                        '例: --tfot_methods T-FoT 只跑最快的多项式版。默认全部(含很慢的 GP/StP)。')
    p.add_argument('--causal_trials', type=int, default=0,
                   help='run strictly-causal CPGD on the first N trials per scenario '
                        '(0 = disabled). Causal mode is ~W/STRIDE x slower.')
    p.add_argument('--causal_ddim', type=int, default=CAUSAL_DDIM_STEPS,
                   help='DDIM steps for causal mode (fewer = faster real-time story)')
    p.add_argument('--scenarios', type=str,   default='easy,medium,hard,blackout',
                   help='comma-separated list of scenarios to run')
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # --only_ours：只跑 CPGD，跳过 T-FoT 与 DL（连 DL 权重都不加载）
    skip_dl   = args.skip_dl or args.only_ours
    skip_tfot = args.skip_tfot or args.only_ours
    if args.only_ours:
        print("[only_ours] 仅跑 CPGD，跳过所有 T-FoT / DL 基线。")

    # 方法子集过滤(None=全部)
    dl_filter  = ({s.strip() for s in args.dl_methods.split(',')}
                  if args.dl_methods else None)
    tfot_filter = ({s.strip() for s in args.tfot_methods.split(',')}
                   if args.tfot_methods else None)
    if dl_filter:
        print(f"[filter] DL baselines = {sorted(dl_filter)}")
    if tfot_filter:
        print(f"[filter] T-FoT methods = {sorted(tfot_filter)}")

    v8_model, diff_v8, v8_cfg, mmen, dl, our = load_models(
        load_dl=not skip_dl, dl_filter=dl_filter)
    if skip_dl:
        dl = {}

    sc_list = [s.strip() for s in args.scenarios.split(',')]
    for sc_name in sc_list:
        if sc_name not in SCENARIOS:
            print(f"  [Warning] Unknown scenario '{sc_name}', skipping."); continue
        sc_cfg   = SCENARIOS[sc_name]
        out_path = os.path.join(args.out_dir, f'results_{sc_name}.mat')
        run_monte_carlo(
            sc_name, sc_cfg, v8_model, diff_v8, v8_cfg, mmen,
            n_mc       = args.n_mc,
            duration   = args.duration,
            base_seed  = args.base_seed,
            run_tfot   = not skip_tfot,
            out_path   = out_path,
            dl         = dl,
            causal_trials = args.causal_trials,
            causal_ddim   = args.causal_ddim,
            our        = our,
            tfot_filter = tfot_filter)

    print("\nAll done.")

if __name__ == '__main__':
    main()
