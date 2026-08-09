"""
train_prior_denoiser_v8.py
==========================
"Curriculum Fine-tuning" — 支持灵活热启动/冷启动的蒙眼训练 (v8)

继承 v7 的全部 4 项核心修复：
  Fix 1: 删除假停车（缺失处置零 Missing Token）
  Fix 2: 强硬物理遮罩（Hard Masking，替换 Leaky Gate）
  Fix 3: 蒙眼训练（Severe Blackout Augmentation）
  Fix 4: Diffusion 条件泄露修复（resid * valid_flag）

v8 新增：灵活训练模式（CLI 参数控制）
  --mode scratch   : 从零冷启动（等同于 v7 逻辑，STAGE1=15000 步）
  --mode finetune  : 热启动微调（加载 v6/v7 权重，STAGE1=1000 步，低 LR）
  --pretrain_ckpt  : 指定预训练权重路径（默认 v6 final.pth）
  --resume         : 从上次断点继续训练（加载同目录 ckpt，恢复 optimizer/scheduler）

热启动微调超参（mode=finetune）：
  TRAIN_STEPS     = 15000   （原本需要 6 万步，现在 1.5 万步收敛）
  LR_PRIOR        = 1e-4    （原 3e-4，保护 v6 已学好的流形）
  LR_DENOISER     = 5e-5    （原 2e-4）
  STAGE1_ONLY_STEPS = 1000  （让 GRU 先用 1000 步适应蒙眼，再联合微调）
  VAL_INTERVAL    = 200     （高频验证，密切观察收敛情况）

用法示例：
  # 从 v6 权重热启动微调（推荐，最快）
  python train_prior_denoiser_v8.py --mode finetune --pretrain_ckpt ./models_prior_denoiser_v6/final.pth

  # 从 v7 权重热启动继续强化蒙眼能力
  python train_prior_denoiser_v8.py --mode finetune --pretrain_ckpt ./models_prior_denoiser_v7/final.pth

  # 从零冷启动（全程 6 万步）
  python train_prior_denoiser_v8.py --mode scratch

  # 断点续训（从 v8 自己的 ckpt 继续）
  python train_prior_denoiser_v8.py --resume ./models_prior_denoiser_v8/ckpt_step5000.pth
"""

import os
import sys
import math
import csv
import argparse
import contextlib
import warnings
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from generate_trajectory import generate_multi_segment_trajectory, generate_ct_trajectory
from scene_configuration import SceneConfigurator, AgentConfig
from trajectory_generation import TrajectoryGenerator, AgentTrajectory
from detection_process import FusedDetectionProcess

warnings.filterwarnings("ignore", category=UserWarning)


# ─────────────────────────────────────────────────────────────────────────────
# CLI 参数
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description='Prior-Denoiser V8 Training')
    p.add_argument('--mode', choices=['scratch', 'finetune', 'refix_prior'], default='finetune',
                   help='scratch=从零训练 | finetune=热启动微调 | refix_prior=冻结denoiser只重训Prior（推荐）')
    p.add_argument('--pretrain_ckpt', type=str,
                   default='./models_prior_denoiser_v6/final.pth',
                   help='热启动时加载的预训练权重路径（v6 或 v7 的 final.pth）')
    p.add_argument('--refix_prior_ckpt', type=str,
                   default='./models_prior_denoiser_v8/final.pth',
                   help='refix_prior 模式：从此 ckpt 加载 denoiser/EMA 权重（冻结），只重训 Prior')
    p.add_argument('--resume', type=str, default=None,
                   help='断点续训：指定 v8 自身的 ckpt_stepXXXX.pth 路径')
    p.add_argument('--save_dir', type=str, default='./models_prior_denoiser_v8',
                   help='模型保存目录')
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# TrainConfig — 冷启动默认值，热启动会动态覆盖
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class TrainConfig:
    # ── 序列与归一化 ──
    WINDOW:       int   = 32
    DT:           float = 0.1
    LONG_MULT:    int   = 50
    STRIDE_DIV:   int   = 2
    ANCHOR_SCALE: float = 1000.0
    VEL_SCALE:    float = 50.0

    # ── 网络结构 ──
    OBS_IN_DIM:   int   = 3
    PRIOR_HIDDEN: int   = 256
    PRIOR_LAYERS: int   = 2
    PRIOR_OUT:    int   = 128
    STATE_DIM:    int   = 4
    COND_DIM:     int   = 3
    UNET_IN_CH:   int   = 7
    BASE_DIM:     int   = 64
    DIM_MULTS:    Tuple = (1, 2, 4, 8)

    # ── 扩散 ──
    DIFF_STEPS:   int   = 1000
    INF_STEPS:    int   = 25

    # ── 训练（冷启动默认值） ──
    TRAIN_STEPS:        int   = 60000
    BATCH_SIZE:         int   = 4
    SAMPLES_PER_SEQ:    int   = 8
    LR_PRIOR:           float = 3e-4
    LR_DENOISER:        float = 2e-4
    INVALID_WEIGHT:     float = 0.3
    BLACKOUT_WEIGHT:    float = 2.5   # 蒙眼帧 prior loss 权重（原1.0→2.5，强制GRU学惯性）
    SMOOTH_LOSS_WEIGHT: float = 0.08  # 蒙眼期平滑正则化权重（新增：抑制GRU在盲区输出抖动）
    EMA_DECAY:          float = 0.9995
    GRAD_CLIP:          float = 1.0
    STAGE1_ONLY_STEPS:  int   = 15000   # 冷启动：前 1.5 万步只训 Prior

    # ── 蒙眼超参 ──
    BLACKOUT_PROB:  float = 0.75   # 每条序列触发蒙眼的概率（原0.4→0.75）
    BLACKOUT_N_MAX: int   = 4      # 每条序列最多叠加几段蒙眼（新增，原来只有1段）
    BLACKOUT_MIN:   int   = 30
    BLACKOUT_MAX:   int   = 120    # 最大蒙眼帧数（原80→120，约7.5%序列长度）

    VAL_INTERVAL: int = 500
    VAL_SEQS:     int = 4

    # ── 物理 / 课程 ──
    DRONE_ALT:    Tuple = (50, 150)
    DRONE_SPEED:  Tuple = (10, 25)
    TX_POWER:     float = 20.0
    SENSITIVITY:  float = -95.0
    FREQ:         float = 2.4e9

    CURRICULUM: List = field(default_factory=lambda: [
        (0,     8, 1,  600, 10, 0.3),
        (10000, 6, 2,  800, 15, 0.5),
        (25000, 5, 3, 1000, 20, 0.8),
        (40000, 4, 4, 1200, 25, 1.2),
    ])

    SAVE_DIR: str = './models_prior_denoiser_v8'
    LOG_CSV:  str = './models_prior_denoiser_v8/val_log.csv'

    def __post_init__(self):
        os.makedirs(self.SAVE_DIR, exist_ok=True)

    @property
    def LONG_T(self) -> int:
        return self.WINDOW * self.LONG_MULT

    @property
    def STRIDE(self) -> int:
        return self.WINDOW // self.STRIDE_DIV

    @property
    def DEVICE(self) -> torch.device:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_finetune_config(save_dir: str) -> TrainConfig:
    """热启动微调专用超参——低 LR、短 STAGE1、高频验证。"""
    cfg = TrainConfig()
    cfg.SAVE_DIR            = save_dir
    cfg.LOG_CSV             = os.path.join(save_dir, 'val_log.csv')
    cfg.TRAIN_STEPS         = 15000
    cfg.LR_PRIOR            = 1e-4
    cfg.LR_DENOISER         = 5e-5
    cfg.STAGE1_ONLY_STEPS   = 1000   # 热启动：只需 1000 步让 GRU 适应蒙眼
    cfg.VAL_INTERVAL        = 200    # 高频验证，密切观察收敛
    cfg.BLACKOUT_PROB       = 0.75   # 保持增强蒙眼概率
    cfg.BLACKOUT_N_MAX      = 4      # 允许多段蒙眼
    cfg.BLACKOUT_MAX        = 120
    cfg.BLACKOUT_WEIGHT     = 2.5
    cfg.SMOOTH_LOSS_WEIGHT  = 0.08
    # 课程表也缩短：微调时直接用难度最高的场景，充分激活蒙眼能力
    cfg.CURRICULUM = [
        (0,    5, 3, 1000, 20, 0.8),
        (5000, 4, 4, 1200, 25, 1.2),
    ]
    os.makedirs(save_dir, exist_ok=True)
    return cfg


def make_refix_prior_config(save_dir: str) -> TrainConfig:
    """refix_prior 专用超参：冻结 Denoiser，专注重训 Prior，强化蒙眼。"""
    cfg = TrainConfig()
    cfg.SAVE_DIR            = save_dir
    cfg.LOG_CSV             = os.path.join(save_dir, 'val_log.csv')
    cfg.TRAIN_STEPS         = 20000   # 只训 Prior，步数适中
    cfg.LR_PRIOR            = 2e-4    # Prior 可以用稍高 LR（从头重训）
    cfg.LR_DENOISER         = 0.0     # Denoiser 冻结，LR 设 0
    cfg.STAGE1_ONLY_STEPS   = 20000   # 全程只训 Prior（Stage1 贯穿始终）
    cfg.VAL_INTERVAL        = 200
    cfg.BLACKOUT_PROB       = 0.80    # 更激进的蒙眼
    cfg.BLACKOUT_N_MAX      = 5
    cfg.BLACKOUT_MAX        = 150
    cfg.BLACKOUT_WEIGHT     = 3.0     # 蒙眼帧权重最高
    cfg.SMOOTH_LOSS_WEIGHT  = 0.10
    cfg.CURRICULUM = [
        (0,     6, 2,  800, 15, 0.5),
        (5000,  5, 3, 1000, 20, 0.8),
        (12000, 4, 4, 1200, 25, 1.2),
    ]
    os.makedirs(save_dir, exist_ok=True)
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────
@contextlib.contextmanager
def _silence():
    with open(os.devnull, 'w') as dn:
        with contextlib.redirect_stdout(dn), contextlib.redirect_stderr(dn):
            yield


def _get_curriculum(cfg, step):
    params = cfg.CURRICULUM[0][1:]
    for entry in cfg.CURRICULUM:
        if step >= entry[0]:
            params = entry[1:]
    return params


# ─────────────────────────────────────────────────────────────────────────────
# Data generation  [Fix 1: 缺失处置零，删除 forward-fill]
# ─────────────────────────────────────────────────────────────────────────────
def generate_long_sequence(cfg: TrainConfig, step: int) -> Optional[Dict]:
    n_drones, n_obstacles, area_side, spd_max, turn_rate_max = _get_curriculum(cfg, step)
    T, dt   = cfg.LONG_T, cfg.DT
    dur     = T * dt
    area    = [area_side, area_side, 200]
    n_seg   = np.random.randint(max(2, T // 60), max(4, T // 30) + 1)
    spd_lo, spd_hi = 2.0, float(spd_max)
    min_seg = max(1.0, dur / (n_seg * 4))

    try:
        states, _, timestamps, _, _, _ = generate_multi_segment_trajectory(
            total_duration=dur, n_segments=n_seg,
            turn_rate_range=(-turn_rate_max, turn_rate_max),
            speed_range=(spd_lo, spd_hi), dt=dt, min_segment_duration=min_seg)
    except Exception:
        try:
            x0, y0 = np.random.uniform(50, area_side - 50, 2)
            spd = np.random.uniform(spd_lo, spd_hi)
            ang = np.random.uniform(0, 2 * np.pi)
            seg = generate_ct_trajectory(
                np.array([x0, y0, spd * np.cos(ang), spd * np.sin(ang)]),
                turn_rate=np.random.uniform(-turn_rate_max, turn_rate_max),
                duration=dur, dt=dt)
            states, timestamps = seg.states, seg.timestamps
        except Exception:
            return None

    states, timestamps = states[:T], timestamps[:T]
    if len(states) < T:
        pad        = T - len(states)
        states     = np.pad(states,     ((0, pad), (0, 0)), mode='edge')
        timestamps = np.pad(timestamps, (0, pad),           mode='edge')

    off_x = np.random.uniform(50, area_side - 50) - states[0, 0]
    off_y = np.random.uniform(50, area_side - 50) - states[0, 1]
    states[:, 0] += off_x
    states[:, 1] += off_y

    try:
        with _silence():
            configurator = SceneConfigurator(
                area_size=tuple(area), n_drones=n_drones, n_targets=1,
                n_obstacles=n_obstacles)
            traj_gen = TrajectoryGenerator(dt=dt, duration=dur)
            scene    = configurator.generate_scene()
            pos3d    = np.column_stack([states[:, :2], np.zeros(T)])
            tgt_traj = AgentTrajectory(
                agent_id=0, agent_type='target', states_2d=states,
                timestamps=timestamps, altitude=0.0, positions_3d=pos3d,
                segments=[], turn_rates=[], speeds=[],
                config=AgentConfig(
                    agent_id=0, agent_type='target', initial_state_2d=states[0],
                    altitude=0.0, is_single_segment=False, n_segments=n_seg,
                    turn_rate_range=(-turn_rate_max, turn_rate_max),
                    speed_range=(spd_lo, spd_hi), min_segment_duration=min_seg,
                    P_tx=cfg.TX_POWER, G_tx=0.0))
            drone_trajs = {
                dc.agent_id: traj_gen.generate_drone_trajectory(
                    dc.initial_state_2d[:2], dc.altitude, list(cfg.DRONE_SPEED),
                    area, scene['obstacles'])
                for dc in scene['drones']}
            det_proc = FusedDetectionProcess(
                tx_power=cfg.TX_POWER, sensitivity=cfg.SENSITIVITY,
                frequency=cfg.FREQ, nlos_penalty=0.05, outlier_sigma=2.5,
                min_los_for_nlos_suppress=1)
            detections = det_proc.generate_detections(
                drone_trajectories=drone_trajs, target_trajectories={0: tgt_traj},
                obstacles=scene['obstacles'], timestamps=timestamps)
            step_map = defaultdict(list)
            for det in detections.get(0, []):
                step_map[int(round(det.timestamp / dt))].append(det)
            raw_obs = np.full((T, 2), np.nan)
            valid   = np.zeros(T, dtype=bool)
            for si in range(T):
                if si in step_map and step_map[si]:
                    raw_obs[si] = step_map[si][0].target_position[:2]
                    valid[si]   = True
    except Exception:
        std     = np.maximum(np.linalg.norm(states[:, 2:], axis=1) * 0.5, 2.0)
        raw_obs = states[:, :2] + np.random.randn(T, 2) * std[:, None]
        valid   = np.ones(T, dtype=bool)

    # [Fix 1] 缺失处直接置零（Missing Token），不再 forward-fill
    dense = np.nan_to_num(raw_obs, nan=0.0)

    return dict(states=states.astype(np.float32),
                raw_obs=raw_obs.astype(np.float32),
                valid=valid,
                dense_obs=dense.astype(np.float32),
                timestamps=timestamps.astype(np.float32))


# ─────────────────────────────────────────────────────────────────────────────
# Model: CausalPriorGRU  [Fix 2: Hard Masking 替换 Leaky Gate]
# ─────────────────────────────────────────────────────────────────────────────
class CausalPriorGRU(nn.Module):
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(cfg.OBS_IN_DIM, cfg.PRIOR_HIDDEN),
            nn.LayerNorm(cfg.PRIOR_HIDDEN),
            nn.SiLU())
        # 无 valid_gate（删除 v6 的漏水门）
        self.gru       = nn.GRU(cfg.PRIOR_HIDDEN, cfg.PRIOR_HIDDEN,
                                cfg.PRIOR_LAYERS, batch_first=True, dropout=0.1)
        self.pos_head  = nn.Linear(cfg.PRIOR_HIDDEN, 2)
        self.feat_head = nn.Linear(cfg.PRIOR_HIDDEN, cfg.PRIOR_OUT)

    def forward(self, obs_norm_seq: torch.Tensor,
                h0: Optional[torch.Tensor] = None):
        # [Fix 2] 强硬遮罩：valid=0 时坐标绝对为 0
        valid_flag              = obs_norm_seq[:, :, 2:3]
        obs_masked              = obs_norm_seq.clone()
        obs_masked[:, :, :2]   = obs_masked[:, :, :2] * valid_flag
        feat                   = self.input_proj(obs_masked)
        out, h_final           = self.gru(feat, h0)
        return self.pos_head(out), self.feat_head(out), h_final


# ─────────────────────────────────────────────────────────────────────────────
# Model: ResidualUNet（与 v6/v7 相同）
# ─────────────────────────────────────────────────────────────────────────────
class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.c1    = nn.Sequential(
            nn.GroupNorm(8, in_ch), nn.SiLU(), nn.Conv1d(in_ch, out_ch, 3, padding=1))
        self.t_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch))
        self.c2    = nn.Sequential(
            nn.GroupNorm(8, out_ch), nn.SiLU(), nn.Dropout(0.1),
            nn.Conv1d(out_ch, out_ch, 3, padding=1))
        self.skip  = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        return self.c2(self.c1(x) + self.t_mlp(t_emb)[:, :, None]) + self.skip(x)


class ResidualUNet(nn.Module):
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        base, mults = cfg.BASE_DIM, cfg.DIM_MULTS
        time_dim    = base * 4
        prior_dim   = cfg.PRIOR_OUT

        self.time_mlp  = nn.Sequential(
            nn.Linear(base, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))
        self.prior_mlp = nn.Sequential(
            nn.Linear(prior_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))
        self.init_conv = nn.Conv1d(cfg.UNET_IN_CH, base, 7, padding=3)
        channels       = [base] + [base * m for m in mults]

        self.downs = nn.ModuleList()
        for i in range(len(mults)):
            self.downs.append(nn.ModuleList([
                ResBlock1D(channels[i],     channels[i + 1], time_dim),
                ResBlock1D(channels[i + 1], channels[i + 1], time_dim),
                nn.Conv1d(channels[i + 1], channels[i + 1], 3, stride=2, padding=1)
                if i < len(mults) - 1 else nn.Identity()
            ]))
        self.mid1 = ResBlock1D(channels[-1], channels[-1], time_dim)
        self.mid2 = ResBlock1D(channels[-1], channels[-1], time_dim)
        self.ups  = nn.ModuleList()
        for i in reversed(range(len(mults))):
            in_ch = (channels[-1] if i == len(mults) - 1
                     else channels[i + 2]) + channels[i + 1]
            self.ups.append(nn.ModuleList([
                ResBlock1D(in_ch,           channels[i + 1], time_dim),
                ResBlock1D(channels[i + 1], channels[i + 1], time_dim),
                nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
                if i > 0 else nn.Identity()
            ]))
        self.final = nn.Sequential(
            nn.GroupNorm(8, channels[1]), nn.SiLU(),
            nn.Conv1d(channels[1], cfg.STATE_DIM, 1))

    def _sinusoidal(self, t, base=32, device=None):
        freqs = torch.exp(
            torch.arange(base, device=device) * -math.log(10000) / (base - 1))
        args  = t[:, None] * freqs[None]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    def forward(self, x, t, cond, pf_anchor):
        t_emb  = self.time_mlp(self._sinusoidal(t, device=x.device))
        pf_emb = self.prior_mlp(pf_anchor)
        t_emb  = t_emb + pf_emb
        h      = self.init_conv(torch.cat([x, cond], dim=1))
        skips  = []
        for r1, r2, down in self.downs:
            h = r2(r1(h, t_emb), t_emb); skips.append(h); h = down(h)
        h = self.mid2(self.mid1(h, t_emb), t_emb)
        for (r1, r2, up), skip in zip(self.ups, reversed(skips)):
            h = torch.cat([F.interpolate(h, size=skip.shape[-1]), skip], dim=1)
            h = r2(r1(h, t_emb), t_emb); h = up(h)
        return self.final(h)


class DualTrackModel(nn.Module):
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.prior    = CausalPriorGRU(cfg)
        self.denoiser = ResidualUNet(cfg)


class EMA:
    def __init__(self, model, decay=0.9995):
        self.decay  = decay
        self.shadow = deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.data.mul_(self.decay).add_(p.data, alpha=1 - self.decay)


# ─────────────────────────────────────────────────────────────────────────────
# GaussianDiffusion（与 v6/v7 相同）
# ─────────────────────────────────────────────────────────────────────────────
class GaussianDiffusion:
    def __init__(self, cfg: TrainConfig, device):
        self.T, self.device = cfg.DIFF_STEPS, device
        x   = torch.linspace(0, self.T, self.T + 1)
        ac  = torch.cos(((x / self.T) + 0.008) / 1.008 * math.pi * 0.5) ** 2
        betas = torch.clamp(1 - (ac[1:] / ac[:-1]), 1e-4, 0.9999).to(device)
        self.alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
        self.sqrt_ac        = torch.sqrt(self.alphas_cumprod)
        self.sqrt_1m_ac     = torch.sqrt(1.0 - self.alphas_cumprod)

    def compute_loss(self, denoiser, x0, cond, pf_anchor):
        B     = x0.shape[0]
        t     = torch.randint(0, self.T, (B,), device=self.device)
        noise = torch.randn_like(x0)
        if torch.rand(1).item() < 0.10:
            cond = torch.zeros_like(cond); pf_anchor = torch.zeros_like(pf_anchor)
        x_t        = (self.sqrt_ac[t].view(B, 1, 1) * x0
                      + self.sqrt_1m_ac[t].view(B, 1, 1) * noise)
        noise_pred = denoiser(x_t, t, cond, pf_anchor)
        return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def sample_ddim(self, denoiser, cond, pf_anchor,
                    num_steps=25, guidance_scale=1.5):
        B, _, W  = cond.shape
        x_t      = torch.randn((B, 4, W), device=self.device)
        step_idx = torch.linspace(self.T - 1, 0, num_steps,
                                  dtype=torch.long, device=self.device)
        for i in range(num_steps):
            t      = step_idx[i].expand(B)
            t_next = (step_idx[i + 1].expand(B) if i < num_steps - 1
                      else torch.full((B,), -1, device=self.device))
            nc  = denoiser(x_t, t, cond, pf_anchor)
            nuc = denoiser(x_t, t, torch.zeros_like(cond), torch.zeros_like(pf_anchor))
            np_ = nuc + guidance_scale * (nc - nuc)
            alpha_t    = self.alphas_cumprod[t].view(B, 1, 1)
            alpha_next = (self.alphas_cumprod[t_next].view(B, 1, 1)
                          if i < num_steps - 1 else torch.ones_like(alpha_t))
            x0_pred = ((x_t - torch.sqrt(1 - alpha_t) * np_)
                       / torch.sqrt(alpha_t).clamp(min=1e-4))
            x0_pred = torch.clamp(x0_pred, -4.0, 4.0)
            x_t     = torch.sqrt(alpha_next) * x0_pred + torch.sqrt(1 - alpha_next) * np_
        return x0_pred


def anchor_normalize(obs_xy, anchor, scale):
    return (obs_xy - anchor) / scale

def anchor_denormalize(norm_xy, anchor, scale):
    return norm_xy * scale + anchor

def hann_window(W, device):
    return torch.hann_window(W, periodic=False, device=device).clamp(min=1e-3)


# ─────────────────────────────────────────────────────────────────────────────
# 权重加载工具（热启动 / 断点续训）
# ─────────────────────────────────────────────────────────────────────────────
def load_pretrain_weights(m_ref, ema, ckpt_path: str, device, is_master: bool):
    """
    从 v6 / v7 / v8 的 ckpt 加载权重，strict=False 以兼容架构差异
    （v6 的 CausalPriorGRU 有 valid_gate，v7/v8 已删除）。
    返回 (start_step, opt_state, sched_state)，若无则返回 (0, None, None)。
    """
    if not os.path.exists(ckpt_path):
        if is_master:
            print(f"  [Warning] 预训练权重不存在: {ckpt_path}，将从零开始训练。")
        return 0, None, None

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # 加载主模型权重
    model_sd = ckpt.get('model', ckpt)
    missing, unexpected = m_ref.load_state_dict(model_sd, strict=False)
    if is_master:
        print(f"  [Pretrain] 加载模型权重: {ckpt_path}")
        if missing:
            print(f"    缺失参数（新增层，随机初始化）: {missing}")
        if unexpected:
            print(f"    多余参数（已忽略，如 valid_gate）: {unexpected}")

    # 加载 EMA 权重（用于 denoiser）
    if 'ema' in ckpt:
        ema_missing, _ = ema.shadow.load_state_dict(ckpt['ema'], strict=False)
        if is_master and ema_missing:
            print(f"    EMA 缺失参数: {ema_missing}")
    else:
        # EMA 没有单独存储时，用主模型 denoiser 权重初始化
        denoiser_sd = {k[len('denoiser.'):]: v
                       for k, v in model_sd.items()
                       if k.startswith('denoiser.')}
        if denoiser_sd:
            ema.shadow.load_state_dict(denoiser_sd, strict=False)

    start_step = ckpt.get('step', 0)
    opt_state  = ckpt.get('opt',   None)
    sched_state = ckpt.get('sched', None)
    return start_step, opt_state, sched_state


# ─────────────────────────────────────────────────────────────────────────────
# Validation（含蒙眼专项评估 + Fix 4 条件泄露修复）
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_validation(model: DualTrackModel, diff: GaussianDiffusion,
                   cfg: TrainConfig, device, step: int) -> Optional[Dict]:
    model.eval()
    metrics = defaultdict(list)
    hann_w  = hann_window(cfg.WINDOW, device)

    for val_idx in range(cfg.VAL_SEQS):
        seq = generate_long_sequence(cfg, step)
        if seq is None:
            continue

        obs_long = torch.from_numpy(
            np.concatenate([seq['dense_obs'],
                            seq['valid'][:, None].astype(np.float32)], axis=1)
        ).float().unsqueeze(0).to(device)
        gt_long  = torch.from_numpy(seq['states']).float().unsqueeze(0).to(device)
        anchor   = obs_long[:, 0:1, :2]
        h_val    = None

        # 蒙眼验证（奇数序列）：强制中段 30 帧缺失，测试推演能力
        obs_eval = obs_long.clone()
        do_blackout = (val_idx % 2 == 1)
        if do_blackout:
            T_seq  = obs_long.shape[1]
            bo_len = min(30, T_seq // 3)
            bo_st  = T_seq // 3
            obs_eval[0, bo_st:bo_st + bo_len, :] = 0.0

        prior_traj_buf = []
        prior_feat_buf = []
        global_pred    = torch.zeros((1, cfg.LONG_T, 2), device=device)
        global_weight  = torch.zeros((1, cfg.LONG_T, 1), device=device)

        for chunk_start in range(0, cfg.LONG_T, cfg.STRIDE):
            chunk_end   = min(chunk_start + cfg.STRIDE, cfg.LONG_T)
            obs_chunk   = obs_eval[:, chunk_start:chunk_end, :]
            valid_chunk = obs_chunk[:, :, 2:3]
            obs_norm_chunk = torch.cat([
                anchor_normalize(obs_chunk[:, :, :2], anchor, cfg.ANCHOR_SCALE)
                * valid_chunk,
                valid_chunk
            ], dim=-1)
            pt_chunk, pf_chunk, h_val = model.prior(obs_norm_chunk, h0=h_val)
            prior_traj_buf.append(pt_chunk)
            prior_feat_buf.append(pf_chunk)

            if sum(t.shape[1] for t in prior_traj_buf) >= cfg.WINDOW:
                w_end   = chunk_end
                w_start = max(0, w_end - cfg.WINDOW)
                if w_end - w_start < cfg.WINDOW:
                    w_end = w_start + cfg.WINDOW

                all_pt       = torch.cat(prior_traj_buf, dim=1)
                prior_w_norm = all_pt[:, -cfg.WINDOW:, :]
                prior_w_abs  = anchor_denormalize(prior_w_norm, anchor, cfg.ANCHOR_SCALE)
                pf_anchor    = torch.cat(prior_feat_buf, dim=1)[:, -1, :]

                obs_w_abs = obs_eval[:, w_start:w_end, :2]
                valid_w   = obs_eval[:, w_start:w_end, 2:3]

                innov     = torch.norm(obs_w_abs - prior_w_abs,
                                       dim=-1, keepdim=True) * valid_w
                valid_sum = valid_w.sum(dim=1, keepdim=True).clamp(min=1e-5)
                scale_w   = torch.clamp(
                    innov.sum(dim=1, keepdim=True) / valid_sum, min=5.0)

                # [Fix 4] 残差在 valid=0 时强制为 0
                obs_resid_norm = (obs_w_abs - prior_w_abs) / scale_w * valid_w
                cond = torch.cat([
                    obs_resid_norm.permute(0, 2, 1),
                    valid_w.permute(0, 2, 1)
                ], dim=1)

                x0_pred      = diff.sample_ddim(model.denoiser, cond, pf_anchor,
                                                num_steps=cfg.INF_STEPS)
                pred_pos_abs = x0_pred[0, :2, :].T * scale_w[0, 0, 0] + prior_w_abs[0]

                global_pred[:, w_start:w_end, :]   += (pred_pos_abs.unsqueeze(0)
                                                        * hann_w.view(1, cfg.WINDOW, 1))
                global_weight[:, w_start:w_end, :] += hann_w.view(1, cfg.WINDOW, 1)

        global_pred = global_pred / global_weight.clamp(min=1e-6)
        gt_real     = gt_long[0, :, :2].cpu().numpy()
        obs_raw     = obs_long[0, :, :2].cpu().numpy()   # 未蒙眼版用于 obs 误差
        den_real    = global_pred[0].cpu().numpy()
        all_pt_full = torch.cat(prior_traj_buf, dim=1)
        prior_real  = anchor_denormalize(all_pt_full, anchor,
                                         cfg.ANCHOR_SCALE)[0].cpu().numpy()

        T_seq = gt_real.shape[0]
        p1, p2 = T_seq // 3, 2 * T_seq // 3

        def rmse(pred, gt):
            return float(np.sqrt(np.mean(np.sum((pred - gt) ** 2, axis=1))))

        metrics['obs_all'].append(rmse(obs_raw, gt_real))
        metrics['pr_all'].append(rmse(prior_real, gt_real))
        metrics['den_all'].append(rmse(den_real, gt_real))
        metrics['obs_early'].append(rmse(obs_raw[:p1],   gt_real[:p1]))
        metrics['pr_early'].append(rmse(prior_real[:p1], gt_real[:p1]))
        metrics['den_early'].append(rmse(den_real[:p1],  gt_real[:p1]))
        metrics['obs_mid'].append(rmse(obs_raw[p1:p2],   gt_real[p1:p2]))
        metrics['pr_mid'].append(rmse(prior_real[p1:p2], gt_real[p1:p2]))
        metrics['den_mid'].append(rmse(den_real[p1:p2],  gt_real[p1:p2]))
        metrics['obs_late'].append(rmse(obs_raw[p2:],   gt_real[p2:]))
        metrics['pr_late'].append(rmse(prior_real[p2:], gt_real[p2:]))
        metrics['den_late'].append(rmse(den_real[p2:],  gt_real[p2:]))

        if do_blackout:
            T_seq  = gt_real.shape[0]
            bo_len = min(30, T_seq // 3)
            bo_st  = T_seq // 3
            metrics['blackout_pr'].append(
                rmse(prior_real[bo_st:bo_st + bo_len], gt_real[bo_st:bo_st + bo_len]))
            metrics['blackout_den'].append(
                rmse(den_real[bo_st:bo_st + bo_len], gt_real[bo_st:bo_st + bo_len]))

    model.train()
    if not metrics['obs_all']:
        return None

    res = {k: float(np.mean(v)) for k, v in metrics.items()}
    res['supp_all']  = (res['obs_all']  - res['den_all'])  / max(res['obs_all'],  1e-6) * 100.0
    res['supp_late'] = (res['obs_late'] - res['den_late']) / max(res['obs_late'], 1e-6) * 100.0
    return res


# ─────────────────────────────────────────────────────────────────────────────
# Training worker
# ─────────────────────────────────────────────────────────────────────────────
def train_worker(rank, world_size, args):
    is_dist   = world_size > 1
    if is_dist:
        dist.init_process_group(backend='nccl', init_method='env://',
                                world_size=world_size, rank=rank)
        torch.cuda.set_device(rank)

    device    = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    is_master = (rank == 0)

    # ── 选择训练配置 ──────────────────────────────────────────────────────
    mode = args.mode
    if args.resume is not None:
        # 断点续训：使用 scratch 全量配置（步数不变，只恢复权重+优化器）
        cfg  = TrainConfig(SAVE_DIR=args.save_dir,
                           LOG_CSV=os.path.join(args.save_dir, 'val_log.csv'))
        mode = 'resume'
    elif args.mode == 'finetune':
        cfg = make_finetune_config(args.save_dir)
    elif args.mode == 'refix_prior':
        cfg = make_refix_prior_config(args.save_dir)
    else:
        cfg = TrainConfig(SAVE_DIR=args.save_dir,
                          LOG_CSV=os.path.join(args.save_dir, 'val_log.csv'))

    os.makedirs(cfg.SAVE_DIR, exist_ok=True)

    if is_master:
        mode_str = {'scratch': '冷启动（从零训练）',
                    'finetune': f'热启动微调（加载 {args.pretrain_ckpt}）',
                    'refix_prior': f'Prior重训（冻结Denoiser，加载 {args.refix_prior_ckpt}）',
                    'resume':  f'断点续训（{args.resume}）'}[mode]
        print(f"\n{'='*65}\n"
              f"  Prior-Denoiser V8 — Blindfold + Curriculum Fine-tuning\n"
              f"  训练模式  : {mode_str}\n"
              f"  Fix1: 缺失置零 | Fix2: 强遮罩 | Fix3: 多段蒙眼 | Fix4: 残差遮罩\n"
              f"  Fix5: 先归一化再遮罩 | 平滑正则化 weight={cfg.SMOOTH_LOSS_WEIGHT}\n"
              f"  TRAIN_STEPS={cfg.TRAIN_STEPS} | STAGE1={cfg.STAGE1_ONLY_STEPS}\n"
              f"  LR_Prior={cfg.LR_PRIOR:.1e} | LR_Den={cfg.LR_DENOISER:.1e}\n"
              f"  Blackout: prob={cfg.BLACKOUT_PROB} n_max={cfg.BLACKOUT_N_MAX} "
              f"{cfg.BLACKOUT_MIN}~{cfg.BLACKOUT_MAX}f  weight={cfg.BLACKOUT_WEIGHT}\n"
              f"{'='*65}\n")

    # ── 模型初始化 ────────────────────────────────────────────────────────
    model = DualTrackModel(cfg).to(device)
    if is_dist:
        model = DDP(model, device_ids=[rank])
    m_ref = model.module if is_dist else model
    ema   = EMA(m_ref.denoiser, decay=cfg.EMA_DECAY)
    diff  = GaussianDiffusion(cfg, device)

    optimizer = torch.optim.AdamW([
        {'params': m_ref.prior.parameters(),    'lr': cfg.LR_PRIOR},
        {'params': m_ref.denoiser.parameters(), 'lr': cfg.LR_DENOISER},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.TRAIN_STEPS, eta_min=1e-5)
    scaler    = torch.amp.GradScaler('cuda')

    # ── 权重加载 ──────────────────────────────────────────────────────────
    start_step = 0
    if mode == 'resume':
        start_step, opt_sd, sched_sd = load_pretrain_weights(
            m_ref, ema, args.resume, device, is_master)
        if opt_sd is not None:
            optimizer.load_state_dict(opt_sd)
        if sched_sd is not None:
            scheduler.load_state_dict(sched_sd)
        if is_master:
            print(f"  [Resume] 从 step {start_step} 继续训练\n")
    elif mode == 'finetune':
        _, _, _ = load_pretrain_weights(
            m_ref, ema, args.pretrain_ckpt, device, is_master)
        # 热启动不恢复 optimizer（新的 LR schedule 从头开始）
        if is_master:
            print(f"  [Finetune] 权重已加载，以新学习率 {cfg.LR_PRIOR:.1e}/{cfg.LR_DENOISER:.1e} 微调\n")
    elif mode == 'refix_prior':
        # 从 v8 ckpt 加载 Denoiser/EMA 权重，冻结 Denoiser，Prior 随机初始化重训
        ckpt_path = args.refix_prior_ckpt
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model_sd = ckpt.get('model', ckpt)
            # 只加载 denoiser 部分权重
            denoiser_sd = {k[len('denoiser.'):]: v
                           for k, v in model_sd.items() if k.startswith('denoiser.')}
            if denoiser_sd:
                m_ref.denoiser.load_state_dict(denoiser_sd, strict=False)
                if is_master:
                    print(f"  [RefixPrior] Denoiser 权重已从 {ckpt_path} 加载并冻结")
            if 'ema' in ckpt:
                ema.shadow.load_state_dict(ckpt['ema'], strict=False)
            # 冻结 Denoiser 参数（梯度不更新）
            for p in m_ref.denoiser.parameters():
                p.requires_grad_(False)
            if is_master:
                print(f"  [RefixPrior] Prior 从随机初始化开始重训，Denoiser 已冻结\n")
        else:
            if is_master:
                print(f"  [Warning] refix_prior_ckpt 不存在: {ckpt_path}，Prior 和 Denoiser 均从零开始\n")

    # ── 日志 ──────────────────────────────────────────────────────────────
    if is_master:
        log_f = open(cfg.LOG_CSV, 'a', newline='')
        log_w = csv.writer(log_f)
        if os.stat(cfg.LOG_CSV).st_size == 0:
            log_w.writerow([
                'step', 'mode', 'prior_loss', 'ddpm_loss', 'lr_p', 'lr_d', 'valid_ratio',
                'obs_all', 'pr_all', 'den_all', 'supp_all',
                'obs_early', 'pr_early', 'den_early',
                'obs_mid',   'pr_mid',   'den_mid',
                'obs_late',  'pr_late',  'den_late', 'supp_late',
                'blackout_pr', 'blackout_den',
            ])
    else:
        log_f = log_w = None

    pbar      = tqdm(total=cfg.TRAIN_STEPS, initial=start_step, disable=not is_master)
    acc       = defaultdict(float)
    acc_count = 0
    model.train()

    for step in range(start_step, cfg.TRAIN_STEPS):
        # ── 1. 生成数据 ──────────────────────────────────────────────────
        raw_seqs = []
        for _ in range(cfg.BATCH_SIZE):
            for _attempt in range(5):
                seq = generate_long_sequence(cfg, step)
                if seq is not None:
                    raw_seqs.append(seq)
                    break
        if not raw_seqs:
            continue
        B = len(raw_seqs)

        # ── 2. 构建 Tensor ───────────────────────────────────────────────
        obs_seqs = torch.stack([
            torch.from_numpy(
                np.concatenate([s['dense_obs'],
                                s['valid'][:, None].astype(np.float32)], axis=1)
            ).float() for s in raw_seqs
        ]).to(device)
        gt_seqs = torch.stack([
            torch.from_numpy(s['states']).float() for s in raw_seqs
        ]).to(device)
        anchor = obs_seqs[:, 0:1, :2]

        with torch.amp.autocast('cuda'):
            # ── [Fix 3] 蒙眼训练（多段蒙眼增广）────────────────────────
            valid_original = obs_seqs[:, :, 2:3].clone()
            valid_train    = valid_original.clone()

            for b in range(B):
                # 每条序列最多叠加 BLACKOUT_N_MAX 段独立蒙眼，每段独立触发
                n_bo = torch.randint(1, cfg.BLACKOUT_N_MAX + 1, (1,)).item()
                for _ in range(n_bo):
                    if torch.rand(1).item() < cfg.BLACKOUT_PROB:
                        bo_len = torch.randint(cfg.BLACKOUT_MIN,
                                               cfg.BLACKOUT_MAX + 1, (1,)).item()
                        bo_st  = torch.randint(0, max(1, cfg.LONG_T - bo_len), (1,)).item()
                        valid_train[b, bo_st:bo_st + bo_len, 0] = 0.0

            # [预处理修正] 先归一化再遮罩，确保 valid=0 时前两维绝对为 0
            # （不再依赖 GRU.forward 的 Fix2 双重遮罩来拯救此处的错误顺序）
            raw_norm    = anchor_normalize(obs_seqs[:, :, :2], anchor, cfg.ANCHOR_SCALE)
            obs_norm_in = torch.cat([
                raw_norm * valid_train,   # valid=0 时前两通道严格为 0
                valid_train
            ], dim=-1)

            # obs_xy_masked：蒙眼后的坐标，供 DDPM 窗口使用
            obs_xy_masked = obs_seqs[:, :, :2] * valid_train

            # ── Track 1: Prior ──────────────────────────────────────────
            prior_traj_norm, prior_feat_full, _ = m_ref.prior(obs_norm_in, h0=None)

            # Loss 权重：有效=1.0，蒙眼段=BLACKOUT_WEIGHT，真缺失=INVALID_WEIGHT
            weight_long = torch.where(
                valid_original > 0.5,
                torch.ones_like(valid_original),
                torch.full_like(valid_original, cfg.INVALID_WEIGHT))
            blackout_mask = (valid_original > 0.5) & (valid_train < 0.5)
            weight_long   = torch.where(
                blackout_mask,
                torch.full_like(weight_long, cfg.BLACKOUT_WEIGHT),
                weight_long)

            gt_norm    = anchor_normalize(gt_seqs[:, :, :2], anchor, cfg.ANCHOR_SCALE)
            prior_loss = (F.smooth_l1_loss(prior_traj_norm, gt_norm, reduction='none')
                          .mean(-1, keepdim=True) * weight_long).mean()

            # [新增] 蒙眼期平滑正则化：GRU 在盲区应输出平滑惯性轨迹，抑制抖动
            # 对 prior 预测的二阶差分（加速度）在蒙眼帧施加最小化压力
            if cfg.SMOOTH_LOSS_WEIGHT > 0 and blackout_mask.any():
                # shape: (B, T-2, 2)
                prior_accel = (prior_traj_norm[:, 2:, :]
                               - 2.0 * prior_traj_norm[:, 1:-1, :]
                               + prior_traj_norm[:, :-2, :])
                smooth_mask = blackout_mask[:, 1:-1, :]   # 对齐维度
                smooth_loss = (prior_accel.abs() * smooth_mask.float()).mean()
                prior_loss  = prior_loss + cfg.SMOOTH_LOSS_WEIGHT * smooth_loss

            # ── Track 2: DDPM ────────────────────────────────────────────
            ddpm_loss_total  = torch.tensor(0.0, device=device)
            valid_ratio_mean = 0.0

            if step > cfg.STAGE1_ONLY_STEPS:
                prior_abs = anchor_denormalize(
                    prior_traj_norm.detach(), anchor, cfg.ANCHOR_SCALE)

                N_WINS   = cfg.SAMPLES_PER_SEQ
                # 一半窗口随机采样，一半窗口定向覆盖蒙眼区（确保 DDPM 见到盲区条件）
                n_random  = N_WINS // 2
                n_targeted = N_WINS - n_random
                w_starts_rand = torch.randint(
                    0, cfg.LONG_T - cfg.WINDOW + 1, (B, n_random), device=device)

                # 定向窗口：从蒙眼起始点附近采样（确保覆盖盲区过渡段）
                bo_starts_list = []
                for b in range(B):
                    # nonzero() → shape (N, 1)，squeeze 后得到 1D 索引
                    bo_idxs = (valid_train[b, :, 0] < 0.5).nonzero(as_tuple=False).squeeze(1)
                    if len(bo_idxs) >= cfg.WINDOW:
                        # 从蒙眼索引中筛出能容纳完整窗口的起点
                        cands = bo_idxs[bo_idxs <= cfg.LONG_T - cfg.WINDOW]
                        if len(cands) > 0:
                            chosen_idx = torch.randint(0, len(cands), (n_targeted,))
                            chosen = cands[chosen_idx]
                            bo_starts_list.append(chosen.to(device))
                        else:
                            bo_starts_list.append(torch.randint(
                                0, cfg.LONG_T - cfg.WINDOW + 1, (n_targeted,), device=device))
                    else:
                        bo_starts_list.append(torch.randint(
                            0, cfg.LONG_T - cfg.WINDOW + 1, (n_targeted,), device=device))
                w_starts_tgt = torch.stack(bo_starts_list).to(device)
                w_starts = torch.cat([w_starts_rand, w_starts_tgt], dim=1)   # (B, N_WINS)

                obs_w_list, gt_w_list, prior_w_list, pf_list = [], [], [], []
                for b in range(B):
                    for i in range(N_WINS):
                        st = w_starts[b, i].item()
                        ed = st + cfg.WINDOW
                        obs_w_list.append(torch.cat([
                            obs_xy_masked[b, st:ed, :],   # 使用蒙眼后的坐标
                            valid_train[b, st:ed, :]
                        ], dim=-1))
                        gt_w_list.append(gt_seqs[b, st:ed, :])
                        pw = prior_abs[b, st:ed, :]
                        if torch.rand(1).item() < 0.3:
                            pw = pw + torch.randn((1, 2), device=device) * 5.0
                        prior_w_list.append(pw)
                        pf_list.append(prior_feat_full[b, ed - 1, :].detach())

                obs_w    = torch.stack(obs_w_list)
                gt_w     = torch.stack(gt_w_list)
                prior_w  = torch.stack(prior_w_list)
                pf_anchor = torch.stack(pf_list)

                obs_w_abs = obs_w[..., :2]
                valid_w   = obs_w[..., 2:3]

                innov     = torch.norm(obs_w_abs - prior_w,
                                       dim=-1, keepdim=True) * valid_w
                valid_sum = valid_w.sum(dim=1, keepdim=True).clamp(min=1e-5)
                scale_w   = torch.clamp(
                    innov.sum(dim=1, keepdim=True) / valid_sum, min=5.0)

                # [Fix 4] 残差在 valid=0 时强制为 0
                obs_resid_norm = (obs_w_abs - prior_w) / scale_w * valid_w
                gt_resid_norm  = (gt_w[..., :2] - prior_w) / scale_w
                gt_vel_norm    = gt_w[..., 2:] / cfg.VEL_SCALE

                x0   = torch.cat([gt_resid_norm, gt_vel_norm],
                                  dim=-1).permute(0, 2, 1)
                cond = torch.cat([obs_resid_norm, valid_w],
                                  dim=-1).permute(0, 2, 1)

                ddpm_loss_total  = diff.compute_loss(m_ref.denoiser, x0, cond, pf_anchor)
                valid_ratio_mean = valid_w.mean().item()

        total_loss = prior_loss + ddpm_loss_total
        optimizer.zero_grad()
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        ema.update(m_ref.denoiser)
        scheduler.step()

        acc['p']  += float(prior_loss)
        acc['d']  += float(ddpm_loss_total)
        acc['vr'] += valid_ratio_mean
        acc_count += 1

        if is_master:
            pbar.update(1)
            stage = "S1" if step <= cfg.STAGE1_ONLY_STEPS else "S2"
            pbar.set_description(
                f"[{mode}/{stage}] p={acc['p']/acc_count:.4f} "
                f"d={acc['d']/acc_count:.4f}")

            if (step + 1) % cfg.VAL_INTERVAL == 0:
                ap  = acc['p']  / acc_count
                ad  = acc['d']  / acc_count
                avr = acc['vr'] / acc_count
                lr_p = optimizer.param_groups[0]['lr']
                lr_d = optimizer.param_groups[1]['lr']

                val_r = run_validation(m_ref, diff, cfg, device, step)

                if val_r is not None:
                    bo_pr  = val_r.get('blackout_pr',  float('nan'))
                    bo_den = val_r.get('blackout_den', float('nan'))
                    tqdm.write(
                        f"\n  ┌─ Step {step+1:>6} [{mode}] | lr_p={lr_p:.2e} lr_d={lr_d:.2e}\n"
                        f"  │[Loss]   Prior={ap:.5f} | DDPM={ad:.4f} | Valid={avr*100:.1f}%\n"
                        f"  ├─ [ALL]     Obs={val_r['obs_all']:.2f}m | Prior={val_r['pr_all']:.2f}m | Den={val_r['den_all']:.2f}m | Supp={val_r['supp_all']:.1f}%\n"
                        f"  ├─ [Early]   Obs={val_r['obs_early']:.2f}m | Prior={val_r['pr_early']:.2f}m | Den={val_r['den_early']:.2f}m\n"
                        f"  ├─ [Mid]     Obs={val_r['obs_mid']:.2f}m | Prior={val_r['pr_mid']:.2f}m | Den={val_r['den_mid']:.2f}m\n"
                        f"  ├─ [Late]    Obs={val_r['obs_late']:.2f}m | Prior={val_r['pr_late']:.2f}m | Den={val_r['den_late']:.2f}m | Supp={val_r['supp_late']:.1f}%\n"
                        f"  └─ [Blackout] Prior={bo_pr:.2f}m | Den={bo_den:.2f}m\n"
                    )
                    log_w.writerow([
                        step + 1, mode, ap, ad, lr_p, lr_d, avr,
                        val_r['obs_all'],   val_r['pr_all'],   val_r['den_all'],   val_r['supp_all'],
                        val_r['obs_early'], val_r['pr_early'], val_r['den_early'],
                        val_r['obs_mid'],   val_r['pr_mid'],   val_r['den_mid'],
                        val_r['obs_late'],  val_r['pr_late'],  val_r['den_late'],  val_r['supp_late'],
                        bo_pr, bo_den,
                    ])
                    log_f.flush()

                torch.save(
                    dict(step=step + 1, mode=mode,
                         model=m_ref.state_dict(),
                         ema=ema.shadow.state_dict(),
                         opt=optimizer.state_dict(),
                         sched=scheduler.state_dict()),
                    os.path.join(cfg.SAVE_DIR, f'ckpt_step{step+1}.pth'))
                acc       = defaultdict(float)
                acc_count = 0

    if is_master:
        torch.save(
            dict(step=cfg.TRAIN_STEPS, mode=mode,
                 model=m_ref.state_dict(),
                 ema=ema.shadow.state_dict()),
            os.path.join(cfg.SAVE_DIR, 'final.pth'))
        log_f.close()
        print(f"\n[Done] 训练完成，模型已保存至 {cfg.SAVE_DIR}/final.pth\n")

    if is_dist:
        dist.destroy_process_group()


if __name__ == '__main__':
    args       = parse_args()
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    train_worker(int(os.environ.get('RANK', 0)), world_size, args)
