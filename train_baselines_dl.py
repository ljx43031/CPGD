"""
train_baselines_dl.py
=====================
Trains the learning-based baselines (NB-DLSTM, Transformer, CSDI diffusion-
imputation) on the *identical* data distribution, blackout augmentation and
anchor normalisation as CPGD, so the result tables can report a fair
learned-vs-learned comparison (addresses Reviewer 3 / Reviewer 5 / AE: "no
learning-based baselines; the SOTA claim is not tested against any learned method").

Data generator, curriculum and blackout augmentation are imported directly from
train_prior_denoiser_v8 -- there is a single source of truth for the training
distribution, guaranteeing the baselines see exactly what CPGD saw.

Usage
-----
  # NB-DLSTM, faithful Gao 2019 (represents refs [9],[10])
  python train_baselines_dl.py --model dlstm       --steps 60000

  # Transformer denoiser, task-adapted Giuliari 2020 (ref [11])
  python train_baselines_dl.py --model transformer --steps 60000

  # CSDI diffusion imputation (represents ref [16])
  python train_baselines_dl.py --model csdi        --steps 60000

Checkpoints -> ./models_baselines_dl/<model>_final.pth
"""

import env_config  # noqa
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from train_prior_denoiser_v8 import (
    TrainConfig, generate_long_sequence, anchor_normalize)
import baselines_dl as B

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_DIR = './models_baselines_dl'


# ─────────────────────────────────────────────────────────────────────────────
# Blackout augmentation — mirrors train_prior_denoiser_v8 exactly
# ─────────────────────────────────────────────────────────────────────────────
def apply_blackout_aug(valid, cfg):
    """valid: (B,T,1) tensor. Returns valid_train with synthetic blackouts,
    matching CPGD's multi-segment augmentation (prob/n_max/min/max)."""
    valid_train = valid.clone()
    B_, T, _ = valid.shape
    for b in range(B_):
        n_bo = torch.randint(1, cfg.BLACKOUT_N_MAX + 1, (1,)).item()
        for _ in range(n_bo):
            if torch.rand(1).item() < cfg.BLACKOUT_PROB:
                bo_len = torch.randint(cfg.BLACKOUT_MIN, cfg.BLACKOUT_MAX + 1, (1,)).item()
                bo_st  = torch.randint(0, max(1, T - bo_len), (1,)).item()
                valid_train[b, bo_st:bo_st + bo_len, 0] = 0.0
    return valid_train


def build_batch(cfg, step):
    """Generate a batch of long sequences (same generator as CPGD)."""
    seqs = []
    for _ in range(cfg.BATCH_SIZE):
        for _attempt in range(5):
            s = generate_long_sequence(cfg, step)
            if s is not None:
                seqs.append(s)
                break
    if not seqs:
        return None
    obs = torch.stack([
        torch.from_numpy(np.concatenate(
            [s['dense_obs'], s['valid'][:, None].astype(np.float32)], axis=1)).float()
        for s in seqs]).to(DEVICE)                                  # (B,T,3)
    gt = torch.stack([torch.from_numpy(s['states']).float() for s in seqs]).to(DEVICE)  # (B,T,4)
    return obs, gt


# ─────────────────────────────────────────────────────────────────────────────
# CSDI training (conditional diffusion imputation, observation space)
# ─────────────────────────────────────────────────────────────────────────────
def train_csdi(cfg, steps):
    model = B.CSDIDenoiser().to(DEVICE)
    diff = B.CSDIDiffusion(num_steps=50, device=DEVICE)   # CSDI: quadratic schedule, T=50
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-5)
    model.train()
    W = cfg.WINDOW
    pbar = tqdm(range(steps), desc='csdi')
    run = 0.0
    for step in pbar:
        batch = build_batch(cfg, step)
        if batch is None:
            continue
        obs, gt = batch
        Bn = obs.shape[0]
        anchor = obs[:, 0:1, :2]
        valid_orig = obs[:, :, 2:3]
        valid_train = apply_blackout_aug(valid_orig, cfg)

        # sample W-length windows (half random, half covering blackout regions)
        wins_obs, wins_gt, wins_cmask = [], [], []
        for b in range(Bn):
            for _ in range(cfg.SAMPLES_PER_SEQ):
                st = torch.randint(0, cfg.LONG_T - W + 1, (1,)).item()
                ed = st + W
                o_xy = anchor_normalize(obs[b, st:ed, :2], anchor[b], cfg.ANCHOR_SCALE)
                g_xy = anchor_normalize(gt[b, st:ed, :2], anchor[b], cfg.ANCHOR_SCALE)
                cmask = valid_train[b, st:ed, 0]                     # (W,) conditional mask
                wins_obs.append(o_xy.T)                             # (2,W)
                wins_gt.append(g_xy.T)
                wins_cmask.append(cmask)
        x0        = torch.stack(wins_gt)                            # (M,2,W)  clean pos
        cond_obs  = torch.stack(wins_obs) * torch.stack(wins_cmask)[:, None, :]
        cond_mask = torch.stack(wins_cmask)[:, None, :].expand(-1, 2, -1).contiguous()
        # target = points NOT in conditional set (blackouts + random held-out observed)
        target_mask = 1.0 - cond_mask
        # additionally hold out ~15% of conditioned points as CSDI self-supervision
        extra = (torch.rand_like(cond_mask) < 0.15) * cond_mask
        target_mask = torch.clamp(target_mask + extra, 0, 1)
        cond_mask = cond_mask * (1 - extra)
        cond_obs = cond_obs * cond_mask

        loss = diff.compute_loss(model, x0, cond_obs, cond_mask, target_mask)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        run = 0.98 * run + 0.02 * float(loss) if step else float(loss)
        if step % 100 == 0:
            pbar.set_description(f'csdi loss={run:.4f}')
        if (step + 1) % 5000 == 0:
            torch.save({'model': model.state_dict(), 'kind': 'csdi', 'step': step + 1},
                       os.path.join(SAVE_DIR, 'csdi_final.pth'))
    torch.save({'model': model.state_dict(), 'kind': 'csdi', 'step': steps},
               os.path.join(SAVE_DIR, 'csdi_final.pth'))
    print(f'[Done] csdi -> {SAVE_DIR}/csdi_final.pth')


# ─────────────────────────────────────────────────────────────────────────────
# Faithful NB-DLSTM training (Gao 2019) — NO pipeline tricks
# ─────────────────────────────────────────────────────────────────────────────
# Faithful to the paper: raw measurement stream in (zero-order-hold across gaps,
# no validity channel), direct state regression, plain MSE, NO blackout
# augmentation. Trained on the same trajectory/measurement distribution as CPGD.
def train_dlstm(cfg, steps):
    model = B.FaithfulDLSTM().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-5)
    model.train()
    pbar = tqdm(range(steps), desc='dlstm')
    run = 0.0
    for step in pbar:
        batch = build_batch(cfg, step)
        if batch is None:
            continue
        obs, gt = batch
        B_, T, _ = obs.shape
        valid = obs[:, :, 2:3]
        # zero-order hold across naturally-missing frames (measurement stream)
        pos = obs[:, :, :2].clone()
        for b in range(B_):
            last = None
            for i in range(T):
                if valid[b, i, 0] > 0.5:
                    last = pos[b, i].clone()
                elif last is not None:
                    pos[b, i] = last
        shift = pos[:, 0:1, :]                             # plain global de-mean
        x = (pos - shift) / cfg.ANCHOR_SCALE
        st, _ = model(x)
        gt_pos = (gt[:, :, :2] - shift) / cfg.ANCHOR_SCALE
        gt_vel = gt[:, :, 2:] / cfg.VEL_SCALE
        # plain MSE to true state (paper's objective), no per-frame weighting
        loss = (F.mse_loss(st[:, :, :2], gt_pos)
                + 0.5 * F.mse_loss(st[:, :, 2:], gt_vel))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        run = 0.98 * run + 0.02 * float(loss) if step else float(loss)
        if step % 100 == 0:
            pbar.set_description(f'dlstm loss={run:.4f}')
        if (step + 1) % 5000 == 0:
            torch.save({'model': model.state_dict(), 'kind': 'dlstm', 'step': step + 1},
                       os.path.join(SAVE_DIR, 'dlstm_final.pth'))
    torch.save({'model': model.state_dict(), 'kind': 'dlstm', 'step': steps},
               os.path.join(SAVE_DIR, 'dlstm_final.pth'))
    print(f'[Done] dlstm -> {SAVE_DIR}/dlstm_final.pth')


# ─────────────────────────────────────────────────────────────────────────────
# Transformer denoiser training (Giuliari 2020, adapted to denoising)
# ─────────────────────────────────────────────────────────────────────────────
# Faithful architecture, task adapted to denoising. Trained over W-length windows
# with the same blackout augmentation & anchor normalisation as CPGD (the task
# adaptation legitimately uses the validity channel).
def train_transformer(cfg, steps):
    model = B.TransformerDenoiser().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-5)
    model.train()
    W = cfg.WINDOW
    pbar = tqdm(range(steps), desc='transformer')
    run = 0.0
    for step in pbar:
        batch = build_batch(cfg, step)
        if batch is None:
            continue
        obs, gt = batch
        Bn = obs.shape[0]
        anchor = obs[:, 0:1, :2]
        valid_orig = obs[:, :, 2:3]
        valid_train = apply_blackout_aug(valid_orig, cfg)

        # sample W-length windows per sequence
        wins_in, wins_gp, wins_gv, wins_w = [], [], [], []
        for b in range(Bn):
            for _ in range(cfg.SAMPLES_PER_SEQ):
                stp = torch.randint(0, cfg.LONG_T - W + 1, (1,)).item()
                ed = stp + W
                rn = anchor_normalize(obs[b, stp:ed, :2], anchor[b], cfg.ANCHOR_SCALE)
                vt = valid_train[b, stp:ed, :]
                wins_in.append(torch.cat([rn * vt, vt], dim=-1))
                wins_gp.append(anchor_normalize(gt[b, stp:ed, :2], anchor[b], cfg.ANCHOR_SCALE))
                wins_gv.append(gt[b, stp:ed, 2:] / cfg.VEL_SCALE)
                vo = valid_orig[b, stp:ed, :]
                wv = torch.where(vo > 0.5, torch.ones_like(vo),
                                 torch.full_like(vo, cfg.INVALID_WEIGHT))
                bo = (vo > 0.5) & (vt < 0.5)
                wv = torch.where(bo, torch.full_like(wv, cfg.BLACKOUT_WEIGHT), wv)
                wins_w.append(wv)
        x = torch.stack(wins_in)                          # (M,W,3)
        st = model(x)                                     # (M,W,4)
        gp = torch.stack(wins_gp); gv = torch.stack(wins_gv); wt = torch.stack(wins_w)
        pos_loss = (F.smooth_l1_loss(st[:, :, :2], gp, reduction='none')
                    .mean(-1, keepdim=True) * wt).mean()
        vel_loss = (F.smooth_l1_loss(st[:, :, 2:], gv, reduction='none')
                    .mean(-1, keepdim=True) * wt).mean()
        loss = pos_loss + 0.5 * vel_loss
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        run = 0.98 * run + 0.02 * float(loss) if step else float(loss)
        if step % 100 == 0:
            pbar.set_description(f'transformer loss={run:.4f}')
        if (step + 1) % 5000 == 0:
            torch.save({'model': model.state_dict(), 'kind': 'transformer', 'step': step + 1},
                       os.path.join(SAVE_DIR, 'transformer_final.pth'))
    torch.save({'model': model.state_dict(), 'kind': 'transformer', 'step': steps},
               os.path.join(SAVE_DIR, 'transformer_final.pth'))
    print(f'[Done] transformer -> {SAVE_DIR}/transformer_final.pth')


def main():
    global SAVE_DIR   # functions read this module global for checkpoint paths
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=['csdi', 'dlstm', 'transformer'],
                    required=True)
    ap.add_argument('--steps', type=int, default=60000)
    ap.add_argument('--save_dir', type=str, default=SAVE_DIR)
    args = ap.parse_args()

    SAVE_DIR = args.save_dir
    os.makedirs(SAVE_DIR, exist_ok=True)

    cfg = TrainConfig()   # cold-start config = same distribution CPGD trained on
    print(f'Training {args.model} for {args.steps} steps on {DEVICE}')
    print(f'  Data: LONG_T={cfg.LONG_T}, blackout prob={cfg.BLACKOUT_PROB} '
          f'{cfg.BLACKOUT_MIN}-{cfg.BLACKOUT_MAX}f, curriculum={len(cfg.CURRICULUM)} stages')

    if args.model == 'csdi':
        train_csdi(cfg, args.steps)
    elif args.model == 'dlstm':
        train_dlstm(cfg, args.steps)
    elif args.model == 'transformer':
        train_transformer(cfg, args.steps)


if __name__ == '__main__':
    main()
