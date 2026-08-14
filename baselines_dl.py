"""
baselines_dl.py
===============

Three learning-based baselines are provided:

  1. FaithfulDLSTM  -- NB-DLSTM per Gao et al. 2019 (manuscript refs 9,10)
       Faithful to that paper's OWN architecture: stacked LSTM (hidden=128), raw
       measurement input, direct state regression, plain MSE. The literature
       recurrent baseline.

  2. TransformerDenoiser  -- per Giuliari et al. 2020 (manuscript ref 11)
       Faithful to Giuliari's Transformer architecture (linear coord embedding +
       sinusoidal PE + multi-head Transformer encoder), ADAPTED from their
       forecasting task to this paper's denoising task (encoder-only, noisy
       window -> clean state over the same timesteps). Task adaptation is stated
       explicitly (their original task is past->future forecasting).

  3. CSDIDenoiser  -- conditional diffusion imputation, Tashiro et al. 2021 (ref 16)
       Faithful CSDI: 2-channel [cond_obs, noisy_target] input, temporal+feature
       side-info embeddings, temporal & feature transformer residual blocks,
       quadratic noise schedule with T=50. Represents the diffusion-imputation
       family (SSSD ref 17 and PriSTI ref 18 both build directly on CSDI). It has
       NO kinematic prior and NO dynamic residual space -- isolating what CPGD's
       design adds over vanilla diffusion imputation.

All models share CPGD's normalisation constants so checkpoints and .mat outputs
line up with mc_compare_v8.py.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Normalisation constants (must match train_prior_denoiser_v8 / mc_compare) ──
ANCHOR_SCALE = 1000.0
VEL_SCALE    = 50.0


def anchor_normalize(xy, anchor, scale=ANCHOR_SCALE):
    return (xy - anchor) / scale


def anchor_denormalize(norm_xy, anchor, scale=ANCHOR_SCALE):
    return norm_xy * scale + anchor


# ─────────────────────────────────────────────────────────────────────────────
# Baseline: CSDI — conditional diffusion imputation (Tashiro et al., ref [16])
# ─────────────────────────────────────────────────────────────────────────────
# Faithful compact re-implementation of CSDI's core idea:
#   * conditional DDPM whose target is the clean signal x0 (here anchor-norm pos)
#   * conditioning = observed values on the *conditional mask* + the mask itself
#   * transformer backbone alternating a temporal and a feature attention layer
#   * self-supervised training: a random subset of observed points is held out
#     as imputation targets; blackout frames are natural imputation targets.
# Operates in raw (anchor-normalised) OBSERVATION space -- NO GRU prior, NO
# dynamic residual space -- isolating what CPGD's inductive biases add.

def _sinusoidal_embed(vals, dim, device, max_period=10000.0):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) *
                      torch.arange(half, device=device).float() / max(half - 1, 1))
    args = vals.float()[:, None] * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class _CSDIResidualLayer(nn.Module):
    """One CSDI residual block: temporal self-attn + feature self-attn, with the
    diffusion-step embedding added before attention and the side-info (time +
    feature embeddings) projected and added after attention -- as in Tashiro
    et al. 2021 (their DiffWave-derived block)."""

    def __init__(self, channels, side_dim, nheads=8):
        super().__init__()
        self.time_attn = nn.TransformerEncoderLayer(
            channels, nheads, dim_feedforward=channels * 2,
            batch_first=True, dropout=0.0, activation='gelu')
        self.feat_attn = nn.TransformerEncoderLayer(
            channels, nheads, dim_feedforward=channels * 2,
            batch_first=True, dropout=0.0, activation='gelu')
        self.diff_proj = nn.Linear(channels, channels)
        self.side_proj = nn.Conv1d(side_dim, 2 * channels, 1)   # side info -> gate/filter bias
        self.mid_proj  = nn.Conv1d(channels, 2 * channels, 1)
        self.out_proj  = nn.Conv1d(channels, 2 * channels, 1)

    def forward(self, x, diff_emb, side_info):
        # x: (B, C, K, L)  K=feature dim, L=time steps;  side_info: (B, side_dim, K, L)
        B, C, K, L = x.shape
        h = x + self.diff_proj(diff_emb)[:, :, None, None]
        # temporal attention: tokens along L, batch over B*K
        ht = h.permute(0, 2, 3, 1).reshape(B * K, L, C)
        ht = self.time_attn(ht).reshape(B, K, L, C).permute(0, 3, 1, 2)
        # feature attention: tokens along K, batch over B*L
        hf = ht.permute(0, 3, 2, 1).reshape(B * L, K, C)
        hf = self.feat_attn(hf).reshape(B, L, K, C).permute(0, 3, 2, 1)
        y = self.mid_proj(hf.reshape(B, C, K * L)).reshape(B, 2 * C, K, L)
        # inject side info (time + feature embeddings) as in CSDI
        y = y + self.side_proj(side_info.reshape(B, -1, K * L)).reshape(B, 2 * C, K, L)
        gate, filt = y.chunk(2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filt)
        y = self.out_proj(y.reshape(B, C, K * L)).reshape(B, 2 * C, K, L)
        res, skip = y.chunk(2, dim=1)
        return (x + res) / math.sqrt(2.0), skip


class CSDIDenoiser(nn.Module):
    """
    CSDI epsilon-predictor (faithful to Tashiro et al. 2021).
    Signal has K=2 feature channels (x,y position). Per (feature,time) cell the
    network input is the 2 channels [cond_obs, noisy_target]; the conditional
    mask is carried through the SIDE INFO together with a 128-dim temporal
    (position) embedding and a 16-dim feature embedding -- exactly CSDI's design.
    """

    TIME_EMB = 128
    FEAT_EMB = 16

    def __init__(self, target_dim=2, channels=64, layers=4, nheads=8, diff_dim=128):
        super().__init__()
        self.target_dim = target_dim
        self.channels   = channels
        self.diff_dim   = diff_dim
        self.side_dim   = self.TIME_EMB + self.FEAT_EMB + 1   # time + feature + cond_mask
        self.feat_embed = nn.Embedding(target_dim, self.FEAT_EMB)
        self.diff_mlp = nn.Sequential(
            nn.Linear(diff_dim, channels), nn.SiLU(), nn.Linear(channels, channels))
        # input features per cell: [cond_obs, noisy_target] (CSDI: 2-channel input)
        self.in_proj  = nn.Conv1d(2, channels, 1)
        self.layers   = nn.ModuleList(
            [_CSDIResidualLayer(channels, self.side_dim, nheads) for _ in range(layers)])
        self.out_proj = nn.Sequential(
            nn.Conv1d(channels, channels, 1), nn.SiLU(),
            nn.Conv1d(channels, 1, 1))

    def _side_info(self, cond_mask):
        # cond_mask: (B,K,L) -> side info (B, side_dim, K, L)
        B, K, L = cond_mask.shape
        dev = cond_mask.device
        # temporal position embedding along L, broadcast over K
        pos = torch.arange(L, device=dev).float()
        temb = _sinusoidal_embed(pos, self.TIME_EMB, dev)             # (L, TIME_EMB)
        temb = temb.permute(1, 0)[None, :, None, :].expand(B, -1, K, L)
        # feature embedding along K, broadcast over L
        fidx = torch.arange(K, device=dev)
        femb = self.feat_embed(fidx)                                 # (K, FEAT_EMB)
        femb = femb.permute(1, 0)[None, :, :, None].expand(B, -1, K, L)
        return torch.cat([temb, femb, cond_mask[:, None, :, :]], dim=1)

    def forward(self, x_t, cond_obs, cond_mask, diff_step):
        # x_t, cond_obs, cond_mask: (B, K, L);  diff_step: (B,)
        B, K, L = x_t.shape
        inp = torch.stack([cond_obs, x_t], dim=1)                # (B,2,K,L)
        inp = inp.reshape(B, 2, K * L)
        h = self.in_proj(inp).reshape(B, self.channels, K, L)
        de = self.diff_mlp(_sinusoidal_embed(diff_step, self.diff_dim, x_t.device))
        side = self._side_info(cond_mask)                        # (B, side_dim, K, L)
        skip_sum = 0.0
        for layer in self.layers:
            h, skip = layer(h, de, side)
            skip_sum = skip_sum + skip
        h = (skip_sum / math.sqrt(len(self.layers))).reshape(B, self.channels, K * L)
        out = self.out_proj(h).reshape(B, K, L)
        return out                                               # predicted noise, (B,K,L)


class CSDIDiffusion:
    """DDPM wrapper for CSDI. Uses CSDI's QUADRATIC noise schedule with T=50
    (beta = linspace(sqrt(b0), sqrt(bT), T)^2, b0=1e-4, bT=0.5), per Tashiro
    et al. 2021 (their stated setup and official implementation)."""

    def __init__(self, num_steps=50, device='cpu', beta0=1e-4, betaT=0.5):
        self.T = num_steps
        self.device = device
        betas = torch.linspace(beta0 ** 0.5, betaT ** 0.5, num_steps,
                               device=device) ** 2               # quadratic schedule
        self.alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
        self.sqrt_ac    = torch.sqrt(self.alphas_cumprod)
        self.sqrt_1m_ac = torch.sqrt(1.0 - self.alphas_cumprod)

    def compute_loss(self, model, x0, cond_obs, cond_mask, target_mask):
        # x0, cond_obs, cond_mask, target_mask: (B,K,L)
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=self.device)
        noise = torch.randn_like(x0)
        x_t = (self.sqrt_ac[t].view(B, 1, 1) * x0
               + self.sqrt_1m_ac[t].view(B, 1, 1) * noise)
        # conditional points keep their clean value; only target points are noised
        x_in = x_t * target_mask + x0 * (1 - target_mask)
        pred = model(x_in, cond_obs, cond_mask, t)
        # loss only on imputation targets (CSDI convention)
        num = (((pred - noise) ** 2) * target_mask).sum()
        den = target_mask.sum().clamp(min=1.0)
        return num / den

    @torch.no_grad()
    def sample(self, model, cond_obs, cond_mask, num_steps=50):
        B, K, L = cond_obs.shape
        x_t = torch.randn((B, K, L), device=self.device)
        step_idx = torch.linspace(self.T - 1, 0, num_steps,
                                  dtype=torch.long, device=self.device)
        for i in range(num_steps):
            t = step_idx[i].expand(B)
            t_next = (step_idx[i + 1].expand(B) if i < num_steps - 1
                      else torch.full((B,), -1, device=self.device))
            x_in = x_t * (1 - cond_mask) + cond_obs * cond_mask
            eps = model(x_in, cond_obs, cond_mask, t)
            a_t = self.alphas_cumprod[t].view(B, 1, 1)
            a_n = (self.alphas_cumprod[t_next].view(B, 1, 1)
                   if i < num_steps - 1 else torch.ones_like(a_t))
            x0_pred = ((x_t - torch.sqrt(1 - a_t) * eps)
                       / torch.sqrt(a_t).clamp(min=1e-4))
            x0_pred = torch.clamp(x0_pred, -4.0, 4.0)
            x_t = torch.sqrt(a_n) * x0_pred + torch.sqrt(1 - a_n) * eps
        return x0_pred * (1 - cond_mask) + cond_obs * cond_mask


@torch.no_grad()
def run_csdi_inference(model, diff, dense_obs, valid_mask, device,
                       window=32, stride=16, num_steps=50):
    """
    Sliding-window CSDI imputation over a full sequence, Hann-overlap-add
    assembled (same assembly as CPGD's smooth mode, for a fair head-to-head).
    Returns (N,4): imputed [x,y] with zero velocity columns (CSDI imputes
    position only; velocity is not part of its native output).
    """
    N = len(dense_obs)
    obs = dense_obs.copy()
    obs[~valid_mask] = 0.0
    fv = int(np.argmax(valid_mask)) if valid_mask.any() else 0
    anchor = torch.from_numpy(obs[fv:fv + 1]).float().to(device).view(1, 2, 1)

    hann = torch.hann_window(window, periodic=False, device=device).clamp(min=1e-3)
    gp = torch.zeros((2, N), device=device)
    gw = torch.zeros((1, N), device=device)

    obs_t   = torch.from_numpy(obs).float().to(device).T            # (2,N)
    valid_t = torch.from_numpy(valid_mask.astype(np.float32)).to(device)  # (N,)
    obs_norm = (obs_t - anchor[0]) / ANCHOR_SCALE * valid_t[None]    # (2,N)

    for ws in range(0, max(1, N - window + 1), stride):
        we = min(ws + window, N)
        if we - ws < window:
            ws = we - window
        cond_obs  = obs_norm[:, ws:we].unsqueeze(0)                  # (1,2,W)
        cond_mask = valid_t[ws:we].view(1, 1, -1).expand(1, 2, -1).contiguous()
        x0 = diff.sample(model, cond_obs, cond_mask, num_steps=num_steps)  # (1,2,W)
        gp[:, ws:we] += x0[0] * hann.view(1, -1)
        gw[:, ws:we] += hann.view(1, -1)

    imp_norm = (gp / gw.clamp(min=1e-6))
    imp_abs = (imp_norm * ANCHOR_SCALE + anchor[0]).T.cpu().numpy()  # (N,2)
    return np.column_stack([imp_abs, np.zeros((N, 2), dtype=np.float32)])


# ─────────────────────────────────────────────────────────────────────────────
# Baseline: Faithful NB-DLSTM (Gao et al. 2019, Info. Sci., ref [9]/[10])
# ─────────────────────────────────────────────────────────────────────────────
# Re-implementation faithful to the paper's own architecture: stacked LSTM
# (hidden=128), raw measurement input, direct state regression, plain MSE.
#
# Gao 2019 specifics reproduced here:
#   * NB-DLSTM (non-Bayesian, end-to-end) variant -- the paper's recommended one.
#   * Stacked (deep) LSTM, hidden = 128 units.
#   * Input = raw measurement sequence only (NO validity flag, NO anchor
#     normalisation beyond a plain global scaling, NO blackout augmentation).
#   * Output = estimated state; trained with plain MSE to the true state.
# We keep a light global de-meaning (subtract first observation) purely for
# numerical conditioning -- the paper trains in a bounded coordinate range; this
# is the minimal equivalent and is applied identically to all methods' raw obs.
class FaithfulDLSTM(nn.Module):
    """NB-DLSTM per Gao et al. 2019: stacked LSTM (hidden=128), raw-measurement
    input, direct state regression. Faithful literature baseline (no CPGD
    pipeline tricks)."""

    def __init__(self, in_dim=2, hidden=128, layers=4, out_dim=4):
        super().__init__()
        # paper feeds raw measurements straight into the stacked LSTM
        self.lstm = nn.LSTM(in_dim, hidden, layers, batch_first=True)
        self.head = nn.Linear(hidden, out_dim)   # decode state [x,y,vx,vy] from hidden

    def forward(self, obs_seq, h0=None):
        # obs_seq: (B, T, 2) raw (globally-shifted) position measurements
        out, h = self.lstm(obs_seq, h0)
        return self.head(out), h                 # (B,T,4)


@torch.no_grad()
def run_dlstm_inference(model, dense_obs, valid_mask, device):
    """Run faithful NB-DLSTM. It has NO validity channel, so blackout frames are
    filled by zero-order hold (last valid observation) -- the standard way a
    plain measurement-driven tracker sees a gap. Returns (N,4)."""
    N = len(dense_obs)
    obs = dense_obs.copy()
    # zero-order hold across gaps (the paper's LSTM ingests a measurement stream)
    fv = int(np.argmax(valid_mask)) if valid_mask.any() else 0
    last = obs[fv].copy()
    filled = obs.copy()
    for i in range(N):
        if valid_mask[i]:
            last = obs[i].copy()
        else:
            filled[i] = last
    shift = filled[fv:fv + 1].copy()             # plain global de-mean for conditioning
    x = torch.from_numpy((filled - shift) / ANCHOR_SCALE).float().unsqueeze(0).to(device)
    st, _ = model(x)
    st = st[0].cpu().numpy()
    pos = st[:, :2] * ANCHOR_SCALE + shift
    vel = st[:, 2:] * VEL_SCALE
    return np.column_stack([pos, vel])


# ─────────────────────────────────────────────────────────────────────────────
# Baseline 4: Transformer (Giuliari et al. 2020, ICPR, ref [11])
# ─────────────────────────────────────────────────────────────────────────────
# Faithful to Giuliari's Transformer architecture (linear coordinate embedding +
# sinusoidal positional encoding + standard multi-head Transformer encoder,
# d_model / heads / FFN as in the original Transformer), but ADAPTED to this
# paper's task. Giuliari's original task is trajectory FORECASTING (clean past ->
# future); ours is observation DENOISING (noisy current window -> clean current
# state). We therefore use an ENCODER-ONLY, sequence-to-sequence denoiser: the
# noisy observation window (with a validity channel) is mapped to the clean
# state sequence over the SAME timesteps. This task adaptation is stated
# explicitly so the baseline is not a mislabelled forecaster.
#
# Kept faithful: linear R^d embedding of the input, additive sinusoidal PE,
# TransformerEncoder with multi-head self-attention, GELU FFN, dropout 0.1.
# Scaled down d_model=128 (from 512) to match the other baselines' capacity on
# this lower-dimensional (2-D) problem; noted as a capacity choice.
class TransformerDenoiser(nn.Module):
    def __init__(self, in_dim=3, d_model=128, nhead=8, layers=6,
                 dim_ff=512, out_dim=4, dropout=0.1, max_len=4096):
        super().__init__()
        self.in_embed = nn.Linear(in_dim, d_model)              # linear coord embedding
        self.register_buffer('pe', self._build_pe(max_len, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=dim_ff, dropout=dropout,
            batch_first=True, activation='gelu')
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.head = nn.Linear(d_model, out_dim)                 # decode clean state

    @staticmethod
    def _build_pe(max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).float()[:, None]
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe[None]                                         # (1, max_len, d_model)

    def forward(self, obs_norm_seq, causal=False):
        # obs_norm_seq: (B, T, 3) cols [norm_x*valid, norm_y*valid, valid]
        valid = obs_norm_seq[:, :, 2:3]
        x = obs_norm_seq.clone()
        x[:, :, :2] = x[:, :, :2] * valid
        h = self.in_embed(x) + self.pe[:, :x.shape[1]]
        mask = None
        if causal:
            # upper-triangular -inf mask: token t attends only to <= t (no future)
            T = x.shape[1]
            mask = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)
        h = self.encoder(h, mask=mask)
        return self.head(h)                                     # (B,T,4)


@torch.no_grad()
def run_transformer_inference(model, dense_obs, valid_mask, device,
                              window=32, stride=16, mode='smooth'):
    """Run the Transformer denoiser under a UNIFORM inference protocol matched to
    the other methods, so the comparison is fair.

    mode='smooth' (default): sliding window W with Hann overlap-add assembly --
        IDENTICAL windowing protocol to CPGD-Smooth and CSDI. Within each window
        the encoder is bidirectional (sees only the +-W/2 window context, not the
        whole sequence), so its look-ahead matches the other sliding-window
        smoothers. This replaces the earlier full-sequence pass that illegitimately
        attended over ALL future frames.

    mode='causal': single pass with an upper-triangular attention mask so token k
        attends only to frames <= k -- a STRICTLY CAUSAL, zero-lag filter,
        directly comparable to CPGD-Causal / LSTM / GRU / NB-DLSTM.

    Returns (N,4) [x,y,vx,vy]."""
    N = len(dense_obs)
    obs = dense_obs.copy(); obs[~valid_mask] = 0.0
    fv = int(np.argmax(valid_mask)) if valid_mask.any() else 0
    anchor = torch.from_numpy(obs[fv:fv + 1]).float().to(device).view(1, 1, 2)
    obs_t = torch.from_numpy(
        np.column_stack([obs, valid_mask.astype(np.float32)])).float().unsqueeze(0).to(device)
    on = obs_t.clone()
    on[:, :, :2] = anchor_normalize(obs_t[:, :, :2], anchor) * obs_t[:, :, 2:3]

    if mode == 'causal':
        st = model(on, causal=True)                            # strict causal, single pass
        pos = anchor_denormalize(st[:, :, :2], anchor)[0].cpu().numpy()
        vel = st[0, :, 2:].cpu().numpy() * VEL_SCALE
        return np.column_stack([pos, vel])

    # mode == 'smooth': sliding-window Hann overlap-add (matches CPGD-Smooth/CSDI)
    hann = torch.hann_window(window, periodic=False, device=device).clamp(min=1e-3)
    gp = torch.zeros((N, 4), device=device)
    gw = torch.zeros((N, 1), device=device)
    for ws in range(0, max(1, N - window + 1), stride):
        we = min(ws + window, N)
        if we - ws < window:
            ws = we - window
        st_w = model(on[:, ws:we, :])[0]                       # (W,4) bidirectional within window
        gp[ws:we] += st_w * hann.view(-1, 1)
        gw[ws:we] += hann.view(-1, 1)
    st = (gp / gw.clamp(min=1e-6))
    pos = anchor_denormalize(st[None, :, :2], anchor)[0].cpu().numpy()
    vel = st[:, 2:].cpu().numpy() * VEL_SCALE
    return np.column_stack([pos, vel])
