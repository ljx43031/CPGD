"""
imm_tracker.py
==============
IMM-EKF 经典机动目标跟踪基线(回应 R5:补 IMM-EKF/CKF 经典基线)。

结构:Interacting Multiple Model (Blom & Bar-Shalom) + 协同转弯(Coordinated
Turn, CT)EKF 子滤波器。所有子模型共用 5 维状态 [px, py, vx, vy, ω],预测为
CT 非线性运动学(ω→0 时退化为匀速 CV),故用 EKF(数值 Jacobian 传播协方差)。
三个模型以转弯率过程噪声 σ_ω 区分:低(近 CV)/中/高机动。量测为位置 [px,py],
线性 H。盲区帧(valid=0)只预测不更新(coast),与真实跟踪器一致。

输出与其他基线对齐:每帧 (px, py, vx, vy) 的 (N,4),丢弃 ω。

参考:Bar-Shalom, Li, Kirubarajan, "Estimation with Applications to Tracking
and Navigation" (2001), Ch.11(IMM);协同转弯模型 Ch.11.7。
"""

import numpy as np


def _ct_transition(x, T):
    """协同转弯离散状态转移 f(x). x=[px,py,vx,vy,ω]. ω→0 退化 CV。"""
    px, py, vx, vy, w = x
    if abs(w) < 1e-6:
        return np.array([px + vx * T, py + vy * T, vx, vy, w])
    swt, cwt = np.sin(w * T), np.cos(w * T)
    px2 = px + (vx * swt + vy * (cwt - 1.0)) / w
    py2 = py + (vx * (1.0 - cwt) + vy * swt) / w
    vx2 = vx * cwt - vy * swt
    vy2 = vx * swt + vy * cwt
    return np.array([px2, py2, vx2, vy2, w])


def _ct_jacobian(x, T, eps=1e-5):
    """数值 Jacobian ∂f/∂x(5×5),稳健避免手推代数错误(5 维成本可忽略)。"""
    f0 = _ct_transition(x, T)
    J = np.zeros((5, 5))
    for j in range(5):
        dx = np.zeros(5); dx[j] = eps
        J[:, j] = (_ct_transition(x + dx, T) - f0) / eps
    return J


class _CTEKF:
    """单个协同转弯 EKF 子滤波器。"""
    def __init__(self, q_acc, q_omega, T):
        self.T = T
        self.q_acc = q_acc          # 速度白噪声加速度谱密度
        self.q_omega = q_omega      # 转弯率过程噪声(区分模型机动强度)
        self.x = np.zeros(5)
        self.P = np.eye(5) * 100.0

    def _Q(self):
        T, qa, qw = self.T, self.q_acc, self.q_omega
        # 位置/速度用离散白噪声加速度模型;ω 独立随机游走
        Q = np.zeros((5, 5))
        q_pv = qa * np.array([[T**3/3, T**2/2],
                              [T**2/2, T]])
        # px,vx 块
        Q[np.ix_([0, 2], [0, 2])] = q_pv
        # py,vy 块
        Q[np.ix_([1, 3], [1, 3])] = q_pv
        Q[4, 4] = qw * T
        return Q

    def predict(self):
        F = _ct_jacobian(self.x, self.T)
        self.x = _ct_transition(self.x, self.T)
        self.P = F @ self.P @ F.T + self._Q()

    def update(self, z, R):
        H = np.zeros((2, 5)); H[0, 0] = 1.0; H[1, 1] = 1.0
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        try:
            Sinv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            Sinv = np.linalg.pinv(S)
        K = self.P @ H.T @ Sinv
        self.x = self.x + K @ y
        self.P = (np.eye(5) - K @ H) @ self.P
        # 似然(高斯),用于 IMM 模式概率更新
        det = np.linalg.det(S)
        if det <= 0:
            return 1e-12
        L = np.exp(-0.5 * float(y @ Sinv @ y)) / np.sqrt((2 * np.pi) ** 2 * det)
        return max(L, 1e-12)


class IMMEKF:
    """IMM(3 个 CT-EKF 模型:近CV / 中机动 / 高机动)。"""
    def __init__(self, dt, meas_std=10.0,
                 q_acc=1.0, q_omega=(1e-4, 1e-2, 1e-1),
                 p_stay=0.95):
        self.dt = dt
        self.R = np.eye(2) * (meas_std ** 2)
        self.M = len(q_omega)
        self.models = [_CTEKF(q_acc, qw, dt) for qw in q_omega]
        # 模式转移矩阵:对角 p_stay,其余均分
        off = (1.0 - p_stay) / (self.M - 1)
        self.Pi = np.full((self.M, self.M), off)
        np.fill_diagonal(self.Pi, p_stay)
        self.mu = np.full(self.M, 1.0 / self.M)   # 模式概率
        self._init = False

    def initialize(self, pos0, vel0=(0.0, 0.0)):
        x0 = np.array([pos0[0], pos0[1], vel0[0], vel0[1], 0.0])
        for m in self.models:
            m.x = x0.copy()
            m.P = np.diag([100.0, 100.0, 25.0, 25.0, 0.1])
        self._init = True

    def step(self, z=None):
        """一帧 IMM 循环。z=None 表示盲区(只预测不更新)。返回融合 [px,py,vx,vy]。"""
        M = self.M
        # 1) 交互/混合
        cbar = self.Pi.T @ self.mu                       # (M,)
        cbar = np.clip(cbar, 1e-12, None)
        mix = (self.Pi * self.mu[:, None]) / cbar[None, :]   # mix[i,j]=P(i|j)
        xs = [m.x.copy() for m in self.models]
        Ps = [m.P.copy() for m in self.models]
        for j in range(M):
            xj = sum(mix[i, j] * xs[i] for i in range(M))
            Pj = np.zeros((5, 5))
            for i in range(M):
                d = (xs[i] - xj).reshape(-1, 1)
                Pj += mix[i, j] * (Ps[i] + d @ d.T)
            self.models[j].x = xj
            self.models[j].P = Pj
        # 2) 各模型预测(+更新)
        likelihoods = np.ones(M)
        for j, m in enumerate(self.models):
            m.predict()
            if z is not None:
                likelihoods[j] = m.update(z, self.R)
        # 3) 模式概率更新(盲区无量测则只按预测混合,概率不变基于 cbar)
        if z is not None:
            self.mu = cbar * likelihoods
            self.mu = self.mu / max(self.mu.sum(), 1e-12)
        else:
            self.mu = cbar / max(cbar.sum(), 1e-12)
        # 4) 融合输出
        x_out = sum(self.mu[j] * self.models[j].x for j in range(M))
        return np.array([x_out[0], x_out[1], x_out[2], x_out[3]])


def run_imm_inference(dense_obs, valid_mask, timestamps, meas_std=10.0):
    """跑整条序列。dense_obs (N,2) 前向填充观测, valid_mask (N,) bool。
    返回 (N,4) [px,py,vx,vy]。盲区帧只预测(coast)。"""
    N = len(dense_obs)
    dt = float(timestamps[1] - timestamps[0]) if N > 1 else 0.1
    tracker = IMMEKF(dt, meas_std=meas_std)
    # 用首个有效观测初始化
    fv = int(np.argmax(valid_mask)) if valid_mask.any() else 0
    tracker.initialize(dense_obs[fv])
    out = np.zeros((N, 4), dtype=np.float32)
    for k in range(N):
        z = dense_obs[k] if valid_mask[k] else None
        out[k] = tracker.step(z)
    return out
