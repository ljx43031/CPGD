# ------------------------------------------------------------------------------
# 运动模型估计网络 MMEN (修正版)
# 适配: TIMESTEPS=32, INPUT_SIZE=4(只取前两个状态，也就是x y坐标)
# ------------------------------------------------------------------------------
import torch
from torch import nn


class PartialInfoInfer_P1(nn.Module):
    def __init__(self, TIMESTEPS):
        super(PartialInfoInfer_P1, self).__init__()

        # --- 网络结构参数 ---
        # 这里的计算逻辑是为了追踪时间维度(H)的变化，确保最后输出为1
        # H_out = H_in - Kernel_size + 1

        # 第一层参数
        C1_cn = 4
        # CDP1 处理后时间维度减少2 (T-2)
        # Conv1 (Kernel=2): (T-2) - 2 + 1 = T - 3
        self.TIMESTEPS1 = TIMESTEPS - 3

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=C1_cn,
                kernel_size=2,  # (2, 2) 对应 input_width=2 (Cross, Dot)
                stride=1,
                padding=0),
        )
        self.Center1 = nn.Parameter(torch.ones(self.TIMESTEPS1, C1_cn).float())
        self.Sigma1 = nn.Parameter(torch.ones(self.TIMESTEPS1, C1_cn).float())

        # 第二层参数
        C2_cn = 8
        # Conv2 (Kernel=4): T1 - 4 + 1
        self.TIMESTEPS2 = self.TIMESTEPS1 - C1_cn + 1

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=C2_cn,
                kernel_size=C1_cn,  # (4, 4)
                stride=1,
                padding=0),
        )
        self.Center2 = nn.Parameter(torch.ones(self.TIMESTEPS2, C2_cn).float())
        self.Sigma2 = nn.Parameter(torch.ones(self.TIMESTEPS2, C2_cn).float())

        # 第三层参数
        C3_cn = 16
        # Conv3 (Kernel=8): T2 - 8 + 1
        self.TIMESTEPS3 = self.TIMESTEPS2 - C2_cn + 1

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=C3_cn,
                kernel_size=C2_cn,  # (8, 8)
                stride=1,
                padding=0),
        )
        self.Center3 = nn.Parameter(torch.ones(self.TIMESTEPS3, C3_cn).float())
        self.Sigma3 = nn.Parameter(torch.ones(self.TIMESTEPS3, C3_cn).float())

        # 第四层 (输出层)
        StateTrans_num = 8

        # [关键修改] 动态设置卷积核大小
        # 输入维度: [Batch, 1, TIMESTEPS3, C3_cn]
        # 我们需要将 H(Time) 和 W(Feature) 都压缩为 1
        # 原代码 kernel_size=C3_cn (即16)，仅当 TIMESTEPS3==16 时有效
        # 现改为 (TIMESTEPS3, C3_cn)，自适应任何时间步长
        final_kernel_size = (self.TIMESTEPS3, C3_cn)

        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=StateTrans_num,
                kernel_size=final_kernel_size,  # 动态核: (19, 16) for T=32
                stride=1,
                padding=0),
        )

    def CDP1(self, x):
        # shape of x: BATCHSIZE * TIMESTEPS * INPUT_SIZE (4)

        # [关键修改]
        # 几何计算（叉乘/点乘）只基于位置坐标 (前两维)
        # 如果输入包含速度 (x, y, vx, vy)，我们只取 (x, y)
        x_geo = x[:, :, :2]

        x_d0 = x_geo - torch.unsqueeze(x_geo[:, 0, :], 1)
        # 消除第一个点
        x_d0 = x_d0[:, 1:, :]

        # Calculate the 2-Norm
        x_d0_2norm = torch.norm(x_d0, p=2, dim=2)
        TS = x_d0_2norm.size(1)

        # 防止除以零 (可选增强)
        x_d0_2norm = torch.clamp(x_d0_2norm, min=1e-6)

        # Calculate cross and dot product
        x_d0_2norm_c = torch.unsqueeze(x_d0_2norm[:, 0], 1) * x_d0_2norm[:, 1:TS]
        x_d0_et0_0 = torch.unsqueeze(x_d0[:, 0, 0], 1)
        x_d0_et0_1 = torch.unsqueeze(x_d0[:, 0, 1], 1)

        # Cross product
        x_cross = (x_d0_et0_0 * x_d0[:, 1:TS, 1] - x_d0_et0_1 * x_d0[:, 1:TS, 0]) / x_d0_2norm_c
        # Dot product
        x_dot = (x_d0_et0_0 * x_d0[:, 1:TS, 0] + x_d0_et0_1 * x_d0[:, 1:TS, 1]) / x_d0_2norm_c

        # Combine: [Batch, T-2, 2]
        x_comb = torch.stack([x_cross, x_dot], dim=2)

        # 返回原始差分(用于残差等)和几何特征
        # 注意：这里 x_d0 只有两维，如果后续需要用到速度差分，可能需要调整逻辑
        # 但根据原始代码结构，这里主要用于特征提取
        return x_d0, x_comb

    def forward(self, x):
        # x: [Batch, 32, 4]

        # 1. 预处理与特征提取
        x_d0, x_cd = self.CDP1(x)

        # 累积差分消除最后3个时刻值 (这部分在原代码逻辑中似乎未被后续卷积使用，仅作对齐)
        # x_d1 = x_d0[:, 0:x.shape[1]-3, :]
        # x_d1 = torch.unsqueeze(x_d1, 1)

        # 2. 卷积网络处理几何特征
        # [Batch, 1, T-2, 2]
        x_cd = torch.unsqueeze(x_cd, 1)

        # Layer 1
        C1_mix = self.conv1(x_cd)
        C1_tran = torch.transpose(C1_mix, 1, 3)  # [Batch, 1, T-3, 4]
        C1_i = torch.tanh(C1_tran)
        C1_s = torch.exp(-(((C1_tran - self.Center1) / self.Sigma1) ** 2))
        C1_out = C1_i.mul(C1_s)

        # Layer 2
        C2_mix = self.conv2(C1_out)
        C2_tran = torch.transpose(C2_mix, 1, 3)  # [Batch, 1, T-6, 8]
        C2_i = torch.tanh(C2_tran)
        C2_s = torch.exp(-(((C2_tran - self.Center2) / self.Sigma2) ** 2))
        C2_out = C2_i.mul(C2_s)

        # Layer 3
        C3_mix = self.conv3(C2_out)
        C3_tran = torch.transpose(C3_mix, 1, 3)  # [Batch, 1, T-13, 16] (当T=32时, T-13=19)
        C3_i = torch.tanh(C3_tran)
        C3_s = torch.exp(-(((C3_tran - self.Center3) / self.Sigma3) ** 2))
        C3_out = C3_i.mul(C3_s)

        # Layer 4 (Output)
        # Input: [Batch, 1, 19, 16] -> Conv Kernel (19, 16) -> Output [Batch, 8, 1, 1]
        STM_i = self.conv4(C3_out)
        STM_i = torch.squeeze(STM_i)  # [Batch, 8]

        # 3. 构建转移矩阵
        x_TM = STM_i.reshape(x.shape[0], 2, 4)

        # [关键修改] 使用 x.device 确保兼容 CPU/GPU
        x01 = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0]]).float().to(x.device)
        x01 = x01.reshape(1, 2, 4)
        x01_e = x01.repeat(x.shape[0], 1, 1)

        # 拼接: 上半部分是单位阵(位置保持)，下半部分是学习到的(速度/加速度变化)
        x_TM = torch.cat([x01_e, x_TM], 1)  # [Batch, 4, 4]

        return x_TM
