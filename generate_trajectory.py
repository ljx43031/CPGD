"""
改进版二维平面匀速转弯轨迹生成器
Improved Constant Turn (CT) Model Trajectory Generator
- 返回状态转移矩阵F用于训练
- 随机速度范围
- 合并state输出
"""
import env_config  # 统一加载环境配置
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class TrajectorySegment:
    """轨迹段数据类"""
    states: np.ndarray  # (N, 4) - [x, y, vx, vy] states
    transition_matrices: np.ndarray  # (N-1, 4, 4) - 状态转移矩阵F
    timestamps: np.ndarray  # (N,) - time stamps
    turn_rate: float  # 转弯率 (rad/s)
    speed: float  # 速度大小 (m/s)
    duration: float  # 段持续时间 (s)


class CTMotionModel:
    """匀速转弯运动模型 (Constant Turn Motion Model)"""

    def __init__(self, dt: float = 0.1):
        """
        初始化CT运动模型

        Args:
            dt: 采样间隔 (秒)
        """
        self.dt = dt

    def state_transition(self, state: np.ndarray, turn_rate: float, dt: Optional[float] = None) -> np.ndarray:
        """
        状态转移函数

        Args:
            state: 当前状态 [x, y, vx, vy]
            turn_rate: 转弯率 (rad/s)，正值为逆时针，负值为顺时针
            dt: 时间步长，如果为None则使用self.dt

        Returns:
            next_state: 下一时刻状态 [x, y, vx, vy]
        """
        if dt is None:
            dt = self.dt

        x, y, vx, vy = state
        omega = turn_rate

        # 小转弯率阈值，避免数值问题
        omega_threshold = 1e-6

        if abs(omega) < omega_threshold:
            # 直线运动（转弯率接近0）
            x_next = x + dt * vx
            y_next = y + dt * vy
            vx_next = vx
            vy_next = vy
        else:
            # 匀速转弯运动
            omega_dt = omega * dt
            sin_omega_dt = np.sin(omega_dt)
            cos_omega_dt = np.cos(omega_dt)

            # 位置更新
            x_next = x + (sin_omega_dt / omega) * vx - ((1 - cos_omega_dt) / omega) * vy
            y_next = y + ((1 - cos_omega_dt) / omega) * vx + (sin_omega_dt / omega) * vy

            # 速度更新（旋转）
            vx_next = cos_omega_dt * vx - sin_omega_dt * vy
            vy_next = sin_omega_dt * vx + cos_omega_dt * vy

        return np.array([x_next, y_next, vx_next, vy_next])

    def get_transition_matrix(self, turn_rate: float, dt: Optional[float] = None) -> np.ndarray:
        """
        获取状态转移矩阵F

        状态方程: x(k+1) = F * x(k)
        其中 x = [x, y, vx, vy]^T

        Args:
            turn_rate: 转弯率 (rad/s)
            dt: 时间步长

        Returns:
            F: 4x4 状态转移矩阵
        """
        if dt is None:
            dt = self.dt

        omega = turn_rate
        omega_threshold = 1e-6

        if abs(omega) < omega_threshold:
            # 直线运动
            F = np.array([
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
        else:
            # 转弯运动
            omega_dt = omega * dt
            sin_omega_dt = np.sin(omega_dt)
            cos_omega_dt = np.cos(omega_dt)

            F = np.array([
                [1, 0, sin_omega_dt / omega, -(1 - cos_omega_dt) / omega],
                [0, 1, (1 - cos_omega_dt) / omega, sin_omega_dt / omega],
                [0, 0, cos_omega_dt, -sin_omega_dt],
                [0, 0, sin_omega_dt, cos_omega_dt]
            ])

        return F


def generate_ct_trajectory(
        initial_state: np.ndarray,
        turn_rate: float,
        duration: float,
        dt: float = 0.1,
        area_size: Optional[Tuple[float, float]] = None
) -> TrajectorySegment:
    """
    生成单段匀速转弯轨迹

    Args:
        initial_state: 初始状态 [x, y, vx, vy]
        turn_rate: 转弯率 (rad/s)，正值逆时针，负值顺时针
        duration: 段持续时间 (秒)
        dt: 采样间隔 (秒)

    Returns:
        TrajectorySegment: 包含轨迹数据的对象
            - states: (N, 4) 状态矩阵 [x, y, vx, vy]
            - transition_matrices: (N-1, 4, 4) 状态转移矩阵F
            - timestamps: (N,) 时间戳
            - turn_rate: 转弯率
            - speed: 速度大小
            - duration: 持续时间
    """
    model = CTMotionModel(dt)

    # 计算时间步数
    n_steps = int(duration / dt) + 1

    # 初始化数组
    states = np.zeros((n_steps, 4))
    transition_matrices = np.zeros((n_steps - 1, 4, 4))
    # 使用严格均匀步长，保证 timestamps[k] == k * dt，避免 linspace 浮点误差
    timestamps = np.arange(n_steps) * dt

    # 设置初始状态
    state = initial_state.copy()
    states[0] = state

    # 获取状态转移矩阵（对于CT模型，F在每个时间步是相同的）
    F = model.get_transition_matrix(turn_rate, dt)

    # 递推生成轨迹
    for i in range(1, n_steps):
        # 保存转移矩阵
        transition_matrices[i - 1] = F

        # 状态转移
        state = model.state_transition(state, turn_rate, dt)

        states[i] = state

    # 计算速度大小（应该保持恒定）
    speed = np.mean(np.linalg.norm(states[:, 2:], axis=1))

    return TrajectorySegment(
        states=states,
        transition_matrices=transition_matrices,
        timestamps=timestamps,
        turn_rate=turn_rate,
        speed=speed,
        duration=duration
    )


def generate_multi_segment_trajectory(
        total_duration: float,
        n_segments: int,
        turn_rate_range: Tuple[float, float] = (-0.5, 0.5),
        speed_range: Tuple[float, float] = (10.0, 20.0),
        initial_state: Optional[np.ndarray] = None,
        dt: float = 0.1,
        min_segment_duration: float = 5.0,
        area_size: Optional[Tuple[float, float]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[float], List[float], List[TrajectorySegment]]:
    """
    生成多段随机转弯率和速度的轨迹

    Args:
        total_duration: 总时间长度 (秒)
        n_segments: 轨迹段数
        turn_rate_range: 转弯率范围 (rad/s) (min, max)
        speed_range: 速度范围 (m/s) (min, max) - 每段随机选择
        initial_state: 初始状态 [x, y, vx, vy]，如果为None则自动生成
        dt: 采样间隔 (秒)
        min_segment_duration: 最小段持续时间 (秒)

    Returns:
        states: (N, 4) 状态数组 [x, y, vx, vy]
        transition_matrices: (N-1, 4, 4) 状态转移矩阵F
        timestamps: (N,) 时间戳数组
        turn_rates: List[float] 每段的转弯率
        speeds: List[float] 每段的速度
        segments: List[TrajectorySegment] 各段轨迹对象列表
    """
    # 检查参数合理性
    if total_duration < n_segments * min_segment_duration:
        raise ValueError(
            f"总时间({total_duration}s)不足以容纳{n_segments}段"
            f"最小时长为{min_segment_duration}s的轨迹段"
        )

    # 生成每段的持续时间（随机但总和等于total_duration）
    remaining_time = total_duration
    segment_durations = []

    for i in range(n_segments - 1):
        # 随机分配时间，但确保剩余时间足够
        max_duration = remaining_time - (n_segments - i - 1) * min_segment_duration
        min_duration = min_segment_duration

        if max_duration < min_duration:
            raise ValueError("无法分配合理的段持续时间")

        duration = np.random.uniform(min_duration, max_duration)
        segment_durations.append(duration)
        remaining_time -= duration

    # 最后一段使用剩余时间
    segment_durations.append(remaining_time)

    # 随机生成每段的转弯率
    turn_rates = np.random.uniform(
        turn_rate_range[0],
        turn_rate_range[1],
        n_segments
    ).tolist()

    # 随机生成每段的速度
    speeds = np.random.uniform(
        speed_range[0],
        speed_range[1],
        n_segments
    ).tolist()

    # 设置初始状态
    if initial_state is None:
        # 默认从原点出发，初始速度向x方向，使用第一段的速度
        initial_state = np.array([0.0, 0.0, speeds[0], 0.0])
    else:
        # 调整初始速度大小为第一段的速度
        v_current = np.linalg.norm(initial_state[2:])
        if v_current > 0:
            initial_state[2:] = initial_state[2:] / v_current * speeds[0]
        else:
            initial_state[2:] = np.array([speeds[0], 0.0])

    # 生成各段轨迹
    segments = []
    current_state = initial_state.copy()
    current_time = 0.0

    all_states = []
    all_transition_matrices = []
    all_timestamps = []

    for i in range(n_segments):
        # 生成当前段
        segment = generate_ct_trajectory(
            current_state,
            turn_rates[i],
            segment_durations[i],
            dt,
            area_size=area_size
        )

        # 调整时间戳
        segment.timestamps = segment.timestamps + current_time

        # 存储段信息
        segments.append(segment)

        # 累积轨迹数据（避免重复端点）
        if i == 0:
            all_states.append(segment.states)
            all_transition_matrices.append(segment.transition_matrices)
            all_timestamps.append(segment.timestamps)
        else:
            # 跳过第一个点（与上一段最后一个点重复）
            all_states.append(segment.states[1:])
            all_timestamps.append(segment.timestamps[1:])

            # 对于转移矩阵，需要添加从上一段末尾到当前段开头的转移矩阵
            # 这是一个速度变化的转移（如果速度改变了）
            if i > 0:
                # 上一段最后的状态
                prev_final_state = segments[i - 1].states[-1]
                # 当前段第一个状态
                curr_first_state = segment.states[0]

                # 如果速度改变，需要一个速度调整的转移矩阵
                # 这里简化处理：保持位置不变，只改变速度
                # F_transition = [[1, 0, 0, 0],
                #                 [0, 1, 0, 0],
                #                 [0, 0, v_new/v_old * cos(theta), -v_new/v_old * sin(theta)],
                #                 [0, 0, v_new/v_old * sin(theta), v_new/v_old * cos(theta)]]

                # 为简化，我们创建一个速度缩放矩阵
                v_prev = np.linalg.norm(prev_final_state[2:])
                v_curr = np.linalg.norm(curr_first_state[2:])

                if v_prev > 1e-6:
                    scale = v_curr / v_prev
                    # 计算旋转角度（如果有的话）
                    angle_prev = np.arctan2(prev_final_state[3], prev_final_state[2])
                    angle_curr = np.arctan2(curr_first_state[3], curr_first_state[2])
                    delta_angle = angle_curr - angle_prev

                    cos_da = np.cos(delta_angle)
                    sin_da = np.sin(delta_angle)

                    F_transition = np.array([
                        [1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, scale * cos_da, -scale * sin_da],
                        [0, 0, scale * sin_da, scale * cos_da]
                    ])
                else:
                    F_transition = np.eye(4)

                # 添加这个转移矩阵
                all_transition_matrices.append(F_transition.reshape(1, 4, 4))

            # 添加当前段的转移矩阵
            all_transition_matrices.append(segment.transition_matrices)

        # 更新下一段的初始状态
        # 需要调整速度大小为下一段的速度
        if i < n_segments - 1:
            final_position = segment.states[-1, :2]
            final_velocity_direction = segment.states[-1, 2:]
            v_norm = np.linalg.norm(final_velocity_direction)

            if v_norm > 1e-6:
                # 保持速度方向，调整大小为下一段的速度
                velocity_direction = final_velocity_direction / v_norm
                new_velocity = velocity_direction * speeds[i + 1]
            else:
                # 如果速度为0，使用x方向
                new_velocity = np.array([speeds[i + 1], 0.0])

            # ================= 新增：柔性向心引导逻辑（完美保持数学连续性） =================
            if area_size is not None:
                bound_x, bound_y = area_size[0], area_size[1]
                margin = 150.0  # 距离边界 150 米时开始干预

                # 计算当前位置到地图中心的向量
                center_vec = np.array([bound_x / 2 - final_position[0], bound_y / 2 - final_position[1]])
                dist_to_center = np.linalg.norm(center_vec)

                # 如果接近边界，或者已经出界
                if (final_position[0] < margin or final_position[0] > bound_x - margin or
                        final_position[1] < margin or final_position[1] > bound_y - margin):

                    if dist_to_center > 1e-6:
                        center_dir = center_vec / dist_to_center
                        # 强制把下一段的初始速度方向，朝着地图中心的方向进行一定程度的偏转
                        # 越靠近边界，向心拉力越强
                        blend_factor = 0.6  # 0.0 完全保持原方向, 1.0 直接指向中心
                        blended_dir = (1 - blend_factor) * velocity_direction + blend_factor * center_dir
                        blended_dir /= np.linalg.norm(blended_dir)
                        new_velocity = blended_dir * speeds[i + 1]

                        # 同时，强制覆盖下一段的 turn_rates[i+1]，确保其轨迹向内弯曲
                        # 计算当前速度到中心向量的叉乘，决定是左转还是右转能更快朝向中心
                        cross_prod = blended_dir[0] * center_dir[1] - blended_dir[1] * center_dir[0]
                        # 强行赋予一个较大的转弯率，使其掉头
                        turn_rates[i + 1] = 0.25 if cross_prod > 0 else -0.25
            # =========================================================================

            current_state = np.concatenate([final_position, new_velocity])

        current_time = segment.timestamps[-1]

    # 合并所有段
    states = np.vstack(all_states)

    # 合并转移矩阵
    transition_matrices = np.vstack(all_transition_matrices)

    # 重建全局均匀时间戳：保证 timestamps[k] == k * dt
    # 多段 linspace/arange 累积后存在浮点漂移，直接重建可确保
    # int(round(timestamps[k] / dt)) == k 严格成立，避免帧对齐断裂
    total_n = len(states)
    timestamps = np.arange(total_n) * dt

    return states, transition_matrices, timestamps, turn_rates, speeds, segments


def plot_trajectory_v2(
        states: np.ndarray,
        timestamps: np.ndarray,
        turn_rates: List[float],
        speeds: List[float],
        segments: List[TrajectorySegment],
        title: str = "Multi-Segment CT Trajectory"
):
    """
    可视化轨迹（更新版，适配新的数据结构）

    Args:
        states: (N, 4) 状态数组 [x, y, vx, vy]
        timestamps: (N,) 时间戳数组
        turn_rates: List[float] 每段转弯率
        speeds: List[float] 每段速度
        segments: 轨迹段列表
        title: 图标题
    """
    positions = states[:, :2]
    velocities = states[:, 2:]

    fig = plt.figure(figsize=(18, 12))

    # 1. 轨迹平面图
    ax1 = plt.subplot(3, 3, 1)

    # 绘制完整轨迹
    ax1.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2, alpha=0.6, label='Full Trajectory')

    # 用不同颜色标记各段
    colors = plt.cm.rainbow(np.linspace(0, 1, len(segments)))
    start_idx = 0

    for i, (segment, color) in enumerate(zip(segments, colors)):
        end_idx = start_idx + len(segment.states)

        # 绘制段
        ax1.plot(
            positions[start_idx:end_idx, 0],
            positions[start_idx:end_idx, 1],
            color=color,
            linewidth=3,
            alpha=0.7,
            label=f'S{i + 1}: ω={segment.turn_rate:.3f}, v={segment.speed:.1f}'
        )

        # 标记起点
        ax1.plot(positions[start_idx, 0], positions[start_idx, 1],
                 'o', color=color, markersize=8)

        start_idx = end_idx - 1

    # 标记总起点和终点
    ax1.plot(positions[0, 0], positions[0, 1], 'go', markersize=14,
             markerfacecolor='lime', markeredgewidth=2, label='Start', zorder=10)
    ax1.plot(positions[-1, 0], positions[-1, 1], 'rs', markersize=14,
             markerfacecolor='red', markeredgewidth=2, label='End', zorder=10)

    # 绘制速度向量（每隔N个点）
    N_arrows = 30
    arrow_indices = np.linspace(0, len(positions) - 1, N_arrows, dtype=int)
    for idx in arrow_indices:
        ax1.arrow(
            positions[idx, 0], positions[idx, 1],
            velocities[idx, 0] * 0.5, velocities[idx, 1] * 0.5,
            head_width=2, head_length=1.5, fc='red', ec='red', alpha=0.3
        )

    ax1.set_xlabel('X Position (m)', fontsize=11)
    ax1.set_ylabel('Y Position (m)', fontsize=11)
    ax1.set_title('2D Trajectory with Velocity Vectors', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    ax1.legend(loc='best', fontsize=7, ncol=2)

    # 2. X-Y位置随时间变化
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(timestamps, positions[:, 0], 'r-', linewidth=2, label='X Position')
    ax2.plot(timestamps, positions[:, 1], 'b-', linewidth=2, label='Y Position')

    # 标记段边界
    cumulative_time = 0
    for i, segment in enumerate(segments):
        cumulative_time += segment.duration
        ax2.axvline(cumulative_time, color='gray', linestyle='--', alpha=0.5)
        ax2.text(cumulative_time - segment.duration / 2,
                 ax2.get_ylim()[1] * 0.9,
                 f'S{i + 1}',
                 ha='center', fontsize=8, color='gray')

    ax2.set_xlabel('Time (s)', fontsize=11)
    ax2.set_ylabel('Position (m)', fontsize=11)
    ax2.set_title('Position vs Time', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # 3. 速度分量随时间变化
    ax3 = plt.subplot(3, 3, 3)
    ax3.plot(timestamps, velocities[:, 0], 'r-', linewidth=2, label='Vx')
    ax3.plot(timestamps, velocities[:, 1], 'b-', linewidth=2, label='Vy')

    # 计算速度大小
    speed = np.linalg.norm(velocities, axis=1)
    ax3.plot(timestamps, speed, 'g--', linewidth=2, label='Speed')

    # 标记段边界和目标速度
    cumulative_time = 0
    for i, segment in enumerate(segments):
        cumulative_time += segment.duration
        ax3.axvline(cumulative_time, color='gray', linestyle='--', alpha=0.5)
        # 绘制该段的目标速度
        t_start = cumulative_time - segment.duration
        ax3.hlines(segment.speed, t_start, cumulative_time,
                   colors='orange', linestyles=':', linewidth=2, alpha=0.5)

    ax3.set_xlabel('Time (s)', fontsize=11)
    ax3.set_ylabel('Velocity (m/s)', fontsize=11)
    ax3.set_title('Velocity Components vs Time', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # 4. 航向角随时间变化
    ax4 = plt.subplot(3, 3, 4)
    heading = np.arctan2(velocities[:, 1], velocities[:, 0])
    heading_deg = np.degrees(heading)

    ax4.plot(timestamps, heading_deg, 'purple', linewidth=2)

    # 标记段边界和转弯率
    cumulative_time = 0
    for i, segment in enumerate(segments):
        cumulative_time += segment.duration
        ax4.axvline(cumulative_time, color='gray', linestyle='--', alpha=0.5)

        # 计算该段的平均航向
        seg_start_idx = np.searchsorted(timestamps, cumulative_time - segment.duration)
        seg_end_idx = np.searchsorted(timestamps, cumulative_time)

        if seg_start_idx < seg_end_idx:
            avg_heading = np.mean(heading_deg[seg_start_idx:seg_end_idx])

            ax4.text(cumulative_time - segment.duration / 2,
                     avg_heading,
                     f'ω={segment.turn_rate:.3f}',
                     ha='center', fontsize=7,
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax4.set_xlabel('Time (s)', fontsize=11)
    ax4.set_ylabel('Heading Angle (degrees)', fontsize=11)
    ax4.set_title('Heading Angle vs Time', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    # 5. 转弯率分布
    ax5 = plt.subplot(3, 3, 5)
    segment_indices = np.arange(len(segments))
    durations = [seg.duration for seg in segments]

    bars = ax5.bar(segment_indices, turn_rates,
                   color=colors, alpha=0.7, edgecolor='black', linewidth=2)

    # 在柱子上标注持续时间
    for i, (bar, duration) in enumerate(zip(bars, durations)):
        height = bar.get_height()
        y_pos = height + 0.02 if height >= 0 else height - 0.02
        ax5.text(bar.get_x() + bar.get_width() / 2., y_pos,
                 f'{duration:.1f}s',
                 ha='center', va='bottom' if height >= 0 else 'top', fontsize=8)

    ax5.set_xlabel('Segment Index', fontsize=11)
    ax5.set_ylabel('Turn Rate (rad/s)', fontsize=11)
    ax5.set_title('Turn Rate per Segment', fontsize=12, fontweight='bold')
    ax5.axhline(0, color='black', linestyle='-', linewidth=1)
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.set_xticks(segment_indices)
    ax5.set_xticklabels([f'S{i + 1}' for i in range(len(segments))])

    # 6. 速度分布
    ax6 = plt.subplot(3, 3, 6)
    bars = ax6.bar(segment_indices, speeds,
                   color=colors, alpha=0.7, edgecolor='black', linewidth=2)

    # 在柱子上标注转弯率
    for i, (bar, omega) in enumerate(zip(bars, turn_rates)):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                 f'ω={omega:.3f}',
                 ha='center', va='bottom', fontsize=8)

    ax6.set_xlabel('Segment Index', fontsize=11)
    ax6.set_ylabel('Speed (m/s)', fontsize=11)
    ax6.set_title('Speed per Segment', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    ax6.set_xticks(segment_indices)
    ax6.set_xticklabels([f'S{i + 1}' for i in range(len(segments))])

    # 7. 状态转移矩阵可视化（第一个段的F矩阵）
    ax7 = plt.subplot(3, 3, 7)
    if len(segments) > 0:
        F_sample = segments[0].transition_matrices[0]
        im = ax7.imshow(F_sample, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        ax7.set_xticks(range(4))
        ax7.set_yticks(range(4))
        ax7.set_xticklabels(['x', 'y', 'vx', 'vy'])
        ax7.set_yticklabels(['x', 'y', 'vx', 'vy'])
        ax7.set_title(f'Transition Matrix F (Segment 1, ω={turn_rates[0]:.3f})',
                      fontsize=11, fontweight='bold')

        # 在每个格子中显示数值
        for i in range(4):
            for j in range(4):
                text = ax7.text(j, i, f'{F_sample[i, j]:.3f}',
                                ha="center", va="center", color="black", fontsize=8)

        plt.colorbar(im, ax=ax7, fraction=0.046, pad=0.04)

    # 8. 速度大小误差（实际速度 vs 目标速度）
    ax8 = plt.subplot(3, 3, 8)
    actual_speeds = np.linalg.norm(velocities, axis=1)

    # 为每个段创建目标速度数组
    target_speeds = np.zeros_like(actual_speeds)
    cumulative_idx = 0
    for segment in segments:
        seg_len = len(segment.states)
        target_speeds[cumulative_idx:cumulative_idx + seg_len] = segment.speed
        cumulative_idx += seg_len - 1  # -1 因为段之间共享端点

    # 如果长度不匹配，调整
    if len(target_speeds) > len(actual_speeds):
        target_speeds = target_speeds[:len(actual_speeds)]
    elif len(target_speeds) < len(actual_speeds):
        target_speeds = np.pad(target_speeds, (0, len(actual_speeds) - len(target_speeds)),
                               mode='edge')

    speed_error = actual_speeds - target_speeds

    ax8.plot(timestamps, speed_error, 'b-', linewidth=2, label='Speed Error')
    ax8.axhline(0, color='red', linestyle='--', linewidth=1.5)

    # 标记段边界
    cumulative_time = 0
    for segment in segments:
        cumulative_time += segment.duration
        ax8.axvline(cumulative_time, color='gray', linestyle='--', alpha=0.5)

    ax8.set_xlabel('Time (s)', fontsize=11)
    ax8.set_ylabel('Speed Error (m/s)', fontsize=11)
    ax8.set_title('Speed Error (Actual - Target)', fontsize=12, fontweight='bold')
    ax8.grid(True, alpha=0.3)
    ax8.legend()

    # 计算速度误差统计
    speed_rmse = np.sqrt(np.mean(speed_error ** 2))
    speed_mae = np.mean(np.abs(speed_error))

    # 9. 统计信息
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')

    # 计算总距离
    total_distance = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))

    stats_text = f"""
    📊 Trajectory Statistics

    ═══════════════════════════════
    Basic Info:
    ═══════════════════════════════
    • Total Duration:     {timestamps[-1]:.2f} s
    • Number of Segments: {len(segments)}
    • Total Points:       {len(states)}
    • Total Distance:     {total_distance:.2f} m
    • Sampling Interval:  {timestamps[1] - timestamps[0]:.3f} s

    ═══════════════════════════════
    Position Range:
    ═══════════════════════════════
    • X: [{positions[:, 0].min():.2f}, {positions[:, 0].max():.2f}] m
    • Y: [{positions[:, 1].min():.2f}, {positions[:, 1].max():.2f}] m

    ═══════════════════════════════
    Speed Statistics:
    ═══════════════════════════════
    • Mean:   {actual_speeds.mean():.2f} m/s
    • Std:    {actual_speeds.std():.2f} m/s
    • Min:    {actual_speeds.min():.2f} m/s
    • Max:    {actual_speeds.max():.2f} m/s
    • RMSE:   {speed_rmse:.4f} m/s
    • MAE:    {speed_mae:.4f} m/s

    ═══════════════════════════════
    Turn Rate Statistics:
    ═══════════════════════════════
    • Mean:   {np.mean(turn_rates):.4f} rad/s
    • Std:    {np.std(turn_rates):.4f} rad/s
    • Min:    {np.min(turn_rates):.4f} rad/s
    • Max:    {np.max(turn_rates):.4f} rad/s

    ═══════════════════════════════
    Segment Details:
    ═══════════════════════════════
    """

    for i, seg in enumerate(segments):
        stats_text += f"\n  S{i + 1}: {seg.duration:.1f}s | v={seg.speed:.1f} m/s | ω={seg.turn_rate:.4f} rad/s"

    ax9.text(0.05, 0.98, stats_text, transform=ax9.transAxes,
             fontsize=8, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.998)
    plt.tight_layout(rect=[0, 0, 1, 0.995])
    plt.show()


def analyze_transition_matrices(segments: List[TrajectorySegment]):
    """
    分析状态转移矩阵的性质

    Args:
        segments: 轨迹段列表
    """
    print("\n" + "=" * 70)
    print("状态转移矩阵分析 (State Transition Matrix Analysis)")
    print("=" * 70)

    for i, segment in enumerate(segments):
        print(f"\n📌 Segment {i + 1}:")
        print(f"   Turn Rate (ω): {segment.turn_rate:.6f} rad/s")
        print(f"   Speed: {segment.speed:.2f} m/s")
        print(f"   Duration: {segment.duration:.2f} s")
        print(f"   Number of F matrices: {len(segment.transition_matrices)}")

        # 取第一个F矩阵进行分析
        F = segment.transition_matrices[0]

        print(f"\n   State Transition Matrix F:")
        print("   " + "-" * 50)
        for row in F:
            print(f"   [{row[0]:8.5f}, {row[1]:8.5f}, {row[2]:8.5f}, {row[3]:8.5f}]")

        # 计算特征值
        eigenvalues = np.linalg.eigvals(F)
        print(f"\n   Eigenvalues:")
        for j, eigval in enumerate(eigenvalues):
            if np.isreal(eigval):
                print(f"   λ{j + 1} = {eigval.real:.6f}")
            else:
                print(f"   λ{j + 1} = {eigval.real:.6f} + {eigval.imag:.6f}j (magnitude: {np.abs(eigval):.6f})")

        # 计算行列式（应该接近1，表示保持体积）
        det_F = np.linalg.det(F)
        print(f"\n   Determinant: {det_F:.8f}")

        # 检验正交性（对于纯旋转部分）
        velocity_rotation = F[2:, 2:]
        orthogonality = np.dot(velocity_rotation, velocity_rotation.T)
        print(f"\n   Velocity Rotation Orthogonality Check:")
        print(f"   R^T * R =")
        for row in orthogonality:
            print(f"   [{row[0]:8.5f}, {row[1]:8.5f}]")

        # 验证状态转移的正确性
        # 随机生成一个初始状态，验证F * x = x_next
        test_state = np.array([100.0, 50.0, segment.speed, 0.0])
        model = CTMotionModel(dt=segment.timestamps[1] - segment.timestamps[0])

        # 使用F矩阵计算
        next_state_F = F @ test_state

        # 使用状态转移函数计算
        next_state_func = model.state_transition(test_state, segment.turn_rate)

        # 比较误差
        error = np.linalg.norm(next_state_F - next_state_func)
        print(f"\n   Verification (F*x vs state_transition):")
        print(f"   Error: {error:.10f} m (should be ~0)")

        if error < 1e-6:
            print(f"   ✓ Verification PASSED")
        else:
            print(f"   ✗ Verification FAILED (error too large)")

    print("\n" + "=" * 70)


def export_trajectory_data(
        states: np.ndarray,
        transition_matrices: np.ndarray,
        timestamps: np.ndarray,
        turn_rates: List[float],
        speeds: List[float],
        filename: str = "trajectory_data.npz"
):
    """
    导出轨迹数据为NPZ文件（用于机器学习训练）

    Args:
        states: (N, 4) 状态数组
        transition_matrices: (N-1, 4, 4) 状态转移矩阵
        timestamps: (N,) 时间戳
        turn_rates: 转弯率列表
        speeds: 速度列表
        filename: 输出文件名
    """
    np.savez(
        filename,
        states=states,
        transition_matrices=transition_matrices,
        timestamps=timestamps,
        turn_rates=np.array(turn_rates),
        speeds=np.array(speeds)
    )

    print(f"\n✓ 轨迹数据已导出到: {filename}")
    print(f"  - states: {states.shape}")
    print(f"  - transition_matrices: {transition_matrices.shape}")
    print(f"  - timestamps: {timestamps.shape}")
    print(f"  - turn_rates: {len(turn_rates)} segments")
    print(f"  - speeds: {len(speeds)} segments")


def load_trajectory_data(filename: str):
    """
    从NPZ文件加载轨迹数据

    Args:
        filename: 输入文件名

    Returns:
        数据字典
    """
    data = np.load(filename)

    print(f"\n✓ 从 {filename} 加载轨迹数据:")
    print(f"  - states: {data['states'].shape}")
    print(f"  - transition_matrices: {data['transition_matrices'].shape}")
    print(f"  - timestamps: {data['timestamps'].shape}")
    print(f"  - turn_rates: {data['turn_rates'].shape}")
    print(f"  - speeds: {data['speeds'].shape}")

    return data


def main():
    """主函数：演示改进版轨迹生成"""

    print("=" * 80)
    print(" " * 20 + "改进版二维平面匀速转弯轨迹生成器")
    print(" " * 15 + "Improved Constant Turn (CT) Model Trajectory Generator")
    print("=" * 80)

    # 设置随机种子
    np.random.seed(42)

    # ========== 示例1：单段轨迹（含状态转移矩阵） ==========
    print("\n" + "▶" * 40)
    print("[示例1] 生成单段匀速转弯轨迹（含状态转移矩阵F）")
    print("▶" * 40)

    initial_state = np.array([0.0, 0.0, 15.0, 0.0])  # [x, y, vx, vy]
    turn_rate = 0.2  # rad/s (逆时针)
    duration = 30.0  # 秒
    dt = 0.1  # 采样间隔

    segment = generate_ct_trajectory(initial_state, turn_rate, duration, dt)

    print(f"\n📊 轨迹参数:")
    print(f"  • 转弯率 (ω):     {turn_rate} rad/s")
    print(f"  • 持续时间:       {duration} s")
    print(f"  • 采样间隔:       {dt} s")
    print(f"  • 速度:           {segment.speed:.2f} m/s")

    print(f"\n📦 生成数据:")
    print(f"  • 状态点数:       {len(segment.states)}")
    print(f"  • 状态维度:       {segment.states.shape}")
    print(f"  • 转移矩阵数量:   {len(segment.transition_matrices)}")
    print(f"  • 转移矩阵维度:   {segment.transition_matrices.shape}")

    print(f"\n🔍 状态转移矩阵 F (第一个):")
    F_sample = segment.transition_matrices[0]
    print("     [x_k+1]       [x_k  ]")
    print("     [y_k+1]   = F [y_k  ]")
    print("     [vx_k+1]      [vx_k ]")
    print("     [vy_k+1]      [vy_k ]")
    print()
    for i, row in enumerate(F_sample):
        row_label = ['x', 'y', 'vx', 'vy'][i]
        print(f"  {row_label:3s} [{row[0]:7.4f}, {row[1]:7.4f}, {row[2]:7.4f}, {row[3]:7.4f}]")

    # 分析转移矩阵
    analyze_transition_matrices([segment])

    # 可视化单段轨迹
    plot_trajectory_v2(
        segment.states,
        segment.timestamps,
        [segment.turn_rate],
        [segment.speed],
        [segment],
        title="Single Segment CT Trajectory (with Transition Matrix F)"
    )

    # ========== 示例2：多段随机轨迹（随机速度和转弯率） ==========
    print("\n" + "▶" * 40)
    print("[示例2] 生成多段随机转弯率和速度轨迹")
    print("▶" * 40)

    total_duration = 120.0  # 总时长120秒
    n_segments = 6  # 6个段
    turn_rate_range = (-0.3, 0.3)  # 转弯率范围
    speed_range = (10.0, 20.0)  # 速度范围 (m/s)
    min_segment_duration = 15.0  # 最小段时长

    states, transition_matrices, timestamps, turn_rates, speeds, segments = \
        generate_multi_segment_trajectory(
            total_duration=total_duration,
            n_segments=n_segments,
            turn_rate_range=turn_rate_range,
            speed_range=speed_range,
            dt=dt,
            min_segment_duration=min_segment_duration
        )

    print(f"\n📊 轨迹参数:")
    print(f"  • 总时长:         {total_duration} s")
    print(f"  • 段数:           {n_segments}")
    print(f"  • 转弯率范围:     [{turn_rate_range[0]}, {turn_rate_range[1]}] rad/s")
    print(f"  • 速度范围:       [{speed_range[0]}, {speed_range[1]}] m/s")
    print(f"  • 最小段时长:     {min_segment_duration} s")

    print(f"\n📦 生成数据:")
    print(f"  • 总状态点数:     {len(states)}")
    print(f"  • 状态维度:       {states.shape}")
    print(f"  • 转移矩阵总数:   {len(transition_matrices)}")
    print(f"  • 转移矩阵维度:   {transition_matrices.shape}")

    print(f"\n📋 各段详细信息:")
    print(f"  {'Seg':^5} | {'Duration(s)':^12} | {'Speed(m/s)':^12} | {'Turn Rate(rad/s)':^18} | {'Points':^7}")
    print(f"  {'-' * 5}-+-{'-' * 12}-+-{'-' * 12}-+-{'-' * 18}-+-{'-' * 7}")
    for i, seg in enumerate(segments):
        print(
            f"  {i + 1:^5} | {seg.duration:^12.2f} | {seg.speed:^12.2f} | {seg.turn_rate:^18.6f} | {len(seg.states):^7}")

    print(f"\n📊 转弯率列表: {turn_rates}")
    print(f"📊 速度列表:   {speeds}")

    # 分析部分转移矩阵
    print(f"\n🔍 分析前3个段的状态转移矩阵:")
    analyze_transition_matrices(segments[:3])

    # 可视化多段轨迹
    plot_trajectory_v2(
        states,
        timestamps,
        turn_rates,
        speeds,
        segments,
        title="Multi-Segment Random CT Trajectory (Random Speed & Turn Rate)"
    )

    # 导出数据
    export_trajectory_data(
        states,
        transition_matrices,
        timestamps,
        turn_rates,
        speeds,
        filename="multi_segment_trajectory.npz"
    )

    # ========== 示例3：复杂场景（更多段，更大范围） ==========
    print("\n" + "▶" * 40)
    print("[示例3] 生成复杂多段轨迹（10段，大范围）")
    print("▶" * 40)

    total_duration = 180.0
    n_segments = 10
    turn_rate_range = (-0.5, 0.5)
    speed_range = (8.0, 25.0)
    min_segment_duration = 10.0

    states, transition_matrices, timestamps, turn_rates, speeds, segments = \
        generate_multi_segment_trajectory(
            total_duration=total_duration,
            n_segments=n_segments,
            turn_rate_range=turn_rate_range,
            speed_range=speed_range,
            dt=dt,
            min_segment_duration=min_segment_duration
        )

    print(f"\n📊 轨迹参数:")
    print(f"  • 总时长:         {total_duration} s")
    print(f"  • 段数:           {n_segments}")
    print(f"  • 转弯率范围:     [{turn_rate_range[0]}, {turn_rate_range[1]}] rad/s")
    print(f"  • 速度范围:       [{speed_range[0]}, {speed_range[1]}] m/s")

    print(f"\n📦 生成数据:")
    print(f"  • 总状态点数:     {len(states)}")
    print(f"  • 转移矩阵总数:   {len(transition_matrices)}")

    print(f"\n📋 各段信息:")
    for i, (omega, v, seg) in enumerate(zip(turn_rates, speeds, segments)):
        print(f"  S{i + 1:2d}: t={seg.duration:5.1f}s | v={v:5.1f} m/s | ω={omega:+7.4f} rad/s")

    # 可视化复杂轨迹
    plot_trajectory_v2(
        states,
        timestamps,
        turn_rates,
        speeds,
        segments,
        title="Complex Multi-Segment CT Trajectory (10 Segments, Large Range)"
    )

    # 导出数据
    export_trajectory_data(
        states,
        transition_matrices,
        timestamps,
        turn_rates,
        speeds,
        filename="complex_trajectory.npz"
    )

    # ========== 示例4：演示数据加载 ==========
    print("\n" + "▶" * 40)
    print("[示例4] 演示轨迹数据加载")
    print("▶" * 40)

    # 加载刚才保存的数据
    loaded_data = load_trajectory_data("complex_trajectory.npz")

    print(f"\n✓ 数据加载成功！可以用于训练:")
    print(f"  • 训练数据 X: states[:-1] shape = {loaded_data['states'][:-1].shape}")
    print(f"  • 训练标签 Y: states[1:] shape = {loaded_data['states'][1:].shape}")
    print(f"  • 转移矩阵 F: transition_matrices shape = {loaded_data['transition_matrices'].shape}")

    # 验证数据完整性
    print(f"\n🔍 数据完整性验证:")
    print(f"  • 状态点数 N = {len(loaded_data['states'])}")
    print(
        f"  • 转移矩阵数 = {len(loaded_data['transition_matrices'])} (应该等于 N-1 = {len(loaded_data['states']) - 1})")
    print(f"  • 时间戳数 = {len(loaded_data['timestamps'])} (应该等于 N = {len(loaded_data['states'])})")

    if len(loaded_data['transition_matrices']) == len(loaded_data['states']) - 1:
        print(f"  ✓ 转移矩阵数量正确")
    else:
        print(f"  ✗ 转移矩阵数量不匹配")

    # ========== 总结 ==========
    print("\n" + "=" * 80)
    print("✓ 所有示例运行完成！")
    print("=" * 80)

    print("\n📝 输出文件:")
    print("  • multi_segment_trajectory.npz  - 示例2的轨迹数据")
    print("  • complex_trajectory.npz        - 示例3的轨迹数据")

    print("\n🎓 训练数据使用说明:")
    print("  1. 状态序列: states[:-1] → states[1:]")
    print("  2. 转移矩阵: transition_matrices[i] 对应 states[i] → states[i+1]")
    print("  3. 可用于验证神经网络学习的状态转移函数")
    print("  4. 转弯率和速度标签可用于分类或回归任务")

    print("\n📊 数据结构:")
    print("  states:              (N, 4)        - [x, y, vx, vy]")
    print("  transition_matrices: (N-1, 4, 4)   - F matrices")
    print("  timestamps:          (N,)          - time stamps")
    print("  turn_rates:          (n_segments,) - omega for each segment")
    print("  speeds:              (n_segments,) - speed for each segment")

    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    # 运行主函数
    main()

