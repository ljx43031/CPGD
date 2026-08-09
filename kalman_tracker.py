"""
kalman_tracker.py
基于恒定速度(CV)模型的卡尔曼滤波器，用于目标跟踪
适用于二维平面上的目标定位跟踪
"""
import env_config  # 统一加载环境配置
import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass

from localization_process import LocalizationResult


@dataclass
class TrackingResult:
    """跟踪结果"""
    timestamp: float
    target_id: int

    # 滤波后的状态 [x, y, vx, vy]
    filtered_state: np.ndarray

    # 滤波后的位置和速度
    filtered_position_2d: np.ndarray  # [x, y]
    filtered_velocity_2d: np.ndarray  # [vx, vy]
    filtered_speed: float  # sqrt(vx^2 + vy^2)

    # 状态协方差矩阵 (4x4)
    state_covariance: np.ndarray

    # 位置协方差 (2x2)
    position_covariance: np.ndarray

    # 真实位置（用于评估）
    true_position_3d: np.ndarray

    # 跟踪误差
    tracking_error_2d: float

    # 原始观测
    observation_position_2d: np.ndarray  # 原始定位结果
    observation_error_2d: float  # 原始定位误差

    # 改进效果
    improvement: float  # 相比原始定位的改进百分比


class KalmanTracker:
    """
    恒定速度(CV)模型的卡尔曼滤波器

    状态向量: x = [x, y, vx, vy]^T
    - (x, y): 位置
    - (vx, vy): 速度

    观测向量: z = [x, y]^T
    - 只观测位置
    """

    def __init__(self,
                 process_noise_std: float = 1.0,
                 initial_velocity_std: float = 10.0):
        """
        初始化卡尔曼滤波器

        Args:
            process_noise_std: 过程噪声标准差 (m/s^2)
            initial_velocity_std: 初始速度不确定性 (m/s)
        """
        self.process_noise_std = process_noise_std
        self.initial_velocity_std = initial_velocity_std

        # 状态维度和观测维度
        self.state_dim = 4  # [x, y, vx, vy]
        self.obs_dim = 2  # [x, y]

        # 滤波器状态
        self.is_initialized = False
        self.state = None  # 状态向量 (4,)
        self.covariance = None  # 状态协方差 (4, 4)
        self.last_update_time = None

    def reset(self):
        """重置滤波器"""
        self.is_initialized = False
        self.state = None
        self.covariance = None
        self.last_update_time = None

    def initialize(self,
                   initial_position: np.ndarray,
                   initial_covariance: np.ndarray,
                   timestamp: float):
        """
        初始化滤波器

        Args:
            initial_position: 初始位置 [x, y]
            initial_covariance: 初始位置协方差 (2, 2)
            timestamp: 时间戳
        """
        # 初始状态：位置来自观测，速度初始化为0
        self.state = np.array([
            initial_position[0],
            initial_position[1],
            0.0,  # vx
            0.0  # vy
        ])

        # 初始协方差
        # 位置不确定性来自观测，速度不确定性较大
        self.covariance = np.zeros((4, 4))
        self.covariance[:2, :2] = initial_covariance
        self.covariance[2, 2] = self.initial_velocity_std ** 2
        self.covariance[3, 3] = self.initial_velocity_std ** 2

        self.last_update_time = timestamp
        self.is_initialized = True

    def predict(self, dt: float):
        """
        预测步骤

        Args:
            dt: 时间间隔 (秒)
        """
        if not self.is_initialized:
            return

        # 状态转移矩阵 (恒定速度模型)
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # 预测状态
        self.state = F @ self.state

        # 过程噪声协方差矩阵 Q
        # 使用连续白噪声加速度模型
        q = self.process_noise_std ** 2
        Q = np.array([
            [dt ** 4 / 4, 0, dt ** 3 / 2, 0],
            [0, dt ** 4 / 4, 0, dt ** 3 / 2],
            [dt ** 3 / 2, 0, dt ** 2, 0],
            [0, dt ** 3 / 2, 0, dt ** 2]
        ]) * q

        # 预测协方差
        self.covariance = F @ self.covariance @ F.T + Q

    def update(self,
               observation: np.ndarray,
               observation_covariance: np.ndarray):
        """
        更新步骤

        Args:
            observation: 观测位置 [x, y]
            observation_covariance: 观测协方差 (2, 2)
        """
        if not self.is_initialized:
            return

        # 观测矩阵
        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        # 观测噪声协方差
        R = observation_covariance

        # 创新(innovation)
        innovation = observation - H @ self.state

        # 创新协方差
        S = H @ self.covariance @ H.T + R

        # 卡尔曼增益
        try:
            K = self.covariance @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # 如果S不可逆，使用伪逆
            K = self.covariance @ H.T @ np.linalg.pinv(S)

        # 更新状态
        self.state = self.state + K @ innovation

        # 更新协方差
        I = np.eye(self.state_dim)
        self.covariance = (I - K @ H) @ self.covariance

        # 确保协方差对称
        self.covariance = (self.covariance + self.covariance.T) / 2

    def get_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取当前状态

        Returns:
            (state, covariance)
        """
        return self.state.copy(), self.covariance.copy()

    def get_position(self) -> np.ndarray:
        """获取当前位置估计"""
        return self.state[:2].copy()

    def get_velocity(self) -> np.ndarray:
        """获取当前速度估计"""
        return self.state[2:].copy()

    def get_position_covariance(self) -> np.ndarray:
        """获取位置协方差"""
        return self.covariance[:2, :2].copy()


class MultiTargetTracker:
    """多目标跟踪器"""

    def __init__(self,
                 process_noise_std: float = 1.0,
                 initial_velocity_std: float = 10.0):
        """
        初始化多目标跟踪器

        Args:
            process_noise_std: 过程噪声标准差
            initial_velocity_std: 初始速度不确定性
        """
        self.process_noise_std = process_noise_std
        self.initial_velocity_std = initial_velocity_std

        # 为每个目标创建一个跟踪器
        self.trackers = {}  # {target_id: KalmanTracker}

    def track_all_targets(self,
                          localization_results: dict) -> dict:
        """
        对所有目标进行卡尔曼滤波跟踪

        Args:
            localization_results: {target_id: [LocalizationResult, ...]}

        Returns:
            tracking_results: {target_id: [TrackingResult, ...]}
        """
        print("\n" + "=" * 70)
        print("执行卡尔曼滤波跟踪")
        print("=" * 70)

        tracking_results = {}

        for target_id, loc_results in localization_results.items():
            if len(loc_results) == 0:
                continue

            print(f"\n跟踪目标 {target_id}...")

            # 按时间排序
            loc_results_sorted = sorted(loc_results, key=lambda r: r.timestamp)

            # 创建跟踪器
            tracker = KalmanTracker(
                process_noise_std=self.process_noise_std,
                initial_velocity_std=self.initial_velocity_std
            )

            # 跟踪结果列表
            track_results = []

            for i, loc_result in enumerate(loc_results_sorted):
                if i == 0:
                    # 初始化
                    tracker.initialize(
                        initial_position=loc_result.estimated_position_2d,
                        initial_covariance=loc_result.covariance,
                        timestamp=loc_result.timestamp
                    )
                else:
                    # 计算时间间隔
                    dt = loc_result.timestamp - tracker.last_update_time

                    # 预测
                    tracker.predict(dt)

                # 更新
                tracker.update(
                    observation=loc_result.estimated_position_2d,
                    observation_covariance=loc_result.covariance
                )

                tracker.last_update_time = loc_result.timestamp

                # 保存跟踪结果
                state, cov = tracker.get_state()
                filtered_pos = tracker.get_position()
                filtered_vel = tracker.get_velocity()
                filtered_speed = np.linalg.norm(filtered_vel)
                pos_cov = tracker.get_position_covariance()

                # 计算跟踪误差
                tracking_error = np.linalg.norm(
                    filtered_pos - loc_result.true_position_3d[:2]
                )

                # 计算改进效果
                improvement = (loc_result.error_2d - tracking_error) / loc_result.error_2d * 100

                track_result = TrackingResult(
                    timestamp=loc_result.timestamp,
                    target_id=target_id,
                    filtered_state=state,
                    filtered_position_2d=filtered_pos,
                    filtered_velocity_2d=filtered_vel,
                    filtered_speed=filtered_speed,
                    state_covariance=cov,
                    position_covariance=pos_cov,
                    true_position_3d=loc_result.true_position_3d,
                    tracking_error_2d=tracking_error,
                    observation_position_2d=loc_result.estimated_position_2d,
                    observation_error_2d=loc_result.error_2d,
                    improvement=improvement
                )

                track_results.append(track_result)

            tracking_results[target_id] = track_results

            # 统计信息
            if len(track_results) > 0:
                tracking_errors = [r.tracking_error_2d for r in track_results]
                observation_errors = [r.observation_error_2d for r in track_results]
                improvements = [r.improvement for r in track_results]

                print(f"  跟踪点数: {len(track_results)}")
                print(f"  原始定位平均误差: {np.mean(observation_errors):.2f} m")
                print(f"  跟踪平均误差: {np.mean(tracking_errors):.2f} m")
                print(f"  平均改进: {np.mean(improvements):.1f}%")
                print(f"  最大改进: {np.max(improvements):.1f}%")

        return tracking_results


def compare_tracking_performance(localization_results: dict,
                                 tracking_results: dict) -> dict:
    """
    比较跟踪前后的性能

    Args:
        localization_results: 原始定位结果
        tracking_results: 跟踪结果

    Returns:
        comparison: 性能对比统计
    """
    comparison = {}

    for target_id in tracking_results.keys():
        loc_results = localization_results[target_id]
        track_results = tracking_results[target_id]

        # 提取误差
        loc_errors = np.array([r.error_2d for r in loc_results])
        track_errors = np.array([r.tracking_error_2d for r in track_results])

        # 统计
        comparison[target_id] = {
            'n_points': len(track_results),
            'localization': {
                'mean_error': np.mean(loc_errors),
                'median_error': np.median(loc_errors),
                'std_error': np.std(loc_errors),
                'min_error': np.min(loc_errors),
                'max_error': np.max(loc_errors),
                'percentile_90': np.percentile(loc_errors, 90),
                'percentile_95': np.percentile(loc_errors, 95)
            },
            'tracking': {
                'mean_error': np.mean(track_errors),
                'median_error': np.median(track_errors),
                'std_error': np.std(track_errors),
                'min_error': np.min(track_errors),
                'max_error': np.max(track_errors),
                'percentile_90': np.percentile(track_errors, 90),
                'percentile_95': np.percentile(track_errors, 95)
            },
            'improvement': {
                'mean_error_reduction': np.mean(loc_errors) - np.mean(track_errors),
                'mean_error_reduction_pct': (np.mean(loc_errors) - np.mean(track_errors)) / np.mean(loc_errors) * 100,
                'median_error_reduction': np.median(loc_errors) - np.median(track_errors),
                'std_error_reduction': np.std(loc_errors) - np.std(track_errors)
            }
        }

    return comparison
