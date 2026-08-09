"""
localization_process.py
基于监测数据的单帧目标定位过程（不含滤波跟踪）
"""
import env_config  # 统一加载环境配置
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.optimize import least_squares

from detection_process import Detection


@dataclass
class LocalizationResult:
    """单个时刻的定位结果"""
    timestamp: float
    target_id: int

    # 定位位置（2D，因为目标在地面）
    estimated_position_2d: np.ndarray  # [x, y]

    # 3D位置（添加高度=0）
    estimated_position_3d: np.ndarray  # [x, y, 0]

    # 真实位置
    true_position_3d: np.ndarray

    # 定位误差
    error_2d: float
    error_3d: float

    # 参与定位的无人机数量
    n_drones: int

    # 协方差矩阵
    covariance: np.ndarray  # (2, 2)

    # 使用的检测数据
    detections: List[Detection]


class LocalizationProcess:
    """单帧定位过程类"""

    def __init__(self, method: str = 'wls'):
        """
        初始化定位过程

        Args:
            method: 定位方法
                - 'wls': 加权最小二乘（仅TOA）
                - 'hybrid': 混合定位（TOA + AOA）
        """
        self.method = method

    def localize_all_targets(self,
                             detections_by_target: Dict[int, List[Detection]],
                             timestamps: np.ndarray) -> Dict[int, List[LocalizationResult]]:
        """
        对所有目标进行单帧定位

        Args:
            detections_by_target: {target_id: [Detection, ...]}
            timestamps: 时间戳数组

        Returns:
            results_by_target: {target_id: [LocalizationResult, ...]}
        """
        print("\n" + "=" * 70)
        print("执行单帧目标定位")
        print("=" * 70)

        results_by_target = {}

        for target_id, detections in detections_by_target.items():
            print(f"\n定位目标 {target_id}...")

            # 按时间分组
            detections_by_time = {}
            for det in detections:
                t = det.timestamp
                if t not in detections_by_time:
                    detections_by_time[t] = []
                detections_by_time[t].append(det)

            # 对每个时刻定位
            results = []
            for t in sorted(detections_by_time.keys()):
                dets = detections_by_time[t]

                if len(dets) < 3:  # 至少需要3个观测
                    continue

                result = self._localize_single_frame(dets, t)
                if result is not None:
                    results.append(result)

            results_by_target[target_id] = results

            if len(results) > 0:
                avg_error = np.mean([r.error_2d for r in results])
                median_error = np.median([r.error_2d for r in results])
                print(f"  定位次数: {len(results)}")
                print(f"  平均误差: {avg_error:.2f} m")
                print(f"  中位误差: {median_error:.2f} m")
            else:
                print(f"  警告：目标{target_id}没有成功定位")

        return results_by_target

    def _localize_single_frame(self,
                               detections: List[Detection],
                               timestamp: float) -> Optional[LocalizationResult]:
        """
        单个时刻的定位

        Args:
            detections: 该时刻的监测列表
            timestamp: 时间戳

        Returns:
            定位结果（如果成功）
        """
        if len(detections) == 0:
            return None

        target_id = detections[0].target_id

        # 提取观测数据
        drone_positions = np.array([d.drone_position for d in detections])
        distances = np.array([d.distance for d in detections])
        distance_stds = np.array([d.distance_std for d in detections])

        # 初始猜测（无人机位置的加权中心）
        weights = 1.0 / (distances + 1e-6)
        weights /= np.sum(weights)
        x0 = np.sum(drone_positions[:, :2] * weights[:, np.newaxis], axis=0)

        # 选择定位方法
        if self.method == 'wls':
            result = self._wls_localization(
                x0, drone_positions, distances, distance_stds)
        elif self.method == 'hybrid':
            azimuths = np.array([d.azimuth for d in detections])
            result = self._hybrid_localization(
                x0, drone_positions, distances, distance_stds, azimuths)
        else:
            result = self._wls_localization(
                x0, drone_positions, distances, distance_stds)

        if result is None:
            return None

        estimated_pos_2d, cov = result
        estimated_pos_3d = np.array([estimated_pos_2d[0], estimated_pos_2d[1], 0.0])

        # 真实位置
        true_pos_3d = detections[0].target_position

        # 计算误差
        error_2d = np.linalg.norm(estimated_pos_2d - true_pos_3d[:2])
        error_3d = np.linalg.norm(estimated_pos_3d - true_pos_3d)

        return LocalizationResult(
            timestamp=timestamp,
            target_id=target_id,
            estimated_position_2d=estimated_pos_2d,
            estimated_position_3d=estimated_pos_3d,
            true_position_3d=true_pos_3d,
            error_2d=error_2d,
            error_3d=error_3d,
            n_drones=len(detections),
            covariance=cov,
            detections=detections
        )

    def _wls_localization(self,
                          x0: np.ndarray,
                          drone_positions: np.ndarray,
                          distances: np.ndarray,
                          distance_stds: np.ndarray) -> Optional[Tuple]:
        """
        加权最小二乘定位（仅TOA）

        Args:
            x0: 初始猜测 [x, y]
            drone_positions: 无人机位置数组 (N, 3)
            distances: 测量距离数组 (N,)
            distance_stds: 距离标准差数组 (N,)

        Returns:
            (estimated_position, covariance) 或 None
        """

        def residuals(x):
            """残差函数"""
            pos_2d = x[:2]
            pos_3d = np.array([pos_2d[0], pos_2d[1], 0.0])

            # 预测距离
            predicted_distances = np.linalg.norm(
                drone_positions - pos_3d, axis=1)

            # 加权残差
            weights = 1.0 / (distance_stds + 1e-6)
            return weights * (distances - predicted_distances)

        # 使用Levenberg-Marquardt算法求解
        result = least_squares(residuals, x0, method='lm', max_nfev=100)

        if not result.success:
            return None

        estimated_pos_2d = result.x[:2]

        # 计算协方差（使用雅可比矩阵）
        try:
            J = result.jac
            # 协方差 = (J^T * J)^-1 * sigma^2
            cov = np.linalg.inv(J.T @ J) * np.mean(distance_stds ** 2)
        except:
            # 如果计算失败，使用大协方差
            cov = np.eye(2) * 100.0

        return estimated_pos_2d, cov

    def _hybrid_localization(self,
                             x0: np.ndarray,
                             drone_positions: np.ndarray,
                             distances: np.ndarray,
                             distance_stds: np.ndarray,
                             azimuths: np.ndarray) -> Optional[Tuple]:
        """
        混合定位（TOA + AOA）

        Args:
            x0: 初始猜测 [x, y]
            drone_positions: 无人机位置数组 (N, 3)
            distances: 测量距离数组 (N,)
            distance_stds: 距离标准差数组 (N,)
            azimuths: 方位角数组 (N,) [rad]

        Returns:
            (estimated_position, covariance) 或 None
        """

        def residuals(x):
            """混合残差函数"""
            pos_2d = x[:2]
            pos_3d = np.array([pos_2d[0], pos_2d[1], 0.0])

            # TOA残差
            predicted_distances = np.linalg.norm(
                drone_positions - pos_3d, axis=1)
            distance_weights = 1.0 / (distance_stds + 1e-6)
            distance_residuals = distance_weights * (distances - predicted_distances)

            # AOA残差
            delta = pos_3d - drone_positions
            predicted_azimuths = np.arctan2(delta[:, 1], delta[:, 0])

            # 角度差（考虑周期性）
            azimuth_diff = azimuths - predicted_azimuths
            azimuth_diff = np.arctan2(np.sin(azimuth_diff), np.cos(azimuth_diff))

            # 角度权重（相对TOA较小）
            azimuth_weight = 0.1
            azimuth_residuals = azimuth_weight * azimuth_diff

            # 合并残差
            return np.concatenate([distance_residuals, azimuth_residuals])

        result = least_squares(residuals, x0, method='lm', max_nfev=100)

        if not result.success:
            return None

        estimated_pos_2d = result.x[:2]

        try:
            J = result.jac
            cov = np.linalg.inv(J.T @ J) * np.mean(distance_stds ** 2)
        except:
            cov = np.eye(2) * 100.0

        return estimated_pos_2d, cov
