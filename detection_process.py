"""
detection_process.py
无人机对目标的射频监测过程
充分考虑建筑物遮挡、LOS/NLOS条件、信号强度等因素
"""

import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple
from dataclasses import dataclass

from trajectory_generation import AgentTrajectory
from scene_configuration import ObstacleConfig


@dataclass
class Detection:
    """单次监测数据"""
    timestamp: float
    drone_id: int
    target_id: int

    # 3D位置（2D + 高度）
    drone_position: np.ndarray  # [x, y, z]
    target_position: np.ndarray  # [x, y, z]

    # 测量值
    RSSI: float  # 接收信号强度 (dBm)
    distance: float  # TOA测距 (m)
    azimuth: float  # 方位角 (rad)
    elevation: float  # 俯仰角 (rad)

    # 标志
    is_LOS: bool  # 是否视距
    is_valid: bool  # 是否有效监测
    is_NLOS_detected: bool  # 是否检测到NLOS

    # 不确定性
    distance_std: float
    azimuth_std: float
    elevation_std: float

    # 真实值（用于评估）
    true_distance: float
    true_azimuth: float
    true_elevation: float


# ==================== 几何判断函数 ====================

def point_inside_building(pos: np.ndarray, building: ObstacleConfig) -> bool:
    """
    判断点是否在建筑体内（轴对齐长方体）

    Args:
        pos: [x, y, z] 位置
        building: 建筑物配置

    Returns:
        是否在建筑内
    """
    x, y, z = pos

    return (building.x_min <= x <= building.x_max and
            building.y_min <= y <= building.y_max and
            0.0 <= z <= building.height)


def is_inside_any_building(pos: np.ndarray, buildings: List[ObstacleConfig]) -> bool:
    """
    判断点是否在任意建筑内

    Args:
        pos: [x, y, z] 位置
        buildings: 建筑物列表

    Returns:
        是否在任意建筑内
    """
    for building in buildings:
        if point_inside_building(pos, building):
            return True
    return False


def segment_intersects_aabb(p1: np.ndarray, p2: np.ndarray,
                            building: ObstacleConfig) -> bool:
    """
    判断线段与轴对齐包围盒（建筑体）是否相交
    使用 Liang-Barsky / Slab 算法

    Args:
        p1: 起点 [x, y, z]
        p2: 终点 [x, y, z]
        building: 建筑物配置

    Returns:
        是否相交
    """
    xmin, xmax = building.x_min, building.x_max
    ymin, ymax = building.y_min, building.y_max
    zmin, zmax = 0.0, building.height

    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    d = p2 - p1

    tmin, tmax = 0.0, 1.0

    def update_slab(p, dp, lo, hi, tmin, tmax):
        """更新slab"""
        if abs(dp) < 1e-12:
            # 与该轴平行
            if p < lo or p > hi:
                return None, None
            return tmin, tmax

        t1 = (lo - p) / dp
        t2 = (hi - p) / dp
        t_enter = min(t1, t2)
        t_exit = max(t1, t2)

        tmin = max(tmin, t_enter)
        tmax = min(tmax, t_exit)

        if tmax < tmin:
            return None, None
        return tmin, tmax

    # X slab
    tmin, tmax = update_slab(p1[0], d[0], xmin, xmax, tmin, tmax)
    if tmin is None:
        return False

    # Y slab
    tmin, tmax = update_slab(p1[1], d[1], ymin, ymax, tmin, tmax)
    if tmin is None:
        return False

    # Z slab
    tmin, tmax = update_slab(p1[2], d[2], zmin, zmax, tmin, tmax)
    if tmin is None:
        return False

    return tmax >= tmin


def check_LOS(drone_pos: np.ndarray, target_pos: np.ndarray,
              buildings: List[ObstacleConfig]) -> bool:
    """
    检查无人机和目标之间是否视距（LOS）

    Args:
        drone_pos: 无人机位置 [x, y, z]
        target_pos: 目标位置 [x, y, z]
        buildings: 建筑物列表

    Returns:
        是否视距
    """
    # 如果任一端点在建筑内，则NLOS
    if is_inside_any_building(drone_pos, buildings) or \
            is_inside_any_building(target_pos, buildings):
        return False

    # 检查是否被任何建筑物遮挡
    for building in buildings:
        if segment_intersects_aabb(drone_pos, target_pos, building):
            return False

    return True


# ==================== 射频传播模型 ====================

def compute_path_loss(distance: float, is_los: bool,
                      frequency: float = 2.4e9,
                      shadowing_std: float = 4.0) -> float:
    """
    计算路径损耗

    Args:
        distance: 距离 (m)
        is_los: 是否视距
        frequency: 频率 (Hz)
        shadowing_std: 阴影衰落标准差 (dB)

    Returns:
        路径损耗 (dB)
    """
    d = max(float(distance), 1.0)
    f = frequency

    # 自由空间路径损耗
    PL_fs = 20 * np.log10(d) + 20 * np.log10(f) - 147.55

    if is_los:
        # LOS: 自由空间 + 阴影衰落
        X_shadow = np.random.normal(0, shadowing_std)
        PL = PL_fs + X_shadow
    else:
        # NLOS: 额外损耗
        excess_dist = max(0.0, 10 * np.log10(d / 100.0))
        PL_excess = 20 + excess_dist + np.random.normal(15, 8)
        PL = PL_fs + PL_excess

    return PL


# ==================== 监测过程类 ====================

class DetectionProcess:
    """监测过程类"""

    def __init__(self,
                 noise_params: Dict = None,
                 # 兼容参数（用于main_swarm.py）
                 tx_power: float = None,
                 sensitivity: float = None,
                 frequency: float = None):
        """
        初始化监测过程

        Args:
            noise_params: 噪声参数字典
            tx_power: 发射功率 (dBm) - 兼容参数
            sensitivity: 接收灵敏度 (dBm) - 兼容参数
            frequency: 载波频率 (Hz) - 兼容参数
        """
        default_noise = {
            'rssi_std': 3.0,
            'toa_std_los': 1e-9,
            'toa_std_nlos': 5e-9,
            'toa_excess_min': 20e-9,
            'toa_excess_max': 200e-9,
            'aoa_std_los': 5.0,  # degrees
            'aoa_std_nlos': 15.0,  # degrees
            'shadowing_std': 4.0,
        }

        self.noise_params = {**default_noise, **(noise_params or {})}

        # 兼容参数（如果提供）
        self.tx_power = tx_power if tx_power is not None else 20.0
        self.sensitivity = sensitivity if sensitivity is not None else -95.0
        self.frequency = frequency if frequency is not None else 2.4e9

        # 角度噪声转弧度
        self.aoa_std_los_rad = self.noise_params['aoa_std_los'] * np.pi / 180
        self.aoa_std_nlos_rad = self.noise_params['aoa_std_nlos'] * np.pi / 180

    def generate_detections(self,
                            drone_trajectories: Dict[int, AgentTrajectory],
                            target_trajectories: Dict[int, AgentTrajectory],
                            obstacles: List[ObstacleConfig],
                            timestamps: np.ndarray) -> Dict[int, List[Detection]]:
        """
        生成所有监测数据

        Args:
            drone_trajectories: 无人机轨迹字典 {drone_id: AgentTrajectory}
            target_trajectories: 目标轨迹字典 {target_id: AgentTrajectory}
            obstacles: 障碍物列表
            timestamps: 时间戳数组

        Returns:
            detections_by_target: {target_id: [Detection, ...]}
        """
        print("\n" + "=" * 70)
        print("生成监测数据")
        print("=" * 70)

        detections_by_target = {tid: [] for tid in target_trajectories.keys()}

        # ✅ 修正：使用实际轨迹的长度，而不是外部传入的 timestamps
        # 找出最短的轨迹长度
        min_length = min(
            min(len(traj.timestamps) for traj in drone_trajectories.values()),
            min(len(traj.timestamps) for traj in target_trajectories.values())
        )

        # 使用最短长度来避免越界
        n_timestamps = min(len(timestamps), min_length)

        print(f"  时间戳数组长度: {len(timestamps)}")
        print(f"  实际使用长度: {n_timestamps}")

        n_detections_total = 0

        for step_idx in range(n_timestamps):  # ✅ 使用 range 而不是 enumerate
            t = timestamps[step_idx]

            if step_idx % 500 == 0:
                print(f"  进度: {step_idx}/{n_timestamps} "
                      f"(检测数: {n_detections_total})")

            # 对每个目标
            for target_id, target_traj in target_trajectories.items():
                # ✅ 添加边界检查
                if step_idx >= len(target_traj.positions_3d):
                    continue

                target_pos = target_traj.positions_3d[step_idx]
                target_config = target_traj.config

                # 对每个无人机
                for drone_id, drone_traj in drone_trajectories.items():
                    # ✅ 添加边界检查
                    if step_idx >= len(drone_traj.positions_3d):
                        continue

                    drone_pos = drone_traj.positions_3d[step_idx]
                    drone_config = drone_traj.config

                    # 检查是否在建筑内
                    if is_inside_any_building(drone_pos, obstacles) or \
                            is_inside_any_building(target_pos, obstacles):
                        continue

                    # 计算真实距离
                    d_true = np.linalg.norm(drone_pos - target_pos)

                    # 检查是否超出探测范围
                    if d_true > drone_config.sensor_range:
                        continue

                    # LOS判定
                    geometric_LOS = check_LOS(drone_pos, target_pos, obstacles)
                    P_LOS = 0.95 if geometric_LOS else 0.0
                    LOS_flag = np.random.rand() < P_LOS

                    # 生成RSSI
                    PL = compute_path_loss(
                        d_true,
                        LOS_flag,
                        getattr(target_config, 'frequency', self.frequency),
                        self.noise_params['shadowing_std']
                    )

                    P_rx = (target_config.P_tx +
                            target_config.G_tx +
                            drone_config.G_rx -
                            PL)

                    RSSI = P_rx + np.random.normal(0, self.noise_params['rssi_std'])

                    # 可见性检测（低于灵敏度则无法检测）
                    if RSSI < drone_config.sensitivity:
                        continue

                    # 生成TOA测距
                    if LOS_flag:
                        tau_excess = 0
                        sigma_toa = self.noise_params['toa_std_los']
                    else:
                        tau_excess = np.random.uniform(
                            self.noise_params['toa_excess_min'],
                            self.noise_params['toa_excess_max']
                        )
                        sigma_toa = self.noise_params['toa_std_nlos']

                    tau = d_true / 3e8 + tau_excess + np.random.normal(0, sigma_toa)
                    d_meas = tau * 3e8

                    # 生成AOA
                    delta = target_pos - drone_pos
                    azi_true = np.arctan2(delta[1], delta[0])
                    ele_true = np.arctan2(-delta[2], np.linalg.norm(delta[:2]))

                    sigma_aoa = self.aoa_std_los_rad if LOS_flag else self.aoa_std_nlos_rad
                    azi_meas = azi_true + np.random.normal(0, sigma_aoa)
                    ele_meas = ele_true + np.random.normal(0, sigma_aoa)

                    # NLOS检测（基于距离误差）
                    NLOS_detected = False
                    if not LOS_flag and abs(d_meas - d_true) > 10:
                        NLOS_detected = True

                    # 记录监测
                    detection = Detection(
                        timestamp=t,
                        drone_id=drone_id,
                        target_id=target_id,
                        drone_position=drone_pos.copy(),
                        target_position=target_pos.copy(),
                        RSSI=RSSI,
                        distance=d_meas,
                        azimuth=azi_meas,
                        elevation=ele_meas,
                        is_LOS=LOS_flag,
                        is_valid=True,
                        is_NLOS_detected=NLOS_detected,
                        distance_std=sigma_toa * 3e8,
                        azimuth_std=sigma_aoa,
                        elevation_std=sigma_aoa,
                        true_distance=d_true,
                        true_azimuth=azi_true,
                        true_elevation=ele_true
                    )

                    detections_by_target[target_id].append(detection)
                    n_detections_total += 1

        print(f"\n✓ 监测数据生成完成")
        print(f"  总监测数: {n_detections_total}")

        # 统计每个目标的监测数
        for target_id, detections in detections_by_target.items():
            n_los = sum(1 for d in detections if d.is_LOS)
            n_nlos = len(detections) - n_los
            print(f"  目标 {target_id}: {len(detections)}次监测 "
                  f"(LOS: {n_los}, NLOS: {n_nlos})")

        return detections_by_target


# ==================== 改进融合检测类 ====================

class FusedDetectionProcess(DetectionProcess):
    """
    Improved multi-drone detection fusion that produces a single virtual
    Detection per time-step per target.

    Fusion improvements over the naive RSSI-weighted average used elsewhere:

    1. LOS/NLOS gating
       NLOS measurements carry a systematic positive range bias (excess delay
       20-200 ns -> 6-60 m) and much larger AOA error (15° vs 5°).  They are
       down-weighted by `nlos_penalty` (default 0.05) instead of being treated
       equally to LOS measurements.  When a LOS measurement is available in
       the same time-step, NLOS contributions are suppressed even further.

    2. Geometry-aware position uncertainty
       True position uncertainty from AOA measurement is
           σ_xy = d_horiz * tan(σ_aoa)
       The optimal fusion weight is 1/σ_xy², which equals
           w_geom = 1 / (d_horiz * σ_aoa)²
       This is combined with the RSSI-based reliability signal:
           w_final = w_rssi_linear * w_geom * los_flag_factor

    3. Outlier rejection (NLOS bias correction attempt)
       After computing a preliminary weighted centroid, measurements whose
       reconstructed position deviates more than `outlier_sigma` standard
       deviations from the centroid are discarded and the centroid is
       recomputed.  This removes gross NLOS outliers when enough LOS peers
       are present.

    4. Uncertainty propagation to output Detection fields
       The fused Detection carries meaningful distance_std, azimuth_std, and
       elevation_std fields derived from the actual spread of per-drone
       position estimates, rather than single-drone noise parameters.

    Output format
    -------------
    Identical to DetectionProcess.generate_detections():
        Dict[target_id, List[Detection]]
    One Detection object per (time-step, target) pair where at least one
    drone had a valid measurement.  All Detection fields are populated so
    that downstream code (phys_pipeline_test, train_diffusion_phys, etc.)
    can use it as a drop-in replacement.
    """

    # AOA standard deviations (rad) – mirror DetectionProcess defaults
    _AOA_STD_LOS_DEG  = 5.0
    _AOA_STD_NLOS_DEG = 15.0
    _AOA_STD_LOS_RAD  = _AOA_STD_LOS_DEG  * np.pi / 180.0   # ~0.0873 rad
    _AOA_STD_NLOS_RAD = _AOA_STD_NLOS_DEG * np.pi / 180.0   # ~0.2618 rad

    def __init__(self,
                 noise_params: Dict = None,
                 tx_power: float = None,
                 sensitivity: float = None,
                 frequency: float = None,
                 nlos_penalty: float = 0.05,
                 outlier_sigma: float = 2.5,
                 min_los_for_nlos_suppress: int = 1):
        """
        Parameters
        ----------
        nlos_penalty : float
            Additional weight multiplier for NLOS measurements (0-1).
            Default 0.05 means NLOS contributes at most 5% compared with a
            same-distance LOS measurement.
        outlier_sigma : float
            Measurements whose position estimate is more than this many
            standard deviations from the preliminary centroid are rejected.
            Set to np.inf to disable outlier rejection.
        min_los_for_nlos_suppress : int
            Minimum number of LOS measurements required in a time-step before
            NLOS measurements are suppressed entirely (weight set to zero).
        """
        super().__init__(
            noise_params=noise_params,
            tx_power=tx_power,
            sensitivity=sensitivity,
            frequency=frequency,
        )
        self.nlos_penalty              = nlos_penalty
        self.outlier_sigma             = outlier_sigma
        self.min_los_for_nlos_suppress = min_los_for_nlos_suppress

    # ------------------------------------------------------------------
    # Public API – same signature as DetectionProcess.generate_detections
    # ------------------------------------------------------------------
    def generate_detections(self,
                            drone_trajectories: Dict,
                            target_trajectories: Dict,
                            obstacles: List,
                            timestamps: np.ndarray) -> Dict:
        """
        First collect per-drone raw detections (via the parent class), then
        fuse them per time-step into a single virtual Detection per target.

        Returns
        -------
        Dict[target_id, List[Detection]]
            One fused Detection per time-step where coverage exists.
        """
        # ── Step 1: collect raw per-drone detections ──────────────────
        raw_by_target = super().generate_detections(
            drone_trajectories=drone_trajectories,
            target_trajectories=target_trajectories,
            obstacles=obstacles,
            timestamps=timestamps,
        )

        # ── Step 2: fuse per time-step ────────────────────────────────
        fused_by_target: Dict[int, List[Detection]] = {
            tid: [] for tid in raw_by_target
        }

        dt = timestamps[1] - timestamps[0] if len(timestamps) > 1 else 0.1

        for target_id, raw_dets in raw_by_target.items():
            # Group raw detections by time-step index
            step_map: Dict[int, List[Detection]] = defaultdict(list)
            for det in raw_dets:
                si = int(round(det.timestamp / dt))
                step_map[si].append(det)

            for si in sorted(step_map.keys()):
                dets = step_map[si]
                fused = self._fuse_step(dets, timestamps[si] if si < len(timestamps)
                                        else si * dt)
                if fused is not None:
                    fused_by_target[target_id].append(fused)

        # ── Step 3: report ────────────────────────────────────────────
        print("\n[FusedDetectionProcess] Fusion summary:")
        for tid, fused_dets in fused_by_target.items():
            raw_count = len(raw_by_target[tid])
            print(f"  Target {tid}: {raw_count} raw detections -> "
                  f"{len(fused_dets)} fused time-steps")

        return fused_by_target

    # ------------------------------------------------------------------
    # Internal: fuse a list of same-step Detection objects into one
    # ------------------------------------------------------------------
    def _fuse_step(self, dets: List[Detection], timestamp: float):
        """
        Fuse all per-drone detections at one time-step into a single
        virtual Detection.

        Returns None if no measurements survive quality filtering.
        """
        if not dets:
            return None

        n_los = sum(1 for d in dets if d.is_LOS)

        # ── Reconstruct per-drone 2D position estimates ───────────────
        pos_list   = []   # reconstructed [x, y]
        w_list     = []   # fusion weights
        rssi_list  = []
        dist_list  = []
        los_flags  = []

        for det in dets:
            dx, dy, dz = det.drone_position
            azi = det.azimuth
            ele = det.elevation
            d   = det.distance

            # Horizontal projected distance
            d_horiz = max(d * np.cos(ele), 1.0)

            # 2D position reconstruction
            x_est = dx + d_horiz * np.cos(azi)
            y_est = dy + d_horiz * np.sin(azi)

            # ── Weight components ──────────────────────────────────────
            # 1. RSSI linear power (∝ 1/d² for free-space)
            w_rssi = max(10 ** (det.RSSI / 10.0), 1e-6)

            # 2. Geometry: optimal weight ∝ 1/(d_horiz * σ_aoa)²
            sigma_aoa = (self._AOA_STD_LOS_RAD if det.is_LOS
                         else self._AOA_STD_NLOS_RAD)
            w_geom = 1.0 / max((d_horiz * sigma_aoa) ** 2, 1.0)

            # 3. LOS/NLOS factor
            if det.is_LOS:
                los_factor = 1.0
            else:
                # Suppress NLOS entirely when enough LOS peers exist
                if n_los >= self.min_los_for_nlos_suppress:
                    los_factor = 0.0
                else:
                    los_factor = self.nlos_penalty

            w_final = w_rssi * w_geom * los_factor

            pos_list.append([x_est, y_est])
            w_list.append(w_final)
            rssi_list.append(det.RSSI)
            dist_list.append(det.distance)
            los_flags.append(det.is_LOS)

        pos_arr = np.array(pos_list)    # (K, 2)
        w_arr   = np.array(w_list)      # (K,)

        # If all weights are zero (all NLOS and LOS suppression active),
        # fall back to uniform NLOS weights with nlos_penalty
        if w_arr.sum() < 1e-12:
            w_arr = np.array([
                max(10 ** (d.RSSI / 10.0), 1e-6) * self.nlos_penalty
                for d in dets
            ], dtype=float)

        w_arr = w_arr / w_arr.sum()

        # ── Preliminary centroid ──────────────────────────────────────
        centroid = (w_arr[:, None] * pos_arr).sum(axis=0)

        # ── Outlier rejection ─────────────────────────────────────────
        if len(dets) >= 3 and np.isfinite(self.outlier_sigma):
            deviations = np.linalg.norm(pos_arr - centroid, axis=1)
            weighted_std = np.sqrt(
                (w_arr * deviations ** 2).sum()
            )
            keep = deviations <= self.outlier_sigma * max(weighted_std, 1.0)
            if keep.sum() >= 1:
                w_arr_filt = w_arr[keep] / w_arr[keep].sum()
                centroid   = (w_arr_filt[:, None] * pos_arr[keep]).sum(axis=0)
                # Restrict metadata to surviving measurements
                active_idx = np.where(keep)[0]
            else:
                active_idx = np.arange(len(dets))
        else:
            active_idx = np.arange(len(dets))

        # ── Build fused Detection metadata ────────────────────────────
        # Use the original Detection that has the highest weight as the
        # "representative" for drone_position, true_* fields, etc.
        best_idx   = active_idx[
            np.argmax(w_list[i] for i in active_idx)
            if hasattr(active_idx, '__len__') else 0
        ]
        # Simpler: pick highest-weight among survivors
        weights_active = np.array([w_list[i] for i in active_idx])
        best_local     = int(np.argmax(weights_active))
        best_det       = dets[active_idx[best_local]]

        # Fused RSSI: weighted average in dB domain
        rssi_fused = float(np.average(
            [dets[i].RSSI for i in active_idx],
            weights=weights_active / weights_active.sum()
        ))

        # Fused distance: derive from reconstructed horizontal distance
        # using best_det's drone altitude for consistency
        fused_x, fused_y = float(centroid[0]), float(centroid[1])
        dx, dy, dz = best_det.drone_position
        d_horiz_fused = np.sqrt((fused_x - dx) ** 2 + (fused_y - dy) ** 2)
        d_fused       = np.sqrt(d_horiz_fused ** 2 + dz ** 2)

        # Fused azimuth / elevation from centroid position
        delta_x = fused_x - dx
        delta_y = fused_y - dy
        azi_fused = float(np.arctan2(delta_y, delta_x))
        ele_fused = float(np.arctan2(-dz, max(d_horiz_fused, 1.0)))

        # Uncertainty estimate: spread of active position estimates
        pos_active = pos_arr[active_idx]
        spread     = np.linalg.norm(pos_active - centroid, axis=1)
        pos_std    = float(spread.std()) if len(spread) > 1 else float(spread[0])

        # Propagate to angle std: σ_aoa_fused ≈ pos_std / d_horiz_fused
        aoa_std_fused = float(pos_std / max(d_horiz_fused, 1.0))

        # is_LOS: True if any surviving measurement is LOS
        is_fused_los = any(dets[i].is_LOS for i in active_idx)

        # Create a dummy target position [fused_x, fused_y, 0]
        target_pos_fused = np.array([fused_x, fused_y, 0.0])

        fused_det = Detection(
            timestamp         = timestamp,
            drone_id          = -1,           # virtual fused sensor
            target_id         = best_det.target_id,
            drone_position    = best_det.drone_position.copy(),
            target_position   = target_pos_fused,
            RSSI              = rssi_fused,
            distance          = d_fused,
            azimuth           = azi_fused,
            elevation         = ele_fused,
            is_LOS            = is_fused_los,
            is_valid          = True,
            is_NLOS_detected  = not is_fused_los,
            distance_std      = pos_std,       # repurposed: 2D position spread (m)
            azimuth_std       = aoa_std_fused,
            elevation_std     = aoa_std_fused,
            true_distance     = best_det.true_distance,
            true_azimuth      = best_det.true_azimuth,
            true_elevation    = best_det.true_elevation,
        )
        return fused_det

