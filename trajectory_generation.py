"""
trajectory_generation.py
调用 generate_trajectory.py 生成无人机和目标的轨迹
"""
import env_config  # 统一加载环境配置
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# 导入原始的轨迹生成函数（不修改）
from generate_trajectory import (
    generate_ct_trajectory,
    generate_multi_segment_trajectory,
    TrajectorySegment
)

from scene_configuration import SceneConfiguration, AgentConfig


@dataclass
class AgentTrajectory:
    """智能体轨迹数据"""
    agent_id: int
    agent_type: str  # 'drone' 或 'target'

    # 2D轨迹数据（来自generate_trajectory.py）
    states_2d: np.ndarray  # (N, 4) - [x, y, vx, vy]
    timestamps: np.ndarray  # (N,)

    # 高度信息（固定）
    altitude: float

    # 3D位置（方便后续使用）
    positions_3d: np.ndarray  # (N, 3) - [x, y, z]

    # 轨迹段信息
    segments: List[TrajectorySegment]
    turn_rates: List[float]
    speeds: List[float]

    # 配置信息
    config: AgentConfig


def generate_all_trajectories(scene: SceneConfiguration) -> Tuple[Dict, Dict]:
    """
    生成场景中所有智能体的轨迹

    Args:
        scene: 场景配置对象

    Returns:
        drone_trajectories: {drone_id: AgentTrajectory}
        target_trajectories: {target_id: AgentTrajectory}
    """
    print("\n" + "=" * 70)
    print("生成轨迹数据")
    print("=" * 70)

    drone_trajectories = {}
    target_trajectories = {}

    # 生成无人机轨迹（单段）
    print("\n生成无人机轨迹...")
    for drone_config in scene.drones:
        traj = _generate_drone_trajectory(drone_config, scene.dt)
        drone_trajectories[drone_config.agent_id] = traj
        print(f"  无人机 {drone_config.agent_id}: "
              f"{len(traj.states_2d)}个点, 高度={traj.altitude:.1f}m")

    # 生成目标轨迹（多段）
    print("\n生成目标轨迹...")
    for target_config in scene.targets:
        traj = _generate_target_trajectory(target_config, scene.duration, scene.dt)
        target_trajectories[target_config.agent_id] = traj
        print(f"  目标 {target_config.agent_id}: "
              f"{len(traj.states_2d)}个点, {len(traj.segments)}段, 高度={traj.altitude:.1f}m")

    print("\n✓ 轨迹生成完成")
    return drone_trajectories, target_trajectories


def _generate_drone_trajectory(config: AgentConfig, dt: float,
                               area_size: Optional[Tuple[float, float]] = None) -> AgentTrajectory:
    """生成单个无人机轨迹（单段）"""

    # 调用原始函数生成单段轨迹
    segment = generate_ct_trajectory(
        initial_state=config.initial_state_2d,
        turn_rate=config.turn_rate,
        duration=config.duration,
        dt=dt,
        area_size=area_size
    )

    # 构建3D位置（添加高度）
    positions_2d = segment.states[:, :2]
    n_points = len(positions_2d)
    positions_3d = np.column_stack([
        positions_2d,
        np.full(n_points, config.altitude)
    ])

    return AgentTrajectory(
        agent_id=config.agent_id,
        agent_type='drone',
        states_2d=segment.states,
        timestamps=segment.timestamps,
        altitude=config.altitude,
        positions_3d=positions_3d,
        segments=[segment],
        turn_rates=[segment.turn_rate],
        speeds=[segment.speed],
        config=config
    )


def _generate_target_trajectory(config: AgentConfig,
                                total_duration: float,
                                dt: float, area_size: Optional[Tuple[float, float]] = None
                                ) -> AgentTrajectory:
    """生成单个目标轨迹（多段）"""

    # 调用原始函数生成多段轨迹
    states, transition_matrices, timestamps, turn_rates, speeds, segments = \
        generate_multi_segment_trajectory(
            total_duration=total_duration,
            n_segments=config.n_segments,
            turn_rate_range=config.turn_rate_range,
            speed_range=config.speed_range,
            initial_state=config.initial_state_2d,
            dt=dt,
            min_segment_duration=config.min_segment_duration,
            area_size=area_size
        )

    # 构建3D位置（地面，高度=0）
    positions_2d = states[:, :2]
    n_points = len(positions_2d)
    positions_3d = np.column_stack([
        positions_2d,
        np.full(n_points, config.altitude)  # 应该是0
    ])

    return AgentTrajectory(
        agent_id=config.agent_id,
        agent_type='target',
        states_2d=states,
        timestamps=timestamps,
        altitude=config.altitude,
        positions_3d=positions_3d,
        segments=segments,
        turn_rates=turn_rates,
        speeds=speeds,
        config=config
    )


# ===== 兼容层：为 main_swarm.py 提供 TrajectoryGenerator 类 =====

class TrajectoryGenerator:
    """
    TrajectoryGenerator 兼容类
    包装 generate_all_trajectories 函数以适配 main_swarm.py 的接口
    """

    def __init__(self, dt: float = 0.1, duration: float = 200.0):
        """
        初始化轨迹生成器

        Args:
            dt: 时间步长（秒）
            duration: 总时长（秒）
        """
        self.dt = dt
        self.duration = duration
        self.timestamps = np.arange(0, duration, dt)

    def generate_drone_trajectory(self,
                                  start_position: np.ndarray,
                                  altitude: float,
                                  speed_range: Tuple[float, float],
                                  area_size: Tuple[float, float, float],
                                  obstacles: List) -> AgentTrajectory:
        """
        生成单个无人机轨迹

        Args:
            start_position: 起始位置 [x, y]
            altitude: 飞行高度
            speed_range: 速度范围
            area_size: 区域大小
            obstacles: 障碍物列表

        Returns:
            AgentTrajectory 对象
        """
        # 创建临时的无人机配置
        speed = np.random.uniform(speed_range[0], speed_range[1])
        heading = np.random.uniform(0, 2 * np.pi)
        vx = speed * np.cos(heading)
        vy = speed * np.sin(heading)

        initial_state_2d = np.array([
            start_position[0],
            start_position[1],
            vx,
            vy
        ])

        turn_rate = np.random.choice([-1, 1]) * np.random.uniform(0.08, 0.25)

        config = AgentConfig(
            agent_id=0,
            agent_type='drone',
            initial_state_2d=initial_state_2d,
            altitude=altitude,
            is_single_segment=True,
            turn_rate=turn_rate,
            speed=speed,
            duration=self.duration
        )

        # 提取 2D 边界 (因为 area_size 传进来是 x, y, z)
        area_size_2d = (area_size[0], area_size[1])

        # 把 area_size_2d 传进去
        return _generate_drone_trajectory(config, self.dt, area_size=area_size_2d)

    def generate_target_trajectory(self,
                                   start_position: np.ndarray,
                                   speed_range: Tuple[float, float],
                                   area_size: Tuple[float, float, float],
                                   obstacles: List) -> AgentTrajectory:
        """
        生成单个目标轨迹

        Args:
            start_position: 起始位置 [x, y]
            speed_range: 速度范围
            area_size: 区域大小
            obstacles: 障碍物列表

        Returns:
            AgentTrajectory 对象
        """
        # 创建临时的目标配置
        speed_0 = np.random.uniform(speed_range[0], speed_range[1])
        heading_0 = np.random.uniform(0, 2 * np.pi)
        vx0 = speed_0 * np.cos(heading_0)
        vy0 = speed_0 * np.sin(heading_0)

        initial_state_2d = np.array([
            start_position[0],
            start_position[1],
            vx0,
            vy0
        ])

        n_segments = np.random.randint(4, 10)

        config = AgentConfig(
            agent_id=0,
            agent_type='target',
            initial_state_2d=initial_state_2d,
            altitude=0.0,  # 地面
            is_single_segment=False,
            n_segments=n_segments,
            turn_rate_range=(-0.2, 0.2),
            speed_range=speed_range,
            min_segment_duration=self.duration / (n_segments * 2),
            P_tx=20.0,
            G_tx=0.0
        )

        # 提取 2D 边界
        area_size_2d = (area_size[0], area_size[1])

        # 把 area_size_2d 传进去
        return _generate_target_trajectory(config, self.duration, self.dt, area_size=area_size_2d)


# 为了完全兼容，也提供一个简化的 Trajectory 类别名
class Trajectory:
    """简化的 Trajectory 类，映射到 AgentTrajectory"""

    def __init__(self, agent_traj: AgentTrajectory):
        self.timestamps = agent_traj.timestamps
        self.positions_3d = agent_traj.positions_3d
        self.velocities = agent_traj.states_2d[:, 2:4]  # vx, vy

        # 计算航向角
        self.headings = np.arctan2(
            agent_traj.states_2d[:, 3],  # vy
            agent_traj.states_2d[:, 2]  # vx
        )
