"""
scene_configuration.py
无人机蜂群对地面目标定位的场景配置
基于2D平面轨迹 + 高度属性
"""
import env_config  # 统一加载环境配置
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class AgentConfig:
    """智能体配置（无人机或目标）"""
    agent_id: int
    agent_type: str  # 'drone' 或 'target'

    # 2D平面初始状态 [x, y, vx, vy]
    initial_state_2d: np.ndarray

    # 高度信息（独立于2D状态）
    altitude: float  # 运动平面的高度 (m)

    # 单段或多段轨迹参数
    is_single_segment: bool  # True: 单段（无人机），False: 多段（目标）

    # 单段参数（无人机用）
    turn_rate: Optional[float] = None
    speed: Optional[float] = None
    duration: Optional[float] = None

    # 多段参数（目标用）
    n_segments: Optional[int] = None
    turn_rate_range: Optional[Tuple[float, float]] = None
    speed_range: Optional[Tuple[float, float]] = None
    min_segment_duration: Optional[float] = None

    # 射频参数（仅目标）
    P_tx: float = 20.0  # 发射功率 (dBm)
    G_tx: float = 0.0  # 天线增益 (dBi)
    frequency: float = 2.4e9  # 载波频率 (Hz) - 新增

    # 传感器参数（仅无人机）
    sensor_range: float = 500.0  # 探测范围 (m)
    G_rx: float = 5.0  # 接收天线增益 (dBi)
    sensitivity: float = -95.0  # 接收灵敏度 (dBm)


@dataclass
class ObstacleConfig:
    """障碍物配置"""
    obstacle_id: int
    obstacle_type: str  # 'building', 'wall', etc.

    # 2D平面几何（轴对齐矩形）
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    height: float  # 障碍物高度 (m)

    # 射频衰减
    penetration_loss: float = 20.0  # 穿透损耗 (dB)


class SceneConfiguration:
    """场景配置类"""

    def __init__(self,
                 area_size: Tuple[float, float] = (1000.0, 1000.0),
                 duration: float = 300.0,
                 dt: float = 0.1):
        """
        初始化场景配置

        Args:
            area_size: 监测区域大小 (width, height) in meters
            duration: 仿真时长 (秒)
            dt: 采样间隔 (秒)
        """
        self.area_size = area_size
        self.duration = duration
        self.dt = dt

        self.drones: List[AgentConfig] = []
        self.targets: List[AgentConfig] = []
        self.obstacles: List[ObstacleConfig] = []

    def add_drone_swarm(self,
                        n_drones: int,
                        altitude_range: Tuple[float, float] = (50.0, 150.0),
                        speed_range: Tuple[float, float] = (15.0, 25.0),
                        turn_rate_range: Tuple[float, float] = (0.05, 0.3),
                        coverage_pattern: str = 'random'):
        """
        添加无人机蜂群（单段长时间CT轨迹）

        Args:
            n_drones: 无人机数量
            altitude_range: 高度范围 (min, max)
            speed_range: 速度范围 (min, max)
            turn_rate_range: 转弯率范围 (min, max)
            coverage_pattern: 初始位置分布 'random', 'grid', 'circular'
        """
        print(f"\n配置无人机蜂群: {n_drones}架")

        for i in range(n_drones):
            # 随机高度（每架无人机在不同高度平面）
            altitude = np.random.uniform(altitude_range[0], altitude_range[1])

            # 2D平面初始位置（根据覆盖模式）
            if coverage_pattern == 'grid':
                rows = int(np.ceil(np.sqrt(n_drones)))
                cols = int(np.ceil(n_drones / rows))
                row = i // cols
                col = i % cols
                x0 = (col + 0.5) * self.area_size[0] / cols
                y0 = (row + 0.5) * self.area_size[1] / rows
            elif coverage_pattern == 'circular':
                angle = 2 * np.pi * i / n_drones
                radius = min(self.area_size) / 3
                x0 = self.area_size[0] / 2 + radius * np.cos(angle)
                y0 = self.area_size[1] / 2 + radius * np.sin(angle)
            else:  # random
                x0 = np.random.uniform(50, self.area_size[0] - 50)
                y0 = np.random.uniform(50, self.area_size[1] - 50)

            # 2D速度（随机方向）
            speed = np.random.uniform(speed_range[0], speed_range[1])
            heading = np.random.uniform(0, 2 * np.pi)
            vx = speed * np.cos(heading)
            vy = speed * np.sin(heading)

            # 转弯率（正负随机）
            turn_rate = np.random.choice([-1, 1]) * np.random.uniform(
                turn_rate_range[0], turn_rate_range[1])

            # 2D初始状态
            initial_state_2d = np.array([x0, y0, vx, vy])

            drone = AgentConfig(
                agent_id=i,
                agent_type='drone',
                initial_state_2d=initial_state_2d,
                altitude=altitude,
                is_single_segment=True,
                turn_rate=turn_rate,
                speed=speed,
                duration=self.duration  # 单段持续整个仿真时长
            )

            self.drones.append(drone)

            print(f"  无人机 {i}: 高度={altitude:.1f}m, "
                  f"速度={speed:.1f}m/s, 转弯率={turn_rate:.3f}rad/s")

    def add_ground_targets(self,
                           n_pedestrians: int = 3,
                           n_vehicles: int = 2,
                           n_segments_range: Tuple[int, int] = (3, 8)):
        """
        添加地面目标（行人和车辆，复杂多段轨迹）

        Args:
            n_pedestrians: 行人数量
            n_vehicles: 车辆数量
            n_segments_range: 每个目标的轨迹段数范围
        """
        print(f"\n配置地面目标: {n_pedestrians}行人 + {n_vehicles}车辆")

        target_id = 0

        # 添加行人
        for i in range(n_pedestrians):
            target = self._create_target(
                target_id=target_id,
                target_type='pedestrian',
                speed_range=(1.0, 2.5),  # 步行速度
                turn_rate_range=(0.05, 0.2),
                n_segments_range=n_segments_range
            )
            self.targets.append(target)
            target_id += 1

        # 添加车辆
        for i in range(n_vehicles):
            target = self._create_target(
                target_id=target_id,
                target_type='vehicle',
                speed_range=(5.0, 15.0),  # 车辆速度
                turn_rate_range=(0.05, 0.3),
                n_segments_range=n_segments_range
            )
            self.targets.append(target)
            target_id += 1

    def _create_target(self,
                       target_id: int,
                       target_type: str,
                       speed_range: Tuple[float, float],
                       turn_rate_range: Tuple[float, float],
                       n_segments_range: Tuple[int, int]) -> AgentConfig:
        """创建单个目标配置"""

        # 2D平面初始位置（地面，高度=0）
        x0 = np.random.uniform(50, self.area_size[0] - 50)
        y0 = np.random.uniform(50, self.area_size[1] - 50)

        # 初始速度（第一段的速度）
        speed_0 = np.random.uniform(speed_range[0], speed_range[1])
        heading_0 = np.random.uniform(0, 2 * np.pi)
        vx0 = speed_0 * np.cos(heading_0)
        vy0 = speed_0 * np.sin(heading_0)

        initial_state_2d = np.array([x0, y0, vx0, vy0])

        # 多段轨迹参数
        n_segments = np.random.randint(n_segments_range[0], n_segments_range[1] + 1)
        min_segment_duration = self.duration / (n_segments * 2)  # 确保合理

        # 射频参数
        if target_type == 'pedestrian':
            P_tx = np.random.uniform(15, 20)
            G_tx = 0.0
        else:  # vehicle
            P_tx = np.random.uniform(20, 25)
            G_tx = np.random.choice([0, 2, 3])

        target = AgentConfig(
            agent_id=target_id,
            agent_type='target',
            initial_state_2d=initial_state_2d,
            altitude=0.0,  # 地面
            is_single_segment=False,
            n_segments=n_segments,
            turn_rate_range=turn_rate_range,
            speed_range=speed_range,
            min_segment_duration=min_segment_duration,
            P_tx=P_tx,
            G_tx=G_tx
        )

        print(f"  目标 {target_id} ({target_type}): "
              f"{n_segments}段轨迹, P_tx={P_tx:.1f}dBm")

        return target

    def add_obstacles(self, obstacle_configs: Optional[List[Dict]] = None):
        """
        添加障碍物

        Args:
            obstacle_configs: 障碍物配置列表，如果为None则生成随机障碍物
        """
        if obstacle_configs is None:
            # 生成随机建筑物
            n_buildings = np.random.randint(3, 8)
            print(f"\n生成随机障碍物: {n_buildings}栋建筑")

            for i in range(n_buildings):
                width = np.random.uniform(30, 80)
                length = np.random.uniform(30, 80)
                height = np.random.uniform(20, 100)

                x_min = np.random.uniform(100, self.area_size[0] - width - 100)
                y_min = np.random.uniform(100, self.area_size[1] - length - 100)

                obstacle = ObstacleConfig(
                    obstacle_id=i,
                    obstacle_type='building',
                    x_min=x_min,
                    x_max=x_min + width,
                    y_min=y_min,
                    y_max=y_min + length,
                    height=height,
                    penetration_loss=np.random.uniform(15, 25)
                )

                self.obstacles.append(obstacle)
                print(f"  建筑 {i}: [{x_min:.0f},{x_min + width:.0f}]x"
                      f"[{y_min:.0f},{y_min + length:.0f}], H={height:.0f}m")
        else:
            print(f"\n添加指定障碍物: {len(obstacle_configs)}个")
            for i, config in enumerate(obstacle_configs):
                obstacle = ObstacleConfig(
                    obstacle_id=i,
                    obstacle_type=config.get('type', 'building'),
                    x_min=config['x_min'],
                    x_max=config['x_max'],
                    y_min=config['y_min'],
                    y_max=config['y_max'],
                    height=config['height'],
                    penetration_loss=config.get('penetration_loss', 20.0)
                )
                self.obstacles.append(obstacle)

    def print_summary(self):
        """打印配置摘要"""
        print("\n" + "=" * 70)
        print("场景配置摘要")
        print("=" * 70)
        print(f"监测区域: {self.area_size[0]}m × {self.area_size[1]}m")
        print(f"仿真时长: {self.duration}s")
        print(f"采样间隔: {self.dt}s")
        print(f"无人机数量: {len(self.drones)}")
        print(f"  - 高度范围: [{min(d.altitude for d in self.drones):.1f}, "
              f"{max(d.altitude for d in self.drones):.1f}]m")
        print(f"目标数量: {len(self.targets)}")
        print(f"  - 行人: {sum(1 for t in self.targets if 'pedestrian' in str(t.agent_id))}")
        print(f"  - 车辆: {sum(1 for t in self.targets if 'vehicle' in str(t.agent_id))}")
        print(f"障碍物数量: {len(self.obstacles)}")
        print("=" * 70)

    def get_config_dict(self) -> Dict:
        """获取配置字典"""
        return {
            'area_size': self.area_size,
            'duration': self.duration,
            'dt': self.dt,
            'drones': self.drones,
            'targets': self.targets,
            'obstacles': self.obstacles
        }


# ===== 预定义场景 =====

def create_default_scene() -> SceneConfiguration:
    """创建默认场景"""
    scene = SceneConfiguration(
        area_size=(1000.0, 1000.0),
        duration=200.0,
        dt=0.1
    )

    scene.add_drone_swarm(
        n_drones=6,
        altitude_range=(60.0, 120.0),
        speed_range=(18.0, 25.0),
        turn_rate_range=(0.08, 0.25),
        coverage_pattern='random'
    )

    scene.add_ground_targets(
        n_pedestrians=4,
        n_vehicles=3,
        n_segments_range=(4, 10)
    )

    scene.add_obstacles()

    scene.print_summary()
    return scene


def create_urban_scene() -> SceneConfiguration:
    """创建城市场景"""
    scene = SceneConfiguration(
        area_size=(800.0, 800.0),
        duration=250.0,
        dt=0.1
    )

    scene.add_drone_swarm(
        n_drones=8,
        altitude_range=(70.0, 140.0),
        speed_range=(15.0, 22.0),
        turn_rate_range=(0.1, 0.3),
        coverage_pattern='grid'
    )

    scene.add_ground_targets(
        n_pedestrians=6,
        n_vehicles=4,
        n_segments_range=(5, 12)
    )

    # 密集建筑
    obstacles = [
        {'x_min': 100, 'x_max': 180, 'y_min': 100, 'y_max': 180, 'height': 80},
        {'x_min': 250, 'x_max': 320, 'y_min': 150, 'y_max': 220, 'height': 60},
        {'x_min': 400, 'x_max': 480, 'y_min': 200, 'y_max': 280, 'height': 70},
        {'x_min': 550, 'x_max': 620, 'y_min': 300, 'y_max': 380, 'height': 90},
    ]
    scene.add_obstacles(obstacles)

    scene.print_summary()
    return scene


# ===== 兼容层：为 main_swarm.py 提供 SceneConfigurator 别名 =====

class SceneConfigurator:
    """
    SceneConfigurator 兼容类
    包装 SceneConfiguration 以适配 main_swarm.py 的接口
    """

    def __init__(self,
                 area_size: Tuple[float, float, float],
                 n_drones: int = 5,
                 n_targets: int = 2,
                 n_obstacles: int = 3):
        """
        初始化场景配置器

        Args:
            area_size: 区域大小 (x, y, z) - 注意：z会被忽略，只用(x,y)
            n_drones: 无人机数量
            n_targets: 目标数量
            n_obstacles: 障碍物数量
        """
        # 创建内部的 SceneConfiguration 对象
        self.scene = SceneConfiguration(
            area_size=(area_size[0], area_size[1]),  # 只使用x,y
            duration=200.0,  # 默认时长
            dt=0.1
        )

        self.n_drones = n_drones
        self.n_targets = n_targets
        self.n_obstacles = n_obstacles
        self.area_size = area_size

    def generate_scene(self) -> Dict:
        """
        生成场景配置

        Returns:
            包含 drones, targets, obstacles 的字典
        """
        # 添加无人机
        self.scene.add_drone_swarm(
            n_drones=self.n_drones,
            altitude_range=(50.0, 150.0),
            speed_range=(15.0, 25.0),
            turn_rate_range=(0.08, 0.25),
            coverage_pattern='random'
        )

        # 添加目标（分配为行人和车辆）
        n_pedestrians = max(1, self.n_targets // 2)
        n_vehicles = self.n_targets - n_pedestrians

        self.scene.add_ground_targets(
            n_pedestrians=n_pedestrians,
            n_vehicles=n_vehicles,
            n_segments_range=(4, 10)
        )

        # 添加障碍物
        if self.n_obstacles > 0:
            self.scene.add_obstacles()  # 生成随机障碍物

        # 返回兼容格式
        return {
            'drones': self.scene.drones,
            'targets': self.scene.targets,
            'obstacles': self.scene.obstacles
        }

    def print_configuration(self):
        """打印配置信息"""
        self.scene.print_summary()

    def _generate_random_position_2d(self) -> Tuple[float, float]:
        """
        生成随机2D位置

        Returns:
            (x, y) 坐标元组
        """
        x = np.random.uniform(50, self.area_size[0] - 50)
        y = np.random.uniform(50, self.area_size[1] - 50)
        return (x, y)