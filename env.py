from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from components.pipeline import Pipeline
from components.station import Station, Compressor
from components.soil import Soil

FLOW_TABLE = [0, 10000, 20000, 30000]  # 注入流量表
DT_HOURS = 5 / 60  # 每个时间步长（小时）

class PipelineMaintenanceEnv:
    def __init__(self, seed: Optional[int] = None, power_unit: float = 2.0):
        """
        初始化管道维护仿真环境
        :param power_unit: 功率单位（用于能耗计算）
        """
        self.power_unit = power_unit
        self.rng = np.random.default_rng(seed)
        self.pipeline = Pipeline()  
        # 仅一个启动的压缩机
        self.station = Station([Compressor(on=True)]) 
        self.soil = Soil(stability=0.8)  # 土壤稳定性初始化
        self.time_idx = 0
        self.reset()

        # 记录上一轮的动作，初始默认“压缩机状态不变”
        self.last_comp = [1]

        # 故障状态初始化
        self.leak_risk = False  # 是否发生泄漏
        self.compressor_faults = [False]  # 记录压缩机是否故障
        self.insufficient_gas = False  # 场站气量不足

        # 新增故障状态初始化
        self.pipe_corrosion = 0.0  # 管道腐蚀程度
        self.temperature_fluctuation = 0.0  # 温度波动

    def reset(self) -> Dict[str, Any]:
        """
        重置环境状态
        :return: 初始状态的观测字典
        """
        self.time_idx = 0
        self.leak_risk = False
        self.compressor_faults = [False]
        self.insufficient_gas = False
        self.pipe_corrosion = 0.0
        self.temperature_fluctuation = 0.0
        return self._observe([{"type": "reset"}])

    def _norm(self, lst, n):
        """
        :param lst: 输入列表
        :param n: 目标长度
        :return: 标准化后的列表
        """
        if isinstance(lst, (int, float)):
            return [int(lst)] * n
        if len(lst) != n:
            raise ValueError("bad len")
        return list(lst)

    def step(self, action: Dict[str, Any]) -> Tuple[Dict, float, bool, Dict]:
        """
        :param action: 代理执行的动作，包括压缩机控制命令
        :return: 更新后的观测、奖励、是否结束和附加信息
        """
        self.time_idx += 1

        # 只控制一台压缩机
        comp = self._norm(action.get("compressor_cmd", self.last_comp), 1)

        # 更新压缩机的开关状态
        self.station.compressors[0].on = bool(comp[0])

        # 检查管道故障：泄漏、压缩机故障、气量不足、管道腐蚀、温度波动
        self._check_faults()

        energy = self.station.energy_use()
        leak_penalty = 8.0 if self.leak_risk else 0.0  # 泄漏惩罚
        compressor_penalty = sum([5 for fault in self.compressor_faults if fault])  # 每个故障压缩机罚款
        gas_penalty = 10.0 if self.insufficient_gas else 0.0  # 气量不足惩罚
        corrosion_penalty = 5.0 if self.pipe_corrosion > 0.5 else 0.0  # 管道腐蚀惩罚
        temperature_penalty = 3.0 if abs(self.temperature_fluctuation) > 2.0 else 0.0  # 温度波动惩罚

        reward = -0.05 * energy - leak_penalty - compressor_penalty - gas_penalty - corrosion_penalty - temperature_penalty
        if energy <= 10 and not self.leak_risk and not self.insufficient_gas:
            print("nothing wrong")
            reward += 10  # 系统状态良好时奖励

        done = False

        self.last_comp = comp
        obs = self._observe([])  
        info = {
            "energy_use": energy,
            "leak_risk": self.leak_risk,
            "compressor_faults": self.compressor_faults,
            "insufficient_gas": self.insufficient_gas,
            "pipe_corrosion": self.pipe_corrosion,
            "temperature_fluctuation": self.temperature_fluctuation
        }
        return obs, reward, done, info

    def _observe(self, events: List) -> Dict[str, Any]:
        """
        :param events: 事件列表（可扩展）
        :return: 环境观测字典
        """
        return {
            "t": self.time_idx,
            "station_energy": self.station.energy_use(),
            "pipeline_out": self.pipeline.pressure_out,
            "events": events
        }

    def _check_faults(self):
        """
        检测并模拟管道的故障情况，包括泄漏、压缩机故障、气量不足、管道腐蚀和温度波动
        """
        # 模拟管道泄漏：5% 概率发生泄漏
        self.leak_risk = np.random.rand() < 0.05

        # 模拟压缩机故障：10% 概率发生压缩机故障
        if np.random.rand() < 0.05:
            self.compressor_faults[0] = True
        else:
            self.compressor_faults[0] = False

        # 模拟气量不足：3% 概率发生气量不足
        self.insufficient_gas = np.random.rand() < 0.03

        # 模拟管道腐蚀：每个时间步管道腐蚀程度可能增加
        self.pipe_corrosion = min(self.pipe_corrosion + 0.01, 1.0)  # 腐蚀程度逐步增加，最大为 1.0

        # 模拟温度波动：随机温度波动影响管道压力
        self.temperature_fluctuation = np.random.normal(0, 2)  # 允许温度波动在 ±2°C 之间

    def apply_action(self, payload):  # 执行动作（代理的控制决策）
        """
        :param payload: 动作内容
        :return: 经过一轮操作后的观测、奖励、是否结束
        """
        return self.step(payload)

    def apply_plan(self, payload):  # 应用计划（代理的维护计划）
        """
        :param payload: 计划内容
        :return: 经过一轮操作后的观测、奖励、是否结束
        """
        payload.setdefault("compressor_cmd", [2] * 5)
        return self.step(payload)
