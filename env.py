from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from components.pipeline import Pipeline
from components.station import Station, Compressor
from components.soil import Soil

FLOW_TABLE = [0, 10000, 20000, 30000]
DT_HOURS = 5/60

class InjectorEnv:
    def __init__(self, seed: Optional[int] = None, power_unit: float = 2.0):
        self.power_unit = power_unit
        self.rng = np.random.default_rng(seed)
        self.pipeline = Pipeline()
        self.station = Station([Compressor(on=True) for _ in range(2)] +
                               [Compressor(on=False) for _ in range(3)])
        self.soil = Soil(stability=0.8)
        self.n_well = 5
        self.time_idx = 0
        self.reset()
        # 记录上一轮的动作，初始默认“全部关井，压缩机保持”
        self.last_inj  = [0]*self.n_well
        self.last_prod = [0]*self.n_well
        self.last_comp = [2]*5

    def reset(self)->Dict[str,Any]:
        self.time_idx = 0
        self.P_well = np.full(self.n_well, 45.0)
        return self._observe([{"type":"reset"}])

    @staticmethod
    def _norm(lst, n):
        if isinstance(lst, (int, float)):
            return [int(lst)]*n
        if len(lst) != n:
            raise ValueError("bad len")
        return list(lst)

    def step(self, action: Dict[str, Any]) -> Tuple[Dict, float, bool, Dict]:
        self.time_idx += 1

        inj = self._norm(action.get("inj_level", self.last_inj), self.n_well)
        prod = self._norm(action.get("prod_level", self.last_prod), self.n_well)
        comp = self._norm(action.get("compressor_cmd", self.last_comp), 5)

        # 更新每个压缩机的开关状态
        for j, cmd in enumerate(comp):
            if cmd in (0, 1):
                self.station.compressors[j].on = bool(cmd)

        q_in_tot, q_out_tot = 0, 0
        for i in range(self.n_well):
            q_in_tot += FLOW_TABLE[inj[i]]
            q_out_tot += FLOW_TABLE[prod[i]]
            alpha = 1e-3  # bar /(m³/h)
            dq = FLOW_TABLE[inj[i]] - FLOW_TABLE[prod[i]]
            dP = alpha * dq - 0.01 * self.P_well[i]  # 根据每个井口的注入和生产量差异调整压力变化
            self.P_well[i] += dP  # 每个井口的压力变化应该不同

        # 更新土壤扰动和管道压力
        soil_shift = self.soil.random_shift(self.rng)
        pin = self.station.outlet_pressure(pin=50.0)
        outlet_stats = self.pipeline.step(q_in_tot - q_out_tot, soil_shift, pin=pin)

        # 强制开启压缩机以避免泄漏
        if outlet_stats.get("leak_risk"):
            self._cool = 3

        if getattr(self, "_cool", 0) > 0:
            self.station.compressors[0].on = True
            self.station.compressors[1].on = True
            self._cool -= 1

        # 计算奖励
        press_dev = float(np.mean(np.abs(self.P_well - 50)))  # 根据所有井口的压力计算偏差
        energy = self.station.energy_use()

        leak_penalty = 8.0 if outlet_stats["leak_risk"] else 0.0
        reward = -press_dev - 0.05 * energy - leak_penalty
        if press_dev < 1.0 and energy <= 10:
            reward += 1.0

        done = False

        # 返回更新后的观测结果
        self.last_inj, self.last_prod, self.last_comp = inj, prod, comp
        obs = self._observe([])  # 更新观测信息
        info = {"press_dev": press_dev, "energy_use": energy, **outlet_stats}
        return obs, reward, done, info

    def _observe(self, events:List)->Dict[str,Any]:
        return {
            "t":self.time_idx,
            "P_well":self.P_well.round(2).tolist(),
            "station_energy":self.station.energy_use(),
            "pipeline_out":self.pipeline.pressure_out,
            "events":events
        }

    def apply_action(self,payload):
        return self.step(payload)
    
    def apply_plan(self,payload):
        payload.setdefault("inj_level",[0]*self.n_well)
        payload.setdefault("prod_level",[0]*self.n_well)
        return self.step(payload)
