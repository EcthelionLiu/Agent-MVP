from dataclasses import dataclass, field
import numpy as np
from typing import Optional

@dataclass
class Pipeline:
    length_km: float = 10.0
    diameter_m: float = 0.5
    roughness: float = 1e-5
    pressure_in: float = 50.0
    pressure_out: float = 48.0
    temperature_c: float = 20.0
    flow_sm3_h: float = 30000.0
    corrosion: float = 0.0  # 管道腐蚀程度，0.0表示没有腐蚀，1.0表示完全腐蚀
    temperature_fluctuation: float = 0.0  # 温度波动，单位为°C

    def step(self, q_in: float, soil_shift: float = 0.0, pin: Optional[float] = None):
        if pin is not None:
            self.pressure_in = pin
        self.flow_sm3_h = q_in

        # 根据腐蚀和温度波动计算管道压力损失
        dp = 0.0005 * (q_in / 10000) ** 2 * (self.length_km / self.diameter_m)
        
        # 温度波动对压力的影响（简单模型：温度波动会影响压降）
        temperature_effect = 1 + (self.temperature_fluctuation / 100)  # 每1°C波动影响1%
        dp *= temperature_effect

        # 管道腐蚀影响：腐蚀越严重，压降越大
        dp *= (1 + self.corrosion)

        self.pressure_out = max(0.0, self.pressure_in - dp - soil_shift)

        leak_risk = dp > 3.0 or abs(soil_shift) > 1.0
        return {
            "pressure_out": self.pressure_out,
            "dp": dp,
            "leak_risk": leak_risk
        }
