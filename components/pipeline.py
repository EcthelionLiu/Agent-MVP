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

    def step(self, q_in: float, soil_shift: float = 0.0, pin: Optional[float] = None):
        # 让 Station 可以传入新的入口压
        if pin is not None:          
            self.pressure_in = pin
        self.flow_sm3_h = q_in

        dp = 0.0005 * (q_in / 10000) ** 2 * (self.length_km / self.diameter_m)
        self.pressure_out = max(0.0, self.pressure_in - dp - soil_shift)

        leak_risk = dp > 3.0 or abs(soil_shift) > 1.0
        return {
            "pressure_out": self.pressure_out,
            "dp": dp,
            "leak_risk": leak_risk
        }
