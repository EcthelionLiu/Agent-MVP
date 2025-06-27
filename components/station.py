from dataclasses import dataclass
import numpy as np

@dataclass
class Compressor:
    ratio: float = 1.3
    on: bool = True

    def apply(self, pin: float):
        return pin * self.ratio if self.on else pin

@dataclass
class Station:
    compressors: list

    def outlet_pressure(self, pin: float):
        p = pin
        for comp in self.compressors:
            p = comp.apply(p)
        return p
    
    def energy_use(self): # 计算当前能量的使用
        running = [c for c in self.compressors if c.on]
        if not running:
            return 0.0
        return 5.0 + 4.0 * (len(running) - 1)
