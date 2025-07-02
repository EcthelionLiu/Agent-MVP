from dataclasses import dataclass
import numpy as np

@dataclass
class Compressor:
    ratio: float = 1.3  # 初始的压缩比
    on: bool = True
    wear: float = 0.0  # 设备磨损程度，0.0 表示没有磨损，1.0 表示完全磨损

    def apply(self, pin: float):
        # 随着磨损，压缩机的效率会下降
        effective_ratio = self.ratio * (1 - self.wear)
        return pin * effective_ratio if self.on else pin

    def wear_out(self, wear_rate: float):
        """模拟压缩机磨损，磨损程度逐步增加"""
        self.wear = min(self.wear + wear_rate, 1.0)  # 磨损程度最大为1.0

@dataclass
class Station:
    compressors: list

    def outlet_pressure(self, pin: float):
        p = pin
        for comp in self.compressors:
            p = comp.apply(p)
        return p
    
    def energy_use(self):  # 计算当前能量的使用
        running = [c for c in self.compressors if c.on]
        if not running:
            return 0.0
        return 5.0 + 4.0 * (len(running) - 1)

    def apply_wear(self, wear_rate: float):
        """逐步增加压缩机的磨损"""
        for comp in self.compressors:
            comp.wear_out(wear_rate)
