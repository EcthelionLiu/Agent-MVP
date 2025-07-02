import numpy as np
from dataclasses import dataclass

@dataclass
class Soil:
    stability: float = 1.0  # 土壤稳定性，1.0表示完全稳定，0.0表示完全不稳定

    def random_shift(self, rng):
        return rng.normal(0, (1 - self.stability) * 0.25)  # 稳定性越低，土壤扰动越大

    def apply_stability_effect(self, stability_factor: float):
        """模拟土壤不稳定对管道的影响，稳定性越差，影响越大"""
        self.stability = max(0, min(self.stability - stability_factor, 1.0))  # 稳定性逐渐降低
