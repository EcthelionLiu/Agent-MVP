import numpy as np
from dataclasses import dataclass

@dataclass
class Soil:
    stability: float = 1.0  # 土壤稳定性，1.0表示完全稳定，0.0表示完全不稳定

    def random_shift(self, rng):
        return rng.normal(0, (1 - self.stability) * 0.25)
