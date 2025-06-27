"""
Training-data loader for pipeline-framework.

Features
--------
1. CSV / Parquet   → pandas.DataFrame
2. Directory tail  → yield new rows as soon as files grow
3. In-memory replay buffer compatible with RL algorithms
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Deque, Dict, Generator, List, Tuple

import pandas as pd
from collections import deque


class EpisodeDataset:
    """
    Treat one DataFrame as a list of (obs, action, reward, info).

    Expected columns
    ----------------
    - obs_*      (JSON/dict 字符串或扁平列)
    - action     (JSON/dict 字符串)
    - reward
    - info_*     (可选)
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[dict, dict, float, dict]:
        row = self.df.iloc[idx]

        # -------- 反序列化 ----------
        obs_cols = {c: row[c] for c in self.df.columns if c.startswith("obs_")}
        action = row["action"]
        reward = float(row["reward"])
        info_cols = {c: row[c] for c in self.df.columns if c.startswith("info_")}

        # 如果字段本身是 JSON 字符串 → 转回 dict
        if isinstance(action, str) and action.lstrip().startswith("{"):
            action = pd.io.json.loads(action)
        if "obs_json" in obs_cols:
            obs_cols = pd.io.json.loads(obs_cols["obs_json"])

        return obs_cols, action, reward, info_cols

    # ------------------------------------------------------------
    # batching helper
    # ------------------------------------------------------------
    def batch(self, batch_size: int):
        for start in range(0, len(self), batch_size):
            yield [self[i] for i in range(start, min(len(self), start + batch_size))]


# ----------------------------------------------------------------
# Offline loader helpers
# ----------------------------------------------------------------
def load_csv(path: str | Path) -> EpisodeDataset:
    df = pd.read_csv(path)
    return EpisodeDataset(df)


def load_parquet(path: str | Path) -> EpisodeDataset:
    df = pd.read_parquet(path)
    return EpisodeDataset(df)


# ----------------------------------------------------------------
# Streaming loader – watch a directory & yield new rows incrementally
# ----------------------------------------------------------------
def stream_directory(
    dir_path: str | Path,
    pattern: str = "*.csv",
    poll_sec: float = 1.0,
) -> Generator[Dict, None, None]:
    """
    Continuously watch directory `dir_path` for CSV files.
    Yield dict rows as soon as they appear / grow.

    Example
    -------
    >>> for row in stream_directory("./logs", "*.csv"):
    ...     train_step(row)
    """
    dir_path = Path(dir_path)
    offsets: Dict[Path, int] = {}
    while True:
        for file in dir_path.glob(pattern):
            last = offsets.get(file, 0)
            size = file.stat().st_size
            if size > last:
                # read only appended part
                df = pd.read_csv(file, skiprows=range(1, last // 100))  # crude skip
                for _, row in df.iterrows():
                    yield row.to_dict()
                offsets[file] = size
        time.sleep(poll_sec)


# ----------------------------------------------------------------
# In-memory replay buffer
# ----------------------------------------------------------------
class ReplayBuffer:
    """
    Simple size-bounded buffer <obs, action, reward, next_obs, done>
    suitable for value-based / policy-gradient RL algorithms.
    """

    def __init__(self, capacity: int = 100_000):
        self.capacity = capacity
        self.buffer: Deque[Tuple] = deque(maxlen=capacity)

    def push(self, *transition):
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> List[Tuple]:
        idx = pd.np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in idx]

    def __len__(self):
        return len(self.buffer)
