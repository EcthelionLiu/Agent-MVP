"""
Components:
1. TraceRecorder
2. MetricEngine
3. JudgeLLM
4. EvaluationAggregator
"""
from __future__ import annotations
import datetime as _dt
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import openai  
except ModuleNotFoundError:  
    openai = None  

_LOG = logging.getLogger(__name__)

@dataclass
class TraceRecorder:

    keep_actions_raw: bool = True
    _entries: List[Dict[str, Any]] = field(default_factory=list, init=False)

    def log(self, observation: Dict[str, Any], action: Dict[str, Any], reward: float,
            info: Dict[str, Any] | None = None) -> None:
        ts = info.get("timestep") if info else None
        if ts is None:
            ts = len(self._entries)
        entry = {
            "t": ts,
            "P_well": observation.get("P_well"),
            "compressor_on": observation.get("compressor_on", 0),
            "reward": reward,
        }
        if self.keep_actions_raw:
            entry["action_raw"] = action
        self._entries.append(entry)

    @property
    def df(self) -> pd.DataFrame:
        return pd.DataFrame(self._entries)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for row in self._entries:
                f.write(json.dumps(row) + "\n")

@dataclass
class MetricEngine:
    target: float = 50.0
    low: float = 48.0
    high: float = 52.0
    settle_tau: int = 10
    power_per_step: float = 1.0

    def compute(self, df: pd.DataFrame) -> Dict[str, float]:
        if df.empty:
            raise ValueError("Trace is empty - nothing to evaluate.")

        P = np.stack(df["P_well"].to_numpy())
        C = df.get("compressor_on", pd.Series([0] * len(df))).to_numpy()
        T = P.shape[0]

        hit_mask = ((P >= self.low) & (P <= self.high)).all(axis=1)
        hit_rate = hit_mask.sum() / T
        mse = np.mean((P - self.target) ** 2)
        max_dev = np.max(np.abs(P - self.target))

        viol = (~hit_mask).sum()
        overshoot = max(P.max() - self.high, 0)
        undershoot = max(self.low - P.min(), 0)

        settle = T
        for t in range(T - self.settle_tau):
            if hit_mask[t:].all():
                settle = t
                break

        energy = (C * self.power_per_step).sum()
        switches = (np.diff(C) != 0).sum()

        if "action_raw" in df.columns:
            actions = df["action_raw"].to_numpy()
            diffs = [np.any(np.asarray(actions[i]) != np.asarray(actions[i-1]))
                     for i in range(1, len(actions))]
            sparsity = 1.0 - np.mean(diffs) if diffs else 1.0
        else:
            sparsity = float("nan")

        return {
            "R_hit": round(hit_rate, 4),
            "MSE": round(mse, 4),
            "Delta_max": round(max_dev, 4),
            "N_viol": int(viol),
            "Overshoot": round(overshoot, 3),
            "Undershoot": round(undershoot, 3),
            "T_settle": int(settle),
            "Energy": round(float(energy), 3),
            "N_switch": int(switches),
            "Sparsity": round(float(sparsity), 4),
        }


@dataclass
class JudgeLLM:
    model: str = "gpt-4o"
    temperature: float = 0.0
    n_consistency: int = 3
    openai_api_key: Optional[str] = None
    timeout: int = 60

    _sys_prompt: str = (
        "You are an industrial control evaluator. Judge the agent's trajectory on "
        "four aspects: (1) Goal Fulfillment, (2) Safety & Compliance, "
        "(3) Decision Efficiency, (4) Reasoning Coherence. "
        "Return ONLY valid JSON with keys: score_goal, score_safety, "
        "score_efficiency, score_reasoning, critique. Each score in [0,10]."
    )

    def _prepare_messages(self, metrics: Dict[str, float], traj_snippet: str) -> List[Dict[str, str]]:
        metrics_txt = json.dumps(metrics, indent=2)
        user_msg = f"METRICS:\n{metrics_txt}\n\nTRAJECTORY (truncated):\n{traj_snippet}"
        return [
            {"role": "system", "content": self._sys_prompt},
            {"role": "user", "content": user_msg},
        ]

    def _call_openai(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        if openai is None:
            raise RuntimeError("openai package not installed.")
        client = openai.chat.completions  
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "timeout": self.timeout,
        }
        if self.openai_api_key:
            openai.api_key = self.openai_api_key  
        resp = client.create(**kwargs)  
        content = resp.choices[0].message.content.strip()
        
        # 对齐json格式
        if content.startswith("```json"):
            content = content[7:].strip()  
        if content.endswith("```"):
            content = content[:-3].strip()  
            
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            _LOG.warning("JudgeLLM produced non-JSON output: %s", content[:120])
            return {"malformed": content}

    def evaluate(self, df: pd.DataFrame, metrics: Dict[str, float]) -> Dict[str, Any]:
        if df.empty:
            return {
                "score_goal": 5.0,
                "score_safety": 5.0,
                "score_efficiency": 5.0,
                "score_reasoning": 5.0,
                "critique": "No data available for evaluation. Default scores assigned."
            }

        first_rows = df.head(5).to_json(orient="records")
        last_rows = df.tail(5).to_json(orient="records")
        traj_snippet = first_rows + "\\n…\\n" + last_rows

        messages = self._prepare_messages(metrics, traj_snippet)
        votes: List[Dict[str, Any]] = []
        for _ in range(self.n_consistency):
            try:
                response = self._call_openai(messages)
                
                # Directly use the response (which is already a dict) without checking for JSON code block
                votes.append(response)  # Just append the dict result
            except Exception as exc:
                logging.error("Judge call failed: %s", exc, exc_info=True)

        keys = ["score_goal", "score_safety", "score_efficiency", "score_reasoning"]
        agg: Dict[str, Any] = {k: 0.0 for k in keys}
        good_votes = [v for v in votes if all(k in v for k in keys)]
        if not good_votes:
            return {"error": "All judge calls failed", "raw": votes}

        for v in good_votes:
            for k in keys:
                agg[k] += float(v[k])
        for k in keys:
            agg[k] = round(agg[k] / len(good_votes), 2)

        agg["critique"] = next((v.get("critique", "") for v in good_votes), "")
        return agg

class EvaluationAggregator:
    def combine(self, metrics: Dict[str, float], judge: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "timestamp": _dt.datetime.now().isoformat(),
            "metrics": metrics,
            "judge": judge,
        }

    def save_json(self, report: Dict[str, Any], path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)