
import json, logging, os
from typing import Any, Dict, List
try:
    import openai
except ImportError:
    openai = None

class LLMAgent:
    def __init__(self, model: str = "gpt-4o", temperature: float = 0.0):
        self.model = model
        self.system_prompt = """ You are GasInjector-Agent. Each decision **must** consider each well's individual pressure.Each well must have its own inj_level and prod_level reflecting its P_well deviation.
        Objective: Keep each P_well in [48,52] bar, minimize compressor use, **and** adjust inj/prod per well.
        Return exactly:
          { "inj_level":[i1,i2,i3,i4,i5],
            "prod_level":[j1,j2,j3,j4,j5],
            "compressor_cmd":[c1,c2,c3,c4,c5] }
        No markdown or extra text.

        **Few-shot examples:**  
        Observation:
        {"P_well":[46,55,49,51,44], "press_dev":2.2, ...}
        →
        {"inj_level":[2,0,1,0,3],
         "prod_level":[0,3,1,0,0],
         "compressor_cmd":[1,1,0,1,1]}

        Observation:
        {"P_well":[48,48,48,48,48], "press_dev":0, ...}
        →
        {"inj_level":[1,1,1,1,1],
         "prod_level":[1,1,1,1,1],
         "compressor_cmd":[1,0,0,0,0]}
        """

    def _chat(self, messages):
        if openai and os.getenv("OPENAI_API_KEY"):
            resp = openai.chat.completions.create(
                model=self.model,
                messages=messages,
            )
            return resp.choices[0].message.content
        # offline stub
        return json.dumps({"inj_level":[2]*5,
                           "prod_level":[1]*5,
                           "compressor_cmd":[1,1,0,0,0]})
    
    #压力低多注气，压力高少注气
    def _rule_based(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        inj, prod = [], []
        for p in obs["P_well"]:
            if p < 48:
                inj.append(min(3, 2 + int((48 - p)//2)))
                prod.append(0)
            elif p > 52:
                inj.append(0)
                prod.append(min(3, 1 + int((p - 52)//2)))
            else:
                inj.append(1)
                prod.append(1)
        # 至少开 1 台
        comp = [1,0,0,0,0]  
        if obs.get("press_dev", 0) > 3:
            comp = [1,1,0,0,0]
        return {"inj_level": inj, "prod_level": prod, "compressor_cmd": comp}

    def decide(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        1. messages = [system, user, (optional) alert]
        2. OpenAI, 去掉 ``` 与 ```json 包裹
        3. 校验字段
        """
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": f"Observation:\n{json.dumps(obs)}"}
        ]
        
        # judge_model的内容
        judge_feedback = obs.get('judge_feedback')
        if judge_feedback:
            messages.append(
                {"role": "user", "content": f"Judge Feedback:\n{json.dumps(judge_feedback)}"}
            )
        
        if obs.get("press_dev", 0) > 3:
            messages.append(
                {"role": "user",
                 "content": "press_dev>3 bar,pressure deviation is too high, please adjust the wellhead pressure."}
            )

        raw = self._chat(messages)

        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.lstrip("`")
            raw = raw.split("\n", 1)[-1]   
        if raw.endswith("```"):
            raw = raw[:-3]

        def _safe_return():
            return self._rule_based(obs)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logging.error("LLM produced bad JSON: %s", raw)
            return _safe_return()

        try:
            for k in ("inj_level", "prod_level", "compressor_cmd"):
                if not (isinstance(data[k], list) and len(data[k]) == 5):
                    raise ValueError
            if not all(isinstance(x, int) and 0 <= x <= 3 for x in data["inj_level"]):
                raise ValueError
            if not all(isinstance(x, int) and 0 <= x <= 3 for x in data["prod_level"]):
                raise ValueError
            if not all(isinstance(x, int) and 0 <= x <= 2 for x in data["compressor_cmd"]):
                raise ValueError
        except (KeyError, ValueError):
            logging.error("LLM JSON fields invalid: %s", data)
            return _safe_return()
        
        if obs.get("press_dev", 0) > 10:
            logging.warning("press_dev too high, use rule fallback")
            return self._rule_based(obs)

        return data
