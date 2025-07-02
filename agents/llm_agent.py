import json
import logging
import os
from typing import Any, Dict, List

try:
    import openai
except ImportError:
    openai = None

class LLMAgent:
    def __init__(self, model: str = "gpt-4o", temperature: float = 0.0):
        """
        初始化LLM代理
        :param model: 使用的模型名称
        :param temperature: 模型温度
        """
        self.model = model
        self.system_prompt = """ 
        你是PipelineMaintenance-Agent。你的目标是确保管道和站点在最佳工作条件下运行。
        你需要监控管道的潜在问题，如泄漏、压力偏差和压缩机故障。
        目标是保持管道压力和气体供应在最佳水平，确保压缩机正常运行，并处理如泄漏或气量不足等故障。
        你将从一个计划开始。该计划是动态的，意味着每次决策后，你必须根据结果反思，并决定是否需要更新计划。
        当发生意外事件或异常时，计划应更新。请据此做出决策。

        **计划示例**: "确保管道压力保持在可接受范围内，压缩机正常运行"
        
        **示例：**  
        观察:
        {"leak_risk": false, "compressor_faults": [false], "insufficient_gas": false, "pipe_corrosion": 0.02, "temperature_fluctuation": 1.5}
        → 
        {"compressor_cmd":[1]}  # 如果压缩机未启动，则启动压缩机

        观察:
        {"leak_risk": true, "compressor_faults": [false], "insufficient_gas": false, "pipe_corrosion": 0.05, "temperature_fluctuation": -3.0}
        →
        {"compressor_cmd":[0]}  # 如果检测到泄漏，则关闭压缩机并调整流量
        """
    
    def _chat(self, messages):
        if openai and os.getenv("OPENAI_API_KEY"):
            resp = openai.chat.completions.create(
                model=self.model,
                messages=messages,
            )
            return resp.choices[0].message.content
        return json.dumps({"compressor_cmd": [1]})  # 默认操作返回

    def update_plan(self, current_plan: str, obs: Dict[str, Any]) -> str:
        """
        根据当前观察更新计划
        :param current_plan: 当前的计划
        :param obs: 当前的环境观测
        :return: 更新后的计划
        """
        if obs.get('leak_risk', False):
            return "检测到管道泄漏，启动压缩机以维持压力。"
        if any(fault for fault in obs.get("compressor_faults", [])):
            return "检测到压缩机故障，进行故障排查并修复。"
        if obs.get("insufficient_gas", False):
            return "场站气量不足，调整管道流量。"
        if obs.get("pipe_corrosion", 0) > 0.1:
            return "检测到管道腐蚀，减少压力并进行维护。"
        if abs(obs.get("temperature_fluctuation", 0)) > 2:
            return "检测到温度波动，调整压缩机或压力。"
        return current_plan

    def decide(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        基于当前观测进行决策
        :param obs: 当前的环境观测
        :return: 代理的行动决策
        """
        # 消息列表，包含系统和用户的交互
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"观察:\n{json.dumps(obs)}"}
        ]

        # 添加反馈信息，帮助代理在多步决策中调整计划
        judge_feedback = obs.get('judge_feedback')
        if judge_feedback:
            messages.append(
                {"role": "user", "content": f"评估反馈:\n{json.dumps(judge_feedback)}"}
            )

        # 更新当前计划
        plan = "确保管道压力保持在可接受范围内，压缩机正常运行。"  # 初始计划
        plan = self.update_plan(plan, obs)  # 更新计划
        messages.append({"role": "system", "content": f"计划: {plan}"})

        raw = self._chat(messages).strip()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logging.error("LLM生成的JSON格式错误: %s", raw)
            return {"compressor_cmd": [1]}  # 默认启动压缩机

        try:
            if not (isinstance(data["compressor_cmd"], list) and len(data["compressor_cmd"]) == 1):
                raise ValueError
            if not all(isinstance(x, int) and 0 <= x <= 1 for x in data["compressor_cmd"]):
                raise ValueError
        except (KeyError, ValueError):
            logging.error("LLM返回的JSON字段无效: %s", data)
            return {"compressor_cmd": [1]}  # 默认返回压缩机操作

        return data

