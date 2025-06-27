from env import InjectorEnv
from agents.llm_agent import LLMAgent
from reward import IdentityReward
from judge_model import TraceRecorder, MetricEngine, JudgeLLM, EvaluationAggregator
import logging
import time
import sys
import json

logging.root.handlers.clear()

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
    stream=sys.stdout,
)

def main(max_steps=10):
    env = InjectorEnv()
    agent = LLMAgent()
    rew = IdentityReward()
    
    rec = TraceRecorder()
    metr = MetricEngine(power_per_step=env.power_unit)
    judge = JudgeLLM(model="gpt-4o", temperature=0)
    agg = EvaluationAggregator()

    obs = env.reset()
    cum = 0
    total_dev, total_energy = 0, 0
    for s in range(max_steps):
        # 跳过第一轮评估 因为rec.log里还没有信息
        if len(rec.df) > 0:  
            judge_feedback = judge.evaluate(rec.df, metr.compute(rec.df))
        else:
            judge_feedback = None

        obs['judge_feedback'] = judge_feedback  
        act = agent.decide(obs) 
        
        obs, r, done, info = env.apply_action(act)
        print(f"Obs: {obs}, Reward: {r}, Done: {done}, Info: {info}")
        
        rec.log(obs, act, r, info)
        cum += rew(r)
        logging.info("step=%d r=%.2f cum=%.2f info=%s", s, r, cum, info)
        total_dev += info["press_dev"]
        total_energy += info["energy_use"]
        time.sleep(0.2)

        if done:
            metrics = metr.compute(rec.df)
            judge_res = judge.evaluate(rec.df, metrics)
            report = agg.combine(metrics, judge_res)
            agg.save_json(report, "report.json")
            logging.info("Evaluation report saved to report.json")

    logging.info(f"Avg press_dev {total_dev/max_steps:.2f}  |  Avg energy {total_energy/max_steps:.2f}")
    print("Finished, cum reward", cum)

if __name__ == "__main__":
    main()

