
"""Simple loop interleaving RL + LLM decisions."""
from agents.rl_agent import SimpleRLAgent
from agents.llm_agent import LLMAgent
from env import InjectorEnv
from reward import IdentityReward

def train(num_iter=50, llm_every=5):
    env = InjectorEnv()
    rl = SimpleRLAgent()
    llm = LLMAgent()
    rew_fn = IdentityReward()
    obs = env.reset()
    cum = 0
    for t in range(num_iter):
        if t % llm_every == 0:
            action = llm.decide(obs)
        else:
            action = rl.decide(obs)
        obs, r, done, info = env.apply_action(action)
        cum += rew_fn(r)
    return cum

if __name__ == "__main__":
    print("cumulative reward:", train())
