# run_env.py
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.agents.agent import RandomAgent

# 1. A zero‑arg MDP factory (the constructor wants a *callable* that returns an MDP)
mdp_fn = lambda outside_info=None: OvercookedGridworld.from_layout_name("cramped_room")

# 2. Pass that factory straight into the constructor
env = OvercookedEnv(mdp_fn, horizon=400)

# 3. One RandomAgent per player
agents = {0: RandomAgent(), 1: RandomAgent()}

# 4. Reset → returns a dict {agent_index: observation}
obs = env.reset()
done = False

# 5. Main loop
while not done:
    actions = {i: agents[i].act(obs[i]) for i in obs}
    obs, reward, done, info = env.step(actions)
    env.render()

env.close()
