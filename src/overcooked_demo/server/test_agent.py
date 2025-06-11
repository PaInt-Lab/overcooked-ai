# # test_agent.py

# from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
# from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
# from llm.agents.action_predictor_agent import ActionPredictorAgent

# def main():
#     # 1. Initialize MDP and Env
#     layout = "cramped_room"
#     mdp = OvercookedGridworld.from_layout_name(layout)
#     env = OvercookedEnv.from_mdp(mdp)

#     # 2. Create agent and configure it
#     agent = ActionPredictorAgent(model_name="action_predictor_model")
#     agent.set_agent_index(1)  # robot is index 1
#     agent.set_mdp(mdp)

#     # 3. Grab the initial state
#     state = env.state
#     print("Player positions:", state.player_positions)

#     # 4. Run one action and print results
#     try:
#         move, info = agent.action(state)
#         print("High-level robot task:", info.get("high_level"))
#         print("Primitive move returned:", move)
#     except Exception as e:
#         print("Agent threw an exception:")
#         raise

# if __name__ == "__main__":
#     main()
import json
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from llm.agents.action_predictor_agent import ActionPredictorAgent

def main():
    # 1) Build the MDP
    layout = "cramped_room"
    mdp = OvercookedGridworld.from_layout_name(layout)

    # 2) Instantiate Env properly via from_mdp
    env = OvercookedEnv.from_mdp(mdp)


    # 3) Grab the initial state
    state = env.state

    # 4) Inspect the raw state
    print("=== OvercookedState.to_dict() ===")
    state_dict = state.to_dict()
    print(json.dumps(state_dict, indent=2))

    print("\n=== player_positions ===")
    print(state.player_positions)

    print("\n=== terrain matrix ===")
    print(mdp.terrain_mtx)

    # 5) Create and configure the agent
    agent = ActionPredictorAgent()
    agent.set_agent_index(1)  # robot is index 1
    agent.set_mdp(mdp)

    print("Planner methods:", [m for m in dir(agent.planner) if not m.startswith("_")])

    # 6) Run the agent once and print
    move, info = agent.action(state)
    move, info = agent.action(state)
    print("Primitive move returned:", move)
    print("High-level task was:", info['high_level'])
    print("Human task was:", info['human_task'])
    print("Planner method used:", info['planner_method'])


if __name__ == "__main__":
    main()
