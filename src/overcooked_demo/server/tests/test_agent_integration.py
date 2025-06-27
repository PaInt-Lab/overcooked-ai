# tests/test_agent_integration.py

import json
from plan_session import PLAN_STORE, PlanSession
from llm.agents.subtask_to_event_sequence import classify_subtasks, group_events, normalize_events
from llm.agents.action_predictor import ActionPredictorAgent
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.actions import Action

def build_test_plan():
    subtasks = [
        "Fetch an Ingredient", "Stage Ingredient at stove",
        "Place Ingredient on stove", "Turn stove on", "Wait for Ingredient to cook",
        "Fetch a bowl", "Stage bowl at stove", "Pour soup into bowl",
        "Bring bowl to serving station"
    ]
    tagged = classify_subtasks(subtasks)
    events = group_events(tagged)
    # norm_events = normalize_events(events)
    session_id = "integration-test"
    PLAN_STORE[session_id] = PlanSession(events)
    return session_id

def run_integration_test():
    # 1) Setup
    mdp = OvercookedGridworld.from_layout_name("cramped_room")
    agent = ActionPredictorAgent()
    agent.set_mdp(mdp)
    session_id = build_test_plan()
    agent.set_plan(session_id)
    agent.set_agent_index(0)

    # 2) Get initial state
    state = mdp.get_standard_start_state()
    pos_history = []

    # 3) Simulate ticks until plan exhausted or max steps
    max_steps = 5
    for step in range(max_steps):
        print(f"\n=== Tick {step + 1} ===")
        move, info = agent.action(state)
        # print("Move action:", move, "Info:", info)
        pos, ori = state.player_positions[0], state.to_dict()["players"][0]["orientation"]
        plan_goal = info["robot_task"]
        # print(f"[Tick {step}] Event {info['event']} → {info['robot_task']} → Move: {move}")
        pos_history.append(pos)

        # Feed the low-level action back into the MDP
        joint_action = [move, Action.STAY]  # assume partner does nothing
        # print(f"Joint action: {joint_action}")
        # print(f"Moving from {pos} with orientation {ori} using action {move}") #Move is empty here
        state, reward = mdp.get_state_transition(state, joint_action)

        # Stop when the final task—Bring Dish to Serving Station—has been chosen
        if plan_goal == "Bring Dish to Serving Station":
            print("Final transport task chosen; ending test.")
            break
    else:
        print("Reached max steps without finishing plan.")

    # print("\nPosition trace:", pos_history)
    # print("Final state:", state, "Reward:", reward)
    # print("Final state:", state)

if __name__ == "__main__":
    run_integration_test()
