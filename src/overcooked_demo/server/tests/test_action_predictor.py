# test_action_predictor.py
import json
from plan_session import PLAN_STORE, PlanSession
from llm.agents.subtask_to_event_sequence import classify_subtasks, group_events
from llm.agents.action_predictor import ActionPredictorAgent
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState

# 1. Build the plan locally
subtasks = [
    "Fetch an onion", "Stage onion at stove",
    "Place onion on stove", "Turn stove on", "Wait for onion to cook", "Fetch a bowl", "Stage bowl at stove",
    "Pour soup into bowl", "Bring bowl to serving station"
]
tagged = classify_subtasks(subtasks)
events = group_events(tagged)

print("Events:")
for idx, ev in enumerate(events):
    print(f"{idx+1}), {idx}, primary: {ev['primary']}, secondary: {ev['secondary']}")

# 2. Store it under a fresh ID
session_id = "test-plan"
PLAN_STORE[session_id] = PlanSession(events)

# 3. Create agent & MDP
agent = ActionPredictorAgent()
mdp = OvercookedGridworld.from_layout_name("cramped_room")
agent.set_mdp(mdp)
agent.set_plan(session_id)
agent.set_agent_index(0)

# 4. Test on start state (no onion anywhere → should still be on event 1)
state = mdp.get_standard_start_state()
move, info = agent.action(state)
print("Returned event_idx:", info.get("event_idx"))
print("High-level primary :", info.get("human_task"))
print("High-level secondary:", info.get("robot_task"))