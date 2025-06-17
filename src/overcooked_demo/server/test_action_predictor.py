# test_action_predictor.py

import json
from plan_session import PLAN_STORE, PlanSession
from llm.agents.subtask_classifier import classify_subtasks, group_events
from llm.agents.action_predictor_agent import ActionPredictorAgent
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld

# 1. Build the plan locally
subtasks = [
    "Fetch an onion", "Stage onion at stove",
    "Place onion on stove", "Turn stove on", "Wait for onion to cook",
    "Pour soup into bowl", "Bring bowl to serving station"
]
tagged = classify_subtasks(subtasks)
events = group_events(tagged)

# 2. Store it under a fresh ID
session_id = "test-plan"
PLAN_STORE[session_id] = PlanSession(events)

# 3. Create agent & MDP
agent = ActionPredictorAgent()
mdp = OvercookedGridworld.from_layout_name("cramped_room")
agent.set_mdp(mdp)
agent.set_plan(session_id)
agent.set_agent_index(0)

# 4. Test on start state
state = mdp.get_standard_start_state()
print("Before:", agent.plan.idx)
move, info = agent.action(state)
print("Action:", move)
print("Info:", json.dumps(info, indent=2))
print("After:", agent.plan.idx)
