# test_coordinated_action_predictor.py
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import json
from llm.agents.coordinated_action_predictor import CoordinatedActionPredictorAgent
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState
from plan_session import PLAN_STORE, PlanSession
from llm.agents.subtask_to_event_sequence import classify_subtasks, group_events

def test_coordinated_action_predictor():
    """Test the coordinated action predictor with dual prediction and planning"""
    
    # Build the plan locally (same as action_predictor test)
    subtasks = [
        "Fetch an onion", "Stage onion at stove",
        "Place onion on stove", "Turn stove on", "Wait for onion to cook", "Fetch a bowl", "Stage bowl at stove",
        "Pour soup into bowl", "Bring bowl to serving station"
    ]
    tagged = classify_subtasks(subtasks)
    events = group_events(tagged)
    
    # Store it under a fresh ID
    session_id = "test-coordinated-plan"
    PLAN_STORE[session_id] = PlanSession(events)
    
    # Create agent & MDP
    agent = CoordinatedActionPredictorAgent()
    mdp = OvercookedGridworld.from_layout_name("cramped_room")
    agent.set_mdp(mdp)
    agent.set_plan(session_id)
    agent.set_agent_index(0)
    
    # Test on start state
    state = mdp.get_standard_start_state()
    print("Testing CoordinatedActionPredictorAgent...")
    print(f"Start state: {agent.summarize_state(state, {})}")
    
    move, info = agent.action(state)
    
    print("\nResults:")
    print(f"Move: {move}")
    print(f"Predicted human action: {info.get('predicted_human_action')}")
    print(f"Best robot action: {info.get('best_robot_action')}")
    print(f"Next planned action: {info.get('next_planned_action')}")
    print(f"Possible robot actions: {info.get('possible_actions')}")
    print(f"Possible human actions: {info.get('possible_human_actions')}")
    print(f"LLM Response: {info.get('llm_response')}")
    
    # Verify that we got all the required information
    assert 'predicted_human_action' in info, "Should have predicted human action"
    assert 'best_robot_action' in info, "Should have best robot action"
    assert 'next_planned_action' in info, "Should have next planned action"
    assert 'possible_human_actions' in info, "Should have possible human actions"
    
    print("\n✅ Test passed! Adaptive planning system is working.")
    
    return info

if __name__ == "__main__":
    test_coordinated_action_predictor() 