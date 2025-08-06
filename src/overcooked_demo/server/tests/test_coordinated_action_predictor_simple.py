#!/usr/bin/env python3
"""
Simplified test for coordinated action predictor without LLM dependencies.
"""

import sys
import os
import json
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Mock the LLM client to avoid dependencies
class MockOllamaClient:
    @staticmethod
    def query_ollama(prompt, model="llama3.2", temperature=0.0):
        # Return a mock response that matches expected format
        return """
        predicted_human_action: human_pickup_onion
        best_robot_action: pickup_onion
        """

# Patch the import
import llm.ollama.ollama_client
llm.ollama.ollama_client.query_ollama = MockOllamaClient.query_ollama

from llm.agents.coordinated_action_predictor import CoordinatedActionPredictorAgent
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from plan_session import PLAN_STORE, PlanSession
from llm.agents.subtask_to_event_sequence import classify_subtasks, group_events

def test_coordinated_action_predictor_simple():
    """Test the coordinated action predictor with mocked LLM"""
    
    print("🧪 Testing CoordinatedActionPredictorAgent (Simplified)")
    print("=" * 50)
    
    # Build the plan locally
    subtasks = [
        "Fetch an onion", "Stage onion at stove",
        "Place onion on stove", "Turn stove on", "Wait for onion to cook", 
        "Fetch a bowl", "Stage bowl at stove",
        "Pour soup into bowl", "Bring bowl to serving station"
    ]
    tagged = classify_subtasks(subtasks)
    events = group_events(tagged)
    
    # Store it under a fresh ID
    session_id = "test-coordinated-plan-simple"
    PLAN_STORE[session_id] = PlanSession(events)
    
    # Create agent & MDP
    print("🔧 Creating agent and MDP...")
    agent = CoordinatedActionPredictorAgent()
    mdp = OvercookedGridworld.from_layout_name("cramped_room")
    agent.set_mdp(mdp)
    agent.set_plan(session_id)
    agent.set_agent_index(0)
    
    # Test on start state
    state = mdp.get_standard_start_state()
    print(f"📊 Start state: {agent.summarize_state(state, {})}")
    
    print("🎯 Testing action prediction...")
    move, info = agent.action(state)
    
    print("\n📋 Results:")
    print(f"   Move: {move}")
    print(f"   Predicted human action: {info.get('predicted_human_action')}")
    print(f"   Best robot action: {info.get('best_robot_action')}")
    print(f"   Next planned action: {info.get('next_planned_action')}")
    print(f"   Possible robot actions: {len(info.get('possible_actions', []))}")
    print(f"   Possible human actions: {len(info.get('possible_human_actions', []))}")
    print(f"   Blocking prevention: {info.get('blocking_prevention', False)}")
    print(f"   Reasoning: {info.get('reasoning', 'N/A')}")
    
    # Verify that we got all the required information
    assert 'predicted_human_action' in info, "Should have predicted human action"
    assert 'best_robot_action' in info, "Should have best robot action"
    assert 'next_planned_action' in info, "Should have next planned action"
    assert 'possible_human_actions' in info, "Should have possible human actions"
    
    print("\n✅ Test passed! Coordinated action predictor is working with caching.")
    
    return info

if __name__ == "__main__":
    test_coordinated_action_predictor_simple() 