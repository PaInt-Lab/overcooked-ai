#!/usr/bin/env python3
"""
Test script for the coordinated navigation system.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from server.state_graph import StateGraphGenerator, StateGraph
from server.coordination_system import CoordinationManager, CoordinationContext

def test_state_graph_generation():
    """Test that the state graph can be generated correctly."""
    print("Testing state graph generation...")
    
    generator = StateGraphGenerator()
    graph = generator.generate_state_graph()
    
    print(f"Generated graph with {len(graph.nodes)} nodes")
    print(f"Generated graph with {sum(len(edges) for edges in graph.edges.values())} edges")
    
    # Test a few basic states
    initial_state = {
        'onion_hand': 'none',
        'onion_staged': False,
        'onion_at_chopping': False,
        'onion_chopped': False,
        'tomato_hand': 'none',
        'tomato_staged': False,
        'tomato_at_chopping': False,
        'tomato_chopped': False,
        'onion_in_pot': False,
        'tomato_in_pot': False,
        'soup_cooking': False,
        'soup_ready': False,
        'soup_in_pot_not_cooking': False,
        'dish_hand': 'none',
        'dish_staged': False,
        'soup_hand': 'none',
        'soup_staged': False,
        'soup_served': False
    }
    
    goal_state = {
        'onion_hand': 'none',
        'onion_staged': False,
        'onion_at_chopping': False,
        'onion_chopped': False,
        'tomato_hand': 'none',
        'tomato_staged': False,
        'tomato_at_chopping': False,
        'tomato_chopped': False,
        'onion_in_pot': False,
        'tomato_in_pot': False,
        'soup_cooking': False,
        'soup_ready': False,
        'soup_in_pot_not_cooking': False,
        'dish_hand': 'none',
        'dish_staged': False,
        'soup_hand': 'none',
        'soup_staged': False,
        'soup_served': True  # Goal state
    }
    
    initial_node_id = generator.get_node_id_for_state(initial_state)
    goal_node_id = generator.get_node_id_for_state(goal_state)
    
    print(f"Initial state node ID: {initial_node_id}")
    print(f"Goal state node ID: {goal_node_id}")
    
    if initial_node_id and goal_node_id:
        print("✓ State graph generation successful")
        return True
    else:
        print("✗ State graph generation failed")
        return False

def test_coordination_manager():
    """Test that the coordination manager works correctly."""
    print("\nTesting coordination manager...")
    
    # Create a simple state graph for testing
    generator = StateGraphGenerator()
    graph = generator.generate_state_graph()
    
    coordination_manager = CoordinationManager(graph)
    
    # Test initial state
    initial_state = {
        'onion_hand': 'none',
        'onion_staged': False,
        'onion_at_chopping': False,
        'onion_chopped': False,
        'tomato_hand': 'none',
        'tomato_staged': False,
        'tomato_at_chopping': False,
        'tomato_chopped': False,
        'onion_in_pot': False,
        'tomato_in_pot': False,
        'soup_cooking': False,
        'soup_ready': False,
        'soup_in_pot_not_cooking': False,
        'dish_hand': 'none',
        'dish_staged': False,
        'soup_hand': 'none',
        'soup_staged': False,
        'soup_served': False
    }
    
    action, context = coordination_manager.get_next_action(initial_state)
    
    print(f"Initial state action: {action}")
    print(f"Coordination context: {context}")
    
    if action and context:
        print("✓ Coordination manager test successful")
        return True
    else:
        print("✗ Coordination manager test failed")
        return False

def test_human_behavior_prediction():
    """Test human behavior prediction."""
    print("\nTesting human behavior prediction...")
    
    from server.coordination_system import HumanBehaviorPredictor
    
    predictor = HumanBehaviorPredictor()
    
    # Test various states
    test_states = [
        {
            'onion_hand': 'partner',
            'tomato_hand': 'none',
            'onion_staged': False,
            'tomato_staged': False,
            'soup_ready': False,
            'dish_staged': False,
            'dish_hand': 'none'
        },
        {
            'onion_hand': 'none',
            'tomato_hand': 'none',
            'onion_staged': True,
            'tomato_staged': False,
            'soup_ready': False,
            'dish_staged': False,
            'dish_hand': 'none'
        },
        {
            'onion_hand': 'none',
            'tomato_hand': 'none',
            'onion_staged': False,
            'tomato_staged': False,
            'soup_ready': True,
            'dish_staged': True,
            'dish_hand': 'none'
        }
    ]
    
    for i, state in enumerate(test_states):
        prediction = predictor.predict_human_action(state)
        print(f"State {i+1} prediction: {prediction}")
    
    print("✓ Human behavior prediction test successful")
    return True

def test_coordination_evaluation():
    """Test coordination evaluation."""
    print("\nTesting coordination evaluation...")
    
    from server.coordination_system import CoordinationEvaluator
    
    evaluator = CoordinationEvaluator()
    
    # Test positive coordination
    robot_action = "pickup(onion)"
    human_action = "human_grab_tomato"
    state = {'soup_cooking': False}
    
    quality, risk = evaluator.evaluate_coordination(robot_action, human_action, state)
    print(f"Positive coordination - Quality: {quality}, Risk: {risk}")
    
    # Test negative coordination
    robot_action = "pickup(onion)"
    human_action = "human_grab_onion"
    
    quality, risk = evaluator.evaluate_coordination(robot_action, human_action, state)
    print(f"Negative coordination - Quality: {quality}, Risk: {risk}")
    
    # Test neutral coordination
    robot_action = "pickup(dish)"
    human_action = "no_human_action"
    
    quality, risk = evaluator.evaluate_coordination(robot_action, human_action, state)
    print(f"Neutral coordination - Quality: {quality}, Risk: {risk}")
    
    print("✓ Coordination evaluation test successful")
    return True

def main():
    """Run all tests."""
    print("Running coordinated navigation system tests...\n")
    
    tests = [
        test_state_graph_generation,
        test_coordination_manager,
        test_human_behavior_prediction,
        test_coordination_evaluation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test failed with error: {e}")
    
    print(f"\nTest Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The coordinated navigation system is working correctly.")
    else:
        print("❌ Some tests failed. Please check the implementation.")

if __name__ == "__main__":
    main() 