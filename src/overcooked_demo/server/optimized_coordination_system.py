"""
Optimized Coordination System

This module integrates the primary actions state graph with the secondary action selector
to create a complete, high-performance coordination system.
"""

import json
from typing import Dict, List, Optional, Tuple
from primary_actions_state_graph import (
    PrimaryActionsStateGraph, 
    create_primary_actions_state_graph,
    get_possible_primary_actions
)
from secondary_action_selector import select_secondary_action, get_relevant_secondary_actions


class OptimizedCoordinationSystem:
    """
    High-performance coordination system that uses:
    1. Primary actions state graph for fast LLM prediction
    2. Secondary action selector for robot coordination
    """
    
    def __init__(self):
        self.primary_graph = None
        self.recipe_cache = {}  # Cache for recipe parsing
        
    def initialize(self):
        """Initialize the coordination system"""
        print("🚀 Initializing Optimized Coordination System...")
        
        # Create the primary actions state graph
        self.primary_graph = create_primary_actions_state_graph()
        
        print(f"✅ Primary actions state graph loaded: {len(self.primary_graph.nodes)} nodes")
        print(f"✅ Secondary action selector ready")
        print("🎯 System ready for high-performance coordination!")
        
        return self
    
    def get_possible_primary_actions(self, game_state: Dict) -> List[str]:
        """
        Get possible primary actions for the current game state.
        This is what the LLM will choose from.
        """
        if not self.primary_graph:
            raise RuntimeError("System not initialized. Call initialize() first.")
        
        return get_possible_primary_actions(self.primary_graph, game_state)
    
    def coordinate_actions(self, 
                          game_state: Dict, 
                          recipe: str, 
                          predicted_primary_action: str) -> str:
        """
        Main coordination method:
        1. Takes predicted primary action from LLM
        2. Returns robot secondary action
        
        Returns:
            - robot_action: The secondary action the robot should perform
        """
        if not self.primary_graph:
            raise RuntimeError("System not initialized. Call initialize() first.")
        
        # Get robot action using the secondary action selector
        robot_action = self._get_robot_action(game_state, recipe, predicted_primary_action)
        
        return robot_action
    
    def _get_robot_action(self, game_state: Dict, recipe: str, predicted_primary_action: str) -> str:
        """
        Get robot action using the secondary action selector for consistent behavior.
        """
        return select_secondary_action(game_state, recipe, predicted_primary_action)
    
    def get_coordination_summary(self, 
                                game_state: Dict, 
                                recipe: str, 
                                predicted_primary_action: str) -> Dict:
        """
        Get a comprehensive summary of the coordination decision
        """
        robot_action = self.coordinate_actions(
            game_state, recipe, predicted_primary_action
        )
        
        # Get the relevant secondary actions for the predicted primary action
        relevant_secondary_actions = get_relevant_secondary_actions(predicted_primary_action)
        
        return {
            'predicted_primary_action': predicted_primary_action,
            'robot_secondary_action': robot_action,
            'relevant_secondary_actions': relevant_secondary_actions,
            'game_state': game_state,
            'recipe': recipe
        }
    
    def simulate_workflow(self, 
                         recipe: str, 
                         primary_action_sequence: List[str]) -> List[Dict]:
        """
        Simulate a complete workflow given a sequence of primary actions.
        Useful for testing and validation.
        """
        if not self.primary_graph:
            raise RuntimeError("System not initialized. Call initialize() first.")
        
        workflow_steps = []
        current_state = self._get_initial_state()
        
        for i, primary_action in enumerate(primary_action_sequence):
            # Get coordination for this step
            coordination = self.get_coordination_summary(
                current_state, recipe, primary_action
            )
            
            workflow_steps.append({
                'step': i + 1,
                'primary_action': primary_action,
                'robot_action': coordination['robot_secondary_action'],
                'game_state': current_state.copy(),
                'coordination_summary': coordination
            })
            
            # Update state based on primary action (simplified)
            current_state = self._simulate_state_transition(current_state, primary_action)
        
        return workflow_steps
    
    def _get_initial_state(self) -> Dict:
        """Get the initial game state"""
        return {
            'onion_hand': 'none',
            'tomato_hand': 'none',
            'dish_hand': 'none',
            'soup_hand': 'none',
            'onion_staged': False,
            'tomato_staged': False,
            'dish_staged': False,
            'soup_staged': False,
            'onion_in_pot': False,
            'tomato_in_pot': False,
            'soup_cooking': False,
            'soup_ready': False,
            'soup_served': False,
            'onion_at_chopping': False,
            'tomato_at_chopping': False,
            'onion_chopped': False,
            'tomato_chopped': False
        }
    
    def _simulate_state_transition(self, current_state: Dict, primary_action: str) -> Dict:
        """Simulate state transition based on primary action (simplified)"""
        new_state = current_state.copy()
        
        if primary_action == 'human_grab_onion':
            if new_state['onion_staged']:
                new_state['onion_hand'] = 'partner'
                new_state['onion_staged'] = False
        elif primary_action == 'human_grab_chopped_onion':
            if new_state['onion_chopped']:
                new_state['onion_hand'] = 'partner'
        elif primary_action == 'human_place_onion_in_pot':
            if new_state['onion_hand'] == 'partner':
                new_state['onion_hand'] = 'none'
                new_state['onion_in_pot'] = True
        elif primary_action == 'human_turn_stove_on':
            if new_state['onion_in_pot'] or new_state['tomato_in_pot']:
                new_state['soup_cooking'] = True
        
        return new_state
    
    def get_performance_metrics(self) -> Dict:
        """Get performance metrics for the system"""
        if not self.primary_graph:
            return {'error': 'System not initialized'}
        
        total_nodes = len(self.primary_graph.nodes)
        total_edges = sum(len(edges) for edges in self.primary_graph.edges.values())
        goal_states = len([n for n in self.primary_graph.nodes.values() if n.is_goal_state])
        
        return {
            'total_states': total_nodes,
            'total_transitions': total_edges,
            'goal_states': goal_states,
            'state_space_density': total_edges / total_nodes if total_nodes > 0 else 0,
            'optimization_level': 'primary_actions_only'
        }


# Global instance for easy access
_coordination_system = None


def get_coordination_system() -> OptimizedCoordinationSystem:
    """Get or create the global coordination system instance"""
    global _coordination_system
    if _coordination_system is None:
        _coordination_system = OptimizedCoordinationSystem()
        _coordination_system.initialize()
    return _coordination_system


def quick_coordinate(game_state: Dict, recipe: str, primary_action: str) -> str:
    """
    Quick coordination function for simple use cases.
    Returns just the robot action.
    """
    system = get_coordination_system()
    robot_action = system.coordinate_actions(game_state, recipe, primary_action)
    return robot_action


def get_available_actions(game_state: Dict) -> List[str]:
    """
    Quick function to get available primary actions for a game state.
    """
    system = get_coordination_system()
    return system.get_possible_primary_actions(game_state)


# Convenience functions for testing
def test_coordination_system():
    """Test the coordination system with sample scenarios"""
    print("🧪 Testing Optimized Coordination System...")
    
    system = get_coordination_system()
    
    # Test 1: Basic coordination
    print("\n📋 Test 1: Basic Coordination")
    game_state = {
        'onion_hand': 'none',
        'onion_staged': True,
        'onion_chopped': False,
        'tomato_hand': 'none',
        'tomato_staged': False,
        'tomato_chopped': False,
        'dish_hand': 'none',
        'dish_staged': False,
        'soup_hand': 'none',
        'soup_staged': False,
        'onion_in_pot': False,
        'tomato_in_pot': False,
        'soup_cooking': False,
        'soup_ready': False,
        'soup_served': False,
        'onion_at_chopping': False,
        'tomato_at_chopping': False
    }
    
    available_actions = system.get_possible_primary_actions(game_state)
    print(f"   Available primary actions: {available_actions}")
    
    if 'human_grab_onion' in available_actions:
        coordination = system.get_coordination_summary(
            game_state, 'Serving Onion Soup', 'human_grab_onion'
        )
        print(f"   Robot action for 'human_grab_onion': {coordination['robot_secondary_action']}")
    
    # Test 2: Workflow simulation
    print("\n📋 Test 2: Workflow Simulation")
    workflow = system.simulate_workflow(
        'Serving Onion Soup',
        ['human_grab_onion', 'human_place_onion_in_pot', 'human_turn_stove_on']
    )
    
    for step in workflow:
        print(f"   Step {step['step']}: {step['primary_action']} → Robot: {step['robot_action']}")
    
    # Test 3: Performance metrics
    print("\n📋 Test 3: Performance Metrics")
    metrics = system.get_performance_metrics()
    for key, value in metrics.items():
        print(f"   {key}: {value}")
    
    print("\n✅ Coordination system test completed!")


if __name__ == "__main__":
    test_coordination_system()
