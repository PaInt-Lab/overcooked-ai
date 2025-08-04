from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json
from .state_graph import StateGraph, StateNode

@dataclass
class CoordinationContext:
    """Context for coordination decisions"""
    current_state: Dict
    predicted_human_action: str

class CoordinatedActionSelector:
    """Selects actions using LLM prediction"""
    
    def __init__(self, state_graph: StateGraph):
        self.state_graph = state_graph
        
    def select_action(self, current_state: Dict, goal_state: Dict) -> Tuple[str, CoordinationContext]:
        """Select the best action (will be replaced by LLM)"""
        # Get current and goal node IDs
        current_node_id = self._get_node_id_for_state(current_state)
        goal_node_id = self._get_node_id_for_state(goal_state)
        
        if not current_node_id or not goal_node_id:
            return 'NOOP', CoordinationContext(
                current_state=current_state,
                predicted_human_action='no_human_action'
            )
        
        # Get all possible robot actions from current state
        possible_actions = self._get_possible_robot_actions(current_node_id)
        
        # For now, just pick the first available action
        # This will be replaced by LLM logic
        best_action = possible_actions[0] if possible_actions else 'NOOP'
        
        context = CoordinationContext(
            current_state=current_state,
            predicted_human_action='no_human_action'  # Will be filled by LLM
        )
        
        return best_action, context
    
    def _get_node_id_for_state(self, state: Dict) -> Optional[str]:
        """Get node ID for a state using the state graph mapping"""
        # Use the state graph's mapping if available
        if hasattr(self.state_graph, 'state_to_node_id'):
            state_key = json.dumps(state, sort_keys=True)
            return self.state_graph.state_to_node_id.get(state_key)
        
        # Fallback to simple hash-based ID if mapping not available
        state_key = json.dumps(state, sort_keys=True)
        return f"state_{hash(state_key) % 1000000:06d}"
    
    def _get_possible_robot_actions(self, node_id: str) -> List[str]:
        """Get all possible robot actions from current node"""
        edges = self.state_graph.get_edges_from(node_id)
        robot_actions = []
        
        for edge in edges:
            action = edge.action
            # Only include robot actions (not human or environmental)
            if not action.startswith('human_') and not action.startswith('cooking_') and action != 'soup_ready':
                robot_actions.append(action)
        
        return robot_actions

class CoordinationManager:
    """Main coordination manager that orchestrates the entire system"""
    
    def __init__(self, state_graph: StateGraph):
        self.state_graph = state_graph
        self.action_selector = CoordinatedActionSelector(state_graph)
        self.current_context = None
        
    def get_next_action(self, current_state: Dict) -> Tuple[str, CoordinationContext]:
        """Get the next coordinated action"""
        # Define goal state (soup served)
        goal_state = {
            'onion_hand': 'none',
            'onion_staged': False,
            'onion_at_chopping': False,
            'onion_chopped': False,  # Reset after serving
            'tomato_hand': 'none',
            'tomato_staged': False,
            'tomato_at_chopping': False,
            'tomato_chopped': False,  # Reset after serving
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
        
        action, context = self.action_selector.select_action(current_state, goal_state)
        self.current_context = context
        
        return action, context
    
    def get_coordination_status(self) -> Optional[CoordinationContext]:
        """Get the current coordination status"""
        return self.current_context