"""
Primary Actions Only State Graph Generator

This module generates a dramatically simplified state graph containing only primary actions.
This reduces state space complexity while maintaining all strategic decision points.
"""

import json
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class PrimaryActionNode:
    """Node representing a game state with primary actions only"""
    node_id: str
    state_summary: Dict
    is_goal_state: bool = False


@dataclass
class PrimaryActionEdge:
    """Edge representing a primary action transition"""
    from_node: str
    to_node: str
    primary_action: str  # e.g., 'human_grab_onion', 'human_place_onion_in_pot'
    weight: float = 1.0


class PrimaryActionsStateGraph:
    """State graph containing only primary actions for fast LLM prediction"""
    
    def __init__(self):
        self.nodes: Dict[str, PrimaryActionNode] = {}
        self.edges: Dict[str, List[PrimaryActionEdge]] = {}
        self.reverse_edges: Dict[str, List[PrimaryActionEdge]] = {}
        self.state_to_node_id: Dict[str, str] = {}
        
    def add_node(self, node: PrimaryActionNode):
        """Add a node to the graph"""
        self.nodes[node.node_id] = node
        if node.node_id not in self.edges:
            self.edges[node.node_id] = []
        if node.node_id not in self.reverse_edges:
            self.reverse_edges[node.node_id] = []
        
        # Add to state mapping for fast lookup
        state_key = json.dumps(node.state_summary, sort_keys=True)
        self.state_to_node_id[state_key] = node.node_id
    
    def add_edge(self, edge: PrimaryActionEdge):
        """Add an edge to the graph"""
        self.edges[edge.from_node].append(edge)
        self.reverse_edges[edge.to_node].append(edge)
    
    def get_node(self, node_id: str) -> Optional[PrimaryActionNode]:
        """Get a node by ID"""
        return self.nodes.get(node_id)
    
    def get_edges_from(self, node_id: str) -> List[PrimaryActionEdge]:
        """Get all edges from a node"""
        return self.edges.get(node_id, [])
    
    def get_possible_primary_actions(self, node_id: str) -> List[str]:
        """Get all possible primary actions from a node"""
        edges = self.get_edges_from(node_id)
        return [edge.primary_action for edge in edges]
    
    def get_state_mapping(self) -> Dict[str, str]:
        """Get the state to node ID mapping"""
        return self.state_to_node_id


class PrimaryActionsStateGraphGenerator:
    """Generates state graph with primary actions only"""
    
    def __init__(self):
        self.graph = PrimaryActionsStateGraph()
        self.state_to_node_id = {}
        
    def generate_state_graph(self):
        """Generate all possible states and primary action transitions"""
        print(f"Generating PRIMARY ACTIONS ONLY state graph...")
        
        # Generate all possible state combinations
        self._generate_all_states()
        print(f"   Generated {len(self.graph.nodes)} states")
        
        # Generate all possible primary action transitions
        self._generate_all_primary_transitions()
        total_edges = sum(len(edges) for edges in self.graph.edges.values())
        print(f"   Generated {total_edges} primary action transitions")
        
        # Make the state mapping accessible through the graph
        self.graph.state_to_node_id = self.state_to_node_id
        
        # Debug: Check if goal states exist
        goal_states = [node for node in self.graph.nodes.values() if node.is_goal_state]
        print(f"   Found {len(goal_states)} goal states")
        if goal_states:
            print(f"   Sample goal state: {goal_states[0].state_summary}")
        
        return self.graph
    
    def _generate_all_states(self):
        """Generate all possible game states with more restrictive validation"""
        # Start with initial state
        initial_state = {
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
        
        # Add initial state
        node_id = self._get_state_id(initial_state)
        node = PrimaryActionNode(
            node_id=node_id,
            state_summary=initial_state,
            is_goal_state=self._is_goal_state(initial_state)
        )
        self.graph.add_node(node)
        self.state_to_node_id[json.dumps(initial_state, sort_keys=True)] = node_id
        
        # Generate states by applying primary actions to existing states
        # This ensures we only generate reachable states
        self._generate_reachable_states()
    
    def _generate_reachable_states(self):
        """Generate states by applying primary actions to existing states"""
        from collections import deque
        
        # Start with initial states
        queue = deque(list(self.graph.nodes.keys()))
        processed = set()
        
        while queue:
            current_node_id = queue.popleft()
            if current_node_id in processed:
                continue
                
            processed.add(current_node_id)
            current_state = self.graph.nodes[current_node_id].state_summary
            
            # Generate all possible next states from current state
            next_states = self._generate_next_states(current_state)
            
            for next_state, action in next_states:
                next_state_key = json.dumps(next_state, sort_keys=True)
                
                # Add new state if we haven't seen it
                if next_state_key not in self.state_to_node_id:
                    node_id = self._get_state_id(next_state)
                    node = PrimaryActionNode(
                        node_id=node_id,
                        state_summary=next_state,
                        is_goal_state=self._is_goal_state(next_state)
                    )
                    self.graph.add_node(node)
                    self.state_to_node_id[next_state_key] = node_id
                    queue.append(node_id)
    
    def _generate_next_states(self, current_state: Dict) -> List[tuple]:
        """Generate all possible next states from current state"""
        next_states = []
        
        # Robot actions (these generate states that enable human actions)
        
        # Robot can fetch and stage raw onion
        if (current_state['onion_hand'] == 'none' and 
            not current_state['onion_staged'] and 
            not current_state['onion_chopped']):
            new_state = current_state.copy()
            new_state['onion_staged'] = True
            next_states.append((new_state, 'robot_fetch_onion'))
        
        # Robot can fetch and stage raw tomato
        if (current_state['tomato_hand'] == 'none' and 
            not current_state['tomato_staged'] and 
            not current_state['tomato_chopped']):
            new_state = current_state.copy()
            new_state['tomato_staged'] = True
            next_states.append((new_state, 'robot_fetch_tomato'))
        
        # Robot can fetch and stage dish
        if (current_state['dish_hand'] == 'none' and 
            not current_state['dish_staged']):
            new_state = current_state.copy()
            new_state['dish_staged'] = True
            next_states.append((new_state, 'robot_fetch_dish'))
        
        # Robot can handle chopping workflow for onion
        if (current_state['onion_hand'] == 'none' and 
            not current_state['onion_chopped'] and
            not current_state['onion_at_chopping']):
            new_state = current_state.copy()
            new_state['onion_at_chopping'] = True
            next_states.append((new_state, 'robot_start_chopping_onion'))
        
        # Robot can complete chopping for onion
        if (current_state['onion_at_chopping'] and 
            not current_state['onion_chopped']):
            new_state = current_state.copy()
            new_state['onion_chopped'] = True
            next_states.append((new_state, 'robot_complete_chopping_onion'))
        
        # Robot can handle chopping workflow for tomato
        if (current_state['tomato_hand'] == 'none' and 
            not current_state['tomato_chopped'] and
            not current_state['tomato_at_chopping']):
            new_state = current_state.copy()
            new_state['tomato_at_chopping'] = True
            next_states.append((new_state, 'robot_start_chopping_tomato'))
        
        # Robot can complete chopping for tomato
        if (current_state['tomato_at_chopping'] and 
            not current_state['tomato_chopped']):
            new_state = current_state.copy()
            new_state['tomato_chopped'] = True
            next_states.append((new_state, 'robot_complete_chopping_tomato'))
        
        # Human actions (these consume robot-prepared states)
        
        # Human can grab raw onion from staging
        if current_state['onion_staged'] and current_state['onion_hand'] == 'none' and not current_state['onion_chopped']:
            new_state = current_state.copy()
            new_state['onion_hand'] = 'partner'
            new_state['onion_staged'] = False
            next_states.append((new_state, 'human_grab_onion'))
        
        # Human can grab chopped onion
        if current_state['onion_chopped'] and current_state['onion_hand'] == 'none' and not current_state['onion_staged']:
            new_state = current_state.copy()
            new_state['onion_hand'] = 'partner'
            next_states.append((new_state, 'human_grab_chopped_onion'))
        
        # Human can grab raw tomato from staging
        if current_state['tomato_staged'] and current_state['tomato_hand'] == 'none' and not current_state['tomato_chopped']:
            new_state = current_state.copy()
            new_state['tomato_hand'] = 'partner'
            new_state['tomato_staged'] = False
            next_states.append((new_state, 'human_grab_tomato'))
        
        # Human can grab chopped tomato
        if current_state['tomato_chopped'] and current_state['tomato_hand'] == 'none' and not current_state['tomato_staged']:
            new_state = current_state.copy()
            new_state['tomato_hand'] = 'partner'
            next_states.append((new_state, 'human_grab_chopped_tomato'))
        
        # Human can place onion in pot
        if current_state['onion_hand'] == 'partner':
            new_state = current_state.copy()
            new_state['onion_hand'] = 'none'
            new_state['onion_in_pot'] = True
            next_states.append((new_state, 'human_place_onion_in_pot'))
        
        # Human can place tomato in pot
        if current_state['tomato_hand'] == 'partner':
            new_state = current_state.copy()
            new_state['tomato_hand'] = 'none'
            new_state['tomato_in_pot'] = True
            next_states.append((new_state, 'human_place_tomato_in_pot'))
        
        # Human can turn stove on
        if (current_state['onion_in_pot'] or current_state['tomato_in_pot']) and not current_state['soup_cooking'] and not current_state['soup_ready']:
            new_state = current_state.copy()
            new_state['soup_cooking'] = True
            next_states.append((new_state, 'human_turn_stove_on'))
        
        # Human can pour soup
        if current_state['soup_ready'] and not current_state['soup_staged']:
            new_state = current_state.copy()
            new_state['soup_staged'] = True
            next_states.append((new_state, 'human_pour_soup'))
        
        # Human can grab dish from staging
        if current_state['dish_staged'] and current_state['dish_hand'] == 'none':
            new_state = current_state.copy()
            new_state['dish_hand'] = 'partner'
            new_state['dish_staged'] = False
            next_states.append((new_state, 'human_grab_dish'))
        
        # Environmental transitions
        if current_state['soup_cooking'] and not current_state['soup_ready']:
            new_state = current_state.copy()
            new_state['soup_cooking'] = False
            new_state['soup_ready'] = True
            next_states.append((new_state, 'cooking_complete'))
        
        if current_state['soup_served']:
            new_state = current_state.copy()
            new_state['soup_served'] = False
            new_state['onion_in_pot'] = False
            new_state['tomato_in_pot'] = False
            new_state['soup_ready'] = False
            new_state['soup_staged'] = False
            next_states.append((new_state, 'reset_after_serving'))
        
        return next_states
    
    def _is_valid_state(self, state: Dict) -> bool:
        """Check if a state is valid"""
        # Basic validation rules
        
        # Can't have ingredient in hand if it's staged
        if state['onion_hand'] != 'none' and state['onion_staged']:
            return False
        if state['tomato_hand'] != 'none' and state['tomato_staged']:
            return False
        if state['dish_hand'] != 'none' and state['dish_staged']:
            return False
        if state['soup_hand'] != 'none' and state['soup_staged']:
            return False
        
        # Can't have ingredient in hand if it's in pot
        if state['onion_hand'] != 'none' and state['onion_in_pot']:
            return False
        if state['tomato_hand'] != 'none' and state['tomato_in_pot']:
            return False
        
        # Can't have ingredient in hand if it's at chopping
        if state['onion_hand'] != 'none' and state['onion_at_chopping']:
            return False
        if state['tomato_hand'] != 'none' and state['tomato_at_chopping']:
            return False
        
        # Can't have soup ready if not cooking and not in pot
        if state['soup_ready'] and not state['soup_cooking'] and not (state['onion_in_pot'] or state['tomato_in_pot']):
            return False
        
        # Can't have soup cooking if no ingredients in pot
        if state['soup_cooking'] and not (state['onion_in_pot'] or state['tomato_in_pot']):
            return False
        
        return True
    
    def _is_goal_state(self, state: Dict) -> bool:
        """Check if a state is a goal state (soup served)"""
        # A goal state is when soup has been served
        # This can happen when soup is staged and ready to be served
        return (state['soup_staged'] and state['soup_ready'] and 
                (state['onion_in_pot'] or state['tomato_in_pot']))
    
    def _get_state_id(self, state: Dict) -> str:
        """Generate a unique ID for a state"""
        state_str = json.dumps(state, sort_keys=True)
        return f"state_{hash(state_str) % 1000000}"
    
    def _generate_all_primary_transitions(self):
        """Generate all possible primary action transitions"""
        # Transitions are now generated during state generation
        # We just need to create edges between the states
        for node_id, node in self.graph.nodes.items():
            current_state = node.state_summary
            
            # Generate next states and create edges
            next_states = self._generate_next_states(current_state)
            
            for next_state, action in next_states:
                next_state_key = json.dumps(next_state, sort_keys=True)
                to_node_id = self.state_to_node_id.get(next_state_key)
                
                if to_node_id and to_node_id != node_id:
                    edge = PrimaryActionEdge(
                        from_node=node_id,
                        to_node=to_node_id,
                        primary_action=action,
                        weight=1.0
                    )
                    self.graph.add_edge(edge)
    
    def get_state_mapping(self) -> Dict[str, str]:
        """Get the state to node ID mapping"""
        return self.state_to_node_id


def create_primary_actions_state_graph() -> PrimaryActionsStateGraph:
    """Factory function to create the primary actions state graph"""
    generator = PrimaryActionsStateGraphGenerator()
    return generator.generate_state_graph()


# Convenience functions for integration
def get_possible_primary_actions(graph: PrimaryActionsStateGraph, state: Dict) -> List[str]:
    """Get possible primary actions for a given state"""
    state_key = json.dumps(state, sort_keys=True)
    node_id = graph.state_to_node_id.get(state_key)
    
    if node_id:
        return graph.get_possible_primary_actions(node_id)
    else:
        return []


def find_path_to_goal(graph: PrimaryActionsStateGraph, start_state: Dict, goal_state: Dict) -> List[str]:
    """Find path of primary actions to reach goal state"""
    start_key = json.dumps(start_state, sort_keys=True)
    goal_key = json.dumps(goal_state, sort_keys=True)
    
    start_node_id = graph.state_to_node_id.get(start_key)
    goal_node_id = graph.state_to_node_id.get(goal_key)
    
    if not start_node_id or not goal_node_id:
        return []
    
    # Simple BFS pathfinding
    from collections import deque
    
    queue = deque([(start_node_id, [])])
    visited = set()
    
    while queue:
        current_node, path = queue.popleft()
        
        if current_node == goal_node_id:
            return path
        
        if current_node in visited:
            continue
            
        visited.add(current_node)
        
        for edge in graph.get_edges_from(current_node):
            new_path = path + [edge.primary_action]
            queue.append((edge.to_node, new_path))
    
    return []
