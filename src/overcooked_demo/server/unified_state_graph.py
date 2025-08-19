from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass
import json
from collections import deque
import heapq

@dataclass
class StateNode:
    """Represents a node in the unified state graph"""
    node_id: str
    state_summary: Dict
    is_goal_state: bool = False
    
    def __hash__(self):
        return hash(self.node_id)
    
    def __eq__(self, other):
        return self.node_id == other.node_id

@dataclass
class StateEdge:
    """Represents an edge between states"""
    from_node: str
    to_node: str
    action: str  # Primary actions only: 'Human Grab Onion', 'Place onion in pot', etc.
    weight: float = 1.0  # For weighted path finding

class UnifiedStateGraph:
    """Manages the unified state graph for all recipe types"""
    
    def __init__(self):
        self.nodes: Dict[str, StateNode] = {}
        self.edges: Dict[str, List[StateEdge]] = {}  # node_id -> list of edges
        self.reverse_edges: Dict[str, List[StateEdge]] = {}  # for backward navigation
        self.state_to_node_id: Dict[str, str] = {}  # state JSON -> node_id mapping for fast lookup
        
    def add_node(self, node: StateNode):
        """Add a node to the graph"""
        self.nodes[node.node_id] = node
        if node.node_id not in self.edges:
            self.edges[node.node_id] = []
        if node.node_id not in self.reverse_edges:
            self.reverse_edges[node.node_id] = []
        
        # Add to state mapping for fast lookup
        state_key = json.dumps(node.state_summary, sort_keys=True)
        self.state_to_node_id[state_key] = node.node_id
    
    def add_edge(self, edge: StateEdge):
        """Add an edge to the graph"""
        self.edges[edge.from_node].append(edge)
        self.reverse_edges[edge.to_node].append(edge)
    
    def get_node(self, node_id: str) -> Optional[StateNode]:
        """Get a node by ID"""
        return self.nodes.get(node_id)
    
    def get_edges_from(self, node_id: str) -> List[StateEdge]:
        """Get all edges from a node"""
        return self.edges.get(node_id, [])
    
    def get_edges_to(self, node_id: str) -> List[StateEdge]:
        """Get all edges to a node"""
        return self.reverse_edges.get(node_id, [])
    
    def get_node_id_for_state(self, state: Dict) -> Optional[str]:
        """Get the node ID for a given state"""
        state_key = json.dumps(state, sort_keys=True)
        return self.state_to_node_id.get(state_key)
    
    def get_state_mapping(self) -> Dict[str, str]:
        """Get the state to node ID mapping"""
        return self.state_to_node_id


class UnifiedStateGraphGenerator:
    """Generates the unified state graph for all Overcooked recipes"""
    
    def __init__(self):
        self.graph = UnifiedStateGraph()
        self.state_to_node_id = {}  # Maps state dict to node_id
        
        # Define all primary actions from training data
        self.primary_actions = [
            'Human Grab Onion',
            'Human Grab Chopped Onion', 
            'Human Grab Tomato',
            'Human Grab Chopped Tomato',
            'Place onion in pot',
            'Place tomato in pot',
            'Turn stove on',
            'Wait for ingredients to cook',
            'Human Grab dish',
            'Pour soup',
            'Human Stage Soup',
            'NOOP'
        ]
        
    def generate_state_graph(self):
        """Generate all possible states and transitions"""
        print(f"Generating unified state graph...")
        
        # Generate all possible state combinations
        self._generate_all_states()
        print(f"   Generated {len(self.graph.nodes)} states")
        
        # Generate all possible transitions
        self._generate_all_transitions()
        total_edges = sum(len(edges) for edges in self.graph.edges.values())
        print(f"   Generated {total_edges} transitions")
        
        # Make the state mapping accessible through the graph
        self.graph.state_to_node_id = self.state_to_node_id
        
        # Debug: Check if goal states exist
        goal_states = [node for node in self.graph.nodes.values() if node.is_goal_state]
        print(f"   Found {len(goal_states)} goal states")
        if goal_states:
            print(f"   Sample goal state: {goal_states[0].state_summary}")
        
        return self.graph
    
    def _generate_all_states(self):
        """Generate all possible state combinations"""
        # Define the possible values for each state variable
        hand_values = ['none', 'agent', 'partner']
        bool_values = [False, True]
        
        # Generate all combinations
        state_count = 0
        for onion_hand in hand_values:
            for onion_staged in bool_values:
                for onion_at_chopping in bool_values:
                    for onion_chopped in bool_values:
                        for tomato_hand in hand_values:
                            for tomato_staged in bool_values:
                                for tomato_at_chopping in bool_values:
                                    for tomato_chopped in bool_values:
                                        for onion_in_pot in bool_values:
                                            for tomato_in_pot in bool_values:
                                                for soup_cooking in bool_values:
                                                    for soup_ready in bool_values:
                                                        for soup_in_pot_not_cooking in bool_values:
                                                            for dish_hand in hand_values:
                                                                for dish_staged in bool_values:
                                                                    for soup_hand in hand_values:
                                                                        for soup_staged in bool_values:
                                                                            for soup_served in bool_values:
                                                                                # Create state summary
                                                                                state_summary = {
                                                                                    'onion_hand': onion_hand,
                                                                                    'onion_staged': onion_staged,
                                                                                    'onion_at_chopping': onion_at_chopping,
                                                                                    'onion_chopped': onion_chopped,
                                                                                    'tomato_hand': tomato_hand,
                                                                                    'tomato_staged': tomato_staged,
                                                                                    'tomato_at_chopping': tomato_at_chopping,
                                                                                    'tomato_chopped': tomato_chopped,
                                                                                    'onion_in_pot': onion_in_pot,
                                                                                    'tomato_in_pot': tomato_in_pot,
                                                                                    'soup_cooking': soup_cooking,
                                                                                    'soup_ready': soup_ready,
                                                                                    'soup_in_pot_not_cooking': soup_in_pot_not_cooking,
                                                                                    'dish_hand': dish_hand,
                                                                                    'dish_staged': dish_staged,
                                                                                    'soup_hand': soup_hand,
                                                                                    'soup_staged': soup_staged,
                                                                                    'soup_served': soup_served
                                                                                }
                                                                                
                                                                                # Check if state is valid (not all combinations are valid)
                                                                                if self._is_valid_state(state_summary):
                                                                                    node_id = f"unified_state_{state_count:06d}"
                                                                                    is_goal = soup_served
                                                                                    
                                                                                    node = StateNode(
                                                                                        node_id=node_id,
                                                                                        state_summary=state_summary,
                                                                                        is_goal_state=is_goal
                                                                                    )
                                                                                    
                                                                                    self.graph.add_node(node)
                                                                                    self.state_to_node_id[json.dumps(state_summary, sort_keys=True)] = node_id
                                                                                    state_count += 1
    
    def _is_valid_state(self, state: Dict) -> bool:
        """Check if a state is valid according to game rules"""
        # Basic validity checks
        # 1. Can't have soup cooking and ready at same time
        if state['soup_cooking'] and state['soup_ready']:
            return False
        
        # 2. Can't have soup ready and in pot not cooking
        if state['soup_ready'] and state['soup_in_pot_not_cooking']:
            return False
        
        # 3. Can't have soup cooking and in pot not cooking
        if state['soup_cooking'] and state['soup_in_pot_not_cooking']:
            return False
        
        # 4. If soup is served, it can't be in other states
        if state['soup_served']:
            if state['soup_cooking'] or state['soup_ready'] or state['soup_in_pot_not_cooking']:
                return False
        
        # 5. Can't have onion in multiple places at once
        onion_locations = 0
        if state['onion_hand'] != 'none':
            onion_locations += 1
        if state['onion_staged']:
            onion_locations += 1
        if state['onion_at_chopping']:
            onion_locations += 1
        if state['onion_in_pot']:
            onion_locations += 1
        if onion_locations > 1:
            return False
        
        # 6. Can't have tomato in multiple places at once
        tomato_locations = 0
        if state['tomato_hand'] != 'none':
            tomato_locations += 1
        if state['tomato_staged']:
            tomato_locations += 1
        if state['tomato_at_chopping']:
            tomato_locations += 1
        if state['tomato_in_pot']:
            tomato_locations += 1
        if tomato_locations > 1:
            return False
        
        # 7. Can't have dish in multiple places at once
        dish_locations = 0
        if state['dish_hand'] != 'none':
            dish_locations += 1
        if state['dish_staged']:
            dish_locations += 1
        if dish_locations > 1:
            return False
        
        # 8. Can't have soup in multiple places at once
        soup_locations = 0
        if state['soup_hand'] != 'none':
            soup_locations += 1
        if state['soup_staged']:
            soup_locations += 1
        if state['soup_cooking'] or state['soup_ready'] or state['soup_in_pot_not_cooking']:
            soup_locations += 1
        if soup_locations > 1:
            return False
        
        # 9. Can't have soup cooking without both ingredients in pot
        if state['soup_cooking'] and (not state['onion_in_pot'] or not state['tomato_in_pot']):
            return False
        
        # 10. Can't have soup ready without both ingredients in pot
        if state['soup_ready'] and (not state['onion_in_pot'] or not state['tomato_in_pot']):
            return False
        
        # 11. Can't have soup in pot not cooking without at least one ingredient
        if state['soup_in_pot_not_cooking'] and (not state['onion_in_pot'] and not state['tomato_in_pot']):
            return False
        
        return True
    
    def _generate_all_transitions(self):
        """Generate all possible transitions between states"""
        print(f"   Generating transitions...")
        
        for node_id, node in self.graph.nodes.items():
            state = node.state_summary
            
            # Generate human action transitions (primary actions)
            self._generate_human_transitions(node_id, state)
            
            # Generate environmental transitions
            self._generate_environmental_transitions(node_id, state)
    
    def _generate_human_transitions(self, node_id: str, state: Dict):
        """Generate transitions for human primary actions"""
        # Human Grab Onion (from staging)
        if state['onion_staged'] and state['onion_hand'] == 'none':
            new_state = state.copy()
            new_state['onion_hand'] = 'partner'
            new_state['onion_staged'] = False
            self._add_transition(node_id, new_state, 'Human Grab Onion')
        
        # Human Grab Chopped Onion (from chopping station)
        if state['onion_at_chopping'] and state['onion_chopped'] and state['onion_hand'] == 'none':
            new_state = state.copy()
            new_state['onion_hand'] = 'partner'
            new_state['onion_at_chopping'] = False
            self._add_transition(node_id, new_state, 'Human Grab Chopped Onion')
        
        # Human Grab Tomato (from staging)
        if state['tomato_staged'] and state['tomato_hand'] == 'none':
            new_state = state.copy()
            new_state['tomato_hand'] = 'partner'
            new_state['tomato_staged'] = False
            self._add_transition(node_id, new_state, 'Human Grab Tomato')
        
        # Human Grab Chopped Tomato (from chopping station)
        if state['tomato_at_chopping'] and state['tomato_chopped'] and state['tomato_hand'] == 'none':
            new_state = state.copy()
            new_state['tomato_hand'] = 'partner'
            new_state['tomato_at_chopping'] = False
            self._add_transition(node_id, new_state, 'Human Grab Chopped Tomato')
        
        # Human Grab dish (from staging)
        if state['dish_staged'] and state['dish_hand'] == 'none':
            new_state = state.copy()
            new_state['dish_hand'] = 'partner'
            new_state['dish_staged'] = False
            self._add_transition(node_id, new_state, 'Human Grab dish')
        
        # Place onion in pot (when human has onion)
        if state['onion_hand'] == 'partner':
            new_state = state.copy()
            new_state['onion_hand'] = 'none'
            new_state['onion_in_pot'] = True
            new_state['soup_in_pot_not_cooking'] = True
            self._add_transition(node_id, new_state, 'Place onion in pot')
        
        # Place tomato in pot (when human has tomato)
        if state['tomato_hand'] == 'partner':
            new_state = state.copy()
            new_state['tomato_hand'] = 'none'
            new_state['tomato_in_pot'] = True
            new_state['soup_in_pot_not_cooking'] = True
            self._add_transition(node_id, new_state, 'Place tomato in pot')
        
        # Turn stove on (when both ingredients are in pot)
        if state['onion_in_pot'] and state['tomato_in_pot'] and state['soup_in_pot_not_cooking']:
            new_state = state.copy()
            new_state['soup_in_pot_not_cooking'] = False
            new_state['soup_cooking'] = True
            self._add_transition(node_id, new_state, 'Turn stove on')
        
        # Pour soup (when soup is ready and human has dish)
        if state['soup_ready'] and state['dish_hand'] == 'partner':
            new_state = state.copy()
            new_state['soup_hand'] = 'partner'
            new_state['onion_in_pot'] = False
            new_state['tomato_in_pot'] = False
            new_state['dish_hand'] = 'none'
            new_state['soup_ready'] = False
            self._add_transition(node_id, new_state, 'Pour soup')
        
        # Human Stage Soup (when human has soup)
        if state['soup_hand'] == 'partner':
            new_state = state.copy()
            new_state['soup_hand'] = 'none'
            new_state['soup_staged'] = True
            self._add_transition(node_id, new_state, 'Human Stage Soup')
        
        # ADD MISSING TRANSITIONS: Ingredients becoming available from dispensers
        
        # Onion becomes available at staging (from dispenser)
        if (not state['onion_staged'] and not state['onion_hand'] == 'partner' and 
            not state['onion_at_chopping'] and not state['onion_in_pot']):
            new_state = state.copy()
            new_state['onion_staged'] = True
            self._add_transition(node_id, new_state, 'Onion Available')
        
        # Tomato becomes available at staging (from dispenser)
        if (not state['tomato_staged'] and not state['tomato_hand'] == 'partner' and 
            not state['tomato_at_chopping'] and not state['tomato_in_pot']):
            new_state = state.copy()
            new_state['tomato_staged'] = True
            self._add_transition(node_id, new_state, 'Tomato Available')
        
        # Dish becomes available at staging (from dispenser)
        if (not state['dish_staged'] and not state['dish_hand'] == 'partner'):
            new_state = state.copy()
            new_state['dish_staged'] = True
            self._add_transition(node_id, new_state, 'Dish Available')
        
        # ADD MISSING CHOPPING TRANSITIONS
        
        # Place onion at chopping station (auto-chops)
        if (state['onion_hand'] == 'partner' and not state['onion_chopped'] and 
            not state['onion_at_chopping']):
            new_state = state.copy()
            new_state['onion_hand'] = 'none'
            new_state['onion_at_chopping'] = True
            new_state['onion_chopped'] = True
            self._add_transition(node_id, new_state, 'Place onion at chopping')
        
        # Place tomato at chopping station (auto-chops)
        if (state['tomato_hand'] == 'partner' and not state['tomato_chopped'] and 
            not state['tomato_at_chopping']):
            new_state = state.copy()
            new_state['tomato_hand'] = 'none'
            new_state['tomato_at_chopping'] = True
            new_state['tomato_chopped'] = True
            self._add_transition(node_id, new_state, 'Place tomato at chopping')
    
    def _generate_environmental_transitions(self, node_id: str, state: Dict):
        """Generate transitions for environmental changes"""
        # Cooking starts automatically when both ingredients are in pot
        if state['onion_in_pot'] and state['tomato_in_pot'] and state['soup_in_pot_not_cooking']:
            new_state = state.copy()
            new_state['soup_in_pot_not_cooking'] = False
            new_state['soup_cooking'] = True
            self._add_transition(node_id, new_state, 'cooking_start')
        
        # Soup becomes ready when cooking completes
        if state['soup_cooking']:
            new_state = state.copy()
            new_state['soup_cooking'] = False
            new_state['soup_ready'] = True
            self._add_transition(node_id, new_state, 'soup_ready')
    
    def _add_transition(self, from_node_id: str, to_state: Dict, action: str):
        """Add a transition between states"""
        to_state_key = json.dumps(to_state, sort_keys=True)
        to_node_id = self.state_to_node_id.get(to_state_key)
        
        if to_node_id and to_node_id != from_node_id:
            edge = StateEdge(
                from_node=from_node_id,
                to_node=to_node_id,
                action=action,
                weight=1.0
            )
            self.graph.add_edge(edge)


# Factory function to get the unified state graph
def get_unified_state_graph() -> UnifiedStateGraph:
    """Get the unified state graph for all recipe types"""
    generator = UnifiedStateGraphGenerator()
    return generator.generate_state_graph()
