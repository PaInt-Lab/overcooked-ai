from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass
import json
from collections import deque
import heapq

@dataclass
class StateNode:
    """Represents a node in the state graph"""
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
    action: str  # 'pickup(onion)', 'human_grab_onion', etc.
    weight: float = 1.0  # For weighted path finding

class StateGraph:
    """Manages the state graph for coordinated navigation"""
    
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
    
    def find_path_to_goal(self, start_node_id: str, goal_node_id: str) -> List[str]:
        """Find shortest path from start to goal using A*"""
        try:
            if start_node_id not in self.nodes or goal_node_id not in self.nodes:
                print(f"Pathfinding failed: start_node_id={start_node_id}, goal_node_id={goal_node_id}")
                print(f"   Available nodes: {list(self.nodes.keys())[:5]}...")
                return []
            
            # print(f"Starting pathfinding from {start_node_id} to {goal_node_id}")
            
            # Priority queue for A*: (f_score, node_id, path)
            open_set = [(0, start_node_id, [start_node_id])]
            closed_set = set()
            
            # g_score[node] = cost from start to node
            g_score = {start_node_id: 0}
            
            iterations = 0
            max_iterations = 50000  # Reasonable limit
            
            while open_set and iterations < max_iterations:
                iterations += 1
                f_score, current_id, path = heapq.heappop(open_set)
                
                if current_id == goal_node_id:
                    # print(f"Path found in {iterations} iterations: {path}")
                    return path
                
                if current_id in closed_set:
                    continue
                    
                closed_set.add(current_id)
                
                edges = self.get_edges_from(current_id)
                if iterations == 1:  # Debug first iteration
                    # print(f"   First node edges: {[edge.action for edge in edges]}")
                    pass
                
                for edge in edges:
                    neighbor_id = edge.to_node
                    tentative_g_score = g_score[current_id] + edge.weight
                    
                    if neighbor_id not in g_score or tentative_g_score < g_score[neighbor_id]:
                        g_score[neighbor_id] = tentative_g_score
                        new_path = path + [neighbor_id]
                        h_score = self._heuristic(neighbor_id, goal_node_id)
                        f_score = tentative_g_score + h_score
                        heapq.heappush(open_set, (f_score, neighbor_id, new_path))
            
            # Progress reporting
            if iterations % 1000 == 0:
                # print(f"   Pathfinding progress: {iterations} iterations, open set size: {len(open_set)}")
                pass
            
            if iterations >= max_iterations:
                # print(f"Pathfinding timed out after {iterations} iterations")
                # print(f"   Open set size: {len(open_set)}, Closed set size: {len(closed_set)}")
                pass
            else:
                # print(f"No path found after {iterations} iterations")
                # print(f"   Open set size: {len(open_set)}, Closed set size: {len(closed_set)}")
                pass
            
            return []  # No path found
            
        except Exception as e:
            # print(f"Exception during pathfinding: {e}")
            # import traceback
            # traceback.print_exc()
            return []
    
    def _heuristic(self, node_id: str, goal_id: str) -> float:
        """Heuristic function for A* - distance to goal"""
        # Get the states for both nodes
        current_node = self.get_node(node_id)
        goal_node = self.get_node(goal_id)
        
        if not current_node or not goal_node:
            return 0
        
        current_state = current_node.state_summary
        goal_state = goal_node.state_summary
        
        # Calculate heuristic based on key differences
        # Prioritize soup_served as the main goal
        if goal_state['soup_served'] and not current_state['soup_served']:
            # We need to get soup served - this is the main goal
            if current_state['soup_hand'] == 'agent' and not current_state['soup_served']:
                return 1.0  # Just need to place soup
            elif current_state['soup_ready'] and current_state['soup_hand'] == 'none':
                return 2.0  # Need to pour soup then place it
            elif current_state['soup_cooking']:
                return 3.0  # Need to wait for cooking, then pour, then place
            elif current_state['onion_in_pot'] and current_state['tomato_in_pot']:
                return 4.0  # Need to start cooking, wait, pour, place
            else:
                # Need to get ingredients, put in pot, cook, etc.
                missing_ingredients = 0
                if not current_state['onion_in_pot']:
                    missing_ingredients += 1
                if not current_state['tomato_in_pot']:
                    missing_ingredients += 1
                return 5.0 + missing_ingredients
        
        # For other goals, use simple difference counting
        differences = 0
        for key in current_state:
            if current_state[key] != goal_state[key]:
                differences += 1
        
        return differences * 0.5
    
    def get_next_action(self, current_node_id: str, goal_node_id: str) -> Optional[str]:
        """Get the next action to take toward the goal"""
        path = self.find_path_to_goal(current_node_id, goal_node_id)
        if len(path) < 2:
            return None
        
        # Find the edge from current to next node
        for edge in self.get_edges_from(current_node_id):
            if edge.to_node == path[1]:
                return edge.action
        
        return None

class StateGraphGenerator:
    """Generates the complete state graph for Overcooked"""
    
    def __init__(self):
        self.graph = StateGraph()
        self.state_to_node_id = {}  # Maps state dict to node_id
        
    def generate_state_graph(self):
        """Generate all possible states and transitions"""
        print(f"Generating state graph...")
        
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
        
        # Test connectivity
        self.test_connectivity()
        
        return self.graph
    
    def test_connectivity(self):
        """Test if the graph is well-connected"""
        print(f"Testing graph connectivity...")
        
        # Find initial and goal states
        initial_states = []
        goal_states = []
        
        for node in self.graph.nodes.values():
            if (node.state_summary['onion_hand'] == 'none' and 
                node.state_summary['tomato_hand'] == 'none' and
                node.state_summary['dish_hand'] == 'none' and
                node.state_summary['soup_hand'] == 'none' and
                not node.state_summary['soup_served']):
                initial_states.append(node)
            
            if node.is_goal_state:
                goal_states.append(node)
        
        print(f"   Found {len(initial_states)} initial states")
        print(f"   Found {len(goal_states)} goal states")
        
        if initial_states and goal_states:
            # Test path from first initial to first goal
            start_node = initial_states[0]
            goal_node = goal_states[0]
            
            print(f"   Testing path from {start_node.node_id} to {goal_node.node_id}")
            print(f"   Start state: {start_node.state_summary}")
            print(f"   Goal state: {goal_node.state_summary}")
            
            path = self.graph.find_path_to_goal(start_node.node_id, goal_node.node_id)
            if path:
                print(f"   Path found with {len(path)} steps")
                return True
            else:
                print(f"   No path found")
                return False
        
        return False
    
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
                                                                                    node_id = f"state_{state_count:06d}"
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
        
        # 10. Can't have soup cooking without both ingredients in pot
        if state['soup_cooking'] and (not state['onion_in_pot'] or not state['tomato_in_pot']):
            return False
        
        # 11. Can't have soup ready without both ingredients in pot
        if state['soup_ready'] and (not state['onion_in_pot'] or not state['tomato_in_pot']):
            return False
        
        # 12. Can't have soup in pot not cooking without at least one ingredient
        if state['soup_in_pot_not_cooking'] and (not state['onion_in_pot'] and not state['tomato_in_pot']):
            return False
        
        return True
    
    def _generate_all_transitions(self):
        """Generate all possible transitions between states"""
        for node_id, node in self.graph.nodes.items():
            state = node.state_summary
            
            # Generate robot action transitions
            self._generate_robot_transitions(node_id, state)
            
            # Generate human action transitions
            self._generate_human_transitions(node_id, state)
            
            # Generate environmental transitions
            self._generate_environmental_transitions(node_id, state)
    
    def _generate_robot_transitions(self, node_id: str, state: Dict):
        """Generate transitions for robot actions"""
        # Robot pickup actions
        # Pickup onion from dispenser
        if (state['onion_hand'] == 'none' and not state['onion_staged'] and 
            not state['onion_at_chopping'] and not state['onion_in_pot']):
            new_state = state.copy()
            new_state['onion_hand'] = 'agent'
            self._add_transition(node_id, new_state, 'pickup(onion)')
        
        # Pickup tomato from dispenser
        if (state['tomato_hand'] == 'none' and not state['tomato_staged'] and 
            not state['tomato_at_chopping'] and not state['tomato_in_pot']):
            new_state = state.copy()
            new_state['tomato_hand'] = 'agent'
            self._add_transition(node_id, new_state, 'pickup(tomato)')
        
        # Pickup chopped onion from chopping station
        if (state['onion_chopped'] and state['onion_at_chopping'] and 
            state['onion_hand'] == 'none'):
            new_state = state.copy()
            new_state['onion_hand'] = 'agent'
            new_state['onion_at_chopping'] = False
            self._add_transition(node_id, new_state, 'pickup(chopped_onion)')
        
        # Pickup chopped tomato from chopping station
        if (state['tomato_chopped'] and state['tomato_at_chopping'] and 
            state['tomato_hand'] == 'none'):
            new_state = state.copy()
            new_state['tomato_hand'] = 'agent'
            new_state['tomato_at_chopping'] = False
            self._add_transition(node_id, new_state, 'pickup(chopped_tomato)')
        
        # Pickup dish from dispenser (when soup is cooking - proactive behavior)
        if (state['dish_hand'] == 'none' and not state['dish_staged'] and 
            state['soup_cooking']):
            new_state = state.copy()
            new_state['dish_hand'] = 'agent'
            self._add_transition(node_id, new_state, 'pickup(dish)')
        
        # Pickup dish from dispenser (when soup is ready - backup behavior)
        if (state['dish_hand'] == 'none' and not state['dish_staged'] and 
            state['soup_ready']):
            new_state = state.copy()
            new_state['dish_hand'] = 'agent'
            self._add_transition(node_id, new_state, 'pickup(dish_ready)')
        
        # Pickup soup from staging (when soup is ready and staged)
        if (state['soup_ready'] and state['soup_staged'] and 
            state['soup_hand'] == 'none'):
            new_state = state.copy()
            new_state['soup_hand'] = 'agent'
            new_state['soup_staged'] = False
            self._add_transition(node_id, new_state, 'pickup(soup)')
        
        # Pickup soup from staging (when soup is staged, even if not ready - for proactive behavior)
        if (state['soup_staged'] and state['soup_hand'] == 'none' and 
            not state['soup_ready']):
            new_state = state.copy()
            new_state['soup_hand'] = 'agent'
            new_state['soup_staged'] = False
            self._add_transition(node_id, new_state, 'pickup(soup)')
        
        # Robot place actions
        # Place onion at chopping station (auto-chops)
        if state['onion_hand'] == 'agent' and not state['onion_chopped']:
            new_state = state.copy()
            new_state['onion_hand'] = 'none'
            new_state['onion_at_chopping'] = True
            new_state['onion_chopped'] = True
            self._add_transition(node_id, new_state, 'place(onion, chopping_station)')
        
        # Place onion at staging station
        if state['onion_hand'] == 'agent' and not state['onion_staged']:
            new_state = state.copy()
            new_state['onion_hand'] = 'none'
            new_state['onion_staged'] = True
            self._add_transition(node_id, new_state, 'place(onion, staging_station)')
        
        # Place tomato at chopping station (auto-chops)
        if state['tomato_hand'] == 'agent' and not state['tomato_chopped']:
            new_state = state.copy()
            new_state['tomato_hand'] = 'none'
            new_state['tomato_at_chopping'] = True
            new_state['tomato_chopped'] = True
            self._add_transition(node_id, new_state, 'place(tomato, chopping_station)')
        
        # Place tomato at staging station
        if state['tomato_hand'] == 'agent' and not state['tomato_staged']:
            new_state = state.copy()
            new_state['tomato_hand'] = 'none'
            new_state['tomato_staged'] = True
            self._add_transition(node_id, new_state, 'place(tomato, staging_station)')
        
        # Place chopped onion at staging station
        if (state['onion_hand'] == 'agent' and state['onion_chopped'] and 
            not state['onion_staged']):
            new_state = state.copy()
            new_state['onion_hand'] = 'none'
            new_state['onion_staged'] = True
            self._add_transition(node_id, new_state, 'place(chopped_onion)')
        
        # Place chopped tomato at staging station
        if (state['tomato_hand'] == 'agent' and state['tomato_chopped'] and 
            not state['tomato_staged']):
            new_state = state.copy()
            new_state['tomato_hand'] = 'none'
            new_state['tomato_staged'] = True
            self._add_transition(node_id, new_state, 'place(chopped_tomato)')
        
        # Place dish at staging station
        if state['dish_hand'] == 'agent' and not state['dish_staged']:
            new_state = state.copy()
            new_state['dish_hand'] = 'none'
            new_state['dish_staged'] = True
            self._add_transition(node_id, new_state, 'place(dish)')
        
        # Place soup at serving station
        if state['soup_hand'] == 'agent' and not state['soup_served']:
            new_state = state.copy()
            new_state['soup_hand'] = 'none'
            new_state['soup_served'] = True
            # Reset chopped flags when soup is served
            new_state['onion_chopped'] = False
            new_state['tomato_chopped'] = False
            self._add_transition(node_id, new_state, 'robot_serve_soup')
        
    def _generate_human_transitions(self, node_id: str, state: Dict):
        """Generate transitions for human actions"""
        # Human pickup actions
        # Human can grab onion from dispenser
        if (state['onion_hand'] == 'none' and not state['onion_staged'] and 
            not state['onion_at_chopping'] and not state['onion_in_pot']):
            new_state = state.copy()
            new_state['onion_hand'] = 'partner'
            self._add_transition(node_id, new_state, 'human_pickup_onion')
        
        # Human can grab tomato from dispenser 
        if (state['tomato_hand'] == 'none' and not state['tomato_staged'] and 
            not state['tomato_at_chopping'] and not state['tomato_in_pot']):
            new_state = state.copy()
            new_state['tomato_hand'] = 'partner'
            self._add_transition(node_id, new_state, 'human_pickup_tomato')
        
        # Human can grab chopped onion from chopping station
        if (state['onion_chopped'] and state['onion_at_chopping'] and 
            state['onion_hand'] == 'none'):
            new_state = state.copy()
            new_state['onion_hand'] = 'partner'
            new_state['onion_at_chopping'] = False
            self._add_transition(node_id, new_state, 'human_pickup_chopped_onion')
        
        # Human can grab chopped tomato from chopping station
        if (state['tomato_chopped'] and state['tomato_at_chopping'] and 
            state['tomato_hand'] == 'none'):
            new_state = state.copy()
            new_state['tomato_hand'] = 'partner'
            new_state['tomato_at_chopping'] = False
            self._add_transition(node_id, new_state, 'human_pickup_chopped_tomato')
        
        # Human can grab dish from dispenser
        if (state['dish_hand'] == 'none' and not state['dish_staged']):
            new_state = state.copy()
            new_state['dish_hand'] = 'partner'
            self._add_transition(node_id, new_state, 'human_pickup_dish')
        
        # Human can grab soup from staging
        if (state['soup_ready'] and state['soup_staged'] and 
            state['soup_hand'] == 'none'):
            new_state = state.copy()
            new_state['soup_hand'] = 'partner'
            new_state['soup_staged'] = False
            self._add_transition(node_id, new_state, 'human_pickup_soup')
        
        # Human can grab staged items
        if state['onion_staged'] and state['onion_hand'] == 'none':
            new_state = state.copy()
            new_state['onion_staged'] = False
            new_state['onion_hand'] = 'partner'
            self._add_transition(node_id, new_state, 'human_grab_onion')
        
        if state['tomato_staged'] and state['tomato_hand'] == 'none':
            new_state = state.copy()
            new_state['tomato_staged'] = False
            new_state['tomato_hand'] = 'partner'
            self._add_transition(node_id, new_state, 'human_grab_tomato')
        
        if state['dish_staged'] and state['dish_hand'] == 'none':
            new_state = state.copy()
            new_state['dish_staged'] = False
            new_state['dish_hand'] = 'partner'
            self._add_transition(node_id, new_state, 'human_grab_dish')
        
        if state['soup_staged'] and state['soup_hand'] == 'none':
            new_state = state.copy()
            new_state['soup_staged'] = False
            new_state['soup_hand'] = 'partner'
            self._add_transition(node_id, new_state, 'human_grab_soup')
        
        # Human place actions
        # Human can place onion at chopping station
        if state['onion_hand'] == 'partner' and not state['onion_chopped']:
            new_state = state.copy()
            new_state['onion_hand'] = 'none'
            new_state['onion_at_chopping'] = True
            new_state['onion_chopped'] = True
            self._add_transition(node_id, new_state, 'human_place_onion_chopping')
        
        # Human can place onion at staging station
        if state['onion_hand'] == 'partner' and not state['onion_staged']:
            new_state = state.copy()
            new_state['onion_hand'] = 'none'
            new_state['onion_staged'] = True
            self._add_transition(node_id, new_state, 'human_place_onion_staging')
        
        # Human can place tomato at chopping station
        if state['tomato_hand'] == 'partner' and not state['tomato_chopped']:
            new_state = state.copy()
            new_state['tomato_hand'] = 'none'
            new_state['tomato_at_chopping'] = True
            new_state['tomato_chopped'] = True
            self._add_transition(node_id, new_state, 'human_place_tomato_chopping')
        
        # Human can place tomato at staging station
        if state['tomato_hand'] == 'partner' and not state['tomato_staged']:
            new_state = state.copy()
            new_state['tomato_hand'] = 'none'
            new_state['tomato_staged'] = True
            self._add_transition(node_id, new_state, 'human_place_tomato_staging')
        
        # Human can place chopped onion at staging station
        if (state['onion_hand'] == 'partner' and state['onion_chopped'] and 
            not state['onion_staged']):
            new_state = state.copy()
            new_state['onion_hand'] = 'none'
            new_state['onion_staged'] = True
            self._add_transition(node_id, new_state, 'human_place_chopped_onion')
        
        # Human can place chopped tomato at staging station
        if (state['tomato_hand'] == 'partner' and state['tomato_chopped'] and 
            not state['tomato_staged']):
            new_state = state.copy()
            new_state['tomato_hand'] = 'none'
            new_state['tomato_staged'] = True
            self._add_transition(node_id, new_state, 'human_place_chopped_tomato')
        
        # Human can place dish at staging station
        if state['dish_hand'] == 'partner' and not state['dish_staged']:
            new_state = state.copy()
            new_state['dish_hand'] = 'none'
            new_state['dish_staged'] = True
            self._add_transition(node_id, new_state, 'human_place_dish')
        
        # Human can place soup at staging station
        if state['soup_hand'] == 'partner' and not state['soup_staged']:
            new_state = state.copy()
            new_state['soup_hand'] = 'none'
            new_state['soup_staged'] = True
            self._add_transition(node_id, new_state, 'human_stage_soup')
        
        # Human can place soup at serving station
        if state['soup_hand'] == 'partner' and not state['soup_served']:
            new_state = state.copy()
            new_state['soup_hand'] = 'none'
            new_state['soup_served'] = True
            # Reset chopped flags when soup is served
            new_state['onion_chopped'] = False
            new_state['tomato_chopped'] = False
            self._add_transition(node_id, new_state, 'human_place_soup')
        
        # Human can place items in pot (only if they were chopped and staged)
        if state['onion_hand'] == 'partner' and state['onion_chopped']:
            new_state = state.copy()
            new_state['onion_hand'] = 'none'
            new_state['onion_in_pot'] = True
            # If it was at chopping, remove it from there
            if state['onion_at_chopping']:
                new_state['onion_at_chopping'] = False
            # Set soup_in_pot_not_cooking to True when ANY ingredient is placed in pot
            new_state['soup_in_pot_not_cooking'] = True
            self._add_transition(node_id, new_state, 'human_place_onion_in_pot')
        
        if state['tomato_hand'] == 'partner' and state['tomato_chopped']:
            new_state = state.copy()
            new_state['tomato_hand'] = 'none'
            new_state['tomato_in_pot'] = True
            # If it was at chopping, remove it from there
            if state['tomato_at_chopping']:
                new_state['tomato_at_chopping'] = False
            # Set soup_in_pot_not_cooking to True when ANY ingredient is placed in pot
            new_state['soup_in_pot_not_cooking'] = True
            self._add_transition(node_id, new_state, 'human_place_tomato_in_pot')
        
        # Human can pour soup from pot into their hand (when soup is ready)
        if state['soup_ready'] and state['soup_hand'] == 'none':
            new_state = state.copy()
            new_state['soup_hand'] = 'partner'
            # Remove soup from pot states
            new_state['onion_in_pot'] = False
            new_state['tomato_in_pot'] = False
            new_state['soup_ready'] = False
            self._add_transition(node_id, new_state, 'human_pour_soup')
        
        # Human can turn stove on (when both ingredients are in pot but not cooking)
        if state['onion_in_pot'] and state['tomato_in_pot'] and state['soup_in_pot_not_cooking']:
            new_state = state.copy()
            new_state['soup_in_pot_not_cooking'] = False
            new_state['soup_cooking'] = True
            self._add_transition(node_id, new_state, 'human_turn_stove_on')
    
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
        
        # When soup is served, reset the chopped flags and pot states
        if state['soup_served']:
            # This is handled in the place actions, but we can also have environmental reset
            new_state = state.copy()
            new_state['onion_chopped'] = False
            new_state['tomato_chopped'] = False
            new_state['onion_in_pot'] = False
            new_state['tomato_in_pot'] = False
            new_state['soup_cooking'] = False
            new_state['soup_ready'] = False
            new_state['soup_in_pot_not_cooking'] = False
            self._add_transition(node_id, new_state, 'reset_after_serving')
    
    def _add_transition(self, from_node_id: str, to_state: Dict, action: str):
        """Add a transition between states with strategic weighting"""
        to_state_key = json.dumps(to_state, sort_keys=True)
        to_node_id = self.state_to_node_id.get(to_state_key)
        
        if to_node_id and to_node_id != from_node_id:
            # Strategic weighting system:
            # - Robot secondary tasks: 0.5 (BEST - preferred)
            # - Human primary tasks: 0.5 (SAME AS ROBOT SECONDARY - also preferred)
            # - Human secondary tasks: 1.0 (HIGHER - less preferred)
            # - Robot primary tasks: 1000.0 (BLOCKED - impossible)
            
            weight = 1.0  # Default weight
            
            # Robot actions
            if action.startswith('pickup(') or action.startswith('place('):
                # Robot secondary tasks (fetching, staging, serving)
                if any(keyword in action for keyword in ['pickup(', 'place(', 'staging_station', 'delivery']):
                    weight = 0.5  # Preferred robot secondary tasks
                else:
                    weight = 1000.0  # Block robot primary tasks
            
            # Human actions
            elif action.startswith('human_'):
                # Human primary tasks (cooking, placing in pot, pouring)
                if any(keyword in action for keyword in ['place_onion_in_pot', 'place_tomato_in_pot', 'pour_soup', 'turn_stove_on']):
                    weight = 0.5  # Preferred human primary tasks
                else:
                    weight = 1.0  # Human secondary tasks (less preferred but allowed)
            
            # Environmental actions (normal weight)
            else:
                weight = 1.0
            
            edge = StateEdge(
                from_node=from_node_id,
                to_node=to_node_id,
                action=action,
                weight=weight
            )
            self.graph.add_edge(edge)
        else:
            # Debug: Log missing transitions (commented out to reduce noise)
            if not to_node_id:
                # print(f"Missing target state for transition: {action}")
                # print(f"   From: {from_node_id}")
                # print(f"   To state: {to_state}")
                # print(f"   To state key: {to_state_key}")
                # print(f"   Available states: {len(self.state_to_node_id)}")
                # Check if this state should be valid
                if self._is_valid_state(to_state):
                    # print(f"   State is valid but not in mapping!")
                    pass
                else:
                    # print(f"   State is invalid according to rules")
                    pass
    
    def get_node_id_for_state(self, state: Dict) -> Optional[str]:
        """Get the node ID for a given state"""
        state_key = json.dumps(state, sort_keys=True)
        return self.state_to_node_id.get(state_key)
    
    def get_state_mapping(self) -> Dict[str, str]:
        """Get the state to node ID mapping"""
        return self.state_to_node_id 


class BaseStateGraphGenerator:
    """Base class for state graph generators with common functionality"""
    
    def __init__(self, soup_type: str):
        self.soup_type = soup_type
        self.state_graph = StateGraph()
        
    def generate_state_graph(self):
        """Generate state graph for specific soup type"""
        raise NotImplementedError
        
    def _is_valid_state(self, state: Dict) -> bool:
        """Check if state is valid for this soup type"""
        raise NotImplementedError
        
    def _generate_robot_transitions(self, node_id: str, state: Dict):
        """Generate robot transitions for this soup type"""
        raise NotImplementedError
        
    def _generate_human_transitions(self, node_id: str, state: Dict):
        """Generate human transitions for this soup type"""
        raise NotImplementedError


class PlainOnionSoupStateGraphGenerator(BaseStateGraphGenerator):
    """State graph generator for plain onion soup (no chopping)"""
    
    def __init__(self):
        super().__init__("plain_onion_soup")
        
    def generate_state_graph(self):
        """Generate state graph for plain onion soup"""
        print(f"Generating state graph for {self.soup_type}...")
        
        # Generate all valid states
        all_states = self._generate_all_states()
        print(f"Generated {len(all_states)} valid states")
        
        # Add nodes to graph
        for state in all_states:
            node_id = self._get_state_id(state)
            node = StateNode(
                node_id=node_id,
                state_summary=state,
                is_goal_state=self._is_goal_state(state)
            )
            self.state_graph.add_node(node)
        
        # Generate transitions
        self._generate_all_transitions()
        print(f"Generated transitions between states")
        
        return self.state_graph
        
    def _generate_all_states(self):
        """Generate all valid states for plain onion soup"""
        states = []
        
        # State variables for plain onion soup (includes all required variables)
        onion_hand_options = ['none', 'agent', 'partner']
        onion_staged_options = [True, False]
        onion_at_chopping_options = [True, False]
        onion_chopped_options = [True, False]
        onion_in_pot_options = [True, False]
        tomato_hand_options = ['none', 'agent', 'partner']
        tomato_staged_options = [True, False]
        tomato_at_chopping_options = [True, False]
        tomato_chopped_options = [True, False]
        tomato_in_pot_options = [True, False]
        soup_cooking_options = [True, False]
        soup_ready_options = [True, False]
        soup_hand_options = ['none', 'agent', 'partner']
        soup_staged_options = [True, False]
        soup_in_pot_not_cooking_options = [True, False]
        dish_hand_options = ['none', 'agent', 'partner']
        dish_staged_options = [True, False]
        soup_served_options = [True, False]
        
        # Generate all combinations
        for onion_hand in onion_hand_options:
            for onion_staged in onion_staged_options:
                for onion_at_chopping in onion_at_chopping_options:
                    for onion_chopped in onion_chopped_options:
                        for onion_in_pot in onion_in_pot_options:
                            for tomato_hand in tomato_hand_options:
                                for tomato_staged in tomato_staged_options:
                                    for tomato_at_chopping in tomato_at_chopping_options:
                                        for tomato_chopped in tomato_chopped_options:
                                            for tomato_in_pot in tomato_in_pot_options:
                                                for soup_cooking in soup_cooking_options:
                                                    for soup_ready in soup_ready_options:
                                                        for soup_hand in soup_hand_options:
                                                            for soup_staged in soup_staged_options:
                                                                for soup_in_pot_not_cooking in soup_in_pot_not_cooking_options:
                                                                    for dish_hand in dish_hand_options:
                                                                        for dish_staged in dish_staged_options:
                                                                            for soup_served in soup_served_options:
                                                                                state = {
                                                                                    'onion_hand': onion_hand,
                                                                                    'onion_staged': onion_staged,
                                                                                    'onion_at_chopping': onion_at_chopping,
                                                                                    'onion_chopped': onion_chopped,
                                                                                    'onion_in_pot': onion_in_pot,
                                                                                    'tomato_hand': tomato_hand,
                                                                                    'tomato_staged': tomato_staged,
                                                                                    'tomato_at_chopping': tomato_at_chopping,
                                                                                    'tomato_chopped': tomato_chopped,
                                                                                    'tomato_in_pot': tomato_in_pot,
                                                                                    'soup_cooking': soup_cooking,
                                                                                    'soup_ready': soup_ready,
                                                                                    'soup_hand': soup_hand,
                                                                                    'soup_staged': soup_staged,
                                                                                    'soup_in_pot_not_cooking': soup_in_pot_not_cooking,
                                                                                    'dish_hand': dish_hand,
                                                                                    'dish_staged': dish_staged,
                                                                                    'soup_served': soup_served
                                                                                }
                                                                                
                                                                                if self._is_valid_state(state):
                                                                                    states.append(state)
        
        return states
        
    def _is_valid_state(self, state: Dict) -> bool:
        """Check if state is valid for plain onion soup"""
        # Basic validity checks
        if state['soup_served'] and not state['soup_ready']:
            return False  # Can't serve unready soup
            
        if state['soup_cooking'] and state['soup_in_pot_not_cooking']:
            return False  # Can't be cooking and not cooking at same time
            
        if state['onion_in_pot'] and state['onion_hand'] != 'none':
            return False  # Can't hold onion if it's in pot
            
        # For plain onion soup, tomato variables should be false/none since no tomato is needed
        if state['tomato_hand'] != 'none':
            return False  # No tomato in plain onion soup
        if state['tomato_staged'] or state['tomato_at_chopping'] or state['tomato_chopped'] or state['tomato_in_pot']:
            return False  # No tomato processing in plain onion soup
            
        # For plain onion soup, onion chopping variables should be false since no chopping needed
        if state['onion_at_chopping'] or state['onion_chopped']:
            return False  # No onion chopping in plain onion soup
            
        # Logical constraints
        if state['onion_in_pot'] and state['onion_staged']:
            return False  # Can't be in pot and staged at same time
            
        if state['soup_cooking'] and state['soup_ready']:
            return False  # Can't be cooking and ready at same time
            
        if state['soup_ready'] and not state['onion_in_pot']:
            return False  # Need onion in pot for soup to be ready
            
        if state['soup_ready'] and state['soup_in_pot_not_cooking']:
            return False  # If soup is ready, it's not "not cooking" - it's already done
            
        if state['soup_in_pot_not_cooking'] and state['soup_cooking']:
            return False  # Can't be in pot not cooking AND cooking
                
        return True
        
    def _is_goal_state(self, state: Dict) -> bool:
        """Check if state is a goal state (soup served)"""
        return state['soup_served']
        
    def _get_state_id(self, state: Dict) -> str:
        """Generate unique ID for state"""
        return f"onion_plain_{hash(str(sorted(state.items())))}"
        
    def _generate_all_transitions(self):
        """Generate all transitions between states"""
        for node_id, node in self.state_graph.nodes.items():
            self._generate_robot_transitions(node_id, node.state_summary)
            self._generate_human_transitions(node_id, node.state_summary)
            
    def _generate_robot_transitions(self, node_id: str, state: Dict):
        """Generate robot transitions for plain onion soup"""
        # Robot can pick up onion
        if state['onion_hand'] == 'none' and not state['onion_staged'] and not state['onion_in_pot']:
            new_state = state.copy()
            new_state['onion_hand'] = 'agent'
            self._add_transition(node_id, new_state, 'pickup(onion)')
            
        # Robot can place onion at staging
        if state['onion_hand'] == 'agent':
            new_state = state.copy()
            new_state['onion_hand'] = 'none'
            new_state['onion_staged'] = True
            self._add_transition(node_id, new_state, 'place(onion, staging_station)')
            
        # Robot can pick up dish
        if state['dish_hand'] == 'none' and not state['dish_staged']:
            new_state = state.copy()
            new_state['dish_hand'] = 'agent'
            self._add_transition(node_id, new_state, 'pickup(dish)')
            
        # Robot can place dish at staging
        if state['dish_hand'] == 'agent':
            new_state = state.copy()
            new_state['dish_hand'] = 'none'
            new_state['dish_staged'] = True
            self._add_transition(node_id, new_state, 'place(dish)')
            
        # Robot can pick up soup
        if state['soup_ready'] and state['soup_hand'] == 'none' and state['soup_staged']:
            new_state = state.copy()
            new_state['soup_hand'] = 'agent'
            self._add_transition(node_id, new_state, 'pickup(soup)')
            
        # Robot can serve soup
        if state['soup_hand'] == 'agent':
            new_state = state.copy()
            new_state['soup_hand'] = 'none'
            new_state['soup_served'] = True
            self._add_transition(node_id, new_state, 'place(soup)')
            
    def _generate_human_transitions(self, node_id: str, state: Dict):
        """Generate human transitions for plain onion soup"""
        # Human can grab onion from staging
        if state['onion_staged'] and state['onion_hand'] == 'none':
            new_state = state.copy()
            new_state['onion_hand'] = 'partner'
            new_state['onion_staged'] = False
            self._add_transition(node_id, new_state, 'human_grab_onion')
            
        # Human can place onion in pot
        if state['onion_hand'] == 'partner':
            new_state = state.copy()
            new_state['onion_hand'] = 'none'
            new_state['onion_in_pot'] = True
            self._add_transition(node_id, new_state, 'human_place_onion_in_pot')
            
        # Human can turn stove on
        if state['onion_in_pot'] and not state['soup_cooking'] and not state['soup_ready']:
            new_state = state.copy()
            new_state['soup_cooking'] = True
            self._add_transition(node_id, new_state, 'human_turn_stove_on')
            
        # Cooking completes automatically (environmental transition)
        if state['soup_cooking'] and not state['soup_ready']:
            new_state = state.copy()
            new_state['soup_cooking'] = False
            new_state['soup_ready'] = True
            self._add_transition(node_id, new_state, 'cooking_complete')
            
        # Human can pour soup
        if state['soup_ready'] and not state['soup_staged']:
            new_state = state.copy()
            new_state['soup_staged'] = True
            self._add_transition(node_id, new_state, 'human_pour_soup')
            
        # Human can grab dish from staging
        if state['dish_staged'] and state['dish_hand'] == 'none':
            new_state = state.copy()
            new_state['dish_hand'] = 'partner'
            new_state['dish_staged'] = False
            self._add_transition(node_id, new_state, 'human_grab_dish')
            
    def _add_transition(self, from_node_id: str, to_state: Dict, action: str):
        """Add a transition to the state graph"""
        to_node_id = self._get_state_id(to_state)
        if to_node_id in self.state_graph.nodes:
            edge = StateEdge(
                from_node=from_node_id,
                to_node=to_node_id,
                action=action,
                weight=1.0
            )
            self.state_graph.add_edge(edge)


class PlainTomatoSoupStateGraphGenerator(BaseStateGraphGenerator):
    """State graph generator for plain tomato soup (no chopping)"""
    
    def __init__(self):
        super().__init__("plain_tomato_soup")
        
    def generate_state_graph(self):
        """Generate state graph for plain tomato soup"""
        print(f"Generating state graph for {self.soup_type}...")
        
        # Generate all valid states
        all_states = self._generate_all_states()
        print(f"Generated {len(all_states)} valid states")
        
        # Add nodes to graph
        for state in all_states:
            node_id = self._get_state_id(state)
            node = StateNode(
                node_id=node_id,
                state_summary=state,
                is_goal_state=self._is_goal_state(state)
            )
            self.state_graph.add_node(node)
        
        # Generate transitions
        self._generate_all_transitions()
        print(f"Generated transitions between states")
        
        return self.state_graph
        
    def _generate_all_states(self):
        """Generate all valid states for plain tomato soup"""
        states = []
        
        # State variables for plain tomato soup (includes all required variables)
        onion_hand_options = ['none', 'agent', 'partner']
        onion_staged_options = [True, False]
        onion_at_chopping_options = [True, False]
        onion_chopped_options = [True, False]
        onion_in_pot_options = [True, False]
        tomato_hand_options = ['none', 'agent', 'partner']
        tomato_staged_options = [True, False]
        tomato_at_chopping_options = [True, False]
        tomato_chopped_options = [True, False]
        tomato_in_pot_options = [True, False]
        soup_cooking_options = [True, False]
        soup_ready_options = [True, False]
        soup_hand_options = ['none', 'agent', 'partner']
        soup_staged_options = [True, False]
        soup_in_pot_not_cooking_options = [True, False]
        dish_hand_options = ['none', 'agent', 'partner']
        dish_staged_options = [True, False]
        soup_served_options = [True, False]
        
        # Generate all combinations
        for onion_hand in onion_hand_options:
            for onion_staged in onion_staged_options:
                for onion_at_chopping in onion_at_chopping_options:
                    for onion_chopped in onion_chopped_options:
                        for onion_in_pot in onion_in_pot_options:
                            for tomato_hand in tomato_hand_options:
                                for tomato_staged in tomato_staged_options:
                                    for tomato_at_chopping in tomato_at_chopping_options:
                                        for tomato_chopped in tomato_chopped_options:
                                            for tomato_in_pot in tomato_in_pot_options:
                                                for soup_cooking in soup_cooking_options:
                                                    for soup_ready in soup_ready_options:
                                                        for soup_hand in soup_hand_options:
                                                            for soup_staged in soup_staged_options:
                                                                for soup_in_pot_not_cooking in soup_in_pot_not_cooking_options:
                                                                    for dish_hand in dish_hand_options:
                                                                        for dish_staged in dish_staged_options:
                                                                            for soup_served in soup_served_options:
                                                                                state = {
                                                                                    'onion_hand': onion_hand,
                                                                                    'onion_staged': onion_staged,
                                                                                    'onion_at_chopping': onion_at_chopping,
                                                                                    'onion_chopped': onion_chopped,
                                                                                    'onion_in_pot': onion_in_pot,
                                                                                    'tomato_hand': tomato_hand,
                                                                                    'tomato_staged': tomato_staged,
                                                                                    'tomato_at_chopping': tomato_at_chopping,
                                                                                    'tomato_chopped': tomato_chopped,
                                                                                    'tomato_in_pot': tomato_in_pot,
                                                                                    'soup_cooking': soup_cooking,
                                                                                    'soup_ready': soup_ready,
                                                                                    'soup_hand': soup_hand,
                                                                                    'soup_staged': soup_staged,
                                                                                    'soup_in_pot_not_cooking': soup_in_pot_not_cooking,
                                                                                    'dish_hand': dish_hand,
                                                                                    'dish_staged': dish_staged,
                                                                                    'soup_served': soup_served
                                                                                }
                                                                                
                                                                                if self._is_valid_state(state):
                                                                                    states.append(state)
        
        return states
        
    def _is_valid_state(self, state: Dict) -> bool:
        """Check if state is valid for plain tomato soup"""
        # Basic validity checks
        if state['soup_served'] and not state['soup_ready']:
            return False  # Can't serve unready soup
            
        if state['soup_cooking'] and state['soup_in_pot_not_cooking']:
            return False  # Can't be cooking and not cooking at same time
            
        if state['tomato_in_pot'] and state['tomato_hand'] != 'none':
            return False  # Can't hold tomato if it's in pot
            
        # For plain tomato soup, onion variables should be false/none since no onion is needed
        if state['onion_hand'] != 'none':
            return False  # No onion in plain tomato soup
        if state['onion_staged'] or state['onion_at_chopping'] or state['onion_chopped'] or state['onion_in_pot']:
            return False  # No onion processing in plain tomato soup
            
        # For plain tomato soup, tomato chopping variables should be false since no chopping needed
        if state['tomato_at_chopping'] or state['tomato_chopped']:
            return False  # No tomato chopping in plain tomato soup
            
        # Logical constraints
        if state['tomato_in_pot'] and state['tomato_staged']:
            return False  # Can't be in pot and staged at same time
            
        if state['soup_cooking'] and state['soup_ready']:
            return False  # Can't be cooking and ready at same time
            
        if state['soup_ready'] and not state['tomato_in_pot']:
            return False  # Need tomato in pot for soup to be ready
            
        if state['soup_ready'] and state['soup_in_pot_not_cooking']:
            return False  # If soup is ready, it's not "not cooking" - it's already done
            
        if state['soup_in_pot_not_cooking'] and state['soup_cooking']:
            return False  # Can't be in pot not cooking AND cooking
                
        return True
        
    def _is_goal_state(self, state: Dict) -> bool:
        """Check if state is a goal state (soup served)"""
        return state['soup_served']
        
    def _get_state_id(self, state: Dict) -> str:
        """Generate unique ID for state"""
        return f"tomato_plain_{hash(str(sorted(state.items())))}"
        
    def _generate_all_transitions(self):
        """Generate all transitions between states"""
        for node_id, node in self.state_graph.nodes.items():
            self._generate_robot_transitions(node_id, node.state_summary)
            self._generate_human_transitions(node_id, node.state_summary)
            
    def _generate_robot_transitions(self, node_id: str, state: Dict):
        """Generate robot transitions for plain tomato soup"""
        # Robot can pick up tomato
        if state['tomato_hand'] == 'none' and not state['tomato_staged'] and not state['tomato_in_pot']:
            new_state = state.copy()
            new_state['tomato_hand'] = 'agent'
            self._add_transition(node_id, new_state, 'pickup(tomato)')
            
        # Robot can place tomato at staging
        if state['tomato_hand'] == 'agent':
            new_state = state.copy()
            new_state['tomato_hand'] = 'none'
            new_state['tomato_staged'] = True
            self._add_transition(node_id, new_state, 'place(tomato, staging_station)')
            
        # Robot can pick up dish
        if state['dish_hand'] == 'none' and not state['dish_staged']:
            new_state = state.copy()
            new_state['dish_hand'] = 'agent'
            self._add_transition(node_id, new_state, 'pickup(dish)')
            
        # Robot can place dish at staging
        if state['dish_hand'] == 'agent':
            new_state = state.copy()
            new_state['dish_hand'] = 'none'
            new_state['dish_staged'] = True
            self._add_transition(node_id, new_state, 'place(dish)')
            
        # Robot can pick up soup
        if state['soup_ready'] and state['soup_hand'] == 'none' and state['soup_staged']:
            new_state = state.copy()
            new_state['soup_hand'] = 'agent'
            self._add_transition(node_id, new_state, 'pickup(soup)')
            
        # Robot can serve soup
        if state['soup_hand'] == 'agent':
            new_state = state.copy()
            new_state['soup_hand'] = 'none'
            new_state['soup_served'] = True
            self._add_transition(node_id, new_state, 'place(soup)')
            
    def _generate_human_transitions(self, node_id: str, state: Dict):
        """Generate human transitions for plain tomato soup"""
        # Human can grab tomato from staging
        if state['tomato_staged'] and state['tomato_hand'] == 'none':
            new_state = state.copy()
            new_state['tomato_hand'] = 'partner'
            new_state['tomato_staged'] = False
            self._add_transition(node_id, new_state, 'human_grab_tomato')
            
        # Human can place tomato in pot
        if state['tomato_hand'] == 'partner':
            new_state = state.copy()
            new_state['tomato_hand'] = 'none'
            new_state['tomato_in_pot'] = True
            self._add_transition(node_id, new_state, 'human_place_tomato_in_pot')
            
        # Human can turn stove on
        if state['tomato_in_pot'] and not state['soup_cooking'] and not state['soup_ready']:
            new_state = state.copy()
            new_state['soup_cooking'] = True
            self._add_transition(node_id, new_state, 'human_turn_stove_on')
            
        # Cooking completes automatically (environmental transition)
        if state['soup_cooking'] and not state['soup_ready']:
            new_state = state.copy()
            new_state['soup_cooking'] = False
            new_state['soup_ready'] = True
            self._add_transition(node_id, new_state, 'cooking_complete')
            
        # Human can pour soup
        if state['soup_ready'] and not state['soup_staged']:
            new_state = state.copy()
            new_state['soup_staged'] = True
            self._add_transition(node_id, new_state, 'human_pour_soup')
            
        # Human can grab dish from staging
        if state['dish_staged'] and state['dish_hand'] == 'none':
            new_state = state.copy()
            new_state['dish_hand'] = 'partner'
            new_state['dish_staged'] = False
            self._add_transition(node_id, new_state, 'human_grab_dish')
            
    def _add_transition(self, from_node_id: str, to_state: Dict, action: str):
        """Add a transition to the state graph"""
        to_node_id = self._get_state_id(to_state)
        if to_node_id in self.state_graph.nodes:
            edge = StateEdge(
                from_node=from_node_id,
                to_node=to_node_id,
                action=action,
                weight=1.0
            )
            self.state_graph.add_edge(edge)


class ChoppedTomatoSoupStateGraphGenerator(BaseStateGraphGenerator):
    """State graph generator for chopped tomato soup"""
    
    def __init__(self):
        super().__init__("chopped_tomato_soup")
        
    def generate_state_graph(self):
        """Generate state graph for chopped tomato soup"""
        print(f"Generating state graph for {self.soup_type}...")
        
        # Generate all valid states
        all_states = self._generate_all_states()
        print(f"Generated {len(all_states)} valid states")
        
        # Add nodes to graph
        for state in all_states:
            node_id = self._get_state_id(state)
            node = StateNode(
                node_id=node_id,
                state_summary=state,
                is_goal_state=self._is_goal_state(state)
            )
            self.state_graph.add_node(node)
        
        # Generate transitions
        self._generate_all_transitions()
        print(f"Generated transitions between states")
        
        return self.state_graph
        
    def _generate_all_states(self):
        """Generate all valid states for chopped tomato soup"""
        states = []
        
        # State variables for chopped tomato soup (includes chopping mechanics)
        tomato_hand_options = ['none', 'agent', 'partner']
        tomato_staged_options = [True, False]
        tomato_at_chopping_options = [True, False]
        tomato_chopped_options = [True, False]
        tomato_in_pot_options = [True, False]
        soup_cooking_options = [True, False]
        soup_ready_options = [True, False]
        soup_hand_options = ['none', 'agent', 'partner']
        soup_staged_options = [True, False]
        soup_in_pot_not_cooking_options = [True, False]
        dish_hand_options = ['none', 'agent', 'partner']
        dish_staged_options = [True, False]
        soup_served_options = [True, False]
        
        # Generate all combinations
        for tomato_hand in tomato_hand_options:
            for tomato_staged in tomato_staged_options:
                for tomato_at_chopping in tomato_at_chopping_options:
                    for tomato_chopped in tomato_chopped_options:
                        for tomato_in_pot in tomato_in_pot_options:
                            for soup_cooking in soup_cooking_options:
                                for soup_ready in soup_ready_options:
                                    for soup_hand in soup_hand_options:
                                        for soup_staged in soup_staged_options:
                                            for soup_in_pot_not_cooking in soup_in_pot_not_cooking_options:
                                                for dish_hand in dish_hand_options:
                                                    for dish_staged in dish_staged_options:
                                                        for soup_served in soup_served_options:
                                                            state = {
                                                                'tomato_hand': tomato_hand,
                                                                'tomato_staged': tomato_staged,
                                                                'tomato_at_chopping': tomato_at_chopping,
                                                                'tomato_chopped': tomato_chopped,
                                                                'tomato_in_pot': tomato_in_pot,
                                                                'soup_cooking': soup_cooking,
                                                                'soup_ready': soup_ready,
                                                                'soup_hand': soup_hand,
                                                                'soup_staged': soup_staged,
                                                                'soup_in_pot_not_cooking': soup_in_pot_not_cooking,
                                                                'dish_hand': dish_hand,
                                                                'dish_staged': dish_staged,
                                                                'soup_served': soup_served
                                                            }
                                                            
                                                            if self._is_valid_state(state):
                                                                states.append(state)
        
        return states
        
    def _is_valid_state(self, state: Dict) -> bool:
        """Check if state is valid for chopped tomato soup"""
        # Basic validity checks
        if state['soup_served'] and not state['soup_ready']:
            return False  # Can't serve unready soup
            
        if state['soup_cooking'] and state['soup_in_pot_not_cooking']:
            return False  # Can't be cooking and not cooking at same time
            
        if state['tomato_in_pot'] and state['tomato_hand'] != 'none':
            return False  # Can't hold tomato if it's in pot
            
        # Chopping-specific checks
        if state['tomato_at_chopping'] and not state['tomato_chopped']:
            return False  # If at chopping, must be chopped
            
        if state['tomato_chopped'] and not state['tomato_at_chopping'] and not state['tomato_in_pot']:
            return False  # Chopped tomato must be at chopping or in pot
                
        return True
        
    def _is_goal_state(self, state: Dict) -> bool:
        """Check if state is a goal state (soup served)"""
        return state['soup_served']
        
    def _get_state_id(self, state: Dict) -> str:
        """Generate unique ID for state"""
        return f"tomato_chopped_{hash(str(sorted(state.items())))}"
        
    def _generate_all_transitions(self):
        """Generate all transitions between states"""
        for node_id, node in self.state_graph.nodes.items():
            self._generate_robot_transitions(node_id, node.state_summary)
            self._generate_human_transitions(node_id, node.state_summary)
            
    def _generate_robot_transitions(self, node_id: str, state: Dict):
        """Generate robot transitions for chopped tomato soup"""
        # Robot can pick up raw tomato
        if state['tomato_hand'] == 'none' and not state['tomato_staged'] and not state['tomato_at_chopping'] and not state['tomato_in_pot']:
            new_state = state.copy()
            new_state['tomato_hand'] = 'agent'
            self._add_transition(node_id, new_state, 'pickup(tomato)')
            
        # Robot can place tomato at chopping station
        if state['tomato_hand'] == 'agent':
            new_state = state.copy()
            new_state['tomato_hand'] = 'none'
            new_state['tomato_at_chopping'] = True
            new_state['tomato_chopped'] = True
            self._add_transition(node_id, new_state, 'place(tomato, chopping_station)')
            
        # Robot can pick up chopped tomato
        if state['tomato_at_chopping'] and state['tomato_hand'] == 'none':
            new_state = state.copy()
            new_state['tomato_hand'] = 'agent'
            new_state['tomato_at_chopping'] = False
            self._add_transition(node_id, new_state, 'pickup(chopped_tomato)')
            
        # Robot can place chopped tomato at staging
        if state['tomato_hand'] == 'agent' and state['tomato_chopped']:
            new_state = state.copy()
            new_state['tomato_hand'] = 'none'
            new_state['tomato_staged'] = True
            self._add_transition(node_id, new_state, 'place(chopped_tomato)')
            
        # Robot can pick up dish
        if state['dish_hand'] == 'none' and not state['dish_staged']:
            new_state = state.copy()
            new_state['dish_hand'] = 'agent'
            self._add_transition(node_id, new_state, 'pickup(dish)')
            
        # Robot can place dish at staging
        if state['dish_hand'] == 'agent':
            new_state = state.copy()
            new_state['dish_hand'] = 'none'
            new_state['dish_staged'] = True
            self._add_transition(node_id, new_state, 'place(dish)')
            
        # Robot can pick up soup
        if state['soup_ready'] and state['soup_hand'] == 'none' and state['soup_staged']:
            new_state = state.copy()
            new_state['soup_hand'] = 'agent'
            self._add_transition(node_id, new_state, 'pickup(soup)')
            
        # Robot can serve soup
        if state['soup_hand'] == 'agent':
            new_state = state.copy()
            new_state['soup_hand'] = 'none'
            new_state['soup_served'] = True
            self._add_transition(node_id, new_state, 'place(soup)')
            
    def _generate_human_transitions(self, node_id: str, state: Dict):
        """Generate human transitions for chopped tomato soup"""
        # Human can grab chopped tomato from staging
        if state['tomato_staged'] and state['tomato_hand'] == 'none' and state['tomato_chopped']:
            new_state = state.copy()
            new_state['tomato_hand'] = 'partner'
            new_state['tomato_staged'] = False
            self._add_transition(node_id, new_state, 'human_grab_tomato')
            
        # Human can place chopped tomato in pot
        if state['tomato_hand'] == 'partner' and state['tomato_chopped']:
            new_state = state.copy()
            new_state['tomato_hand'] = 'none'
            new_state['tomato_in_pot'] = True
            self._add_transition(node_id, new_state, 'human_place_tomato_in_pot')
            
        # Human can turn stove on
        if state['tomato_in_pot'] and not state['soup_cooking'] and not state['soup_ready']:
            new_state = state.copy()
            new_state['soup_cooking'] = True
            self._add_transition(node_id, new_state, 'human_turn_stove_on')
            
        # Human can pour soup
        if state['soup_ready'] and not state['soup_staged']:
            new_state = state.copy()
            new_state['soup_staged'] = True
            self._add_transition(node_id, new_state, 'human_pour_soup')
            
        # Human can grab dish from staging
        if state['dish_staged'] and state['dish_hand'] == 'none':
            new_state = state.copy()
            new_state['dish_hand'] = 'partner'
            new_state['dish_staged'] = False
            self._add_transition(node_id, new_state, 'human_grab_dish')
            
    def _add_transition(self, from_node_id: str, to_state: Dict, action: str):
        """Add a transition to the state graph"""
        to_node_id = self._get_state_id(to_state)
        if to_node_id in self.state_graph.nodes:
            edge = StateEdge(
                from_node=from_node_id,
                to_node=to_node_id,
                action=action,
                weight=1.0
            )
            self.state_graph.add_edge(edge)


class ChoppedOnionSoupStateGraphGenerator(BaseStateGraphGenerator):
    """State graph generator for chopped onion soup"""
    
    def __init__(self):
        super().__init__("chopped_onion_soup")
        
    def generate_state_graph(self):
        """Generate state graph for chopped onion soup"""
        print(f"Generating state graph for {self.soup_type}...")
        
        # Generate all valid states
        all_states = self._generate_all_states()
        print(f"Generated {len(all_states)} valid states")
        
        # Add nodes to graph
        for state in all_states:
            node_id = self._get_state_id(state)
            node = StateNode(
                node_id=node_id,
                state_summary=state,
                is_goal_state=self._is_goal_state(state)
            )
            self.state_graph.add_node(node)
        
        # Generate transitions
        self._generate_all_transitions()
        print(f"Generated transitions between states")
        
        return self.state_graph
        
    def _generate_all_states(self):
        """Generate all valid states for chopped onion soup"""
        states = []
        
        # State variables for chopped onion soup (includes chopping mechanics)
        onion_hand_options = ['none', 'agent', 'partner']
        onion_staged_options = [True, False]
        onion_at_chopping_options = [True, False]
        onion_chopped_options = [True, False]
        onion_in_pot_options = [True, False]
        soup_cooking_options = [True, False]
        soup_ready_options = [True, False]
        soup_hand_options = ['none', 'agent', 'partner']
        soup_staged_options = [True, False]
        soup_in_pot_not_cooking_options = [True, False]
        dish_hand_options = ['none', 'agent', 'partner']
        dish_staged_options = [True, False]
        soup_served_options = [True, False]
        
        # Generate all combinations
        for onion_hand in onion_hand_options:
            for onion_staged in onion_staged_options:
                for onion_at_chopping in onion_at_chopping_options:
                    for onion_chopped in onion_chopped_options:
                        for onion_in_pot in onion_in_pot_options:
                            for soup_cooking in soup_cooking_options:
                                for soup_ready in soup_ready_options:
                                    for soup_hand in soup_hand_options:
                                        for soup_staged in soup_staged_options:
                                            for soup_in_pot_not_cooking in soup_in_pot_not_cooking_options:
                                                for dish_hand in dish_hand_options:
                                                    for dish_staged in dish_staged_options:
                                                        for soup_served in soup_served_options:
                                                            state = {
                                                                'onion_hand': onion_hand,
                                                                'onion_staged': onion_staged,
                                                                'onion_at_chopping': onion_at_chopping,
                                                                'onion_chopped': onion_chopped,
                                                                'onion_in_pot': onion_in_pot,
                                                                'soup_cooking': soup_cooking,
                                                                'soup_ready': soup_ready,
                                                                'soup_hand': soup_hand,
                                                                'soup_staged': soup_staged,
                                                                'soup_in_pot_not_cooking': soup_in_pot_not_cooking,
                                                                'dish_hand': dish_hand,
                                                                'dish_staged': dish_staged,
                                                                'soup_served': soup_served
                                                            }
                                                            
                                                            if self._is_valid_state(state):
                                                                states.append(state)
        
        return states
        
    def _is_valid_state(self, state: Dict) -> bool:
        """Check if state is valid for chopped onion soup"""
        # Basic validity checks
        if state['soup_served'] and not state['soup_ready']:
            return False  # Can't serve unready soup
            
        if state['soup_cooking'] and state['soup_in_pot_not_cooking']:
            return False  # Can't be cooking and not cooking at same time
            
        if state['onion_in_pot'] and state['onion_hand'] != 'none':
            return False  # Can't hold onion if it's in pot
            
        # Chopping-specific checks
        if state['onion_at_chopping'] and not state['onion_chopped']:
            return False  # If at chopping, must be chopped
            
        if state['onion_chopped'] and not state['onion_at_chopping'] and not state['onion_in_pot']:
            return False  # Chopped onion must be at chopping or in pot
                
        return True
        
    def _is_goal_state(self, state: Dict) -> bool:
        """Check if state is a goal state (soup served)"""
        return state['soup_served']
        
    def _get_state_id(self, state: Dict) -> str:
        """Generate unique ID for state"""
        return f"onion_chopped_{hash(str(sorted(state.items())))}"
        
    def _generate_all_transitions(self):
        """Generate all transitions between states"""
        for node_id, node in self.state_graph.nodes.items():
            self._generate_robot_transitions(node_id, node.state_summary)
            self._generate_human_transitions(node_id, node.state_summary)
            
    def _generate_robot_transitions(self, node_id: str, state: Dict):
        """Generate robot transitions for chopped onion soup"""
        # Robot can pick up raw onion
        if state['onion_hand'] == 'none' and not state['onion_staged'] and not state['onion_at_chopping'] and not state['onion_in_pot']:
            new_state = state.copy()
            new_state['onion_hand'] = 'agent'
            self._add_transition(node_id, new_state, 'pickup(onion)')
            
        # Robot can place onion at chopping station
        if state['onion_hand'] == 'agent':
            new_state = state.copy()
            new_state['onion_hand'] = 'none'
            new_state['onion_at_chopping'] = True
            new_state['onion_chopped'] = True
            self._add_transition(node_id, new_state, 'place(onion, chopping_station)')
            
        # Robot can pick up chopped onion
        if state['onion_at_chopping'] and state['onion_hand'] == 'none':
            new_state = state.copy()
            new_state['onion_hand'] = 'agent'
            new_state['onion_at_chopping'] = False
            self._add_transition(node_id, new_state, 'pickup(chopped_onion)')
            
        # Robot can place chopped onion at staging
        if state['onion_hand'] == 'agent' and state['onion_chopped']:
            new_state = state.copy()
            new_state['onion_hand'] = 'none'
            new_state['onion_staged'] = True
            self._add_transition(node_id, new_state, 'place(chopped_onion)')
            
        # Robot can pick up dish
        if state['dish_hand'] == 'none' and not state['dish_staged']:
            new_state = state.copy()
            new_state['dish_hand'] = 'agent'
            self._add_transition(node_id, new_state, 'pickup(dish)')
            
        # Robot can place dish at staging
        if state['dish_hand'] == 'agent':
            new_state = state.copy()
            new_state['dish_hand'] = 'none'
            new_state['dish_staged'] = True
            self._add_transition(node_id, new_state, 'place(dish)')
            
        # Robot can pick up soup
        if state['soup_ready'] and state['soup_hand'] == 'none' and state['soup_staged']:
            new_state = state.copy()
            new_state['soup_hand'] = 'agent'
            self._add_transition(node_id, new_state, 'pickup(soup)')
            
        # Robot can serve soup
        if state['soup_hand'] == 'agent':
            new_state = state.copy()
            new_state['soup_hand'] = 'none'
            new_state['soup_served'] = True
            self._add_transition(node_id, new_state, 'place(soup)')
            
    def _generate_human_transitions(self, node_id: str, state: Dict):
        """Generate human transitions for chopped onion soup"""
        # Human can grab chopped onion from staging
        if state['onion_staged'] and state['onion_hand'] == 'none' and state['onion_chopped']:
            new_state = state.copy()
            new_state['onion_hand'] = 'partner'
            new_state['onion_staged'] = False
            self._add_transition(node_id, new_state, 'human_grab_onion')
            
        # Human can place chopped onion in pot
        if state['onion_hand'] == 'partner' and state['onion_chopped']:
            new_state = state.copy()
            new_state['onion_hand'] = 'none'
            new_state['onion_in_pot'] = True
            self._add_transition(node_id, new_state, 'human_place_onion_in_pot')
            
        # Human can turn stove on
        if state['onion_in_pot'] and not state['soup_cooking'] and not state['soup_ready']:
            new_state = state.copy()
            new_state['soup_cooking'] = True
            self._add_transition(node_id, new_state, 'human_turn_stove_on')
            
        # Human can pour soup
        if state['soup_ready'] and not state['soup_staged']:
            new_state = state.copy()
            new_state['soup_staged'] = True
            self._add_transition(node_id, new_state, 'human_pour_soup')
            
        # Human can grab dish from staging
        if state['dish_staged'] and state['dish_hand'] == 'none':
            new_state = state.copy()
            new_state['dish_hand'] = 'partner'
            new_state['dish_staged'] = False
            self._add_transition(node_id, new_state, 'human_grab_dish')
            
    def _add_transition(self, from_node_id: str, to_state: Dict, action: str):
        """Add a transition to the state graph"""
        to_node_id = self._get_state_id(to_state)
        if to_node_id in self.state_graph.nodes:
            edge = StateEdge(
                from_node=from_node_id,
                to_node=to_node_id,
                action=action,
                weight=1.0
            )
            self.state_graph.add_edge(edge)


# Factory function to get the appropriate state graph generator
def get_state_graph_generator(soup_type: str) -> BaseStateGraphGenerator:
    """Get the appropriate state graph generator based on soup type"""
    soup_type_lower = soup_type.lower()
    
    # Check for specific scenarios first (most specific to least specific)
    
    # "Serving Chopped Onion and Chopped Tomato Soup" - both ingredients chopped
    if ("chopped" in soup_type_lower and "onion" in soup_type_lower and 
        "chopped" in soup_type_lower and "tomato" in soup_type_lower):
        return StateGraphGenerator()  # Use original for complex mixed chopped
    
    # "Serving Onion and Chopped Tomato Soup" - onion plain, tomato chopped
    if ("onion" in soup_type_lower and "chopped" in soup_type_lower and 
        "tomato" in soup_type_lower and "chopped onion" not in soup_type_lower):
        return StateGraphGenerator()  # Use original for mixed plain+chopped
    
    # "Serving Chopped Onion and Tomato Soup" - onion chopped, tomato plain
    if ("chopped onion" in soup_type_lower and "tomato" in soup_type_lower and 
        "chopped tomato" not in soup_type_lower):
        return StateGraphGenerator()  # Use original for mixed chopped+plain
    
    # Single ingredient soups
    if "chopped" in soup_type_lower and "onion" in soup_type_lower and "tomato" not in soup_type_lower:
        return ChoppedOnionSoupStateGraphGenerator()
    elif "chopped" in soup_type_lower and "tomato" in soup_type_lower and "onion" not in soup_type_lower:
        return ChoppedTomatoSoupStateGraphGenerator()
    elif "onion" in soup_type_lower and "chopped" not in soup_type_lower and "tomato" not in soup_type_lower:
        return PlainOnionSoupStateGraphGenerator()
    elif "tomato" in soup_type_lower and "chopped" not in soup_type_lower and "onion" not in soup_type_lower:
        return PlainTomatoSoupStateGraphGenerator()
    
    # Default to the original generator for mixed soups or complex cases
    print(f"No specific generator found for '{soup_type}', using default StateGraphGenerator")
    return StateGraphGenerator() 