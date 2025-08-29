"""
Complete State Graph Generator for Washing + Chopping Recipe

This module implements a comprehensive state graph for recipes that include:
- Ingredient washing (onions and tomatoes)
- Ingredient chopping (onions and tomatoes)  
- Complete cooking workflow (staging, cooking, serving)

The state graph handles flexible ingredient processing order - humans can pick up
and process ingredients in any sequence, not just wash→chop→place.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import itertools
import json
import pickle
import os
import hashlib
import time


@dataclass
class CompleteRecipeState:
    """
    State representation for the complete washing+chopping recipe.
    Built on top of existing game state variables from summarize_state().
    
    Supports flexible ingredient processing where humans can:
    - Pick up raw ingredients and wash them
    - Pick up washed ingredients and chop them
    - Pick up chopped+washed ingredients and place them
    - Process ingredients in any order (onion first, tomato first, etc.)
    """
    
    # Hand states (existing) - human can only hold one item
    onion_hand: str = "none"  # none|agent|partner
    tomato_hand: str = "none"
    dish_hand: str = "none" 
    soup_hand: str = "none"
    
    # Staging states (existing)
    onion_staged: bool = False
    tomato_staged: bool = False
    dish_staged: bool = False
    soup_staged: bool = False
    
    # Processing states (existing + new washing states)
    onion_at_chopping: bool = False
    onion_chopped: bool = False
    onion_at_sink: bool = False
    onion_washed: bool = False  
    
    tomato_at_chopping: bool = False
    tomato_chopped: bool = False
    tomato_at_sink: bool = False
    tomato_washed: bool = False  
    
    # Cooking states (existing)
    onion_in_pot: bool = False
    tomato_in_pot: bool = False
    soup_cooking: bool = False
    soup_ready: bool = False
    soup_in_pot_not_cooking: bool = False
    
    # Serving states (existing)
    soup_served: bool = False
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization and state graph mapping"""
        return asdict(self)
    
    def to_json_key(self) -> str:
        """Create a consistent JSON key for state-to-node mapping"""
        return json.dumps(self.to_dict(), sort_keys=True)
    
    @classmethod
    def from_game_state(cls, game_state_summary: dict) -> 'CompleteRecipeState':
        """
        Create state from existing game state summary.
        
        The game state summary already includes all needed variables from CoordinatedActionPredictorAgent.
        """
        state = cls()
        
        # Copy all existing state variables 
        for key, value in game_state_summary.items():
            if hasattr(state, key):
                setattr(state, key, value)
        
        return state
    
    def __hash__(self) -> int:
        """Make state hashable for use in sets and dictionaries"""
        return hash(self.to_json_key())
    
    def __eq__(self, other) -> bool:
        """State equality comparison"""
        if not isinstance(other, CompleteRecipeState):
            return False
        return self.to_dict() == other.to_dict()


@dataclass
class CompleteRecipeNode:
    """
    Node in the complete recipe state graph.
    Each node represents a unique valid state in the washing+chopping recipe.
    """
    node_id: str
    state: CompleteRecipeState
    is_goal_state: bool = False      # True if state.soup_served=True
    is_initial_state: bool = False   # True if all default values


@dataclass  
class CompleteRecipeEdge:
    """
    Edge in the complete recipe state graph.
    Each edge represents a primary action that transitions between states.
    """
    from_node: str          # Source node ID (e.g., "state_001234")
    to_node: str           # Target node ID (e.g., "state_005678")
    primary_action: str    # One of our 14 primary actions
    weight: float = 1.0    # For pathfinding (all equal weight for now)


class CompleteRecipeGraph:
    """
    Main graph class that holds all nodes and edges for the complete recipe.
    Provides fast lookup and navigation for real-time gameplay.
    """
    
    def __init__(self):
        self.nodes: Dict[str, CompleteRecipeNode] = {}
        self.edges: Dict[str, List[CompleteRecipeEdge]] = {}        # from_node -> [edges]
        self.state_to_node_id: Dict[str, str] = {}                 # state_json -> node_id
    
    def add_node(self, node: CompleteRecipeNode):
        """Add a node to the graph and update mappings"""
        self.nodes[node.node_id] = node
        
        # Initialize edge list
        if node.node_id not in self.edges:
            self.edges[node.node_id] = []
        
        # Add state-to-node mapping for fast lookup
        state_key = node.state.to_json_key()
        self.state_to_node_id[state_key] = node.node_id
    
    def add_edge(self, edge: CompleteRecipeEdge):
        """Add an edge to the graph (only forward mapping needed)"""
        self.edges[edge.from_node].append(edge)
        # No reverse edges needed since we don't use to_node
    
    def get_possible_actions(self, node_id: str) -> List[str]:
        """Get all valid primary actions from this node"""
        if node_id not in self.edges:
            return []
        
        return [edge.primary_action for edge in self.edges[node_id]]
    
    def get_node_for_state(self, state: CompleteRecipeState) -> Optional[str]:
        """Find node ID for a given state (for real-time game state lookup)"""
        state_key = state.to_json_key()
        return self.state_to_node_id.get(state_key)
    
    def get_edges_from(self, node_id: str) -> List[CompleteRecipeEdge]:
        """Get all edges from a node"""
        return self.edges.get(node_id, [])


def is_valid_state(state: CompleteRecipeState) -> bool:
    """
    Validate that a state is logically possible and reachable.
    Cleaned up version with no duplicates and comprehensive checks.
    """
    
    # 1. CRITICAL: Human can only hold one item at a time
    held_items = [
        state.onion_hand == "partner",
        state.tomato_hand == "partner", 
        state.dish_hand == "partner",
        state.soup_hand == "partner"
    ]
    if sum(held_items) > 1:
        return False
            
    # 1. CRITICAL: Agent can only hold one item at a time
    held_items = [
        state.onion_hand == "agent",
        state.tomato_hand == "agent", 
        state.dish_hand == "agent",
        state.soup_hand == "agent"
    ]
    if sum(held_items) > 1:
        return False
    
    # 2. CRITICAL: Cooking states are mutually exclusive
    cooking_states = [
        state.soup_cooking,
        state.soup_ready,
        state.soup_in_pot_not_cooking
    ]
    if sum(cooking_states) > 1:
        return False
    
    # 3. CRITICAL: Can't have soup ready without both ingredients in pot
    if state.soup_ready and not (state.onion_in_pot and state.tomato_in_pot):
        return False
    
    # 4. LOCATION EXCLUSIVITY: Ingredients can't be in multiple places
    onion_locations = [
        state.onion_staged,
        state.onion_at_chopping,
        state.onion_at_sink,
        state.onion_in_pot
    ]
    if sum(onion_locations) > 1:
        return False
    
    tomato_locations = [
        state.tomato_staged,
        state.tomato_at_chopping,
        state.tomato_at_sink,
        state.tomato_in_pot
    ]
    if sum(tomato_locations) > 1:
        return False
    
    # 6. STATION CAPACITY: Only one ingredient per station
    if state.onion_at_chopping and state.tomato_at_chopping:
        return False
    if state.onion_at_sink and state.tomato_at_sink:
        return False
    
    # 5. HAND vs LOCATION CONSISTENCY: Can't hold what's placed somewhere
    # Onion consistency
    if state.onion_hand != "none" and (
        state.onion_staged or 
        state.onion_at_chopping or 
        state.onion_at_sink or 
        state.onion_in_pot
    ):
        return False
    
    # Tomato consistency  
    if state.tomato_hand != "none" and (
        state.tomato_staged or 
        state.tomato_at_chopping or 
        state.tomato_at_sink or 
        state.tomato_in_pot
    ):
        return False
    
    # Dish consistency
    if state.dish_hand != "none" and state.dish_staged:
        return False
    
    # Soup consistency
    if state.soup_hand != "none" and state.soup_staged:
        return False
    
    # 9. SOUP SERVING: If served, soup shouldn't be held or staged
    if state.soup_served and (state.soup_hand != "none" or state.soup_staged):
        return False
    
    # 11. DISH SERVING REQUIREMENT: Can't serve without dish available
    # (This depends on your game logic - uncomment if needed)
    # if state.soup_served and not (state.dish_staged or state.dish_hand != "none"):
    #     return False
    
    return True


def generate_all_valid_states() -> List[CompleteRecipeState]:
    """
    Generate all possible valid state combinations for the washing+chopping recipe.
    
    Uses itertools.product to create all possible combinations of state variables,
    then filters out invalid states using is_valid_state().
    
    This approach ensures we capture all reachable states while allowing flexible
    ingredient processing order.
    
    Returns:
        List of all valid and reachable CompleteRecipeState objects
    """
    
    print("Generating all valid state combinations...")
    
    # Define possible values for each state variable
    hand_values = ["none", "agent", "partner"]
    bool_values = [True, False]
    
    valid_states = []
    total_combinations = 0
    
    # Generate all possible combinations using itertools.product
    # This creates a Cartesian product of all possible values
    for combination in itertools.product(
        # Hand states
        hand_values,  # onion_hand
        hand_values,  # tomato_hand
        hand_values,  # dish_hand
        hand_values,  # soup_hand
        
        # Staging states
        bool_values,  # onion_staged
        bool_values,  # tomato_staged
        bool_values,  # dish_staged
        bool_values,  # soup_staged
        
        # Onion processing states
        bool_values,  # onion_at_chopping
        bool_values,  # onion_chopped
        bool_values,  # onion_at_sink
        bool_values,  # onion_washed
        
        # Tomato processing states
        bool_values,  # tomato_at_chopping
        bool_values,  # tomato_chopped
        bool_values,  # tomato_at_sink
        bool_values,  # tomato_washed
        
        # Cooking states
        bool_values,  # onion_in_pot
        bool_values,  # tomato_in_pot
        bool_values,  # soup_cooking
        bool_values,  # soup_ready
        bool_values,  # soup_in_pot_not_cooking
        
        # Serving states
        bool_values,  # soup_served
    ):
        total_combinations += 1
        
        # Create state from combination
        state = CompleteRecipeState(
            # Hand states
            onion_hand=combination[0],
            tomato_hand=combination[1],
            dish_hand=combination[2],
            soup_hand=combination[3],
            
            # Staging states
            onion_staged=combination[4],
            tomato_staged=combination[5],
            dish_staged=combination[6],
            soup_staged=combination[7],
            
            # Onion processing states
            onion_at_chopping=combination[8],
            onion_chopped=combination[9],
            onion_at_sink=combination[10],
            onion_washed=combination[11],
            
            # Tomato processing states
            tomato_at_chopping=combination[12],
            tomato_chopped=combination[13],
            tomato_at_sink=combination[14],
            tomato_washed=combination[15],
            
            # Cooking states
            onion_in_pot=combination[16],
            tomato_in_pot=combination[17],
            soup_cooking=combination[18],
            soup_ready=combination[19],
            soup_in_pot_not_cooking=combination[20],
            
            # Serving states
            soup_served=combination[21],
        )
        
        # Validate the state
        if is_valid_state(state):
            valid_states.append(state)
        
        # Progress reporting for large state spaces
        if total_combinations % 100000 == 0:
            print(f"   Processed {total_combinations} combinations, found {len(valid_states)} valid states...")
    
    print(f"Generated {len(valid_states)} valid states from {total_combinations} total combinations")
    print(f"State space reduction: {len(valid_states)/total_combinations*100:.2f}% of combinations are valid")
    
    return valid_states

class CompleteStateGraphGenerator:
    """
    Generates the complete state graph for washing+chopping recipe.
    
    This class creates all nodes and edges for the state graph, implementing
    the transition logic for each of the 14 primary actions.
    """
    
    def __init__(self):
        self.graph = CompleteRecipeGraph()
        
        # All 14 primary actions for the complete recipe
        self.primary_actions = [
            "Wash Onion",                  # Robot washes raw onion at sink
            "Chop Onion",                  # Robot chops washed onion at chopping station
            "Human Grab Onion",            # Human grabs processed onion
            "Place Onion in Pot",          # Human places onion in cooking pot
            
            "Wash Tomato",                 # Robot washes raw tomato at sink
            "Chop Tomato",                 # Robot chops washed tomato at chopping station
            "Human Grab Tomato",           # Human grabs processed tomato
            "Place Tomato in Pot",         # Human places tomato in cooking pot
            
            "Turn Stove On",               # Human starts cooking when both ingredients in pot
            "Wait For Ingredients to Cook", # System state - cooking in progress
            "Human Grab Dish",             # Human grabs clean dish for serving
            "Pour Soup in Dish",           # Human pours ready soup into dish
            "Serve Soup",                  # Human serves soup (goal action)
            "NOOP"                         # No action needed (waiting/self-loop)
        ]
        
        # We'll populate this with our 31,296 valid states
        self.all_valid_states = []
        self.state_to_node_mapping = {}
        
        # Caching configuration
        self.cache_dir = "data"
        self.cache_file = os.path.join(self.cache_dir, "complete_state_graph_cache.pkl")
        self.cache_meta_file = os.path.join(self.cache_dir, "complete_state_graph_meta.json")
    
    def generate_complete_graph(self) -> CompleteRecipeGraph:
        """
        Generate the complete state graph with all nodes and edges.
        
        Returns:
            CompleteRecipeGraph with all 31,296 nodes and their transitions
        """
        print("Generating complete recipe state graph...")
        
        # Step 1: Generate all valid states (we already tested this)
        print("Step 1: Generating all valid states...")
        self.all_valid_states = generate_all_valid_states()
        print(f"   Generated {len(self.all_valid_states)} valid states")
        
        # Step 2: Create nodes for all states
        print("Step 2: Creating graph nodes...")
        self._generate_all_nodes()
        print(f"   Created {len(self.graph.nodes)} nodes")
        
        # Step 3: Create edges for all valid transitions
        print("Step 3: Creating graph edges...")
        self._generate_all_edges()
        total_edges = sum(len(edges) for edges in self.graph.edges.values())
        print(f"   Created {total_edges} edges")
        
        print(f"Complete state graph generated!")
        print(f"   {len(self.graph.nodes)} nodes, {total_edges} edges")
        
        return self.graph
    
    def _generate_all_nodes(self):
        """Create nodes for all valid states"""
        
        for i, state in enumerate(self.all_valid_states):
            node_id = f"state_{i:06d}"
            
            # Determine if this is a goal state or initial state
            is_goal = state.soup_served
            is_initial = self._is_initial_state(state)
            
            node = CompleteRecipeNode(
                node_id=node_id,
                state=state,
                is_goal_state=is_goal,
                is_initial_state=is_initial
            )
            
            self.graph.add_node(node)
            self.state_to_node_mapping[state.to_json_key()] = node_id
    
    def _generate_all_edges(self):
        """Create edges for all valid actions from each state"""
        
        total_states = len(self.all_valid_states)
        processed = 0
        
        for state in self.all_valid_states:
            current_node_id = self.state_to_node_mapping[state.to_json_key()]
            
            # Try each primary action from this state
            for action in self.primary_actions:
                if self._is_action_valid(state, action):
                    # Create edge with dummy target (we don't need actual target nodes)
                    edge = CompleteRecipeEdge(
                        from_node=current_node_id,
                        to_node="",  # Not used - we only care about valid actions
                        primary_action=action
                    )
                    
                    self.graph.add_edge(edge)
            
            processed += 1
            if processed % 5000 == 0:
                print(f"   Processed {processed}/{total_states} states...")
    
    def _is_initial_state(self, state: CompleteRecipeState) -> bool:
        """Check if this is the initial state (all default values)"""
        initial_state = CompleteRecipeState()
        return state == initial_state
    
    def _is_action_valid(self, state: CompleteRecipeState, action: str) -> bool:
        """
        Check if a primary action is valid from this state.
        Returns True if the action can be attempted, False otherwise.
        
        This is the core validation logic that defines which actions are possible.
        """
        
        # Implement validation logic for each primary action
        if action == "NOOP":
            # NOOP always valid
            return True
            
        elif action == "Wash Onion":
            # Valid if: sink is available, onion needs washing
            return (not state.onion_at_sink and not state.tomato_at_sink and  # Sink available
                    not state.onion_washed and  # Onion not already washed
                    not state.onion_in_pot)     # Onion not already used
            
        elif action == "Chop Onion":
            # Valid if: chopping station available, onion not already chopped, onion not in pot
            # Can chop raw onion OR already washed onion - chopping is independent of washing
            return (not state.onion_at_chopping and not state.tomato_at_chopping and  # Station available
                    not state.onion_chopped and  # Onion not already chopped
                    not state.onion_in_pot)       # Onion not already used
            
        elif action == "Human Grab Onion":
            # Valid if: human has empty hands AND onion not already in pot (no point grabbing if already used)
            # Locations: dispenser (raw), sink (washed), chopping station (chopped), staging (processed)
            return self._human_hands_empty(state) and not state.onion_in_pot
            
        elif action == "Place Onion in Pot":
            # Valid if: human is holding onion, pot available
            return state.onion_hand == "partner" or state.onion_staged
            
        elif action == "Wash Tomato":
            # Valid if: sink is available, tomato needs washing
            return (not state.onion_at_sink and not state.tomato_at_sink and  # Sink available
                    not state.tomato_washed and  # Tomato not already washed
                    not state.tomato_in_pot)     # Tomato not already used
            
        elif action == "Chop Tomato":
            # Valid if: chopping station available, tomato not already chopped, tomato not in pot
            # Can chop raw tomato OR already washed tomato - chopping is independent of washing
            return (not state.onion_at_chopping and not state.tomato_at_chopping and  # Station available
                    not state.tomato_chopped and  # Tomato not already chopped
                    not state.tomato_in_pot)       # Tomato not already used
            
        elif action == "Human Grab Tomato":
            # Valid if: human has empty hands AND tomato not already in pot (no point grabbing if already used)
            # Locations: dispenser (raw), sink (washed), chopping station (chopped), staging (processed)
            return self._human_hands_empty(state) and not state.tomato_in_pot
            
        elif action == "Place Tomato in Pot":
            # Valid if: human is holding tomato, pot available
            return state.tomato_hand == "partner" or state.tomato_staged
            
        elif action == "Turn Stove On":
            # Valid if: both ingredients in pot, stove not already on
            return (state.onion_in_pot and state.tomato_in_pot and
                    not state.soup_cooking and not state.soup_ready)
            
        elif action == "Wait For Ingredients to Cook":
            # Valid if: soup is cooking, advances to ready state
            return state.soup_cooking
            
        elif action == "Human Grab Dish":
            # Always valid if human has empty hands (they can attempt to grab)
            return self._human_hands_empty(state)
            
        elif action == "Pour Soup in Dish":
            # Valid if: soup is ready, human holding dish
            return state.soup_ready and state.dish_hand == "partner"
            
        elif action == "Serve Soup":
            # Valid if: human holding soup
            return state.soup_hand != "none"
        
        # Unknown action
        return False
    
    def _human_hands_empty(self, state: CompleteRecipeState) -> bool:
        """Check if human has empty hands (not holding any item)"""
        return (state.onion_hand != "partner" and 
                state.tomato_hand != "partner" and
                state.dish_hand != "partner" and
                state.soup_hand != "partner")
    
    def _get_code_hash(self) -> str:
        """Generate a hash of the current code to detect changes"""
        # Hash the key parts of the state graph implementation
        hash_content = ""
        
        # Hash the primary actions
        hash_content += str(self.primary_actions)
        
        # Hash the state dataclass structure (field names and types)
        state_fields = [(field.name, str(field.type)) for field in CompleteRecipeState.__dataclass_fields__.values()]
        hash_content += str(state_fields)
        
        # Hash the validation logic by reading this file's content
        try:
            with open(__file__, 'r') as f:
                # Only hash the validation methods to detect logic changes
                content = f.read()
                # Extract validation-related methods
                for line in content.split('\n'):
                    if ('def _is_action_valid' in line or 
                        'def is_valid_state' in line or
                        'def _human_hands_empty' in line):
                        hash_content += line
        except:
            hash_content += "no_file_access"
        
        return hashlib.md5(hash_content.encode()).hexdigest()
    
    def _save_graph_to_cache(self):
        """Save the generated graph to cache"""
        print("Saving graph to cache...")
        
        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Save the graph data
        cache_data = {
            'graph': self.graph,
            'all_valid_states': self.all_valid_states,
            'state_to_node_mapping': self.state_to_node_mapping
        }
        
        with open(self.cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        
        # Save metadata
        metadata = {
            'code_hash': self._get_code_hash(),
            'generation_time': time.time(),
            'node_count': len(self.graph.nodes),
            'edge_count': sum(len(edges) for edges in self.graph.edges.values()),
            'version': '1.0'
        }
        
        with open(self.cache_meta_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"   Graph cached to {self.cache_file}")
    
    def _load_graph_from_cache(self) -> bool:
        """Load graph from cache if valid. Returns True if successful."""
        
        if not os.path.exists(self.cache_file) or not os.path.exists(self.cache_meta_file):
            return False
        
        try:
            # Check if cache is still valid
            with open(self.cache_meta_file, 'r') as f:
                metadata = json.load(f)
            
            current_hash = self._get_code_hash()
            if metadata.get('code_hash') != current_hash:
                print("Cache invalid (code changed), regenerating...")
                return False
            
            # Load the cached graph
            print("Loading graph from cache...")
            with open(self.cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            self.graph = cache_data['graph']
            self.all_valid_states = cache_data['all_valid_states']
            self.state_to_node_mapping = cache_data['state_to_node_mapping']
            
            print(f"   Loaded {metadata['node_count']:,} nodes, {metadata['edge_count']:,} edges")
            print(f"   Instant loading (cached {time.time() - metadata['generation_time']:.0f}s ago)")
            
            return True
            
        except Exception as e:
            print(f"Failed to load cache: {e}")
            return False
    
    def get_or_generate_graph(self) -> CompleteRecipeGraph:
        """
        Get the complete state graph, using cache if available or generating if needed.
        This is the main entry point for production use.
        """
        print("INITIALIZING COMPLETE STATE GRAPH")
        print("="*50)
        
        # Try to load from cache first
        if self._load_graph_from_cache():
            return self.graph
        
        # Cache miss or invalid - generate fresh
        print("Generating fresh state graph...")
        graph = self.generate_complete_graph()
        
        # Save to cache for next time
        self._save_graph_to_cache()
        
        return graph


# Washing state detection is already implemented in CoordinatedActionPredictorAgent.summarize_state()
# The game state summary already includes onion_washed and tomato_washed flags
# No additional detection functions needed!



