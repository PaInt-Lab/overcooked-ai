from collections import deque
import json
import re
from typing import List
from overcooked_ai_py.agents.agent import Agent
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.planning.planners import MotionPlanner
from overcooked_ai_py.mdp.actions import Action, Direction
import os
from openai import OpenAI
from plan_session import PLAN_STORE
from secondary_action_selector import select_secondary_action
from complete_state_graph import CompleteStateGraphGenerator, CompleteRecipeState  # NEW: Complete state graph with washing
from ..plan_adaptation import ActionTracker, PlanRepository  # NEW: Plan adaptation system

# Use a simple vanilla model instead of fine-tuned ones
DEFAULT_MODEL = "gpt-4o-mini"

def query_openai(prompt: str, model: str = None, temperature: float = 0.0) -> str:
    """Query the OpenAI API with the given prompt and return the response text."""
    # Use the default model if none specified
    if model is None:
        model = DEFAULT_MODEL
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set.")
   
    client = OpenAI(api_key=api_key)
   
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=256,
        )
       
        content = response.choices[0].message.content
        if content is not None:
            return content.strip()
        return ""
       
    except Exception as e:
        print(f"Error querying OpenAI: {e}")
        return ""

def _bfs_fallback(start, goal, terrain, goal_orientation=None):
    """Return a list of (delta_col, delta_row) moves to walk from start to goal on terrain."""
    H, W = len(terrain), len(terrain[0])
    visited = {start}
    parent = {}
    queue = deque([start])

    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    while queue:
        col, row = queue.popleft()
        if (col, row) == goal:
            path = []
            cur = goal
            while cur != start:
                prev = parent[cur]
                dc = cur[0] - prev[0]
                dr = cur[1] - prev[1]
                path.append((dc, dr))
                cur = prev
            path = list(reversed(path))  
            
            if goal_orientation is not None and path[-1] != goal_orientation:
                if goal_orientation == (1, 0): 
                    path.append((1, 0))
                elif goal_orientation == (-1, 0):  
                    path.append((-1, 0))
                elif goal_orientation == (0, 1):  
                    path.append((0, 1))
                elif goal_orientation == (0, -1):  
                    path.append((0, -1))

            path.append(Action.INTERACT)
            return path

        for dc, dr in directions:
            new_col = col + dc
            new_row = row + dr
            if (
                0 <= new_row < H and
                0 <= new_col < W and
                terrain[new_row][new_col] != 'X' and
                (new_col, new_row) not in visited
            ):
                visited.add((new_col, new_row))
                parent[(new_col, new_row)] = (col, row)
                queue.append((new_col, new_row))

    return []

class CoordinatedActionPredictorAgent(Agent):
    """
    An agent that uses coordinated state graph navigation for goal-directed behavior.
    
    This agent:
    1. Uses a state graph to navigate toward the goal
    2. Predicts human behavior and coordinates accordingly
    3. Balances goal progress with coordination quality
    4. Adapts to human actions dynamically
    """
    
    def __init__(self):
        super().__init__()
        self.mdp = None
        self.planner = None
        self.ingredient_spawns = []
        self.onion_spawns = []
        self.tomato_spawns = []
        self.stove_tiles = []
        self.dish_spawns = []
        self.delivery_tiles = []
        self.staging_tiles = []
        self.onion_chopping_stations = []
        self.tomato_chopping_stations = []


        self.last_summary = None
        self.agent_index = None

        self.onion_chopped = False
        self.tomato_chopped = False
        self.onion_washed = False
        self.tomato_washed = False
        
        # Complete state graph with washing capabilities
        self.complete_state_graph_generator = None
        self.complete_state_graph = None
        
        # Task title for coordination
        self.task_title = None
        

        
        # Use simple vanilla model
        self.selected_model = DEFAULT_MODEL
        print(f"Using model: {self.selected_model}")
        
        # NEW: Plan adaptation system
        self.action_tracker = ActionTracker()
        self.plan_repository = PlanRepository()
        print("Plan adaptation system initialized")

    def _initialize_complete_state_graph(self):
        """Initialize the complete state graph"""
        if self.complete_state_graph_generator is None:
            print("Initializing complete state graph...")
            self.complete_state_graph_generator = CompleteStateGraphGenerator()
            # Use cached version for fast loading
            self.complete_state_graph = self.complete_state_graph_generator.get_or_generate_graph()
            print("Complete state graph ready for action selection")
    
    def get_available_primary_actions(self, state, info):
        """Get available primary actions from complete state graph"""
        # Initialize complete state graph if needed
        self._initialize_complete_state_graph()
        
        # Get current state summary
        state_summary = self.summarize_state(state, info)
        
        # Convert to CompleteRecipeState
        complete_state = CompleteRecipeState.from_game_state(state_summary)
        
        # Find corresponding node in graph
        node_id = self.complete_state_graph.get_node_for_state(complete_state)
        
        if node_id:
            # Get available actions from state graph
            available_actions = self.complete_state_graph.get_possible_actions(node_id)
            return available_actions
        else:
            # Fallback to default actions if state not found
            print(f"Warning: State not found in graph, using fallback action")
            return ["NOOP"]
    
    def set_agent_index(self, agent_index: int):
        super().set_agent_index(agent_index)
        self.agent_index = agent_index

    def set_plan(self, session_id: str):
        """Attach the full PlanSession to this agent and filter for primary tasks only."""
        self.plan = PLAN_STORE[session_id]
        
        # NEW: Filter plan to extract only primary tasks for LLM context
        self.primary_tasks = self._extract_primary_tasks_from_plan()
        print(f"Full plan loaded with {len(self.plan.events)} total tasks")
        print(f"Filtered to {len(self.primary_tasks)} primary tasks for LLM")
        
        # Set task title from plan for coordination
        if hasattr(self.plan, 'task_title'):
            self.task_title = self.plan.task_title
            print(f"Task title set from plan: {self.task_title}")
    
    def _extract_primary_tasks_from_plan(self) -> List[str]:
        """
        NEW: Extract only primary tasks from the full plan.
        This filters out secondary tasks so the LLM only sees human actions.
        The plan structure has arrays of primary actions per step.
        """
        if not self.plan or not hasattr(self.plan, 'events'):
            return []
        
        # NEW: Handle the actual plan structure with arrays of primary actions
        primary_tasks = []
        for event in self.plan.events:
            # Extract primary actions from the 'primary' field (which is an array)
            if 'primary' in event and isinstance(event['primary'], list):
                for primary_action in event['primary']:
                    # Skip NOOP actions as they're not meaningful for LLM context
                    if primary_action != 'NOOP':
                        primary_tasks.append(primary_action)
        
        # print(f"🔍 Extracted primary tasks from plan: {primary_tasks}")
        return primary_tasks



    def set_mdp(self, mdp: OvercookedGridworld):
        super().set_mdp(mdp)
        self.mdp = mdp
        terrain = mdp.terrain_mtx
        self.ingredient_spawns = [(j, i)
                            for i, row in enumerate(terrain)
                            for j, c in enumerate(row)
                            if c in ('O', 'T')]
        self.onion_spawns = [(j, i)
                            for i, row in enumerate(terrain)
                            for j, c in enumerate(row)
                            if c == 'O']
        self.tomato_spawns = [(j, i)
                            for i, row in enumerate(terrain)
                            for j, c in enumerate(row)
                            if c == 'T']
        self.stove_tiles = [(j, i)
                            for i, row in enumerate(terrain)
                            for j, c in enumerate(row)
                            if c == 'P']
        self.dish_spawns = [(j, i)
                            for i, row in enumerate(terrain)
                            for j, c in enumerate(row)
                            if c == 'D']
        self.delivery_tiles = [(j, i)
                            for i, row in enumerate(terrain)
                            for j, c in enumerate(row)
                            if c == 'S']
        
        # Assign staging stations per stove with directional logic
        self.onion_staging_tiles = []
        self.tomato_staging_tiles = []
        self.dish_staging_tiles = []
        H, W = len(terrain), len(terrain[0]) 
        
        for (stove_c, stove_r) in self.stove_tiles:
            adjacent_staging = []
            for dc, dr in [(1,0), (-1,0), (0,1), (0,-1)]:
                nc, nr = stove_c + dc, stove_r + dr
                if 0 <= nr < H and 0 <= nc < W and terrain[nr][nc] == 'X':
                    adjacent_staging.append((nc, nr, dc, dr))
            
            left_stations = [(c, r) for c, r, dc, dr in adjacent_staging if dc == -1]
            right_stations = [(c, r) for c, r, dc, dr in adjacent_staging if dc == 1]  
            top_stations = [(c, r) for c, r, dc, dr in adjacent_staging if dr == -1]   
            bottom_stations = [(c, r) for c, r, dc, dr in adjacent_staging if dr == 1]
            
            if left_stations:
                self.onion_staging_tiles.extend(left_stations)
            elif bottom_stations:
                self.onion_staging_tiles.extend(bottom_stations)
            
            # if right_stations:
            #     self.dish_staging_tiles.extend(right_stations)
            # elif top_stations:
            #     self.dish_staging_tiles.extend(top_stations)

            self.tomato_staging_tiles = self.onion_staging_tiles.copy()
            self.dish_staging_tiles = self.onion_staging_tiles.copy() # Moving the dish staging tile to agent side
            self.soup_staging_tiles = self.dish_staging_tiles

        # Get layout name for special handling
        layout_name = getattr(mdp, 'layout_name', 'unknown')
        
        # Create chopping stations
        self.onion_chopping_stations = []
        self.tomato_chopping_stations = []
        
        if layout_name == 'cramped_room_tomato':
            # For cramped_room_tomato: use hardcoded chopping station at (0,2)
            self.onion_chopping_stations = [(0, 2)]
            self.tomato_chopping_stations = [(0, 2)]
        else:
            # Use dynamic detection for other layouts
            for staging_pos in self.onion_staging_tiles:
                staging_c, staging_r = staging_pos
                for dc, dr in [(-1, 0), (0, -1)]:
                    chopping_c, chopping_r = staging_c + dc, staging_r + dr
                    if (0 <= chopping_r < H and 0 <= chopping_c < W and 
                        terrain[chopping_r][chopping_c] == 'X' and
                        (chopping_c, chopping_r) not in self.onion_chopping_stations):
                        self.onion_chopping_stations.append((chopping_c, chopping_r))
                        break
            
            self.tomato_chopping_stations = self.onion_chopping_stations.copy()

        # Create sink stations
        self.sink_stations = []
        
        if layout_name == 'counter_circuit':
            # For counter_circuit: sink at (0,2)
            self.sink_stations.append((0, 2))
        elif layout_name == 'cramped_room_tomato':
            # For cramped_room_tomato: sink at (2,3)
            self.sink_stations.append((2, 3))
        
        # Compute frontiers
        self.ingredient_frontier = self._compute_frontier(self.ingredient_spawns, terrain)
        self.onion_frontier = self._compute_frontier(self.onion_spawns, terrain)
        self.tomato_frontier = self._compute_frontier(self.tomato_spawns, terrain)
        self.stove_frontier = self._compute_frontier(self.stove_tiles, terrain) 
        self.dish_frontier = self._compute_frontier(self.dish_spawns, terrain)
        self.delivery_frontier = self._compute_frontier(self.delivery_tiles, terrain)
        self.onion_staging_frontier = self._compute_frontier(self.onion_staging_tiles, terrain)
        self.tomato_staging_frontier = self._compute_frontier(self.tomato_staging_tiles, terrain)
        self.dish_staging_frontier = self._compute_frontier(self.dish_staging_tiles, terrain)
        self.soup_staging_frontier = self._compute_frontier(self.soup_staging_tiles, terrain)
        self.onion_chopping_frontier = self._compute_frontier(self.onion_chopping_stations, terrain)
        self.tomato_chopping_frontier = self._compute_frontier(self.tomato_chopping_stations, terrain)
        self.sink_frontier = self._compute_frontier(self.sink_stations, terrain)
        
        my_goals = {
            'ingredient': self.ingredient_spawns,
            'pot': self.stove_tiles,
            'dish': self.dish_spawns,
            'delivery': self.delivery_tiles
        } 

        self.planner = MotionPlanner(mdp, counter_goals=my_goals)

    def _compute_frontier(self, tiles, terrain):
        """Return the set of walkable tiles adjacent to any tile in 'tiles'."""
        H, W = len(terrain), len(terrain[0])
        frontier = set()
        WALKABLE = {' '}

        for c, r in tiles:
            for dc, dr in [(1,0), (-1,0), (0,1), (0,-1)]:
                nc, nr = c + dc, r + dr
                if (
                    0 <= nr < H and 
                    0 <= nc < W and 
                    terrain[nr][nc] in WALKABLE
                ):
                    orient = (-dc, -dr)
                    frontier.add(((nc, nr), orient))
        return list(frontier)

    def summarize_state(self, state, info):
        """Extract state predicates for the LLM (same as original)"""
        sd = state.to_dict()

        def held_item(player_dict):
            held = player_dict["held_object"]
            if held is None:
                return "none"
            if held.get("name") in ("onion", "tomato"):
                return held["name"]
            if held.get("ingredient") in ("onion", "tomato"):
                return held["ingredient"]
            if held.get("name") in ("dish", "soup"):
                return held["name"]
            return "none"

        me = sd["players"][self.agent_index]
        them = sd["players"][1 - (self.agent_index or 0)]
        agent_item = held_item(me)
        partner_item = held_item(them)

        onion_hand = "none"
        if agent_item == "onion":
            onion_hand = "agent"
        elif partner_item == "onion":
            onion_hand = "partner"

        tomato_hand = "none"
        if agent_item == "tomato":
            tomato_hand = "agent"
        elif partner_item == "tomato":
            tomato_hand = "partner"

        dish_hand = "none"
        if agent_item == "dish":
            dish_hand = "agent"
        elif partner_item == "dish":
            dish_hand = "partner"

        soup_hand = "none"
        if agent_item == "soup":
            soup_hand = "agent"
        elif partner_item == "soup":
            soup_hand = "partner"

        tile_contents = {}
        onion_in_pot = False
        tomato_in_pot = False
        soup_cooking = False
        soup_ready = False
        soup_in_pot_not_cooking = False
        
        for obj in sd["objects"]:
            p = tuple(obj["position"])
            name = obj.get("ingredient") or obj.get("name")
            tile_contents.setdefault(p, []).append(name)
            
            if obj.get("name") == "soup" and obj.get("_ingredients"):
                for ingredient in obj["_ingredients"]:
                    ing_name = ingredient.get("name")
                    if ing_name:
                        tile_contents.setdefault(p, []).append(ing_name)
                
                if p in self.stove_tiles:
                    for ingredient in obj["_ingredients"]:
                        ing_name = ingredient.get("name")
                        if ing_name in ("onion"):
                            onion_in_pot = True
                        elif ing_name in ("tomato"):
                            tomato_in_pot = True
                    
                    cooking_tick = obj.get("cooking_tick", -1)
                    is_cooking = obj.get("is_cooking", False)
                    is_ready = obj.get("is_ready", False)
                    
                    if is_cooking and cooking_tick >= 1:
                        soup_cooking = True
                    elif is_ready:
                        soup_ready = True
                    elif cooking_tick == -1:
                        soup_in_pot_not_cooking = True

        onion_staged = any(
            "onion" in tile_contents.get(pos, [])
            for pos in self.onion_staging_tiles
        )
        
        onion_at_chopping = any(
            "onion" in tile_contents.get(pos, [])
            for pos in self.onion_chopping_stations
        )
        
        if onion_at_chopping:
            self.onion_chopped = True

        tomato_staged = any(
            "tomato" in tile_contents.get(pos, [])
            for pos in self.tomato_staging_tiles
        )
        
        tomato_at_chopping = any(
            "tomato" in tile_contents.get(pos, [])
            for pos in self.tomato_chopping_stations
        )
        
        if tomato_at_chopping:
            self.tomato_chopped = True

        dish_staged = any(
            "dish" in tile_contents.get(pos, [])
            for pos in self.dish_staging_tiles
        )

        soup_staged = any(
            "soup" in tile_contents.get(pos, [])
            for pos in self.soup_staging_tiles
        )

        soup_served = False
        if info:
            soup_served = any(
                info.get("event_infos", {})
                    .get("soup_delivery", [False, False])
            )

        if soup_served:
            self.onion_chopped = False
            self.tomato_chopped = False
            self.onion_washed = False
            self.tomato_washed = False

        # Check sink-related states
        onion_at_sink = any(
            "onion" in tile_contents.get(pos, [])
            for pos in self.sink_stations
        )
        
        tomato_at_sink = any(
            "tomato" in tile_contents.get(pos, [])
            for pos in self.sink_stations
        )
        
        # Track washing completion state
        # When an ingredient is at the sink, it means it's being washed or has been washed
        if onion_at_sink:
            self.onion_washed = True
            
        if tomato_at_sink:
            self.tomato_washed = True

        return {
            "onion_hand": onion_hand,
            "onion_staged": onion_staged,
            "onion_at_chopping": onion_at_chopping,
            "onion_chopped": self.onion_chopped,
            "onion_at_sink": onion_at_sink,
            "onion_washed": self.onion_washed,
            "tomato_hand": tomato_hand,
            "tomato_staged": tomato_staged,
            "tomato_at_chopping": tomato_at_chopping,
            "tomato_chopped": self.tomato_chopped,
            "tomato_at_sink": tomato_at_sink,
            "tomato_washed": self.tomato_washed,
            "onion_in_pot": onion_in_pot,
            "tomato_in_pot": tomato_in_pot,
            "soup_cooking": soup_cooking,
            "soup_ready": soup_ready,
            "soup_in_pot_not_cooking": soup_in_pot_not_cooking,
            "dish_hand": dish_hand,
            "dish_staged": dish_staged,
            "soup_hand": soup_hand,
            "soup_staged": soup_staged,
            "soup_served": soup_served
        }



    def _is_blocking_important_tile(self, my_pos: tuple, state) -> bool:
        """
        Check if the agent is currently blocking an important staging tile.
        Returns True if blocking, False otherwise.
        """
        # Get all important tiles that shouldn't be blocked
        important_tiles = set()
        
        # Add all staging tiles
        important_tiles.update(self.onion_staging_tiles)
        important_tiles.update(self.tomato_staging_tiles)
        important_tiles.update(self.dish_staging_tiles)
        important_tiles.update(self.soup_staging_tiles)
        
        # Add all chopping stations
        important_tiles.update(self.onion_chopping_stations)
        important_tiles.update(self.tomato_chopping_stations)
        
        # Add all sink stations
        important_tiles.update(self.sink_stations)
        
        # Add all frontier tiles (adjacent to important locations)
        important_tiles.update([pos for pos, _ in self.ingredient_frontier])
        important_tiles.update([pos for pos, _ in self.onion_frontier])
        important_tiles.update([pos for pos, _ in self.tomato_frontier])
        important_tiles.update([pos for pos, _ in self.stove_frontier])
        important_tiles.update([pos for pos, _ in self.dish_frontier])
        important_tiles.update([pos for pos, _ in self.delivery_frontier])
        important_tiles.update([pos for pos, _ in self.onion_staging_frontier])
        important_tiles.update([pos for pos, _ in self.tomato_staging_frontier])
        important_tiles.update([pos for pos, _ in self.dish_staging_frontier])
        important_tiles.update([pos for pos, _ in self.soup_staging_frontier])
        important_tiles.update([pos for pos, _ in self.onion_chopping_frontier])
        important_tiles.update([pos for pos, _ in self.tomato_chopping_frontier])
        
        # Check if current position is blocking an important tile
        is_blocking = my_pos in important_tiles
        
        return is_blocking

    def _find_safe_position(self, my_pos: tuple, state) -> tuple:
        """
        Find a safe position to move to that doesn't block important tiles.
        Returns (new_position, action_plan) or (my_pos, []) if no safe move found.
        """
        if not self.mdp:
            return my_pos, []
            
        terrain = self.mdp.terrain_mtx
        H, W = len(terrain), len(terrain[0])
        
        # Get all important tiles to avoid
        important_tiles = set()
        important_tiles.update(self.onion_staging_tiles)
        important_tiles.update(self.tomato_staging_tiles)
        important_tiles.update(self.dish_staging_tiles)
        important_tiles.update(self.soup_staging_tiles)
        important_tiles.update(self.onion_chopping_stations)
        important_tiles.update(self.tomato_chopping_stations)
        important_tiles.update(self.sink_stations)
        important_tiles.update([pos for pos, _ in self.ingredient_frontier])
        important_tiles.update([pos for pos, _ in self.onion_frontier])
        important_tiles.update([pos for pos, _ in self.tomato_frontier])
        important_tiles.update([pos for pos, _ in self.stove_frontier])
        important_tiles.update([pos for pos, _ in self.dish_frontier])
        important_tiles.update([pos for pos, _ in self.delivery_frontier])
        important_tiles.update([pos for pos, _ in self.onion_staging_frontier])
        important_tiles.update([pos for pos, _ in self.tomato_staging_frontier])
        important_tiles.update([pos for pos, _ in self.dish_staging_frontier])
        important_tiles.update([pos for pos, _ in self.soup_staging_frontier])
        important_tiles.update([pos for pos, _ in self.onion_chopping_frontier])
        important_tiles.update([pos for pos, _ in self.tomato_chopping_frontier])
        important_tiles.update([pos for pos, _ in self.sink_frontier])
        
        # Get other player position to avoid blocking them
        other_player_pos = state.player_positions[1 - self.agent_index]
        
        # Find safe positions within reasonable distance (max 3 steps)
        safe_positions = []
        for distance in range(1, 4):  # Check 1, 2, 3 steps away
            for dc in range(-distance, distance + 1):
                for dr in range(-distance, distance + 1):
                    if abs(dc) + abs(dr) == distance:  # Manhattan distance
                        new_col = my_pos[0] + dc
                        new_row = my_pos[1] + dr
                        
                        # Check bounds
                        if 0 <= new_row < H and 0 <= new_col < W:
                            new_pos = (new_col, new_row)
                            
                            # Check if position is walkable and not important
                            if (terrain[new_row][new_col] == ' ' and 
                                new_pos not in important_tiles and
                                new_pos != other_player_pos):
                                safe_positions.append(new_pos)
            
            # If we found safe positions at this distance, stop searching
            if safe_positions:
                break
        
        # If no safe positions found, stay put
        if not safe_positions:
            return my_pos, []
        
        # Choose the closest safe position
        best_pos = min(safe_positions, key=lambda pos: abs(pos[0] - my_pos[0]) + abs(pos[1] - my_pos[1]))
        
        # Generate action plan to move to safe position
        my_ori = state.to_dict()["players"][self.agent_index]["orientation"]
        start_pair = (my_pos, tuple(my_ori))
        goal_pair = (best_pos, tuple(my_ori))  # Keep same orientation
        
        action_plan = self._get_action_plan(start_pair, goal_pair)
        
        return best_pos, action_plan

    def _parse_primary_action(self, response: str) -> str:
        """
        Parse the LLM response to extract only the human primary action prediction.
        Expected format:
          Primary: <primary_action_name>
        """
        # Default value
        predicted_human_action = "NOOP"
        
        # Parse primary action prediction
        primary_match = re.search(r'Primary:\s*(.+?)(?:\n|$)', 
                                 response, re.IGNORECASE)
        if primary_match:
            predicted_human_action = primary_match.group(1).strip()
        
        return predicted_human_action

    def _get_action_plan(self, start_pair, goal_pair):
        """Get action plan between two position/orientation pairs."""
        if self.mdp is None:
            return []
        terrain = self.mdp.terrain_mtx

        if self.planner is None:
            start_pos, start_ori = start_pair
            goal_pos, goal_ori = goal_pair
            return _bfs_fallback(start_pos, goal_pos, terrain, goal_ori)
        
        try:
            action_plan, _, _ = self.planner.get_plan(start_pair, goal_pair)
            return action_plan
        except KeyError:
            try:
                goal_pos, goal_ori = goal_pair
                action_plan, _, _ = self.planner.action_plan_from_positions(
                    [goal_pos], start_pair, goal_pair
                )
                return action_plan
            except Exception:
                start_pos, start_ori = start_pair
                goal_pos, goal_ori = goal_pair
                action_plan = _bfs_fallback(start_pos, goal_pos, terrain, goal_ori)
                return action_plan



    def _find_nearest_goal(self, choices, my_pos: tuple, my_ori: tuple) -> tuple:
        """Find the nearest goal from a list of choices."""
        if not choices:
            return (my_pos, tuple(my_ori))

        def sort_key(mo):
            (c, r), _ = mo
            # primary: Manhattan distance
            dist = abs(c - my_pos[0]) + abs(r - my_pos[1])
            # secondary: prefer smaller col, then smaller row
            return (dist, c, r)

        goal_pos, goal_orient = min(choices, key=sort_key)
        return (goal_pos, goal_orient)

    def _parse_robot_action(self, robot_action, game_state=None):
        """
        Parse the robot action to get function name and item with state-aware location detection.
        Expected format: pickup(onion) or place(onion, chopping_station) or NOOP
        
        For pickup actions, uses game state to determine the appropriate location:
        - pickup(onion) checks onion_at_chopping, onion_at_sink, or defaults to dispenser
        - pickup(tomato) checks tomato_at_chopping, tomato_at_sink, or defaults to dispenser
        """
        # Check for explicit NOOP
        if robot_action == "NOOP":
            return "NOOP", None

        # Parse pickup actions with state-aware location detection
        pickup_match = re.search(r'pickup\(([^)]+)\)', robot_action)
        if pickup_match:
            item = pickup_match.group(1).strip()
            
            # State-aware location detection for onion/tomato
            if item == "onion" and game_state:
                if game_state.get('onion_at_chopping', False):
                    location = "chopping_station"
                elif game_state.get('onion_at_sink', False):
                    location = "sink"
                else:
                    location = "dispenser"
                return "pickup", (item, location)
                
            elif item == "tomato" and game_state:
                if game_state.get('tomato_at_chopping', False):
                    location = "chopping_station"
                elif game_state.get('tomato_at_sink', False):
                    location = "sink"
                else:
                    location = "dispenser"
                return "pickup", (item, location)
            
            # Other items have fixed/unambiguous locations
            elif item in ["dish", "soup"]:
                return "pickup", item
                
            # Fallback for onion/tomato without game state
            elif item in ["onion", "tomato"]:
                return "pickup", item

        # Parse place actions with destination
        place_match = re.search(r'place\(([^,]+),\s*([^)]+)\)', robot_action)
        if place_match:
            item = place_match.group(1).strip()
            destination = place_match.group(2).strip()
            if item in ["onion", "tomato", "dish"] and destination in ["chopping_station", "staging_station", "sink"]:
                return "place", (item, destination)

        # Parse place actions without destination (for items with fixed destinations)
        place_simple_match = re.search(r'place\(([^)]+)\)', robot_action)
        if place_simple_match:
            item = place_simple_match.group(1).strip()
            if item in ["dish", "soup"]:
                return "place", (item, "default")

        # final fallback
        return "pickup", "onion"

    def _move_to(self, action: str, item_info, start_pos: tuple, start_ori: tuple, destination: str = None):
        """Move to the appropriate location for the given action and item."""
        if action == "pickup":
            # Handle new tuple format (item, location) or legacy format (just item)
            if isinstance(item_info, tuple):
                item, location = item_info
                if location == "chopping_station":
                    choices = self.onion_chopping_frontier if item == "onion" else self.tomato_chopping_frontier
                elif location == "sink":
                    choices = self.sink_frontier
                elif location == "dispenser":
                    choices = self.onion_frontier if item == "onion" else self.tomato_frontier
                else:
                    choices = None
            else:
                # Legacy format: just item name
                item = item_info
                frontier_map = {
                    "onion": self.onion_frontier,
                    "tomato": self.tomato_frontier,
                    "dish": self.dish_frontier,
                    "soup": self.soup_staging_frontier,
                }
                choices = frontier_map.get(item)
        elif action == "place":
            # Extract item name from tuple format if needed
            if isinstance(item_info, tuple):
                item = item_info[0]  # Extract item from (item, location) tuple
            else:
                item = item_info
                
            if destination == "chopping_station":
                frontier_map = {
                    "onion": self.onion_chopping_frontier,
                    "tomato": self.tomato_chopping_frontier,
                }
                choices = frontier_map.get(item)
            elif destination == "staging_station":
                frontier_map = {
                    "onion": self.onion_staging_frontier,
                    "tomato": self.tomato_staging_frontier,
                    "dish": self.dish_staging_frontier,
                }
                choices = frontier_map.get(item)
            elif destination == "sink":
                frontier_map = {
                    "onion": self.sink_frontier,
                    "tomato": self.sink_frontier,
                }
                choices = frontier_map.get(item)
            else:
                frontier_map = {
                    "dish": self.dish_staging_frontier,
                    "soup": self.delivery_frontier,
                }
                choices = frontier_map.get(item)
        else:
            return []
        
        if not choices:
            return []
        
        goal = min(choices, key=lambda mo: abs(mo[0][0] - start_pos[0]) + abs(mo[0][1] - start_pos[1]))
        start_pair = (start_pos, tuple(start_ori))
        return self._get_action_plan(start_pair, goal)

    def PickUp(self, item_info, start_pos, start_ori):
        """Returns an action plan to pick up the specified item (supports both tuple and string format)."""
        action_plan = self._move_to("pickup", item_info, start_pos, start_ori)
        # Only add INTERACT if it's not already in the plan
        if action_plan and Action.INTERACT not in action_plan:
            action_plan.append(Action.INTERACT)
        return action_plan

    def Place(self, item, start_pos, start_ori, destination: str = "default"):
        """Returns an action plan to place the specified item at the destination."""
        if destination == "default":
            action_plan = self._move_to("place", item, start_pos, start_ori, destination)
        else:
            action_plan = self._move_to("place", item, start_pos, start_ori, destination)
        
        # Only add INTERACT if it's not already in the plan
        if action_plan and Action.INTERACT not in action_plan:
            action_plan.append(Action.INTERACT)
        return action_plan

    def action(self, state):
        """Main action selection using optimized coordination system."""
        assert self.agent_index is not None, "agent_index is None in action!"
        
        # Get current state summary
        self.last_summary = self.summarize_state(state, {})
        
        # NEW: Get available primary actions from our complete state graph with washing
        try:
            available_primary_actions = self.get_available_primary_actions(state, {})
            print(f"STATE GRAPH ACTIONS: {available_primary_actions}")
        except Exception as e:
            print(f"Error getting available actions from complete state graph: {e}")
            available_primary_actions = []
        
        # FALLBACK: If no actions available from complete state graph, provide basic actions
        if not available_primary_actions:
            print("WARNING: No actions from complete state graph, using fallback actions")
            available_primary_actions = [
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
                "Pour Soup",                   # Human pours ready soup into dish
                "Human Stage Soup",            # Human stages soup for serving
                "Wait For Robot To Serve Soup"  # Robot serves the soup (robot action)
            ]
        
        plan_text = ""
        if hasattr(self, 'primary_tasks') and self.primary_tasks:
            plan_lines = []
            for idx, primary_task in enumerate(self.primary_tasks):
                plan_lines.append(f"{idx+1}) {primary_task}")
            plan_text = "\n".join(plan_lines)
        else:
            plan_text = "No primary tasks available"

        # NEW: Include historical successful plans in the prompt
        historical_plans_text = ""
        if not self.plan_repository.is_empty():
            historical_plans_text = "\nPREVIOUS SUCCESSFUL PLANS:\n"
            all_plans = self.plan_repository.get_all_plans()
            for i, plan in enumerate(all_plans):
                actions_str = " → ".join(plan['actions'])
                historical_plans_text += f"Plan {plan['sequence_id']}: {actions_str}\n"
            historical_plans_text += "\nUse these successful plans as reference for effective action sequences.\n"
        
        prompt = f"""
        You are helping a human cook soup. Follow the user's plan step-by-step in the correct sequence.
        IMPORTANT: SERVE THE CURRENT SOUP BEFORE STARTING WITH NEW INGREDIENTS.

        CURRENT STATE:
        {self.last_summary}

        USER PLAN (follow in order):
        {plan_text}

        AVAILABLE ACTIONS:
        {available_primary_actions}

        PREVIOUS SUCCESSFUL PLANS:
        {historical_plans_text}

        **CRITICAL: Follow the plan sequence step-by-step!**
        - The plan is designed to be followed in order
        - Don't skip ahead to later steps
        - Only choose actions that are both AVAILABLE and the NEXT LOGICAL STEP in the user's plan

        Select the action that best aligns with the user's plan and current state.

        Return only this line:
        Primary: <action_name>
        """
        
        # Display essential information for testing
        print(f"CURRENT STATE: {self.last_summary}")
        print(f"SUPPLYING {len(available_primary_actions)} ACTIONS TO LLM: {available_primary_actions}")
        
        # NEW: Show plan adaptation info
        if not self.plan_repository.is_empty():
            print(f"PLAN ADAPTATION: Including {self.plan_repository.get_plan_count()} historical plans in prompt")
            print(f"PLAN ADAPTATION: Current sequence has {len(self.action_tracker.get_current_sequence())} actions")
        else:
            print("PLAN ADAPTATION: No historical plans yet, starting fresh")
        # Call LLM to get predictions
        response = query_openai(prompt, self.selected_model)
        print(f"LLM RESPONSE: {response}")
        
        # Parse the response - only need primary action now
        predicted_human_action = self._parse_primary_action(response)
        
        # NEW: Track primary action for plan adaptation
        self.action_tracker.record_action(predicted_human_action)
        print(f"PLAN TRACKING: Recorded action '{predicted_human_action}' (sequence length: {len(self.action_tracker.get_current_sequence())})")
        
        # NEW: Check if soup was served (success detection)
        if self.last_summary and self.last_summary.get('soup_served', False):
            # Finalize and store the successful sequence
            if not self.action_tracker.is_empty():
                successful_plan = {
                    'actions': self.action_tracker.get_current_sequence(),
                    'duration': self.action_tracker.get_sequence_duration(),
                    'recipe_type': 'onion_washed_chopped_tomato_washed_chopped'  # Fixed recipe type for now
                }
                self.plan_repository.add_successful_plan(successful_plan)
                print(f"PLAN SUCCESS: Stored successful plan with {len(successful_plan['actions'])} actions")
                print(f"PLAN REPOSITORY: Now has {self.plan_repository.get_plan_count()} total plans")
                
                # Reset tracker for next sequence
                self.action_tracker.reset()
                print("PLAN TRACKING: Reset tracker for new sequence")
        
        # Get robot action using our smart coordination system
        try:
            robot_action = select_secondary_action(self.last_summary, self.task_title, predicted_human_action)
        except Exception as e:
            print(f"Error getting robot action: {e}")
            robot_action = "NOOP"
        
        # Clean, essential debugging output
        print(f"LLM PREDICTION: {predicted_human_action}")
        print(f"ROBOT ACTION: {robot_action}")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        
        # Convert high-level action to low-level movement
        my_pos = state.player_positions[self.agent_index]
        my_ori = state.to_dict()["players"][self.agent_index]["orientation"]
        
        # Execute the robot action using the EXACT same logic as action_predictor.py
        # FIXED: We now ALWAYS create new action plans instead of reusing old ones
        # This ensures the robot responds immediately to changing game states and predictions
        try:
            # ALWAYS create a new action plan based on the current robot action
            # This ensures we respond to changing game states and predictions
            # We no longer reuse old plans - each prediction gets a fresh action plan
            func_name, item_info = self._parse_robot_action(robot_action, self.last_summary)
            print(f"Parsed robot action: {func_name}({item_info})")
            
            if func_name == "NOOP":
                # Check if the agent is blocking important tiles
                if self._is_blocking_important_tile(my_pos, state):
                    # If blocking, find a safe position to move to
                    safe_pos, action_plan = self._find_safe_position(my_pos, state)
                    if action_plan:
                        # Return first action from the plan
                        move = action_plan[0] if action_plan else Action.STAY
                        return move, {
                            "predicted_human_action": predicted_human_action,
                            "robot_action": robot_action,
                            # "next_planned_action": next_planned_action,  # TODO: Re-enable
                            "llm_response": response,
                            "available_primary_actions": available_primary_actions,
                            "blocking_prevention": True,
                            "action_plan": action_plan
                        }
                    else:
                        # If no safe move found, stay put
                        return Action.STAY, {
                            "predicted_human_action": predicted_human_action,
                            "robot_action": robot_action,
                            # "next_planned_action": next_planned_action,  # TODO: Re-enable
                            "llm_response": response,
                            "available_primary_actions": available_primary_actions,
                            "blocking_prevention": False,
                            "action_plan": []
                        }
                else:
                    # If not blocking, stay put
                    return Action.STAY, {
                        "predicted_human_action": predicted_human_action,
                        "robot_action": robot_action,
                        # "next_planned_action": next_planned_action,  # TODO: Re-enable
                        "llm_response": response,
                        "available_primary_actions": available_primary_actions,
                        "blocking_prevention": False,
                        "action_plan": []
                    }

            # Execute the compound action (EXACTLY like action_predictor.py)
            if func_name == "pickup":
                action_plan = self.PickUp(item_info, my_pos, my_ori)
            elif func_name == "place":
                if isinstance(item_info, tuple):
                    item_to_place, destination = item_info
                    action_plan = self.Place(item_to_place, my_pos, my_ori, destination)
                else:
                    action_plan = self.Place(item_info, my_pos, my_ori, "default")
            else:
                # Fallback to simple movement
                action_plan = [Action.STAY]

            # Return first action from the plan
            if action_plan:
                move = action_plan[0] if action_plan else Action.STAY
                
                return move, {
                    "predicted_human_action": predicted_human_action,
                    "robot_action": robot_action,
                    # "next_planned_action": next_planned_action,  # TODO: Re-enable
                    "llm_response": response,
                    "available_primary_actions": available_primary_actions,
                    "function_call": f"{func_name}({item_info})",
                    "action_plan": action_plan
                }
            else:
                return Action.STAY, {
                    "predicted_human_action": predicted_human_action,
                    "robot_action": robot_action,
                    # "next_planned_action": next_planned_action,  # TODO: Re-enable
                    "llm_response": response,
                    "available_primary_actions": available_primary_actions,
                    "function_call": f"{func_name}({item_info})",
                    "action_plan": []
                }
        
        except Exception as e:
            print(f"Error executing robot action: {e}")
            return Action.STAY, {
                "predicted_human_action": predicted_human_action,
                "robot_action": robot_action,
                # "next_planned_action": next_planned_action,  # TODO: Re-enable 
                "llm_response": response,
                "available_primary_actions": available_primary_actions,
                "reasoning": "Action execution failed, staying in place"
            }

    def actions(self, states, agent_indices):
        return [self.action(s) for s in states]
    


 

 