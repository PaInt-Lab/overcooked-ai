from collections import deque
import json
import re
from overcooked_ai_py.agents.agent import Agent
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.planning.planners import MotionPlanner, NO_COUNTERS_PARAMS
from overcooked_ai_py.mdp.actions import Action, Direction
from llm.ollama.ollama_client import query_ollama
import os
from openai import OpenAI
from llm.memory.vector_memory import VectorMemory
from ...state_graph import StateGraphGenerator, StateGraph
from ...coordination_system import CoordinationManager

OVERCOOKED_GAME_MECHANICS = """
## OVERCOOKED GAME MECHANICS (MDP Knowledge)

### State Variables:
- onion_hand in {none, agent, partner} - Who is holding the onion
- onion_staged in {true, false} - Is onion staged/placed somewhere accessible
- onion_at_chopping in {true, false} - Is onion at chopping station
- onion_chopped in {true, false} - Is the onion chopped (flag set when placed at chopping station)
- tomato_hand in {none, agent, partner} - Who is holding the tomato
- tomato_staged in {true, false} - Is tomato staged/placed somewhere accessible
- tomato_at_chopping in {true, false} - Is tomato at chopping station
- tomato_chopped in {true, false} - Is the tomato chopped (flag set when placed at chopping station)
- onion_in_pot in {true, false} - Is onion placed in cooking pot (raw or chopped)
- tomato_in_pot in {true, false} - Is tomato placed in cooking pot (raw or chopped)
- soup_cooking in {true, false} - Is soup actively cooking (ticker >= 1)
- soup_ready in {true, false} - Is soup ready to serve
- soup_hand in {none, agent, partner} - Who is holding the soup
- soup_staged in {true, false} - Is soup staged/placed somewhere accessible
- soup_in_pot_not_cooking in {true, false} - Is soup in pot but not cooking (ticker = -1)
- dish_hand in {none, agent, partner} - Who is holding the dish
- dish_staged in {true, false} - Is dish staged/placed somewhere accessible
- soup_served in {true, false} - Is soup delivered to serving station

### Goal-Directed Navigation:
You are navigating through a state space toward the goal of serving soup. Your job is to:

1. **Analyze current state** to understand where you are in the state space
2. **Predict human behavior** to understand what the human is likely to do
3. **Choose the best action** that advances toward the goal while coordinating with the human
4. **Balance goal progress and coordination** - don't sacrifice one for the other

### Coordination Strategy:
- **Positive Coordination**: Actions that work well with human behavior
- **Negative Coordination**: Actions that conflict with human behavior
- **Goal Progress**: Actions that advance toward serving soup
- **Adaptive Strategy**: Choose actions based on current coordination context

### Available Actions:
- pickup(onion): Pick up onion from dispenser
- pickup(tomato): Pick up tomato from dispenser
- pickup(chopped_onion): Pick up chopped onion from chopping station
- pickup(chopped_tomato): Pick up chopped tomato from chopping station
- pickup(dish): Pick up dish from dispenser
- pickup(soup): Pick up soup from staging
- place(onion, chopping_station): Place onion at chopping station (auto-chops)
- place(onion, staging_station): Place onion at staging station
- place(tomato, chopping_station): Place tomato at chopping station (auto-chops)
- place(tomato, staging_station): Place tomato at staging station
- place(chopped_onion): Place chopped onion at staging station
- place(chopped_tomato): Place chopped tomato at staging station
- place(dish): Place dish at staging station
- place(soup): Place soup at serving station
- NOOP: No action needed

### IMPORTANT: Robot Restrictions
- The robot CANNOT place ingredients in the pot/stove
- The robot CANNOT turn on the stove
- The robot CANNOT interact with the stove in any way
- Only the human player handles cooking and stove interactions
- The robot focuses on preparation tasks: chopping and staging ingredients

## TASK EXECUTION

Each call you receive has this structure:

STATE SUMMARY:
<current state description>

COORDINATION CONTEXT:
<predicted human action, coordination quality, conflict risk, goal progress>

Your job:

1. **Analyze the current state** and coordination context
2. **Choose the best action** that balances goal progress and coordination
3. **Consider the predicted human action** when making your decision
4. **Return the optimal action** for the current situation

Return **only** the action name (no extra commentary):

<action_name>
"""

OVERCOOKED_MODEL = "ft:gpt-4o-mini-2024-07-18:personal:ap-onion-tomato-chopped:BysMfU2m"

def query_openai(prompt: str, model: str = OVERCOOKED_MODEL, temperature: float = 0.0) -> str:
    """Query the OpenAI API with the given prompt and return the response text."""
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
        self.last_info = None
        self.cleaned_terrain = None
        self.last_summary = None
        self.agent_index = None
        self.memory = VectorMemory()
        self.onion_chopped = False
        self.tomato_chopped = False
        
        # State graph and coordination system
        self.state_graph = None
        self.coordination_manager = None
        self._initialize_state_graph()

    def _initialize_state_graph(self):
        """Initialize the state graph and coordination system"""
        print("Initializing state graph...")
        generator = StateGraphGenerator()
        self.state_graph = generator.generate_state_graph()
        self.coordination_manager = CoordinationManager(self.state_graph)
        print(f"State graph initialized with {len(self.state_graph.nodes)} nodes")

    def set_agent_index(self, agent_index: int):
        super().set_agent_index(agent_index)
        self.agent_index = agent_index

    TERRAIN_MAPPING = {
        "X": "Wall",
        "P": "Stove",
        "O": "Onions",
        "T": "Tomatoes",  
        "D": "DishSpawn",
        "S": "Serving",
        " ": "Empty"
    }

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
            
            self.tomato_staging_tiles = self.onion_staging_tiles.copy()
            
            if right_stations:
                self.dish_staging_tiles.extend(right_stations)
            elif top_stations:
                self.dish_staging_tiles.extend(top_stations)

            self.soup_staging_tiles = self.dish_staging_tiles

        # Create chopping stations
        self.onion_chopping_stations = []
        self.tomato_chopping_stations = []
        
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
        
        my_goals = {
            'ingredient': self.ingredient_spawns,
            'pot': self.stove_tiles,
            'dish': self.dish_spawns,
            'delivery': self.delivery_tiles
        } 
        
        self.cleaned_terrain = [
            [self.TERRAIN_MAPPING.get(cell, "Unknown") for cell in row]
            for row in terrain
        ]

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
                        if ing_name in ("onion", "chopped_onion"):
                            onion_in_pot = True
                        elif ing_name in ("tomato", "chopped_tomato"):
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

        return {
            "onion_hand": onion_hand,
            "onion_staged": onion_staged,
            "onion_at_chopping": onion_at_chopping,
            "onion_chopped": self.onion_chopped,
            "tomato_hand": tomato_hand,
            "tomato_staged": tomato_staged,
            "tomato_at_chopping": tomato_at_chopping,
            "tomato_chopped": self.tomato_chopped,
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

    def _move_to(self, action: str, item: str, start_pos: tuple, start_ori: tuple, destination: str = None):
        """Move to the appropriate location for the given action and item."""
        if action == "pickup":
            frontier_map = {
                "onion": self.onion_frontier,
                "tomato": self.tomato_frontier,
                "chopped_onion": self.onion_chopping_frontier,
                "chopped_tomato": self.tomato_chopping_frontier,
                "dish": self.dish_frontier,
                "soup": self.soup_staging_frontier,
            }
            choices = frontier_map.get(item)
        elif action == "place":
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
                }
                choices = frontier_map.get(item)
            else:
                frontier_map = {
                    "chopped_onion": self.onion_staging_frontier,
                    "chopped_tomato": self.tomato_staging_frontier,
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

    def PickUp(self, item, start_pos, start_ori):
        """Returns an action plan to pick up the specified item."""
        action_plan = self._move_to("pickup", item, start_pos, start_ori)
        if action_plan:
            action_plan.append(Action.INTERACT)
        return action_plan

    def Place(self, item, start_pos, start_ori, destination: str = "default"):
        """Returns an action plan to place the specified item at the destination."""
        if destination == "default":
            action_plan = self._move_to("place", item, start_pos, start_ori, destination)
        else:
            action_plan = self._move_to("place", item, start_pos, start_ori, destination)
        
        if action_plan:
            action_plan.append(Action.INTERACT)
        return action_plan

    def action(self, state):
        """Main action selection using coordinated state graph navigation with LLM."""
        assert self.agent_index is not None, "agent_index is None in action!"
        
        info = getattr(self, "last_info", {})       
        self.last_summary = self.summarize_state(state, self.last_info)
        print(f"Current position: {state.player_positions[self.agent_index]}")
        print(f"State summary: {self.last_summary}")

        # Get possible actions from state graph
        current_node_id = self.coordination_manager.action_selector._get_node_id_for_state(self.last_summary)
        if current_node_id:
            possible_actions = self.coordination_manager.action_selector._get_possible_robot_actions(current_node_id)
        else:
            possible_actions = ['NOOP']
        
        print(f"Possible actions: {possible_actions}")

        # Create LLM prompt with current state and possible actions
        prompt = f"""
            {OVERCOOKED_GAME_MECHANICS}

            CURRENT STATE:
            {self.last_summary}

            POSSIBLE ACTIONS:
            {possible_actions}

            GOAL: Serve soup (soup_served = true)

            Based on the current state and available actions, what is the best robot action to take?
            Choose from the possible actions list above.

            Consider:
            1. Which action advances toward the goal of serving soup?
            2. Which action coordinates well with potential human actions?
            3. Which action is most efficient given the current state?

            Return only the action name (e.g., "pickup(onion)" or "place(onion, chopping_station)" or "NOOP"):
            """

        # Call LLM to get the best action
        response = query_openai(prompt, OVERCOOKED_MODEL)
        print(f"LLM Response: {response}")
        
        # Parse the response to get the action
        action = response.strip()
        
        # Validate that the action is in our possible actions
        if action not in possible_actions:
            print(f"Warning: LLM returned '{action}' but it's not in possible actions. Using first available action.")
            action = possible_actions[0] if possible_actions else 'NOOP'
        
        print(f"Selected action: {action}")

        # Convert high-level action to low-level movement
        my_pos = state.player_positions[self.agent_index]
        my_ori = state.to_dict()["players"][self.agent_index]["orientation"]

        if action == "NOOP":
            return Action.STAY, {
                "action": action,
                "llm_response": response,
                "possible_actions": possible_actions,
                "reasoning": "No action needed"
            }

        # Parse action and execute
        if action.startswith("pickup("):
            item = action[7:-1]  # Extract item from pickup(item)
            action_plan = self.PickUp(item, my_pos, my_ori)
        elif action.startswith("place("):
            # Parse place(action, destination) or place(item)
            if "," in action:
                parts = action[6:-1].split(", ")
                item = parts[0]
                destination = parts[1]
                action_plan = self.Place(item, my_pos, my_ori, destination)
            else:
                item = action[6:-1]  # Extract item from place(item)
                action_plan = self.Place(item, my_pos, my_ori)
        else:
            action_plan = [Action.STAY]

        # Return first action from the plan
        move = action_plan[0] if action_plan else Action.STAY
        print(f"Next Move: {move}")

        # Store in memory for future reference
        self.memory.add_game_memory(
            state_summary=self.last_summary,
            action_info={
                "action": action,
                "llm_response": response,
                "possible_actions": possible_actions
            },
            task_title="Coordinated Navigation"
        )
        
        return move, {
            "action": action,
            "llm_response": response,
            "possible_actions": possible_actions,
            "action_plan": action_plan
        }

    def actions(self, states, agent_indices):
        return [self.action(s) for s in states] 