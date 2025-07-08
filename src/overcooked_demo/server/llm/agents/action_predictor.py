from collections import deque
import json
import re
from overcooked_ai_py.agents.agent import Agent
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.planning.planners import MotionPlanner, NO_COUNTERS_PARAMS
from overcooked_ai_py.mdp.actions import Action, Direction
from llm.ollama.ollama_client import query_ollama
from plan_session import PLAN_STORE

def serialize_state(state, mdp) -> str:
    """
    Convert OvercookedState and MDP into a compact JSON string for the LLM prompt.
    """
    state_info = {
        'terrain': mdp.terrain_mtx,
        'state': state.to_dict()
    }
    return json.dumps(state_info)


def _bfs_fallback(start, goal, terrain):
    """
    Return a list of (delta_row, delta_col) moves to walk from start to goal on terrain,
    ignoring orientation. Guaranteed to find a path if one exists.
    """
    H, W = len(terrain), len(terrain[0])
    visited = {start}
    parent = {}
    queue = deque([start])

    # Four cardinal directions: (delta_row, delta_col)
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    while queue:
        row, col = queue.popleft()
        if (row, col) == goal:
            # Reconstruct path of (dr, dc) tuples
            path = []
            cur = goal
            while cur != start:
                prev = parent[cur]
                dr = cur[0] - prev[0]
                dc = cur[1] - prev[1]
                path.append((dr, dc))
                cur = prev
            path = list(reversed(path))  
            path.append(Action.INTERACT)  
            return path

        for dr, dc in directions:
            new_row = row + dr
            new_col = col + dc
            if (
                0 <= new_row < H and
                0 <= new_col < W and
                terrain[new_row][new_col] != 'X' and
                (new_row, new_col) not in visited
            ):
                visited.add((new_row, new_col))
                parent[(new_row, new_col)] = (row, col)
                queue.append((new_row, new_col))

    return []

class ActionPredictorAgent(Agent):
    def __init__(self):
        super().__init__()
        self.mdp = None
        self.planner = None
        self.ingredient_spawns = []
        self.stove_tiles = []
        self.dish_spawns = []
        self.delivery_tiles = []
        self.staging_tiles = []
        self.last_info = None
        self.cleaned_terrain = None

    def set_agent_index(self, agent_index: int):
        super().set_agent_index(agent_index)

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
        self.ingredient_spawns = [(i, j)
                            for i, row in enumerate(terrain)
                            for j, c in enumerate(row)
                            if c in ('O', 'T')]
        self.stove_tiles       = [(i, j) # Need to manipulate this for the bot to place dishes next to stove
                            for i, row in enumerate(terrain)
                            for j, c in enumerate(row)
                            if c == 'P']
        self.dish_spawns       = [(i, j)
                            for i, row in enumerate(terrain)
                            for j, c in enumerate(row)
                            if c == 'D']
        self.delivery_tiles    = [(i, j)
                            for i, row in enumerate(terrain)
                            for j, c in enumerate(row)
                            if c == 'S']
        # Assign staging stations per stove with directional logic
        self.onion_staging_tiles = []
        self.dish_staging_tiles = []
        H, W = len(terrain), len(terrain[0])
        
        for (stove_r, stove_c) in self.stove_tiles:
            # Find all adjacent staging positions for this stove
            adjacent_staging = []
            for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                nr, nc = stove_r + dr, stove_c + dc
                if 0 <= nr < H and 0 <= nc < W and terrain[nr][nc] == 'X':
                    adjacent_staging.append((nr, nc, dr, dc))
            
            # Categorize by direction relative to stove
            left_stations = [(r, c) for r, c, dr, dc in adjacent_staging if dc == -1]  # left of stove
            right_stations = [(r, c) for r, c, dr, dc in adjacent_staging if dc == 1]  # right of stove  
            top_stations = [(r, c) for r, c, dr, dc in adjacent_staging if dr == -1]   # above stove
            bottom_stations = [(r, c) for r, c, dr, dc in adjacent_staging if dr == 1] # below stove
            
            # Assign onion staging: prefer left, fallback to bottom
            if left_stations:
                self.onion_staging_tiles.extend(left_stations)
            elif bottom_stations:
                self.onion_staging_tiles.extend(bottom_stations)
            
            # Assign dish staging: prefer right, fallback to top
            if right_stations:
                self.dish_staging_tiles.extend(right_stations)
            elif top_stations:
                self.dish_staging_tiles.extend(top_stations)

        self.ingredient_frontier = self._compute_frontier(self.ingredient_spawns, terrain)
        self.stove_frontier      = self._compute_frontier(self.stove_tiles,       terrain) 
        self.dish_frontier       = self._compute_frontier(self.dish_spawns,       terrain)
        self.delivery_frontier   = self._compute_frontier(self.delivery_tiles,    terrain)
        self.onion_staging_frontier = self._compute_frontier(self.onion_staging_tiles, terrain)
        self.dish_staging_frontier  = self._compute_frontier(self.dish_staging_tiles,  terrain)
        
        my_goals = {
            'ingredient': self.ingredient_spawns,
            'pot':        self.stove_tiles,
            'dish':       self.dish_spawns,
            'delivery':   self.delivery_tiles
        } 
        
        self.cleaned_terrain = [
            [ self.TERRAIN_MAPPING.get(cell, "Unknown") for cell in row ]
            for row in terrain
        ]

        self.planner = MotionPlanner(mdp, counter_goals=my_goals) # Eventually, instead of building everytime, can save a pickled version 

    def set_plan(self, session_id: str):
        """Attach the full PlanSession to this agent."""
        self.plan = PLAN_STORE[session_id]

    def _compute_frontier(self, tiles, terrain):
        """
        Return the set of walkable tiles adjacent to any tile in 'tiles'.
        """
        H, W = len(terrain), len(terrain[0])
        frontier = set()
        WALKABLE = {' '}

        for r, c in tiles:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < H and 
                    0 <= nc < W and 
                    terrain[nr][nc] in WALKABLE
                ):
                    orient = (-dc, -dr)
                    frontier.add(((nr, nc), orient))
        return list(frontier)

    def summarize_state(self, state, info):
        """
        Extract exactly the predicates we need for the LLM:
        - agent_holding: one of "onion", "tomato", "dish", "soup", or "none"
        - partner_holding: same for the other player
        - onion_staged:        is raw onion on any onion‐staging tile?
        - ingredient_in_pot:   is any onion/tomato in any stove tile?
        - dish_staged:         is any clean dish on any dish‐staging tile?
        - soup_staged:         is any soup on any dish‐staging tile?
        - soup_served:         did the last transition include a soup_delivery?
        """
        sd = state.to_dict()

        def held_item(player_dict):
            held = player_dict["held_object"]
            if held is None:
                return "none"
            if held.get("ingredient") in ("onion", "tomato"):
                return held["ingredient"]
            if held.get("name") in ("dish", "soup"):
                return held["name"]
            return "none"

        # 1) what each player holds
        me   = sd["players"][self.agent_index]
        them = sd["players"][1 - self.agent_index]
        agent_holding   = held_item(me)
        partner_holding = held_item(them)

        # 2) collect what’s on every tile
        tile_contents = {}
        for obj in sd["objects"]:
            p    = tuple(obj["position"])
            name = obj.get("ingredient") or obj.get("name")
            tile_contents.setdefault(p, []).append(name)

        # 3) onion_staged?
        onion_staged = any(
            "onion" in tile_contents.get(pos, [])
            for pos in self.onion_staging_tiles
        )

        # 4) ingredient_in_pot?
        ingredient_in_pot = any(
            ing in tile_contents.get(pos, [])
            for pos in self.stove_tiles
            for ing in ("onion", "tomato")
        )

        # 5) dish_staged?
        dish_staged = any(
            "dish" in tile_contents.get(pos, [])
            for pos in self.dish_staging_tiles
        )

        # 6) soup_staged?
        soup_staged = any(
            "soup" in tile_contents.get(pos, [])
            for pos in self.dish_staging_tiles
        )

        # 7) soup_served?
        soup_served = False
        if info:
            soup_served = any(
                info.get("event_infos", {})
                    .get("soup_delivery", [False, False])
            )

        return {
            "agent_holding":    agent_holding,
            "partner_holding":  partner_holding,
            "onion_staged":     onion_staged,
            "ingredient_in_pot":ingredient_in_pot,
            "dish_staged":      dish_staged,
            "soup_staged":      soup_staged,
            "soup_served":      soup_served
        }

    
    def _parse_function_call(self, response):
        """
        Parse the LLM response to extract both primary event and function call.
        Expected format:
          primary: <event description>
          secondary: pickup_and_place(onion)
          or
          secondary: NOOP
        """
        # grab the primary
        primary_match = re.search(r'primary:\s*(.+?)(?:\n|secondary:|$)',
                                  response,
                                  re.IGNORECASE)
        primary_event = primary_match.group(1).strip() if primary_match else "Unknown Event"

        # check for explicit NOOP
        if re.search(r'secondary:\s*noop', response, re.IGNORECASE):
            return primary_event, "NOOP", None

        # otherwise fall back to pickup_and_place(...)
        secondary_match = re.search(r'pickup_and_place\s*\(\s*(\w+)\s*\)',
                                    response.lower())
        if secondary_match:
            item = secondary_match.group(1)
            if item not in ("onion", "dish", "soup"):
                item = "onion"
            return primary_event, "pickup_and_place", item

        # final fallback
        return primary_event, "pickup_and_place", "onion"

        
    def _task_to_goal(self, task: str, my_pos: tuple, my_ori: tuple) -> tuple:
        """
        Given a canonical task, return the nearest tile for that goal.
        """
        if task == "Fetch Ingredient":
            choices = self.ingredient_frontier
        elif task == "Stage Ingredient at Stove": 
            choices = self.onion_staging_frontier
        elif task == "Fetch Dish":
            choices = self.dish_frontier
        elif task == "Stage Dish at Stove":  
            choices = self.dish_staging_frontier
        elif task == "Fetch Soup":
            choices = self.dish_staging_frontier
        elif task == "Bring Dish to Serving Station":
            choices = self.delivery_frontier
        else:
            return (my_pos, tuple(my_ori))

        if not choices:
            return (my_pos, tuple(my_ori))

        def sort_key(mo):
            (r, c), _ = mo
            # primary: Manhattan distance
            dist = abs(r - my_pos[0]) + abs(c - my_pos[1])
            # secondary: prefer smaller row, then smaller col
            return (dist, r, c)

        goal_pos, goal_orient = min(choices, key=sort_key)
        return (goal_pos, goal_orient)
    
    def _get_plan_between_goals(self, start_pair, goal_pair):
        """
        Get action plan between two position/orientation pairs.
        Returns the action plan using the motion planner with BFS fallback.
        """
        terrain = self.mdp.terrain_mtx
        
        try:
            # Plan A: orientation‐specific
            action_plan, _, _ = self.planner.get_plan(start_pair, goal_pair)
            print(f"Plan found using get_plan: {action_plan}")
            return action_plan
        except KeyError:
            try:
                # Plan B: orientation‐agnostic  
                goal_pos, goal_ori = goal_pair
                action_plan, _, _ = self.planner.action_plan_from_positions(
                    [goal_pos], start_pair, goal_pair
                )
                print(f"Plan found using action_plan_from_positions: {action_plan}")
                return action_plan
            except Exception:
                # Plan C: guaranteed BFS fallback
                start_pos, _ = start_pair
                goal_pos, _ = goal_pair
                action_plan = _bfs_fallback(start_pos, goal_pos, terrain)
                print(f"Plan found using BFS fallback: {action_plan}")
                return action_plan
            
    def pickup_and_place(self, item, start_pos, start_ori):
        """
        Execute a pickup and place compound action for the given item type.
        Returns combined action plan for fetch + stage operations.
        """
        start_pair = (start_pos, tuple(start_ori))
        
        # Determine fetch task based on item type
        if item == "onion":
            fetch_task = "Fetch Ingredient"
            stage_task = "Stage Ingredient at Stove"
        elif item == "dish":
            fetch_task = "Fetch Dish" 
            stage_task = "Stage Dish at Stove"
        elif item == "soup":
            fetch_task = "Fetch Soup"  # Assuming soup is on a dish at stove
            stage_task = "Bring Soup to Serving Station"
        else:
            # Default fallback
            fetch_task = "Fetch Ingredient"
            stage_task = "Stage Ingredient at Stove"
        
        # Get fetch goal
        fetch_goal_pair = self._task_to_goal(fetch_task, start_pos, start_ori)
        fetch_goal_pos, fetch_goal_ori = fetch_goal_pair
        
        # Get fetch plan
        fetch_plan = self._get_plan_between_goals(start_pair, fetch_goal_pair)
        
        # Get stage goal (starting from fetch goal)
        stage_goal_pair = self._task_to_goal(stage_task, fetch_goal_pos, fetch_goal_ori)
        
        # Get stage plan
        stage_plan = self._get_plan_between_goals(fetch_goal_pair, stage_goal_pair)
        
        # Combine plans
        combined_plan = fetch_plan + stage_plan
        
        print(f"Combined plan for pickup_and_place({item}): {combined_plan}")
        return combined_plan    

    def action(self, state):
        info = getattr(self, "last_info", {})       
        summary = self.summarize_state(state, self.last_info)

        plan_lines = []
        for idx, ev in enumerate(self.plan.events):
            sec = ev["secondary"]
            prim = ev["primary"]
            plan_lines.append(f"{idx+1}) secondary: {sec}, primary: {prim}")
        plan_text = "\n".join(plan_lines)

        prompt = (
            f"TERRAIN:\n{json.dumps(self.cleaned_terrain)}\n\n"
            f"STATE:\n{json.dumps(summary)}\n\n"
            f"Summarize the overcooked state. Go over every detail. Do not mention the orientation of players or explicit coordinates for the players."
            f"Their positions are simply to be referred to relative to landmarks on the terrain.\n\n"
        ) 
        response = query_ollama("mistral", prompt)
        
        prompt = (
        f"STATE SUMMARY:\n{response}\n\n"
        f"PLAN:\n{plan_text}\n\n"
        "Based on the current state and plan, determine what the robot should do. "
        "Respond in this exact format:\n"
        "primary: <description of the primary event from the plan>\n"
        "secondary: pickup_and_place(onion) or pickup_and_place(dish) or pickup_and_place(soup)\n"
        "Choose the appropriate primary event and secondary action based on the current state and plan."
    )

        response = query_ollama("action_predictor", prompt)
        print(f"\nLLM response: {response}")

        # Parse the function call from LLM response
        primary_event, func_name, item = self._parse_function_call(response)
        print(f"Primary event: {primary_event}")
        print(f"Parsed function: {func_name}({item})")

        # Get current position and orientation
        my_pos = state.player_positions[self.agent_index]
        my_ori = state.to_dict()["players"][self.agent_index]["orientation"]
        
        print(f"Agent position: {my_pos}, Human Position: {state.player_positions[1-self.agent_index]}")

        if func_name == "NOOP":
            return Action.STAY, {
                "primary_event": primary_event,
                "function_call": "NOOP",
                "action_plan": [],
                "response": response
            }

        # Execute the compound action
        if func_name == "pickup_and_place":
            action_plan = self.pickup_and_place(item, my_pos, my_ori)
        else:
            # Fallback to simple movement
            action_plan = [Action.STAY]

        # Return first action from the plan
        move = action_plan[0] if action_plan else Action.STAY
        
        return move, {
            "primary_event": primary_event,
            "function_call": f"{func_name}({item})",
            "action_plan": action_plan,
            "response": response
        }

    def actions(self, states, agent_indices):
        return [self.action(s) for s in states]

