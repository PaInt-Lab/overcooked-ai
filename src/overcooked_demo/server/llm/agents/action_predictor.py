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


def _bfs_fallback(start, goal, terrain, goal_orientation=None, start_orientation=None):
    """
    Return a list of (delta_col, delta_row) moves to walk from start to goal on terrain,
    ignoring orientation. Guaranteed to find a path if one exists.
    Always adds goal orientation at the end if provided.
    """
    H, W = len(terrain), len(terrain[0])
    visited = {start}
    parent = {}
    queue = deque([start])

    # Four cardinal directions: (delta_col, delta_row) - right, left, down, up
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    while queue:
        col, row = queue.popleft()
        if (col, row) == goal:
            # Reconstruct path of (dc, dr) tuples
            path = []
            cur = goal
            while cur != start:
                prev = parent[cur]
                dc = cur[0] - prev[0]
                dr = cur[1] - prev[1]
                path.append((dc, dr))
                cur = prev
            path = list(reversed(path))  
            
            # Always add goal orientation if provided (agent can't move through objects anyway)
            if goal_orientation is not None:
                if goal_orientation == [1, 0]:  # Want to face right
                    path.append(Direction.EAST)
                elif goal_orientation == [-1, 0]:  # Want to face left
                    path.append(Direction.WEST)
                elif goal_orientation == [0, 1]:  # Want to face down
                    path.append(Direction.SOUTH)
                elif goal_orientation == [0, -1]:  # Want to face up
                    path.append(Direction.NORTH)
                print(f"BFS: Added goal orientation: {goal_orientation}")
            
            # Add INTERACT at the very end
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
        self.last_summary = None

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
        self.ingredient_spawns = [(j, i)
                            for i, row in enumerate(terrain)
                            for j, c in enumerate(row)
                            if c in ('O', 'T')]
        self.stove_tiles       = [(j, i) # Need to manipulate this for the bot to place dishes next to stove
                            for i, row in enumerate(terrain)
                            for j, c in enumerate(row)
                            if c == 'P']
        self.dish_spawns       = [(j, i)
                            for i, row in enumerate(terrain)
                            for j, c in enumerate(row)
                            if c == 'D']
        self.delivery_tiles    = [(j, i)
                            for i, row in enumerate(terrain)
                            for j, c in enumerate(row)
                            if c == 'S']
        # Assign staging stations per stove with directional logic
        self.onion_staging_tiles = []
        self.dish_staging_tiles = []
        H, W = len(terrain), len(terrain[0])
        
        for (stove_c, stove_r) in self.stove_tiles:
            # Find all adjacent staging positions for this stove
            adjacent_staging = []
            for dc, dr in [(1,0), (-1,0), (0,1), (0,-1)]:
                nc, nr = stove_c + dc, stove_r + dr
                if 0 <= nr < H and 0 <= nc < W and terrain[nr][nc] == 'X':
                    adjacent_staging.append((nc, nr, dc, dr))
            
            # Categorize by direction relative to stove
            left_stations = [(c, r) for c, r, dc, dr in adjacent_staging if dc == -1]  # left of stove
            right_stations = [(c, r) for c, r, dc, dr in adjacent_staging if dc == 1]  # right of stove  
            top_stations = [(c, r) for c, r, dc, dr in adjacent_staging if dr == -1]   # above stove
            bottom_stations = [(c, r) for c, r, dc, dr in adjacent_staging if dr == 1] # below stove
            
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

        for c, r in tiles:
            for dc, dr in [(1,0), (-1,0), (0,1), (0,-1)]:
                nc, nr = c + dc, r + dr
                if (
                    0 <= nr < H and 
                    0 <= nc < W and 
                    terrain[nr][nc] in WALKABLE
                ):
                    orient = (dc, dr)
                    frontier.add(((nc, nr), orient))
        return list(frontier)

    def summarize_state(self, state, info):
        """
        Extract exactly the predicates we need for the LLM based on new MDP:
        - onion_hand: one of "none", "agent", "partner" 
        - onion_staged: is raw onion on any onion‐staging tile?
        - onion_in_pot: is any onion in any stove tile?
        - dish_hand: one of "none", "agent", "partner"
        - dish_staged: is any clean dish on any dish‐staging tile?
        - soup_in_dish: is any soup on any dish‐staging tile?
        - soup_served: did the last transition include a soup_delivery?
        """
        sd = state.to_dict()

        def held_item(player_dict):
            held = player_dict["held_object"]
            if held is None:
                return "none"
            # raw ingredients sometimes come back under "name"
            if held.get("name") in ("onion", "tomato"):
                return held["name"]
            # some objects still use "ingredient"
            if held.get("ingredient") in ("onion", "tomato"):
                return held["ingredient"]
            if held.get("name") in ("dish", "soup"):
                return held["name"]
            return "none"

        # 1) who holds what
        me   = sd["players"][self.agent_index]
        them = sd["players"][1 - (self.agent_index or 0)]
        agent_item   = held_item(me)
        partner_item = held_item(them)

        # Map to new MDP variables
        onion_hand = "none"
        if agent_item == "onion":
            onion_hand = "agent"
        elif partner_item == "onion":
            onion_hand = "partner"

        dish_hand = "none"
        if agent_item == "dish":
            dish_hand = "agent"
        elif partner_item == "dish":
            dish_hand = "partner"

        # 2) collect what's on every tile
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

        # 4) onion_in_pot?
        onion_in_pot = any(
            "onion" in tile_contents.get(pos, [])
            for pos in self.stove_tiles
        )

        # 5) dish_staged?
        dish_staged = any(
            "dish" in tile_contents.get(pos, [])
            for pos in self.dish_staging_tiles
        )

        # 6) soup_in_dish? (renamed from soup_staged)
        soup_in_dish = any(
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
            "onion_hand":       onion_hand,
            "onion_staged":     onion_staged,
            "onion_in_pot":     onion_in_pot,
            "dish_hand":        dish_hand,
            "dish_staged":      dish_staged,
            "soup_in_dish":     soup_in_dish,
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
    
    def _get_action_plan(self, start_pair, goal_pair):
        """
        Get action plan between two position/orientation pairs.
        Returns the action plan using the motion planner with BFS fallback.
        """
        if self.mdp is None:
            return []
        terrain = self.mdp.terrain_mtx
        
        if self.planner is None:
            # Fallback to BFS if no planner available
            start_pos, start_ori = start_pair
            goal_pos, goal_ori = goal_pair
            print(f"BFS fallback (no planner): start={start_pos}, goal={goal_pos}, start_ori={start_ori}, goal_ori={goal_ori}")
            return _bfs_fallback(start_pos, goal_pos, terrain, goal_ori, start_ori)
        
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
                start_pos, start_ori = start_pair
                goal_pos, goal_ori = goal_pair
                print(f"BFS fallback (planner failed): start={start_pos}, goal={goal_pos}, start_ori={start_ori}, goal_ori={goal_ori}")
                action_plan = _bfs_fallback(start_pos, goal_pos, terrain, goal_ori, start_ori)
                print(f"Plan found using BFS fallback: {action_plan}")
                return action_plan
            
    def _get_frontier_for_action(self, action: str, item: str):
        """Get the appropriate frontier based on action and item."""
        if action == "pickup":
            frontier_map = {
                "onion": self.ingredient_frontier,
                "dish": self.dish_frontier,
                "soup": self.dish_staging_frontier,  
            }
        elif action == "place":
            frontier_map = {
                "onion": self.onion_staging_frontier,
                "dish": self.dish_staging_frontier,
                "soup": self.delivery_frontier,
            }
        else:
            return None
        
        return frontier_map.get(item)

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

    def _move_to(self, action: str, item: str, start_pos: tuple, start_ori: tuple):
        """Move to the appropriate location for the given action and item."""
        choices = self._get_frontier_for_action(action, item)
        if choices is None:
            return []
        
        goal = self._find_nearest_goal(choices, start_pos, start_ori)
        start_pair = (start_pos, tuple(start_ori))
        return self._get_action_plan(start_pair, goal)

    def PickUp(self, item, start_pos, start_ori):
        """Returns an action plan to pick up the specified item."""
        return self._move_to("pickup", item, start_pos, start_ori)

    def Place(self, item, start_pos, start_ori):
        """Returns an action plan to place the specified item at the correct location."""
        return self._move_to("place", item, start_pos, start_ori)

    def pickup_and_place(self, item, start_pos, start_ori):
        """Execute a pickup and place compound action for the given item type."""
        # Check if already holding the correct item
        currently_holding = (hasattr(self, "last_summary") and 
                            self.last_summary and 
                            self.last_summary.get("agent_holding"))
        
        if currently_holding == item:
            place_plan = self.Place(item, start_pos, start_ori)
            print(f"Already holding {item}, placing directly: {place_plan}")
            return place_plan
        elif currently_holding and currently_holding != item:
            # Agent is holding wrong item - might need to drop first
            print(f"Warning: Agent holding {currently_holding} but need {item}")
        
        # Full pickup + place sequence
        pickup_plan = self.PickUp(item, start_pos, start_ori)
        
        # Calculate position after pickup
        pickup_choices = self._get_frontier_for_action("pickup", item)
        pickup_goal_pos, pickup_goal_ori = self._find_nearest_goal(pickup_choices, start_pos, start_ori)
        
        # Generate place plan from pickup destination
        place_plan = self.Place(item, pickup_goal_pos, pickup_goal_ori)
        
        return pickup_plan + place_plan

    def action(self, state):

        print(f"onion dispensers: {self.ingredient_spawns} stove tiles: {self.stove_tiles} dish spawns: {self.dish_spawns} delivery tiles: {self.delivery_tiles} onion staging tiles: {self.onion_staging_tiles} dish staging tiles: {self.dish_staging_tiles}")


        info = getattr(self, "last_info", {})       
        self.last_summary = self.summarize_state(state, self.last_info)

        plan_lines = []
        for idx, ev in enumerate(self.plan.events):
            sec = ev["secondary"]
            prim = ev["primary"]
            plan_lines.append(f"{idx+1}) secondary: {sec}, primary: {prim}")
        plan_text = "\n".join(plan_lines)

        prompt = (
            # f"TERRAIN:\n{json.dumps(self.cleaned_terrain)}\n\n"
            f"STATE:\n{json.dumps(self.last_summary)}\n\n"
            f"Summarize the overcooked state. Go over every detail. Do not mention the orientation of players or explicit coordinates for the players."
            f"Their positions are simply to be referred to relative to landmarks on the terrain.\n\n"
        ) 

        print(f"{self.last_summary}")

        response = query_ollama("state_summarizer", prompt)

        prompt = (
        f"STATE SUMMARY:\n{response}\n\n"
        f"PLAN: {plan_text}\n\n"
        "Based on the current state and plan, determine what the robot should do."
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
        
        print(f"Agent position: {my_pos}, orientation: {my_ori}")

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

        print(f"Action plan: {action_plan}")
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

