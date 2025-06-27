from collections import deque
import json
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
        self.staging_tiles = []
        H, W = len(terrain), len(terrain[0])
        for (r, c) in self.stove_tiles:
            for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                nr, nc = r + dr, c + dc
                # ensure we’re in bounds and it’s a counter‐wall ('X')
                if 0 <= nr < H and 0 <= nc < W and terrain[nr][nc] == 'X':
                    self.staging_tiles.append((nr, nc))

        self.ingredient_frontier = self._compute_frontier(self.ingredient_spawns, terrain)
        self.stove_frontier      = self._compute_frontier(self.stove_tiles,       terrain) 
        self.dish_frontier       = self._compute_frontier(self.dish_spawns,       terrain)
        self.delivery_frontier   = self._compute_frontier(self.delivery_tiles,    terrain)
        self.staging_frontier    = self._compute_frontier(self.staging_tiles,          terrain)
        
        my_goals = {
            'ingredient': self.ingredient_spawns,
            'pot':        self.stove_tiles,
            'dish':       self.dish_spawns,
            'delivery':   self.delivery_tiles
        } # for the next step of testing we will test if we can add 'staging': self.staging_tiles as a goal so that when it hits the staging spot at the end of the plan it interacts
        
        self.cleaned_terrain = [
            [ self.TERRAIN_MAPPING.get(cell, "Unknown") for cell in row ]
            for row in terrain
        ]

        self.planner = MotionPlanner(mdp, counter_goals=my_goals) # Eventually, instead of building everytime, can save a pickled version 

    def set_plan(self, session_id: str):
        """Attach the full PlanSession to this agent."""
        self.plan = PLAN_STORE[session_id]

    def summarize_state(self, state, info):
        """
        Return a compact summary of the current OvercookedState + last-transition info,
        with JSON-safe (string) keys for all tile maps.
        """
        sd = state.to_dict()

        # 1) Players
        me   = sd["players"][self.agent_index]
        them = sd["players"][1 - self.agent_index]
        summary = {
            "me": {
                "pos":    me["position"],
                "orient": me["orientation"],
                "hold":   me["held_object"]
            },
            "partner": {
                "pos":    them["position"],
                "orient": them["orientation"],
                "hold":   them["held_object"]
            }
        }

        # 2) Contents on every tile
        tile_contents = {}
        for obj in sd["objects"]:
            p    = tuple(obj["position"])
            name = obj.get("ingredient") or obj.get("name")
            tile_contents.setdefault(p, []).append(name)

        # helper to turn (r,c) -> "r,c"
        def key_str(pos):
            return f"{pos[0]},{pos[1]}"

        # 3) Counters & stations (stringify the keys)
        summary["staging_station"] = {
            key_str(pos): tile_contents.get(pos, [])
            for pos in self.staging_tiles
        }
        summary["pots"] = {
            key_str(pos): tile_contents.get(pos, [])
            for pos in self.stove_tiles
        }

        # 4) Recent event flags for onions & dishes (from last get_state_transition)
        ei = info.get("event_infos", {}) if info else {}
        summary["recent"] = {
            "onion_pickup":        ei.get("onion_pickup", [False, False]),
            "useful_onion_pickup": ei.get("useful_onion_pickup", [False, False]),
            "onion_drop":          ei.get("onion_drop", [False, False]),
            "potting_onion":       ei.get("potting_onion", [False, False]),
            "dish_pickup":         ei.get("dish_pickup", [False, False]),
            "useful_dish_pickup":  ei.get("useful_dish_pickup", [False, False]),
            "dish_drop":           ei.get("dish_drop", [False, False]),
            "soup_delivery":       ei.get("soup_delivery", [False, False]),
        }

        return summary



    def _task_to_goal(self, task: str, my_pos: tuple, my_ori: tuple) -> tuple:
        """
        Given a canonical task, return the nearest tile for that goal.
        """
        if task == "Fetch Ingredient":
            choices = self.ingredient_frontier
        elif task == "Stage Ingredient at Stove": 
            choices = self.staging_frontier
        elif task == "Fetch Dish":
            choices = self.dish_frontier
        elif task == "Stage Dish at Stove":  
            choices = self.staging_frontier
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
        # return ((1, 1), (-1, 0))  # For testing, return a fixed goal position and orientation
    
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

    def action(self, state):
        info = getattr(self, "last_info", {})       
        summary = self.summarize_state(state, self.last_info)
        terrain = self.mdp.terrain_mtx

        plan_lines = []
        for idx, ev in enumerate(self.plan.events):
            sec = ev["secondary"]
            prim = ev["primary"]
            plan_lines.append(f"{idx+1}) secondary: {sec}, primary: {prim}")
        plan_text = "\n".join(plan_lines)

        prompt = (
            f"TERRAIN:\n{json.dumps(self.cleaned_terrain)}\n\n"
            f"STATE SUMMARY:\n{json.dumps(summary)}\n\n"
            f"PLAN:\n{plan_text}\n\n"
        ) 
        response = query_ollama("action_predictor", prompt)

        print("\nPrompt sent to LLM:\n\n", prompt)
        print("\nLLM response:", response)

        try:
            pred = json.loads(response)
            event_idx = pred.get("event_idx", 1) - 1  # 0-based
        except (json.JSONDecodeError, TypeError):
            event_idx = 0
            pred = {}

        # Task validation and fallback    
        event_idx = max(0, min(event_idx, len(self.plan.events)-1))
        human_task = pred.get("primary",   "–")
        robot_task = pred.get("secondary", "–")
        # print(f"Event {event_idx+1} → Human Task: {human_task}, Robot Task: {robot_task}")

        # Position and orientation handling
        my_pos = state.player_positions[self.agent_index]
        print("Agent position:", my_pos, "Human Position:", state.player_positions[1-self.agent_index])
        my_ori = state.to_dict()["players"][self.agent_index]["orientation"]
        start_pair = (my_pos, tuple(my_ori))
        goal_pair = self._task_to_goal(robot_task, my_pos, my_ori)
        goal_pos, goal_ori = goal_pair
        print(f"Start pair: {start_pair}, Goal pair: {goal_pair}")

        try:
            # Plan A: orientation‐specific
            action_plan, _, _ = self.planner.get_plan(start_pair, goal_pair)
            print("Action plan found using get_plan:", action_plan)
        except KeyError:
            try:
                # Plan B: orientation‐agnostic
                action_plan, _, _ = self.planner.action_plan_from_positions(
                    [goal_pos], start_pair, goal_pair
                )
                print("Action plan found using action_plan_from_positions:", action_plan)
            except Exception:
                # Plan C: guaranteed BFS fallback
                action_plan = _bfs_fallback(my_pos, goal_pos, terrain) # Might need to add orientation at end
                print("Action plan found using BFS fallback:", action_plan)
       
        # finally pick the first step or stay
        move = action_plan[0] if action_plan else Action.STAY

        return move, {
            "event":    event_idx+1,  # back to 1-based for clarity
            # "human_task":   human_task,
            "robot_task":   robot_task,
            # "prompt":       prompt,
            "response":     response
        }

    def actions(self, states, agent_indices):
        return [self.action(s) for s in states]

