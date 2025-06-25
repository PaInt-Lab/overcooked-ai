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
            return list(reversed(path))

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

    def set_agent_index(self, agent_index: int):
        super().set_agent_index(agent_index)

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
        staging_tiles = []
        H, W = len(terrain), len(terrain[0])
        for (r, c) in self.stove_tiles:
            for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                nr, nc = r + dr, c + dc
                # ensure we’re in bounds and it’s a counter‐wall ('X')
                if 0 <= nr < H and 0 <= nc < W and terrain[nr][nc] == 'X':
                    staging_tiles.append((nr, nc))

        self.ingredient_frontier = self._compute_frontier(self.ingredient_spawns, terrain)
        self.stove_frontier      = self._compute_frontier(self.stove_tiles,       terrain) 
        self.dish_frontier       = self._compute_frontier(self.dish_spawns,       terrain)
        self.delivery_frontier   = self._compute_frontier(self.delivery_tiles,    terrain)
        self.staging_frontier    = self._compute_frontier(staging_tiles,          terrain)
        
        my_goals = {
            'ingredient': self.ingredient_spawns,
            'pot':        self.stove_tiles,
            'dish':       self.dish_spawns,
            'delivery':   self.delivery_tiles
        } # for the next step of testing we will test if we can add 'staging': self.staging_tiles as a goal so that when it hits the staging spot at the end of the plan it interacts
        
        self.planner = MotionPlanner(mdp, counter_goals=my_goals) # Eventually, instead of building everytime, can save a pickled version 

    def set_plan(self, session_id: str):
        """Attach the full PlanSession to this agent."""
        self.plan = PLAN_STORE[session_id]

    def _task_to_goal(self, task: str, my_pos: tuple) -> tuple:
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
            choices = self.delivery_tiles
        else:
            return my_pos

        if not choices:
            return my_pos

        return min(
            choices,
            key=lambda p: abs(p[0]-my_pos[0]) + abs(p[1]-my_pos[1])
        )
    
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
                    frontier.add((nr, nc))
        return list(frontier)

    def action(self, state):
        serialized = serialize_state(state, self.mdp)
        terrain = self.mdp.terrain_mtx

        plan_lines = []
        for idx, ev in enumerate(self.plan.events):
            sec = ev["secondary"]
            prim = ev["primary"]
            plan_lines.append(f"{idx+1}) secondary: {sec}, primary: {prim}")
        plan_text = "\n".join(plan_lines)

        prompt = (
            f"\n\nTERRAIN:\n{terrain}\n\n"
            f"STATE: {serialized}"
            f"PLAN:\n{plan_text}\n\n"
        ) 
        response = query_ollama("action_predictor", prompt)

        try:
            pred = json.loads(response)
            event_idx = pred.get("event_idx", 1) - 1  # 0-based
        except (json.JSONDecodeError, TypeError):
            event_idx = 0
            pred = {}

        event_idx = max(0, min(event_idx, len(self.plan.events)-1))
        human_task = pred.get("primary",   "–")
        robot_task = pred.get("secondary", "–")
        print(f"Event {event_idx+1} → Human Task: {human_task}, Robot Task: {robot_task}")

        my_pos = state.player_positions[self.agent_index]
        goal = self._task_to_goal(robot_task, my_pos)

        orientations = state.to_dict()["players"][self.agent_index]["orientation"]
        start_pair = (my_pos, tuple(orientations))
        goal_pair  = (goal,  tuple(orientations))
        print(f"Start pair: {start_pair}, Goal pair: {goal_pair}")

        try:
            # Plan A: orientation‐specific
            action_plan, _, _ = self.planner.get_plan(start_pair, goal_pair)
            print("Action plan found using get_plan:", action_plan)
        except KeyError:
            try:
                # Plan B: orientation‐agnostic
                action_plan, _, _ = self.planner.action_plan_from_positions(
                    [goal], start_pair, goal_pair
                )
                print("Action plan found using action_plan_from_positions:", action_plan)
            except Exception:
                # Plan C: guaranteed BFS fallback
                action_plan = _bfs_fallback(my_pos, goal, terrain)
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

