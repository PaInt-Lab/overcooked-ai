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
    Guaranteed path from start (r,c) to goal (r,c) on `terrain`,
    returning a list of Direction.* moves.
    """
    # Map delta→Direction
    moves = {
        ( 0,  1): Direction.EAST,
        ( 0, -1): Direction.WEST,
        ( 1,  0): Direction.SOUTH,
        (-1,  0): Direction.NORTH
    }
    H, W = len(terrain), len(terrain[0])
    visited = {start}
    parent = {}
    q = deque([start])

    while q:
        x, y = q.popleft()
        if (x, y) == goal:
            # Reconstruct move‐list
            path = []
            cur = goal
            while cur != start:
                prev = parent[cur]
                dx, dy = cur[0] - prev[0], cur[1] - prev[1]
                path.append(moves[(dx, dy)])
                cur = prev
            return list(reversed(path))

        for (dx, dy), dir_action in moves.items():
            nx, ny = x + dx, y + dy
            if (
                0 <= nx < H and
                0 <= ny < W and
                terrain[nx][ny] != 'X' and
                (nx, ny) not in visited
            ):
                visited.add((nx, ny))
                parent[(nx, ny)] = (x, y)
                q.append((nx, ny))

    return []  # unreachable

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
        self.stove_tiles       = [(i, j)
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

        my_goals = {
            'ingredient': self.ingredient_spawns,
            'pot':        self.stove_tiles,
            'dish':       self.dish_spawns,
            'delivery':   self.delivery_tiles
        }

        self.planner = MotionPlanner(mdp, counter_goals=my_goals) # Eventually, instead of building everytime, can save a pickled version 


    def set_plan(self, session_id: str):
        """Attach the full PlanSession to this agent."""
        self.plan = PLAN_STORE[session_id]

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
        print(f"Goal for robot task '{robot_task}': {goal}")

        orientations = state.to_dict()["players"][self.agent_index]["orientation"]
        start_pair = (my_pos, tuple(orientations))
        goal_pair  = (goal,  tuple(orientations))


        # try:
        #     action_plan, _, _ = self.planner.get_plan(start_pair, goal_pair)
        #     print(f"Action plan for robot task '{robot_task}': {action_plan}")
        # except KeyError:
        #     action_plan = []
        # move = action_plan[0] if action_plan else Action.STAY

        # try:
        #     action_plan, _, _ = self.planner.get_plan(start_pair, goal_pair)
        # except KeyError:
        #     action_plan = self.planner.action_plan_from_positions(start_pair, goal_pair)

        # inside your ActionPredictorAgent.action(...)
        
        # try:
        #     action_plan, _, _ = self.planner.get_plan(start_pair, goal_pair)
        # except KeyError:
        #     # FALLBACK with correct signature:
        #     # 1) list of positions → [goal]
        #     # 2) full start motion‐state
        #     # 3) full goal motion‐state
        #     action_plan = self.planner.action_plan_from_positions(
        #         [goal],        # <-- raw (x,y) position list
        #         start_pair,    # <-- (pos,orient)
        #         goal_pair      # <-- (pos,orient)
        #     )


        # # 3. Call the orientation-agnostic planner only
        # action_plan = self.planner.action_plan_from_positions(
        #     [goal],      # a list of raw (x,y) goal positions
        #     start_pair,  # (pos,orient) start motion‐state
        #     goal_pair    # (pos,orient) goal motion‐state
        # )

        # try:
        #     # Plan A: exact orientation‐specific lookup
        #     action_plan, _, _ = self.planner.get_plan(start_pair, goal_pair)
        #     planner_method = "get_plan"
        # except KeyError:
        #     try:
        #         # Plan B: orientation‐agnostic fallback
        #         action_plan = self.planner.action_plan_from_positions(
        #             [goal],     # list of raw (x,y) goal positions
        #             start_pair, # (pos,orient)
        #             goal_pair   # (pos,orient)
        #         )
        #         planner_method = "action_plan_from_positions"
        #     except Exception as e:
        #         # Anything goes wrong in Plan B → stay put
        #         print("Planner fallback failed:", e)
        #         action_plan = []
        #         planner_method = "failed_fallback"
        try:
            # Plan A: orientation‐specific
            action_plan, _, _ = self.planner.get_plan(start_pair, goal_pair)
            print("Action plan found using get_plan:", action_plan)
        except KeyError:
            try:
                # Plan B: orientation‐agnostic
                action_plan = self.planner.action_plan_from_positions(
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

    def _task_to_goal(self, task: str, my_pos: tuple) -> tuple:
        """
        Given a canonical task, return the nearest tile for that goal.
        """
        if task == "Fetch Ingredient":
            choices = self.ingredient_spawns
        elif task == "Stage Ingredient at Stove":
            choices = self.stove_tiles
        elif task == "Fetch Dish":
            choices = self.dish_spawns
        elif task == "Stage Dish at Stove":
            choices = self.stove_tiles
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


