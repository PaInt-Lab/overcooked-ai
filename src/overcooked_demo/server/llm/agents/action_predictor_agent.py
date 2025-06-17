import json
from overcooked_ai_py.agents.agent import Agent
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.planning.planners import MotionPlanner, NO_COUNTERS_PARAMS
from overcooked_ai_py.mdp.actions import Action
from llm.ollama.ollama_client import query_ollama
from plan_session import PLAN_STORE


def serialize_state(state, mdp) -> str:
    """
    Convert OvercookedState and MDP into a compact JSON string for the LLM prompt.
    Includes layout terrain and full state dict.
    """
    state_info = {
        'terrain': mdp.terrain_mtx,
        'state': state.to_dict()
    }
    return json.dumps(state_info)


class ActionPredictorAgent(Agent):
    """
    Agent that serializes the game state, calls an Ollama LLM once to predict primary (A1)
    and secondary (A2) tasks, then uses MotionPlanner and raw terrain to convert the robot task into
    a low-level Action.
    """
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
        # initialize planner with no-counter parameters
        self.planner = MotionPlanner.from_pickle_or_compute(
            mdp, counter_goals=NO_COUNTERS_PARAMS
        )
        # Build spawn lists from terrain matrix
        terrain = mdp.terrain_mtx
        self.ingredient_spawns = [ 
            (i, j)
            for i, row in enumerate(terrain)
            for j, c in enumerate(row)
            if c in ('O', 'T')  # any ingredient dispenser
        ]
        self.stove_tiles = [
            (i, j)
            for i, row in enumerate(terrain)
            for j, c in enumerate(row)
            if c == 'P'
        ]
        self.dish_spawns = [
            (i, j)
            for i, row in enumerate(terrain)
            for j, c in enumerate(row)
            if c == 'D'
        ]
        self.delivery_tiles = [
            (i, j)
            for i, row in enumerate(terrain)
            for j, c in enumerate(row)
            if c == 'S'
        ]

    def action(self, state):
        # 0. Grab the current high-level event
        sec_list, prim_list = self.plan.current().values()

        # 1. Serialize state
        serialized = serialize_state(state, self.mdp)

        # 2a. Build prompt with just the current event
        prompt = (
            f"EVENT Action's:\n"
            f"  Secondary: {sec_list}\n"
            f"  Primary:   {prim_list}\n\n"
            f"STATE: {serialized}"
        )

        # 2. Single LLM call
        response = query_ollama("overcooked_action_predictor_model", prompt)

        # 3. Parse JSON output
        pred = json.loads(response)
        complete      = pred.get("complete", False)
        human_task    = pred.get("primary",   "–")
        robot_task    = pred.get("secondary", "–")

        # 3b. Advance the plan if the model says current event is done
        if complete:
            self.plan.advance()

        # 4. Map high-level robot task to goal coordinate
        my_pos = state.player_positions[self.agent_index]
        goal = self._task_to_goal(robot_task, my_pos)

        # pull orientation from state.to_dict()
        orientations = state.to_dict()["players"][self.agent_index]["orientation"]
        start = (my_pos, tuple(orientations))
        # for goal we usually don’t care about final orientation, so repeat it
        goal_pair = (goal, tuple(orientations))

        # 5. Use planner.get_plan or action_plan_from_positions
        if hasattr(self.planner, "get_plan"):
            try:
                action_plan, _, _ = self.planner.get_plan(start, goal_pair)
            except KeyError:
                action_plan = []
            move = action_plan[0] if action_plan else Action.STAY
            planner_method = "get_plan"
        elif hasattr(self.planner, 'action_plan_from_positions'):
            plan = self.planner.action_plan_from_positions(my_pos, goal)
            move = plan[0] if plan else Action.STAY
            planner_method = 'action_plan_from_positions'
        else:
            move = Action.STAY
            planner_method = 'fallback_STAY'

        return move, {
            "high_level": robot_task,
            "human_task": human_task,
            "planner_method": planner_method,
            "prompt": prompt,
            "response": response
        }

    def actions(self, states, agent_indices):
        results = []
        for state, idx in zip(states, agent_indices):
            self.set_agent_index(idx)
            results.append(self.action(state))
        return results

    @staticmethod
    def _parse_response(response: str): 
        a1, a2 = '–', '–'
        for line in response.splitlines():
            if line.startswith('A1:'):
                a1 = line.split('A1:', 1)[1].strip()
            elif line.startswith('A2:'):
                a2 = line.split('A2:', 1)[1].strip()
        return a1, a2

    def _task_to_goal(self, task: str, my_pos: tuple) -> tuple:
        # Map high-level tasks to spawn lists built from raw terrain
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

        # Choose the closest tile by Manhattan distance
        return min(
            choices,
            key=lambda p: abs(p[0] - my_pos[0]) + abs(p[1] - my_pos[1])
        )
    
    def set_plan(self, session_id):
        self.plan = PLAN_STORE[session_id]
