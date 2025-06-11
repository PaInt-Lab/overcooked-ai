import json
from overcooked_ai_py.agents.agent import Agent
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.planning.planners import MotionPlanner
from overcooked_ai_py.mdp.actions import Action
from llm.ollama.ollama_client import query_ollama


def serialize_state(state) -> str:
    """
    Convert OvercookedState into a compact JSON string for the LLM prompt.
    Include only the necessary elements such as terrain, object positions, and player locations.
    """
    terrain = state.terrain_mtx.tolist()
    objects = state.objects_dict() if hasattr(state, 'objects_dict') else {}
    players = state.player_positions
    state_info = {
        'terrain': terrain,
        'objects': objects,
        'player_positions': players
    }
    return json.dumps(state_info)


class ActionPredictorAgent(Agent):
    """
    Agent that serializes the game state, calls an Ollama LLM once to predict both
    primary (human) and secondary (robot) tasks, maps the robot task to a goal
    position based on the current layout, and returns a valid low-level Action.
    """
    def __init__(self, model_name: str = "action_predictor_model"):
        super().__init__()
        self.model_name = model_name
        self.mdp = None
        self.planner = None
        # placeholders for dynamic spawn points
        self.ingredient_spawns = []
        self.stove_tiles = []
        self.dish_spawns = []
        self.delivery_tiles = []

    def set_agent_index(self, agent_index: int):
        super().set_agent_index(agent_index)

    def set_mdp(self, mdp: OvercookedGridworld):
        super().set_mdp(mdp)
        self.mdp = mdp
        # initialize the motion planner for this layout
        self.planner = MotionPlanner.from_pickle_or_compute(mdp)
        # cache spawn point lists for layout-agnostic mapping
        self.ingredient_spawns = mdp.ingredient_locations
        self.stove_tiles       = mdp.stove_locations
        self.dish_spawns       = mdp.dish_locations
        self.delivery_tiles    = mdp.delivery_locations

    def action(self, state):
        # Serialize and query LLM
        serialized = serialize_state(state)
        prompt = f"STATE: {serialized}"
        response = query_ollama(self.model_name, prompt)
        human_task, robot_task = self._parse_response(response)

        # Map high-level robot task to a goal coordinate
        my_pos = state.player_positions[self.agent_index]
        goal = self._task_to_goal(robot_task, my_pos)

        # Plan a path and choose next primitive action
        path = self.planner.get_shortest_path(my_pos, goal)
        action = path[0] if path else Action.STAY

        return action, {"high_level": robot_task, "prompt": prompt, "response": response}

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
        # Choose the nearest relevant tile for the given high-level task
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
        # Return the closest tile by Manhattan distance
        return min(choices, key=lambda p: abs(p[0] - my_pos[0]) + abs(p[1] - my_pos[1]))

