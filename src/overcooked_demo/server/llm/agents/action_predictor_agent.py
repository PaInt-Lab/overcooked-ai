
import json
from overcooked_ai_py.agents.agent import Agent
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.planning.planners import MotionPlanner, NO_COUNTERS_PARAMS
from overcooked_ai_py.mdp.actions import Action
from llm.ollama.ollama_client import query_ollama


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
    and secondary (A2) tasks, then uses MotionPlanner to convert the robot task into
    a low-level Action in a layout-agnostic way.
    """
    def __init__(self, model_name: str = "overcooked_action_predictor_model"):
        super().__init__()
        self.model_name = model_name
        self.mdp = None
        self.planner = None

    def set_agent_index(self, agent_index: int):
        super().set_agent_index(agent_index)

    def set_mdp(self, mdp: OvercookedGridworld):
        super().set_mdp(mdp)
        self.mdp = mdp
        # Initialize the motion planner with no-counter parameters
        self.planner = MotionPlanner.from_pickle_or_compute(mdp, counter_goals=NO_COUNTERS_PARAMS)

    def action(self, state):
        # 1. Serialize state + layout
        serialized = serialize_state(state, self.mdp)
        prompt = f"STATE: {serialized}"

        # 2. Single LLM call
        response = query_ollama(self.model_name, prompt)

        # 3. Parse A1 and A2
        human_task, robot_task = self._parse_response(response)

        # 4. Convert high-level robot task to a goal coordinate
        my_pos = state.player_positions[self.agent_index]
        goal = self._task_to_goal(robot_task, my_pos)

        # 5. Use planner.get_plan to map goal -> primitive actions
        if hasattr(self.planner, 'get_plan'):
            try:
                plan = self.planner.get_plan(my_pos, goal)
            except KeyError:
                plan = []
            if plan:
                move = plan[0]
            else:
                move = Action.STAY
            planner_method = 'get_plan'
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
        # Look up layout-specific counter goals
        counter_goals = getattr(self.planner, 'counter_goals', {})
        if task == "Fetch Ingredient":
            choices = counter_goals.get('ingredient', [])
        elif task == "Stage Ingredient at Stove":
            choices = counter_goals.get('pot', [])
        elif task == "Fetch Dish":
            choices = counter_goals.get('dish', [])
        elif task == "Stage Dish at Stove":
            choices = counter_goals.get('pot', [])
        elif task == "Bring Dish to Serving Station":
            choices = counter_goals.get('delivery', [])
        else:
            return my_pos

        if not choices:
            return my_pos

        # Choose the closest tile by Manhattan distance
        return min(choices, key=lambda p: abs(p[0] - my_pos[0]) + abs(p[1] - my_pos[1]))
