
import json
from overcooked_ai_py.agents.agent import Agent
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.planning.planners import MotionPlanner, NO_COUNTERS_PARAMS
from overcooked_ai_py.mdp.actions import Action
from llm.ollama.ollama_client import query_ollama

class ActionPredictorAgent(Agent):
    """
    Agent that serializes the game state, calls an Ollama LLM once to predict primary (A1)
    and secondary (A2) tasks, then maps the robot task to a goal coordinate via MotionPlanner,
    returning a low-level Action.
    """
    def __init__(self, model_name: str = "overcooked_action_predictor_model"):
        super().__init__()
        self.model_name = model_name
        self.planner = None

    def set_agent_index(self, agent_index: int):
        super().set_agent_index(agent_index)

    def set_mdp(self, mdp: OvercookedGridworld):
        super().set_mdp(mdp)
        # initialize planner with no-counter parameters
        self.planner = MotionPlanner.from_pickle_or_compute(mdp, counter_goals=NO_COUNTERS_PARAMS)
        self.mdp = mdp

    def action(self, state):
        # 1. Serialize using MDP terrain and full state dict
        state_info = {
            'terrain': self.mdp.terrain_mtx,
            'state': state.to_dict()
        }
        serialized = json.dumps(state_info)
        prompt = f"STATE: {serialized}"

        # 2. One LLM call
        response = query_ollama(self.model_name, prompt)

        # 3. Parse A1 and A2
        human_task, robot_task = self._parse_response(response)

        # 4. Map high-level robot task to goal coordinate
        my_pos = state.player_positions[self.agent_index]
        goal = self._task_to_goal(robot_task, my_pos)

        # 5. Plan path and choose primitive action
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
        # Get layout-specific counter goals
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

        return min(choices, key=lambda p: abs(p[0] - my_pos[0]) + abs(p[1] - my_pos[1]))

