import json
from overcooked_ai_py.agents.agent import Agent
from llm.ollama.ollama_client import query_ollama


def serialize_state(state) -> str:
    """
    Convert OvercookedState into a compact JSON string for the LLM prompt.
    Include only the necessary elements such as terrain matrix and object positions.
    """
    terrain = state.terrain_mtx.tolist()
    objects = state.objects_dict() if hasattr(state, 'objects_dict') else {}
    state_info = {
        'terrain': terrain,
        'objects': objects,
    }
    return json.dumps(state_info)


class ActionPredictorAgent(Agent):
    """
    Agent that serializes the game state, calls an Ollama LLM once to predict both
    primary (human) and secondary (robot) tasks, and returns both predictions.
    """
    def __init__(self, model_name: str = "action_predictor_model"):
        super().__init__()
        self.model_name = model_name

    def set_agent_index(self, agent_index: int):
        # Ensure the agent_index is stored for this instance
        super().set_agent_index(agent_index)

    def set_mdp(self, mdp):
        # Store the MDP if future navigation or context is needed
        super().set_mdp(mdp)
        self.mdp = mdp

    def action(self, state):
        # 1. Serialize the Overcooked state
        serialized = serialize_state(state)
        prompt = f"STATE: {serialized}"

        # 2. One LLM call
        response = query_ollama(self.model_name, prompt)

        # 3. Parse A1 and A2
        human_task, robot_task = self._parse_response(response)

        # 4. Return both tasks for inspection
        return (human_task, robot_task), {"prompt": prompt, "response": response}

    def actions(self, states, agent_indices):
        # Batch version: predict both tasks for each state
        results = []
        for state, idx in zip(states, agent_indices):
            self.set_agent_index(idx)
            results.append(self.action(state))
        return results

    @staticmethod
    def _parse_response(response: str):
        # Expect lines:
        # A1: <Primary Task>
        # A2: <Secondary Task>
        a1, a2 = '–', '–'
        for line in response.splitlines():
            if line.startswith('A1:'):
                a1 = line.split('A1:', 1)[1].strip()
            elif line.startswith('A2:'):
                a2 = line.split('A2:', 1)[1].strip()
        return a1, a2