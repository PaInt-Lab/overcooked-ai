# from llm.agents.action_predictor_agent import ActionPredictorAgent
# from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld

# mdp = OvercookedGridworld.from_layout_name("cramped_room")
# agent = ActionPredictorAgent()
# agent.set_mdp(mdp)

# start = ( (1,1), (0,1) )  # arbitrary valid pos/orient
# mapping = {
#     "Fetch Ingredient":     agent.ingredient_spawns,
#     "Stage Ingredient at Stove": agent.stove_tiles,
#     "Fetch Dish":           agent.dish_spawns,
#     "Stage Dish at Stove":  agent.stove_tiles,
#     "Bring Dish to Serving Station": agent.delivery_tiles
# }

# for task, choices in mapping.items():
#     goal = agent._task_to_goal(task, start[0])
#     assert goal in choices, f"{task} → {goal} not in {choices}"
# print("All task→goal mappings valid")

from llm.agents.action_predictor_agent import ActionPredictorAgent
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld

mdp = OvercookedGridworld.from_layout_name("cramped_room")
agent = ActionPredictorAgent()
agent.set_mdp(mdp)

start = ((1, 1), (0, 1))  # arbitrary valid pos/orient

mapping = {
    "Fetch Ingredient":            agent.ingredient_spawns,
    "Stage Ingredient at Stove":   agent.stove_tiles,
    "Fetch Dish":                  agent.dish_spawns,
    "Stage Dish at Stove":         agent.stove_tiles,
    "Bring Dish to Serving Station": agent.delivery_tiles
}

for task, choices in mapping.items():
    goal = agent._task_to_goal(task, start[0])

    # --- debug prints ---
    print(f"\n=== Task: {task} ===")
    print("Start position:", start[0])
    print("Choices:")
    for p in choices:
        d = abs(p[0] - start[0][0]) + abs(p[1] - start[0][1])
        print(f"  {p}  (dist={d})")
    print("Selected goal:", goal)
    # --------------------

    assert goal in choices, f"{task} → {goal} not in {choices}"

print("\nAll task→goal mappings valid")

