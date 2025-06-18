# test_motion_planner.py
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, Action
from llm.agents.action_predictor_agent import ActionPredictorAgent
from overcooked_ai_py.planning.planners import NO_COUNTERS_PARAMS, MotionPlanner

mdp = OvercookedGridworld.from_layout_name("cramped_room")

terrain = mdp.terrain_mtx
ingredient_spawns = [(i, j)
                     for i, row in enumerate(terrain)
                     for j, c in enumerate(row)
                     if c in ('O', 'T')]
stove_tiles       = [(i, j)
                     for i, row in enumerate(terrain)
                     for j, c in enumerate(row)
                     if c == 'P']
dish_spawns       = [(i, j)
                     for i, row in enumerate(terrain)
                     for j, c in enumerate(row)
                     if c == 'D']
delivery_tiles    = [(i, j)
                     for i, row in enumerate(terrain)
                     for j, c in enumerate(row)
                     if c == 'S']

my_goals = {
    'ingredient': ingredient_spawns,
    'pot':        stove_tiles,
    'dish':       dish_spawns,
    'delivery':   delivery_tiles
}

planner = MotionPlanner(mdp, counter_goals=my_goals) # Eventually, instead of building everytime, can save a pickled version 

# print("All computed keys:")
# for k in planner.all_plans.keys():
#     print(k)


start_pos = (1,2)
orient   = (0,1)
start    = (start_pos, orient)


valid_goals = [
    goal_and_or
    for (start_and_or, goal_and_or) in planner.all_plans.keys()
    if start_and_or == start
]
goal_pos, goal_orient = valid_goals[0]
goal = (goal_pos, goal_orient)

print(f"\n=== MOTION PLANNER DEBUG ===")
print(f"Start cell: {start_pos}, orientation: {orient}")
print(f"Goal cell:  {goal_pos}, orientation: {goal_orient}\n")

plan, _, _ = planner.get_plan(start, goal)
assert plan, "Planner returned empty plan!"
print("Plan to fetch ingredient:", plan , "\n")

# Simulate following the plan
state = mdp.get_standard_start_state()
print("Initial full state:")
print("Player positions:", state.player_positions)

step = 0
for a in plan:
    step += 1
    joint_action = [a, Action.STAY]
    state, reward = mdp.get_state_transition(state, joint_action)
    pos = state.player_positions[0]
    print(f" Step {step}: Action = {a}")
    print(f"   New pos = {pos}, reward = {reward}")

print("Final position:", pos, "Expected:", goal_pos)