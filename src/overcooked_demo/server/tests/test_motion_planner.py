# test_motion_planner.py
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
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

print("All computed keys:")
for k in planner.all_plans.keys():
    print(k)


start_pos = (1,2)
orient   = (0,1)
start    = (start_pos, orient)

# pick a fetch‐ingredient goal
goal_pos = ingredient_spawns[0] 
goal     = (goal_pos, orient)

print(f"\n=== MOTION PLANNER DEBUG ===")
print(f"Start cell: {start_pos}, orientation: {orient}")
print(f"Goal cell:  {goal_pos}, orientation: {orient}\n")

plan, _, _ = planner.get_plan(start, goal)
assert plan, "Planner returned empty plan!"
print("Plan to fetch ingredient:", plan , "\n")

# Simulate following the plan
state = mdp.get_standard_start_state()
print("Initial full state:")
print("  Player positions:", state.player_positions)
print("  Pots:", [pot.ingredients for pot in state.pot_states], "\n")

step = 0
for a in plan:
    step += 1
    # unpack reward and done so we can print them
    state, reward, done = mdp.get_state_transition(state, [a])
    pos = state.player_positions[0]
    print(f" Step {step}: Action = {a}")
    print(f"   New pos = {pos}, reward = {reward}, done = {done}")

print("\nFinal position:", pos, "  Expected goal:", goal_pos)
print("Test complete.")