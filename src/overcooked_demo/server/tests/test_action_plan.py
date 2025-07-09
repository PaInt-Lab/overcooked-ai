from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.actions import Action


def apply_action_plan(action_plan, layout="cramped_room", max_steps=None):
    """
    Apply a custom action plan for the first agent, assuming the second agent stays idle.
    Prints state transitions so you can inspect positions, orientations, and rewards.

    Change the `action_plan` list in `test_custom_action_plan` to try different sequences.
    """
    mdp = OvercookedGridworld.from_layout_name(layout)
    state = mdp.get_standard_start_state()

    print(f"Starting state: pos={state.player_positions[0]}, ori={state.to_dict()['players'][0]['orientation']}")

    #gonna move the human around a bit, gonna start him off at (2, 2) so he is out of the way
    joint_action = [Action.STAY, (-1, 0)]      
    state, reward = mdp.get_state_transition(state, joint_action)
    joint_action = [Action.STAY, (0, 1)]      
    state, reward = mdp.get_state_transition(state, joint_action)
    
    for step, action in enumerate(action_plan):
        if max_steps and step >= max_steps:
            break
        joint_action = [action, Action.STAY]
        state, reward = mdp.get_state_transition(state, joint_action)
        pos = state.player_positions[0]
        ori = state.to_dict()["players"][0]["orientation"]
        # print(f"Step {step+1}: Action={action}, New pos={pos}, ori={ori}, \n\nstate={state}, \n\nreward={reward}")
        print(f"Step {step+1}: Action={action}, New pos={pos}, ori={ori}, state={state}\n")
    return state

# Plan Fetch: [(0, -1), (-1, 0), 'interact']
# Plan Stage: [(0, -1), 'interact']

def test_custom_action_plan():
    # === Define your custom action plan here ===
    action_plan = [
        (0, -1),   
        (-1, 0),      
        Action.INTERACT, # Grabbing Onion
        (0, 1),
        (-1, 0),
        Action.INTERACT,  # Placing Onion on Stove
        # Action.INTERACT,  # Turning stove on does not work, for some reason it just picks up the onion
        (0, -1),
        (0, 1),
        Action.INTERACT,  # Grab Dish
    ]
    # ============================================

    final_state = apply_action_plan(action_plan)
    # Example assertion (uncomment and adapt as needed):
    # assert final_state.player_positions[0] == (0, 1)


if __name__ == "__main__":
    # Manual run
    test_custom_action_plan()
