"""
Secondary Action Selection Framework

This module provides algorithmic secondary action selection based on predicted
primary actions. It maps primary actions to relevant secondary actions and 
selects the best one based on game state.
"""

from typing import Dict, Optional, List
from enum import Enum


class PrimaryAction(Enum):
    """Primary actions from actual plans and complete state graph"""
    # Washing actions (from complete state graph)
    WASH_ONION = "Wash Onion"
    WASH_TOMATO = "Wash Tomato"
    
    # Processing actions (from complete state graph)
    CHOP_ONION = "Chop Onion"
    CHOP_TOMATO = "Chop Tomato"
    
    # Seasoning actions (from complete state graph)
    SALT_ONION = "Salt Onion"
    SALT_TOMATO = "Salt Tomato"
    PEPPER_ONION = "Pepper Onion"
    PEPPER_TOMATO = "Pepper Tomato"
    
    # Plan-style primary actions (standardized case)
    STAGE_ONION = "Stage Onion"
    STAGE_TOMATO = "Stage Tomato"
    HUMAN_GRAB_ONION = "Human Grab Onion"
    HUMAN_GRAB_TOMATO = "Human Grab Tomato"
    HUMAN_GRAB_DISH = "Human Grab Dish"
    
    # Cooking actions (standardized case)
    PLACE_ONION_IN_POT = "Place Onion in Pot"
    PLACE_TOMATO_IN_POT = "Place Tomato in Pot"
    TURN_STOVE_ON = "Turn Stove On"
    WAIT_TILL_INGREDIENTS_COOKED = "Wait For Ingredients to Cook"
    POUR_SOUP = "Pour Soup"
    HUMAN_STAGE_SOUP = "Human Stage Soup"
    WAIT_FOR_ROBOT_TO_SERVE_SOUP = "Wait For Robot To Serve Soup"
    
    # State graph style (fallback)
    HUMAN_GRAB_ONION_STATE = "human_grab_onion"
    HUMAN_PLACE_ONION_IN_POT_STATE = "human_place_onion_in_pot"
    HUMAN_GRAB_TOMATO_STATE = "human_grab_tomato"
    HUMAN_PLACE_TOMATO_IN_POT_STATE = "human_place_tomato_in_pot"
    HUMAN_TURN_STOVE_ON_STATE = "human_turn_stove_on"
    HUMAN_POUR_SOUP_STATE = "human_pour_soup"
    HUMAN_GRAB_DISH_STATE = "human_grab_dish"
    
    # Special
    NOOP = "NOOP"
    HUMAN_NOOP = "Human NOOP"


# Primary Action to Secondary Action Sequences (Based on Real Plans)
PRIMARY_TO_SECONDARY_SEQUENCES = {
    # Washing actions - Robot handles the washing workflow
    PrimaryAction.WASH_ONION: [
        "pickup(onion)",                 # Step 1: Fetch raw onion
        "place(onion, sink)"             # Step 2: Place at sink for washing
    ],
    
    PrimaryAction.WASH_TOMATO: [
        "pickup(tomato)",                # Step 1: Fetch raw tomato
        "place(tomato, sink)"            # Step 2: Place at sink for washing
    ],
    
    # Processing actions - Robot handles the chopping workflow
    PrimaryAction.CHOP_ONION: [
        "pickup(onion)",                 # Step 1: Fetch onion (raw or washed)
        "place(onion, chopping_station)" # Step 2: Place at chopping station
    ],
    
    PrimaryAction.CHOP_TOMATO: [
        "pickup(tomato)",                # Step 1: Fetch tomato (raw or washed)
        "place(tomato, chopping_station)" # Step 2: Place at chopping station
    ],
    
    # Seasoning actions - Robot handles the seasoning workflow
    PrimaryAction.SALT_ONION: [
        "pickup(onion)",                 # Step 1: Fetch onion (raw, washed, or chopped)
        "place(onion, salt_station)"     # Step 2: Place at salt station
    ],
    
    PrimaryAction.SALT_TOMATO: [
        "pickup(tomato)",                # Step 1: Fetch tomato (raw, washed, or chopped)
        "place(tomato, salt_station)"    # Step 2: Place at salt station
    ],
    
    PrimaryAction.PEPPER_ONION: [
        "pickup(onion)",                 # Step 1: Fetch onion (raw, washed, or chopped)
        "place(onion, pepper_station)"   # Step 2: Place at pepper station
    ],
    
    PrimaryAction.PEPPER_TOMATO: [
        "pickup(tomato)",                # Step 1: Fetch tomato (raw, washed, or chopped)
        "place(tomato, pepper_station)"  # Step 2: Place at pepper station
    ],
    
    # Plan-style primary actions with sequential workflows
    
    # "Stage Onion" -> Robot: stage onion for human (robot already has onion)
    PrimaryAction.STAGE_ONION: [
        "pickup(onion)",
        "place(onion, staging_station)"  # Step 1: Stage onion for human
    ],
    
    # "Stage Tomato" -> Robot: stage tomato for human (robot already has tomato)
    PrimaryAction.STAGE_TOMATO: [
        "pickup(tomato)",
        "place(tomato, staging_station)" # Step 1: Stage tomato for human
    ],
    
    # "Human Grab Onion" -> Robot: NOOP (human task - grabs staged onion)
    PrimaryAction.HUMAN_GRAB_ONION: [
        "NOOP"                          # Human grabs staged onion
    ],
    
    # "Human Grab Tomato" -> Robot: NOOP (human task - grabs staged tomato)
    PrimaryAction.HUMAN_GRAB_TOMATO: [
        "NOOP"                          # Human grabs staged tomato
    ],
    
    # "Human Grab Dish" -> Robot: fetch dish, then stage it
    PrimaryAction.HUMAN_GRAB_DISH: [
        "pickup(dish)",   # Step 1: Fetch dish for human
        "place(dish)"     # Step 2: Stage dish
    ],
    
    # "Place X in Pot" -> Robot does NOOP (human task)
    PrimaryAction.PLACE_ONION_IN_POT: ["NOOP"],
    PrimaryAction.PLACE_TOMATO_IN_POT: ["NOOP"],
    
    # "Turn stove on" -> Robot does NOOP (human task)
    PrimaryAction.TURN_STOVE_ON: ["NOOP"],
    
    # "Wait Till Ingredients Cooked" -> Robot prepares dish for soup serving
    PrimaryAction.WAIT_TILL_INGREDIENTS_COOKED: [
        "pickup(dish)",                 # Step 1: Fetch dish for soup serving
        "place(dish, staging_station)"  # Step 2: Stage dish for when soup is ready
    ],
    
    # "Pour Soup" -> Robot does NOOP (human task)
    PrimaryAction.POUR_SOUP: ["NOOP"],
    
    # "Serve Soup" -> Robot does NOOP (human task)  
    PrimaryAction.WAIT_FOR_ROBOT_TO_SERVE_SOUP: ["NOOP"],
    
    # "Human Stage Soup" -> Robot does NOOP (human task)
    PrimaryAction.HUMAN_STAGE_SOUP: ["NOOP"],
    
    # Special cases
    PrimaryAction.NOOP: ["pickup(soup)", "place(soup)"],
    
    # When human does nothing, robot follows general priority
    PrimaryAction.HUMAN_NOOP: [
        "pickup(onion)", "pickup(tomato)", "pickup(dish)", "pickup(soup)",
        "place(onion)", "place(tomato)", "place(dish)", "place(soup)",
        "place(onion, chopping_station)", "place(tomato, chopping_station)",
        "place(onion, salt_station)", "place(tomato, salt_station)",
        "place(onion, pepper_station)", "place(tomato, pepper_station)",
        "place(chopped_onion)", "place(chopped_tomato)"
    ]
}


def select_secondary_action(game_state: Dict, task_title: str, 
                          predicted_primary_action: Optional[str] = None) -> str:
    """
    Convenience function to select secondary action for a given task.
    
    Args:
        game_state: Current game state dict (from agent.summarize_state())
        task_title: Task title to determine recipe type
        predicted_primary_action: Predicted human primary action (optional)
        
    Returns:
        String representation of the secondary action
    """
    # Always use smart secondary action selection
    return _smart_select_secondary_action(game_state, predicted_primary_action or "NOOP", task_title)


def _smart_select_secondary_action(game_state: Dict, predicted_primary_action: str, task_title: str) -> str:
    """
    Smart secondary action selection that considers current state and human intent.
    This fixes the issue where robot keeps trying to pickup when it should place.
    Now also considers what's already in the pot to avoid redundant actions.
    """
    # Check for object mismatch first (before handling specific actions)
    object_to_drop = _detect_object_mismatch(game_state, predicted_primary_action)
    if object_to_drop:
        # Check if human already has the required object
        required_object = _get_required_object(predicted_primary_action)
        if required_object and game_state.get(f'{required_object}_hand') != 'partner':
            # Human doesn't have it, so we need to drop our wrong object first
            print(f"OBJECT MISMATCH DETECTED: Agent has {object_to_drop} but needs {required_object} for {predicted_primary_action}")
            return _handle_object_mismatch(game_state, object_to_drop)
    
    # Prioritize soup handling - No Primary Action Expected - So when soup is ready, Robot should serve it
    if game_state.get('soup_hand') == 'agent':
        # Robot has soup, place it (staging or serving station)
        return "place(soup)"
    if game_state.get('soup_staged', False) and game_state.get('soup_hand') == 'none':
        # Soup is staged and ready, robot should pick it up for serving
        return "pickup(soup)"
    
    # Handle NOOP case (when no primary action predicted)
    if predicted_primary_action == "NOOP":
        return "NOOP"
    
    # Handle washing actions
    if predicted_primary_action == "Wash Onion":
        # If sink blocked by tomato, clear it first
        if game_state.get('tomato_at_sink'):
            if game_state.get('onion_hand') == 'agent':
                return "place(onion, counter_tile)"
            return "pickup(tomato)"
        # If onion already at sink:
        # - If it's already washed, pick it up to move to next step.
        # - If not washed yet (confirmation pending), stay put.
        if game_state.get('onion_at_sink'):
            if game_state.get('onion_washed'):
                return "pickup(onion)"
            return "NOOP"
        # If human has the onion, let them wash it themselves
        if game_state.get('onion_hand') == 'partner':
            return "NOOP"
        elif game_state.get('onion_hand') == 'agent':
            # Robot has onion, place it at sink for washing
            return "place(onion, sink)"
        elif game_state.get('onion_hand') == 'none':
            # Robot needs to pick up onion first
            return "pickup(onion)"

    if predicted_primary_action == "Wash Tomato":
        # If sink blocked by onion, clear it first
        if game_state.get('onion_at_sink'):
            # If we're holding the tomato already, park it on a counter before clearing
            if game_state.get('tomato_hand') == 'agent':
                return "place(tomato, counter_tile)"
            return "pickup(onion)"
        # If tomato already at sink:
        # - If it's already washed, pick it up to move to next step.
        # - If not washed yet (confirmation pending), stay put.
        if game_state.get('tomato_at_sink'):
            if game_state.get('tomato_washed'):
                return "pickup(tomato)"
            return "NOOP"
        # If human has the tomato, let them wash it themselves
        if game_state.get('tomato_hand') == 'partner':
            return "NOOP"
        elif game_state.get('tomato_hand') == 'agent':
            # Robot has tomato, place it at sink for washing
            return "place(tomato, sink)"
        elif game_state.get('tomato_hand') == 'none':
            # Robot needs to pick up tomato first
            return "pickup(tomato)"
    
    # Handle chopping actions
    if predicted_primary_action == "Chop Onion":
        # If onion already at chopping station (waiting for confirmation), NOOP
        if game_state.get('onion_at_chopping'):
            return "NOOP"
        # If human has the onion, let them chop it themselves
        if game_state.get('onion_hand') == 'partner':
            return "NOOP"
        elif game_state.get('onion_hand') == 'agent':
            # Robot has onion, place it at chopping station
            return "place(onion, chopping_station)"
        elif game_state.get('onion_hand') == 'none':
            # Robot needs to pick up onion first
            return "pickup(onion)"

    if predicted_primary_action == "Chop Tomato":
        # If tomato already at chopping station (waiting for confirmation), NOOP
        if game_state.get('tomato_at_chopping'):
            return "NOOP"
        # If human has the tomato, let them chop it themselves
        if game_state.get('tomato_hand') == 'partner':
            return "NOOP"
        elif game_state.get('tomato_hand') == 'agent':
            # Robot has tomato, place it at chopping station
            return "place(tomato, chopping_station)"
        elif game_state.get('tomato_hand') == 'none':
            # Robot needs to pick up tomato first
            return "pickup(tomato)"

    # Handle seasoning actions
    if predicted_primary_action == "Salt Onion":
        # If onion already at salt station (waiting for confirmation), NOOP
        if game_state.get('onion_at_salt_station'):
            return "NOOP"
        # If human has the onion, let them salt it themselves
        if game_state.get('onion_hand') == 'partner':
            return "NOOP"
        elif game_state.get('onion_hand') == 'agent':
            # Robot has onion, place it at salt station
            return "place(onion, salt_station)"
        elif game_state.get('onion_hand') == 'none':
            # Robot needs to pick up onion first
            return "pickup(onion)"

    if predicted_primary_action == "Salt Tomato":
        # If tomato already at salt station (waiting for confirmation), NOOP
        if game_state.get('tomato_at_salt_station'):
            return "NOOP"
        # If human has the tomato, let them salt it themselves
        if game_state.get('tomato_hand') == 'partner':
            return "NOOP"
        elif game_state.get('tomato_hand') == 'agent':
            # Robot has tomato, place it at salt station
            return "place(tomato, salt_station)"
        elif game_state.get('tomato_hand') == 'none':
            # Robot needs to pick up tomato first
            return "pickup(tomato)"

    if predicted_primary_action == "Pepper Onion":
        # If onion already at pepper station (waiting for confirmation), NOOP
        if game_state.get('onion_at_pepper_station'):
            return "NOOP"
        # If human has the onion, let them pepper it themselves
        if game_state.get('onion_hand') == 'partner':
            return "NOOP"
        elif game_state.get('onion_hand') == 'agent':
            # Robot has onion, place it at pepper station
            return "place(onion, pepper_station)"
        elif game_state.get('onion_hand') == 'none':
            # Robot needs to pick up onion first
            return "pickup(onion)"

    if predicted_primary_action == "Pepper Tomato":
        # If tomato already at pepper station (waiting for confirmation), NOOP
        if game_state.get('tomato_at_pepper_station'):
            return "NOOP"
        # If human has the tomato, let them pepper it themselves
        if game_state.get('tomato_hand') == 'partner':
            return "NOOP"
        elif game_state.get('tomato_hand') == 'agent':
            # Robot has tomato, place it at pepper station
            return "place(tomato, pepper_station)"
        elif game_state.get('tomato_hand') == 'none':
            # Robot needs to pick up tomato first
            return "pickup(tomato)"
    
    # Handle Stage Onion action
    if predicted_primary_action == "Stage Onion":
        # Robot should stage the onion it's holding
        if game_state.get('onion_hand') == 'agent':
            return "place(onion, staging_station)"
        else:
            # Robot doesn't have onion, can't stage
            if game_state.get('onion_hand') == 'none' and not game_state.get('onion_staged', False):
                return "pickup(onion)"
            else:
                return "NOOP"
    
    # Handle Human Grab Onion (human grabs staged onion)
    elif predicted_primary_action == "Human Grab Onion":
        # Human task - robot does nothing
        return "NOOP"

    # Handle Stage Tomato action
    elif predicted_primary_action == "Stage Tomato":
        # Robot should stage the tomato it's holding
        if game_state.get('tomato_hand') == 'agent':
            return "place(tomato, staging_station)"
        else:
            if game_state.get('tomato_hand') == 'none' and not game_state.get('tomato_staged', False):
                return "pickup(tomato)"
            else:
                return "NOOP"
        
    
    # Handle Human Grab Tomato (human grabs staged tomato)
    elif predicted_primary_action == "Human Grab Tomato":
        # Human task - robot does nothing
        return "NOOP"
    
    # Human wants to grab dish
    elif predicted_primary_action == "Human Grab Dish":
        # Check if there's already a staged dish - if so, do NOOP
        if game_state.get('dish_staged', False):
            return "NOOP"
        elif game_state.get('dish_hand') == 'agent':
            # Robot has dish, should stage it
            return "place(dish, staging_station)"
        else:
            # Need to fetch dish
            return "pickup(dish)"
    
    # Turn stove on - when this happens, robot should do nothing
    elif predicted_primary_action == "Turn Stove On":
        return "NOOP"
    
    # Pour soup action - human task
    elif predicted_primary_action == "Pour Soup":
        # Human task - robot should do nothing
        return "NOOP"
    
    # Wait Till Ingredients Cooked - prepare dish for soup serving
    elif predicted_primary_action == "Wait For Ingredients to Cook":
        # When human is waiting for ingredients to cook, robot should prepare for soup serving
        if game_state.get('dish_hand') == 'agent':
            # Robot has dish, should stage it for when soup is ready
            return "place(dish, staging_station)"
        elif game_state.get('dish_hand') == 'none' and not game_state.get('dish_staged', False):
            # Need to fetch dish for soup serving
            return "pickup(dish)"
        elif game_state.get('dish_hand') == 'none' and game_state.get('dish_staged', False):
            # Dish is already staged, robot can do other helpful tasks
            # Check if we need to prepare other ingredients while waiting
            if "onion" in task_title.lower() and not game_state.get('onion_in_pot', False):
                if game_state.get('onion_hand') == 'none':
                    return "pickup(onion)"
                elif game_state.get('onion_hand') == 'agent':
                    return "place(onion, staging_station)"
            elif "tomato" in task_title.lower() and not game_state.get('tomato_in_pot', False):
                if game_state.get('tomato_hand') == 'none':
                    return "pickup(tomato)"
                elif game_state.get('tomato_hand') == 'agent':
                    return "place(tomato, staging_station)"
            # If all ingredients are prepared, just wait
            return "NOOP"
        else:
            # Default fallback
            return "NOOP"

    # Human Stage Soup action - robot helps with soup workflow  
    elif predicted_primary_action == "Human Stage Soup":
        # Robot wait for human to stage soup
        return "NOOP"
    
    # Wait For Robot To Serve Soup - robot actively serves the soup
    elif predicted_primary_action == "Wait For Robot To Serve Soup":
        if game_state.get('soup_hand') == 'agent':
            # Robot has soup, deliver it
            return "place(soup)"
        elif game_state.get('soup_staged', False) and game_state.get('soup_hand') == 'none':
            # Soup is staged, robot should pick it up
            return "pickup(soup)"
        elif game_state.get('soup_hand') == 'none' and not game_state.get('soup_staged', False):
            # No soup available yet, wait
            return "NOOP"
        else:
            return "NOOP"
    
    # Default fallback
    return "NOOP"

def get_all_primary_actions() -> list:
    """Get all available primary actions"""
    return [action.value for action in PrimaryAction]

def _detect_object_mismatch(game_state: Dict, predicted_primary_action: str) -> Optional[str]:
    """
    Detect if agent is holding the wrong object for the predicted action.
    Only considers it a mismatch if the human is NOT handling the required object.
    
    Args:
        game_state: Current game state dict
        predicted_primary_action: The predicted human primary action
        
    Returns:
        - None if no mismatch
        - Object name if agent should drop it (onion, tomato, dish, soup)
    """
    # Get the required object for the predicted action
    required_object = _get_required_object(predicted_primary_action)
    if not required_object:
        return None
    
    # If human already has the required object, no mismatch - they're handling it
    if game_state.get(f'{required_object}_hand') == 'partner':
        return None
    
    # Define all possible objects the agent can hold
    possible_objects = ['onion', 'tomato', 'dish', 'soup']
    
    # Check if agent is holding any object other than the required one
    for obj in possible_objects:
        if obj != required_object and game_state.get(f'{obj}_hand') == 'agent':
            return obj  # Agent has wrong object, should drop it
    
    return None

def _get_required_object(predicted_primary_action: str) -> Optional[str]:
    """
    Get the object required for the predicted primary action.
    
    Args:
        predicted_primary_action: The predicted human primary action
        
    Returns:
        The required object name or None if not applicable
    """
    action_to_object = {
        "Chop Onion": "onion",
        "Wash Onion": "onion", 
        "Stage Onion": "onion",
        "Human Grab Onion": "onion",
        "Place Onion in Pot": "onion",
        "Salt Onion": "onion",
        "Pepper Onion": "onion",
        "Chop Tomato": "tomato",
        "Wash Tomato": "tomato",
        "Stage Tomato": "tomato", 
        "Human Grab Tomato": "tomato",
        "Place Tomato in Pot": "tomato",
        "Salt Tomato": "tomato",
        "Pepper Tomato": "tomato",
        "Human Grab Dish": "dish",
        "Wait For Ingredients to Cook": "dish",
        "Wait For Robot To Serve Soup": "soup",
        "Pour Soup": "soup"
    }
    
    return action_to_object.get(predicted_primary_action)

def _handle_object_mismatch(game_state: Dict, object_to_drop: str, agent_pos: tuple = None) -> str:
    """
    Handle dropping the wrong object to free up hands.
    Now drops on counter tiles instead of staging stations.
    
    Args:
        game_state: Current game state dict
        object_to_drop: The object that needs to be dropped
        agent_pos: Current agent position for finding nearby counter tiles
        
    Returns:
        The action to drop the object
    """
    if object_to_drop == "onion":
        return "place(onion, counter_tile)"
    elif object_to_drop == "tomato":
        return "place(tomato, counter_tile)"
    elif object_to_drop == "dish":
        return "place(dish, counter_tile)"
    
    return "NOOP"


