"""
Secondary Action Selection Framework

This module provides algorithmic secondary action selection based on predicted
primary actions. It maps primary actions to relevant secondary actions and 
selects the best one based on game state.
"""

from typing import Dict, Optional, List
from enum import Enum


class RecipeType(Enum):
    """Recipe types that align with the coordinated action agent recipe keys"""
    # Single ingredient recipes (raw)
    ONION_RAW = "onion_raw"
    TOMATO_RAW = "tomato_raw"
    
    # Single ingredient recipes (washed)
    ONION_WASHED = "onion_washed"
    TOMATO_WASHED = "tomato_washed"
    
    # Single ingredient recipes (chopped)
    ONION_CHOPPED = "onion_chopped"
    TOMATO_CHOPPED = "tomato_chopped"
    
    # Single ingredient recipes (washed + chopped)
    ONION_WASHED_CHOPPED = "onion_washed_chopped"
    TOMATO_WASHED_CHOPPED = "tomato_washed_chopped"
    
    # Dual ingredient recipes (raw)
    ONION_RAW_TOMATO_RAW = "onion_raw_tomato_raw"
    
    # Dual ingredient recipes (one chopped)
    ONION_CHOPPED_TOMATO_RAW = "onion_chopped_tomato_raw"
    ONION_RAW_TOMATO_CHOPPED = "onion_raw_tomato_chopped"
    
    # Dual ingredient recipes (both chopped)
    ONION_CHOPPED_TOMATO_CHOPPED = "onion_chopped_tomato_chopped"
    
    # Dual ingredient recipes (washed)
    ONION_WASHED_TOMATO_WASHED = "onion_washed_tomato_washed"
    
    # Dual ingredient recipes (washed + chopped combinations)
    ONION_WASHED_CHOPPED_TOMATO_WASHED = "onion_washed_chopped_tomato_washed"
    ONION_WASHED_TOMATO_WASHED_CHOPPED = "onion_washed_tomato_washed_chopped"
    ONION_WASHED_CHOPPED_TOMATO_WASHED_CHOPPED = "onion_washed_chopped_tomato_washed_chopped"


class PrimaryAction(Enum):
    """Primary actions from actual plans and complete state graph"""
    # NEW: Washing actions (from complete state graph)
    WASH_ONION = "Wash Onion"
    WASH_TOMATO = "Wash Tomato"
    
    # NEW: Processing actions (from complete state graph)
    CHOP_ONION = "Chop Onion"
    CHOP_TOMATO = "Chop Tomato"
    
    # Plan-style primary actions (standardized case)
    HUMAN_GRAB_ONION = "Human Grab Onion"
    HUMAN_GRAB_TOMATO = "Human Grab Tomato"
    HUMAN_GRAB_DISH = "Human Grab Dish"
    
    # Cooking actions (standardized case)
    PLACE_ONION_IN_POT = "Place Onion in Pot"
    PLACE_TOMATO_IN_POT = "Place Tomato in Pot"
    TURN_STOVE_ON = "Turn Stove On"
    WAIT_TILL_INGREDIENTS_COOKED = "Wait For Ingredients to Cook"
    POUR_SOUP = "Pour Soup in Dish"
    SERVE_SOUP = "Serve Soup"
    
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
    # NEW: Washing actions - Robot handles the washing workflow
    PrimaryAction.WASH_ONION: [
        "pickup(onion)",                 # Step 1: Fetch raw onion
        "place(onion, sink)"             # Step 2: Place at sink for washing
    ],
    
    PrimaryAction.WASH_TOMATO: [
        "pickup(tomato)",                # Step 1: Fetch raw tomato
        "place(tomato, sink)"            # Step 2: Place at sink for washing
    ],
    
    # NEW: Processing actions - Robot handles the chopping workflow
    PrimaryAction.CHOP_ONION: [
        "pickup(onion)",                 # Step 1: Fetch onion (raw or washed)
        "place(onion, chopping_station)" # Step 2: Place at chopping station
    ],
    
    PrimaryAction.CHOP_TOMATO: [
        "pickup(tomato)",                # Step 1: Fetch tomato (raw or washed)
        "place(tomato, chopping_station)" # Step 2: Place at chopping station
    ],
    
    # Plan-style primary actions with sequential workflows
    
    # "Human Grab Onion" -> Robot: fetch and stage onion (state-aware: raw, washed, or chopped)
    PrimaryAction.HUMAN_GRAB_ONION: [
        "pickup(onion)",                 # Step 1: Fetch onion in current state
        "place(onion, staging_station)"  # Step 2: Stage onion for human
    ],
    
    # "Human Grab Tomato" -> Robot: fetch and stage tomato (state-aware: raw, washed, or chopped)
    PrimaryAction.HUMAN_GRAB_TOMATO: [
        "pickup(tomato)",                # Step 1: Fetch tomato in current state
        "place(tomato, staging_station)" # Step 2: Stage tomato for human
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
    
    # "Pour Soup in Dish" -> Robot does NOOP (human task)
    PrimaryAction.POUR_SOUP: ["NOOP"],
    
    # "Serve Soup" -> Robot does NOOP (human task)  
    PrimaryAction.SERVE_SOUP: ["NOOP"],
    
    # Special cases
    PrimaryAction.NOOP: ["pickup(soup)", "place(soup)"],
    
    # When human does nothing, robot follows general priority
    PrimaryAction.HUMAN_NOOP: [
        "pickup(onion)", "pickup(tomato)", "pickup(dish)", "pickup(soup)",
        "place(onion)", "place(tomato)", "place(dish)", "place(soup)",
        "place(onion, chopping_station)", "place(tomato, chopping_station)",
        "place(chopped_onion)", "place(chopped_tomato)"
    ]
}

# Can replace all of this with a basic llm call one day
def parse_recipe_components(task_title: str) -> str:
    """
    Parse task title to determine recipe key (same logic as coordinated_action_predictor.py)
    
    Args:
        task_title: Task title (e.g., "Serving Onion Soup", "Serving Chopped Onion & Tomato Soup")
        
    Returns:
        Recipe key (e.g., "onion_raw", "onion_chopped_tomato_chopped")
    """
    task_title = task_title.lower()
    
    # Check for single ingredient recipes
    if "onion" in task_title and "tomato" not in task_title:
        if "chopped" in task_title:
            if "washed" in task_title:
                return "onion_washed_chopped"
            else:
                return "onion_chopped"
        elif "washed" in task_title:
            return "onion_washed"
        else:
            return "onion_raw"
    
    elif "tomato" in task_title and "onion" not in task_title:
        if "chopped" in task_title:
            if "washed" in task_title:
                return "tomato_washed_chopped"
            else:
                return "tomato_chopped"
        elif "washed" in task_title:
            return "tomato_washed"
        else:
            return "tomato_raw"
    
    # Check for dual ingredient recipes
    elif "onion" in task_title and "tomato" in task_title:
        # Find positions of key words
        onion_pos = task_title.find("onion")
        tomato_pos = task_title.find("tomato")
        
        # Determine which ingredient comes first
        first_ingredient_pos = min(onion_pos, tomato_pos)
        second_ingredient_pos = max(onion_pos, tomato_pos)
        first_is_onion = onion_pos < tomato_pos
        
        # Search for "chopped" specifically where it should be
        # For first ingredient: before first ingredient
        first_chopped_pos = -1
        pos = 0
        while pos < first_ingredient_pos:
            pos = task_title.find("chopped", pos)
            if pos == -1 or pos >= first_ingredient_pos:
                break
            first_chopped_pos = pos
            pos += 1
        
        # For second ingredient: between first and second ingredient
        second_chopped_pos = -1
        pos = first_ingredient_pos + 1
        while pos < second_ingredient_pos:
            pos = task_title.find("chopped", pos)
            if pos == -1 or pos >= second_ingredient_pos:
                break
            second_chopped_pos = pos
            pos += 1
        
        # Search for "washed" specifically where it should be
        # For first ingredient: before first ingredient
        first_washed_pos = -1
        pos = 0
        while pos < first_ingredient_pos:
            pos = task_title.find("washed", pos)
            if pos == -1 or pos >= first_ingredient_pos:
                break
            first_washed_pos = pos
            pos += 1
        
        # For second ingredient: between first and second ingredient
        second_washed_pos = -1
        pos = first_ingredient_pos + 1
        while pos < second_ingredient_pos:
            pos = task_title.find("washed", pos)
            if pos == -1 or pos >= second_ingredient_pos:
                break
            second_washed_pos = pos
            pos += 1
        
        # Check if processing applies to ingredients
        first_chopped = first_chopped_pos != -1
        second_chopped = second_chopped_pos != -1
        first_washed = first_washed_pos != -1
        second_washed = second_washed_pos != -1
        
        # Map first/second to onion/tomato based on order
        if first_is_onion:
            onion_chopped = first_chopped
            tomato_chopped = second_chopped
            onion_washed = first_washed
            tomato_washed = second_washed
        else:
            tomato_chopped = first_chopped
            onion_chopped = second_chopped
            tomato_washed = first_washed
            onion_washed = second_washed
        
        # Determine recipe type based on processing requirements
        if onion_chopped and tomato_chopped:
            if onion_washed and tomato_washed:
                return "onion_washed_chopped_tomato_washed_chopped"
            else:
                return "onion_chopped_tomato_chopped"
        elif onion_chopped and not tomato_chopped:
            if onion_washed and tomato_washed:
                return "onion_washed_chopped_tomato_washed"
            else:
                return "onion_chopped_tomato_raw"
        elif not onion_chopped and tomato_chopped:
            if onion_washed and tomato_washed:
                return "onion_washed_tomato_washed_chopped"
            else:
                return "onion_raw_tomato_chopped"
        elif onion_washed and tomato_washed:
            return "onion_washed_tomato_washed"
        else:
            return "onion_raw_tomato_raw"
    
    # Fallback for unknown recipes
    return "onion_raw"

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
    
    # Handle NOOP case (when no primary action predicted)
    if predicted_primary_action == "NOOP":
        # Default behavior: prepare ingredients or handle soup
        if game_state.get('soup_hand') == 'agent':
            return "place(soup)"
        elif game_state.get('soup_staged', False) and game_state.get('soup_hand') == 'none':
            return "pickup(soup)"
        elif "onion" in task_title.lower() and not game_state.get('onion_in_pot', False):
            if game_state.get('onion_hand') == 'none':
                return "pickup(onion)"
            elif game_state.get('onion_hand') == 'agent':
                return "place(onion, staging_station)"
        elif "tomato" in task_title.lower() and not game_state.get('tomato_in_pot', False):
            if game_state.get('tomato_hand') == 'none':
                return "pickup(tomato)"
            elif game_state.get('tomato_hand') == 'agent':
                return "place(tomato, staging_station)"
        else:
            return "NOOP"
    
    # ALWAYS prioritize soup handling (highest priority override)
    # If robot has soup in hand, always place it
    if game_state.get('soup_hand') == 'agent':
        return "place(soup)"
    
    # If soup is staged and robot doesn't have it, always pick it up
    if game_state.get('soup_staged', False) and game_state.get('soup_hand') == 'none':
        return "pickup(soup)"
    
    # NEW: Handle washing actions
    if predicted_primary_action == "Wash Onion":
        if game_state.get('onion_hand') == 'agent':
            # Robot has onion, place it at sink for washing
            return "place(onion, sink)"
        elif game_state.get('onion_hand') == 'none':
            # Robot needs to pick up onion first
            return "pickup(onion)"
    
    if predicted_primary_action == "Wash Tomato":
        if game_state.get('tomato_hand') == 'agent':
            # Robot has tomato, place it at sink for washing
            return "place(tomato, sink)"
        elif game_state.get('tomato_hand') == 'none':
            # Robot needs to pick up tomato first
            return "pickup(tomato)"
    
    # NEW: Handle chopping actions  
    if predicted_primary_action == "Chop Onion":
        if game_state.get('onion_hand') == 'agent':
            # Robot has onion, place it at chopping station
            return "place(onion, chopping_station)"
        elif game_state.get('onion_hand') == 'none':
            # Robot needs to pick up onion first
            return "pickup(onion)"
    
    if predicted_primary_action == "Chop Tomato":
        if game_state.get('tomato_hand') == 'agent':
            # Robot has tomato, place it at chopping station
            return "place(tomato, chopping_station)"
        elif game_state.get('tomato_hand') == 'none':
            # Robot needs to pick up tomato first
            return "pickup(tomato)"
    
    # Handle generic Human Grab Onion (state tells us if ingredient is chopped/washed/raw)
    if predicted_primary_action == "Human Grab Onion":
        # NEW: Check if onion is already staged - if so, work on tomato instead
        if game_state.get('onion_staged', False):
            # Onion is already staged for human, check if we should work on tomato
            if "tomato" in task_title.lower():
                # Recipe involves tomato, check tomato status
                if not game_state.get('tomato_staged', False) and not game_state.get('tomato_in_pot', False):
                    # Tomato is not staged and not in pot, work on tomato
                    if game_state.get('tomato_hand') == 'agent':
                        # Robot already has tomato, do NOOP (human will grab onion)
                        return "NOOP"
                    elif game_state.get('tomato_hand') == 'none':
                        # Need to get tomato
                        return "pickup(tomato)"
                # If tomato is staged or in pot, do NOOP
                return "NOOP"
            else:
                # Single ingredient recipe, onion is ready, do NOOP
                return "NOOP"
        
        if game_state.get('onion_hand') == 'agent' and game_state.get('onion_washed', False) and game_state.get('onion_chopped', False):
            # Robot has onion - check if it's chopped or raw
            return "place(onion, staging_station)"
        elif game_state.get('onion_hand') == 'none' and game_state.get('onion_washed', False) and game_state.get('onion_chopped', False):
            # Onion is chopped AND washed and ready, robot should pick it up
            return "pickup(onion)"  # State-aware pickup will find it at chopping station
        elif game_state.get('onion_hand') == 'none':
            # Check if we already have onion in pot - if so, prioritize tomato
            if game_state.get('onion_in_pot', False):
                # Onion already in pot, check if we need tomato instead
                if "tomato" in task_title.lower() and not game_state.get('tomato_in_pot', False):
                    if game_state.get('tomato_hand') == 'none':
                        return "pickup(tomato)"  # Get tomato instead of another onion
                    elif game_state.get('tomato_hand') == 'agent':
                        return "place(tomato, staging_station)"  # Stage tomato
                # If no tomato needed or tomato already in pot, get onion for chopping
                return "pickup(onion)"
            else:
                # No onion in pot, get onion for chopping
                return "pickup(onion)"
        else:
            # Default fallback
            return "pickup(onion)"
    
    # Handle generic Human Grab Tomato (state tells us if ingredient is chopped/washed/raw)
    elif predicted_primary_action == "Human Grab Tomato":
        # NEW: Check if tomato is already staged - if so, work on onion instead
        if game_state.get('tomato_staged', False):
            # Tomato is already staged for human, check if we should work on onion
            if "onion" in task_title.lower():
                # Recipe involves onion, check onion status
                if not game_state.get('onion_staged', False) and not game_state.get('onion_in_pot', False):
                    # Onion is not staged and not in pot, work on onion
                    if game_state.get('onion_hand') == 'agent':
                        # Robot already has onion, do NOOP (human will grab tomato)
                        return "NOOP"
                    elif game_state.get('onion_hand') == 'none':
                        # Need to get onion
                        return "pickup(onion)"
                # If onion is staged or in pot, do NOOP
                return "NOOP"
            else:
                # Single ingredient recipe, tomato is ready, do NOOP
                return "NOOP"
        
        if game_state.get('tomato_hand') == 'agent' and game_state.get('tomato_chopped', False) and game_state.get('tomato_washed', False):
            # Robot has tomato - check if it's chopped or raw
            return "place(tomato, staging_station)"  # State-aware place action
        elif game_state.get('tomato_hand') == 'none' and game_state.get('tomato_washed', False) and game_state.get('tomato_chopped', False):
            # Tomato is chopped and washed and ready, robot should pick it up
            return "pickup(tomato)"  # State-aware pickup will find it at chopping station
        elif game_state.get('tomato_hand') == 'none':
            # Check if we already have tomato in pot - if so, prioritize onion
            if game_state.get('tomato_in_pot', False):
                # Tomato already in pot, check if we need onion instead
                if "onion" in task_title.lower() and not game_state.get('onion_in_pot', False):
                    if game_state.get('onion_hand') == 'none':
                        return "pickup(onion)"  # Get onion instead of another tomato
                    elif game_state.get('onion_hand') == 'agent':
                        return "place(onion, staging_station)"  # Stage onion
                # If no onion needed or tomato already in pot, get tomato for chopping
                return "pickup(tomato)"
            else:
                # No tomato in pot, get tomato for chopping
                return "pickup(tomato)"
        else:
            # Default fallback
            return "pickup(tomato)"
    
    # Additional fallback for Human Grab Onion (shouldn't reach here with our current logic)
    elif "Human Grab Onion" in predicted_primary_action:
        if game_state.get('onion_hand') == 'agent':
            # Robot has onion, should stage it
            return "place(onion, staging_station)"
        else:
            # Check if onion already in pot - if so, prioritize tomato
            if game_state.get('onion_in_pot', False):
                # Onion already in pot, check if we need tomato instead
                if "tomato" in task_title.lower() and not game_state.get('tomato_in_pot', False):
                    if game_state.get('tomato_hand') == 'none':
                        return "pickup(tomato)"  # Get tomato instead of another onion
                    elif game_state.get('tomato_hand') == 'agent':
                        return "place(tomato, staging_station)"  # Stage tomato
                # If no tomato needed or tomato already in pot, get onion
                return "pickup(onion)"
            else:
                # No onion in pot, get onion
                return "pickup(onion)"
    
    # Human wants to grab raw tomato
    elif "Human Grab Tomato" in predicted_primary_action and "Chopped" not in predicted_primary_action:
        if game_state.get('tomato_hand') == 'agent':
            # Robot has tomato, should stage it
            return "place(tomato, staging_station)"
        else:
            # Check if tomato already in pot - if so, prioritize onion
            if game_state.get('tomato_in_pot', False):
                # Tomato already in pot, check if we need onion instead
                if "onion" in task_title.lower() and not game_state.get('onion_in_pot', False):
                    if game_state.get('onion_hand') == 'none':
                        return "pickup(onion)"  # Get onion instead of another tomato
                    elif game_state.get('onion_hand') == 'agent':
                        return "place(onion, staging_station)"  # Stage onion
                # If no onion needed or onion already in pot, get tomato
                return "pickup(tomato)"
            else:
                # No tomato in pot, get tomato
                return "pickup(tomato)"
    
    # Human wants to grab dish
    elif "Human Grab dish" in predicted_primary_action:
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
    elif "Turn stove on" in predicted_primary_action:
        return "NOOP"
    
    # Wait Till Ingredients Cooked - prepare dish for soup serving
    elif "Wait Till Ingredients Cooked" in predicted_primary_action:
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
    
    # Default fallback
    return "NOOP"


def get_relevant_secondary_actions(predicted_primary_action: str) -> List[str]:
    """
    Get the relevant secondary actions for a predicted primary action.
    
    Args:
        predicted_primary_action: The predicted human primary action
        
    Returns:
        List of relevant secondary action strings
    """
    try:
        primary_enum = PrimaryAction(predicted_primary_action)
        return PRIMARY_TO_SECONDARY_SEQUENCES.get(primary_enum, 
                                               PRIMARY_TO_SECONDARY_SEQUENCES[PrimaryAction.HUMAN_NOOP])
    except ValueError:
        return PRIMARY_TO_SECONDARY_SEQUENCES[PrimaryAction.HUMAN_NOOP]

def get_all_recipe_types() -> list:
    """Get all available recipe types"""
    return [recipe_type.value for recipe_type in RecipeType]


def get_all_primary_actions() -> list:
    """Get all available primary actions"""
    return [action.value for action in PrimaryAction]


