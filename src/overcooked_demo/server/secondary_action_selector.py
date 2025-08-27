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
    ONION_RAW = "onion_raw"
    ONION_CHOPPED = "onion_chopped"
    TOMATO_RAW = "tomato_raw"
    TOMATO_CHOPPED = "tomato_chopped"
    ONION_RAW_TOMATO_RAW = "onion_raw_tomato_raw"
    ONION_CHOPPED_TOMATO_RAW = "onion_chopped_tomato_raw"
    ONION_RAW_TOMATO_CHOPPED = "onion_raw_tomato_chopped"
    ONION_CHOPPED_TOMATO_CHOPPED = "onion_chopped_tomato_chopped"


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
    HUMAN_NOOP = "human_NOOP"


# Robot Task to Secondary Action Mappings
ROBOT_TASK_TO_ACTION = {
    "Robot Fetch onion": "pickup(onion)",
    "Robot Fetch tomato": "pickup(tomato)",
    "Robot Fetch dish": "pickup(dish)",
    "Robot fetch soup": "pickup(soup)",
    "Stage onion": "place(onion)",
    "Stage tomato": "place(tomato)",
    "Stage dish": "place(dish)",
    "Human stage soup": "NOOP",  # Human action, robot waits
    "Robot place soup on serving station": "place(soup)",
    "NOOP": "NOOP"
}

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
    
    # State graph style actions (fallback compatibility)
    PrimaryAction.HUMAN_GRAB_ONION_STATE: ["pickup(dish)"],
    PrimaryAction.HUMAN_PLACE_ONION_IN_POT_STATE: ["pickup(dish)"],
    PrimaryAction.HUMAN_GRAB_TOMATO_STATE: ["pickup(dish)"],
    PrimaryAction.HUMAN_PLACE_TOMATO_IN_POT_STATE: ["pickup(dish)"],
    PrimaryAction.HUMAN_TURN_STOVE_ON_STATE: ["pickup(dish)", "place(dish)"],
    PrimaryAction.HUMAN_POUR_SOUP_STATE: ["pickup(soup)", "place(soup)"],
    PrimaryAction.HUMAN_GRAB_DISH_STATE: ["pickup(soup)"],
    
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
        if "chopped" in task_title and task_title.find("chopped") < task_title.find("onion"):
            return "onion_chopped"
        else:
            return "onion_raw"
    
    elif "tomato" in task_title and "onion" not in task_title:
        if "chopped" in task_title and task_title.find("chopped") < task_title.find("tomato"):
            return "tomato_chopped"
        else:
            return "tomato_raw"
    
    # Check for dual ingredient recipes
    elif "onion" in task_title and "tomato" in task_title:
        # Check for specific patterns
        if "chopped onion & chopped tomato" in task_title:
            return "onion_chopped_tomato_chopped"
        elif "chopped onion & tomato" in task_title:
            return "onion_chopped_tomato_raw"
        elif "onion & chopped tomato" in task_title:
            return "onion_raw_tomato_chopped"
        else:
            return "onion_raw_tomato_raw"
    
    # Fallback for unknown recipes
    return "onion_raw"


class SecondaryActionSelector:
    """
    Primary-action-driven secondary action selector.
    
    This selector takes a predicted primary action and game state, then selects
    the best secondary action from the relevant subset of actions for that primary action.
    """
    
    def __init__(self, recipe_type: RecipeType):
        self.recipe_type = recipe_type
        
    def select_secondary_action(self, game_state: Dict, 
                               predicted_primary_action: Optional[str] = None) -> str:
        """
        Select the appropriate secondary action based on predicted primary action and game state.
        
        Args:
            game_state: Current game state (from agent.summarize_state())
            predicted_primary_action: The predicted human primary action (optional)
            
        Returns:
            String representing the best secondary action
        """
        # Get relevant secondary actions based on primary action
        relevant_actions = self._get_relevant_secondary_actions(predicted_primary_action)
        
        # Apply priority-based selection within the relevant actions
        return self._select_from_relevant_actions(relevant_actions, game_state)
        
    def _get_relevant_secondary_actions(self, predicted_primary_action: Optional[str]) -> List[str]:
        """Get the relevant secondary action sequence for a predicted primary action"""
        if not predicted_primary_action:
            # No primary action predicted, use general actions
            return PRIMARY_TO_SECONDARY_SEQUENCES[PrimaryAction.HUMAN_NOOP]
        
        # Try to match the predicted action to our enum
        try:
            primary_enum = PrimaryAction(predicted_primary_action)
            return PRIMARY_TO_SECONDARY_SEQUENCES.get(primary_enum, 
                                                   PRIMARY_TO_SECONDARY_SEQUENCES[PrimaryAction.HUMAN_NOOP])
        except ValueError:
            # Unknown primary action, fall back to general actions
            return PRIMARY_TO_SECONDARY_SEQUENCES[PrimaryAction.HUMAN_NOOP]
    
    def _select_from_relevant_actions(self, relevant_actions: List[str], game_state: Dict) -> str:
        """
        Select the best action from the relevant sequence based on current game state.
        Uses state-driven sequence progression logic.
        """
        # ALWAYS prioritize soup handling (highest priority override)
        # If robot has soup in hand, always place it
        if game_state.get('soup_hand') == 'agent':
            return "place(soup)"
        
        # If soup is staged and robot doesn't have it, always pick it up
        if game_state.get('soup_staged', False) and game_state.get('soup_hand') == 'none':
            return "pickup(soup)"
        
        # State-driven sequence progression for multi-step workflows
        
        # Human Grab Onion workflow (RAW onion only)
        if "pickup(onion)" in relevant_actions and "place(onion, staging_station)" in relevant_actions:
            # Step 1: Need raw onion
            if (game_state.get('onion_hand') == 'none' and 
                not game_state.get('onion_staged', False)):
                return "pickup(onion)"
            # Step 2: Have raw onion, stage it
            elif game_state.get('onion_hand') == 'agent':
                return "place(onion, staging_station)"
        
        # Human Grab Chopped Onion workflow (complete chopping process)
        chopped_onion_actions = ["pickup(onion)", "place(onion, chopping_station)", 
                                "pickup(chopped_onion)", "place(chopped_onion)"]
        if all(action in relevant_actions for action in chopped_onion_actions):
            # Step 1: Need raw onion for chopping
            if (game_state.get('onion_hand') == 'none' and 
                not game_state.get('onion_at_chopping', False)):
                return "pickup(onion)"
            # Step 2: Have raw onion, send to chopping
            elif (game_state.get('onion_hand') == 'agent' and 
                  not game_state.get('onion_chopped', False)):
                return "place(onion, chopping_station)"
            # Step 3: Onion is chopped, pick it up
            elif (game_state.get('onion_at_chopping', False) and 
                  game_state.get('onion_chopped', False) and 
                  game_state.get('onion_hand') == 'none'):
                return "pickup(chopped_onion)"
            # Step 4: Have chopped onion, stage it
            elif (game_state.get('onion_hand') == 'agent' and 
                  game_state.get('onion_chopped', False)):
                return "place(chopped_onion)"
        
        # Fallback onion workflow for simpler sequences
        elif "pickup(onion)" in relevant_actions:
            onion_place_actions = [a for a in relevant_actions if a.startswith("place(onion")]
            if onion_place_actions:
                # If we don't have onion and it's not staged/chopping, do pickup first
                if (game_state.get('onion_hand') == 'none' and 
                    not game_state.get('onion_staged', False) and
                    not game_state.get('onion_at_chopping', False)):
                    return "pickup(onion)"
                # If we have onion, place it at recipe-appropriate destination
                elif game_state.get('onion_hand') == 'agent':
                    # Choose destination based on recipe requirements
                    if (self._needs_chopped_onion() and 
                        "place(onion, chopping_station)" in relevant_actions):
                        return "place(onion, chopping_station)"
                    elif "place(onion, staging_station)" in relevant_actions:
                        return "place(onion, staging_station)"
                    else:
                        return "place(onion)"  # Fallback to generic place
        
        # Human Grab Tomato workflow (RAW tomato only)
        if "pickup(tomato)" in relevant_actions and "place(tomato, staging_station)" in relevant_actions:
            # Step 1: Need raw tomato
            if (game_state.get('tomato_hand') == 'none' and 
                not game_state.get('tomato_staged', False)):
                return "pickup(tomato)"
            # Step 2: Have raw tomato, stage it
            elif game_state.get('tomato_hand') == 'agent':
                return "place(tomato, staging_station)"
        
        # Human Grab Chopped Tomato workflow (complete chopping process)
        chopped_tomato_actions = ["pickup(tomato)", "place(tomato, chopping_station)", 
                                 "pickup(chopped_tomato)", "place(chopped_tomato)"]
        if all(action in relevant_actions for action in chopped_tomato_actions):
            # Step 1: Need raw tomato for chopping
            if (game_state.get('tomato_hand') == 'none' and 
                not game_state.get('tomato_at_chopping', False)):
                return "pickup(tomato)"
            # Step 2: Have raw tomato, send to chopping
            elif (game_state.get('tomato_hand') == 'agent' and 
                  not game_state.get('tomato_chopped', False)):
                return "place(tomato, chopping_station)"
            # Step 3: Tomato is chopped, pick it up
            elif (game_state.get('tomato_at_chopping', False) and 
                  game_state.get('tomato_chopped', False) and 
                  game_state.get('tomato_hand') == 'none'):
                return "pickup(chopped_tomato)"
            # Step 4: Have chopped tomato, stage it
            elif (game_state.get('tomato_hand') == 'agent' and 
                  game_state.get('tomato_chopped', False)):
                return "place(chopped_tomato)"
        
        # Fallback tomato workflow for simpler sequences
        elif "pickup(tomato)" in relevant_actions:
            tomato_place_actions = [a for a in relevant_actions if a.startswith("place(tomato")]
            if tomato_place_actions:
                # If we don't have tomato and it's not staged/chopping, do pickup first
                if (game_state.get('tomato_hand') == 'none' and 
                    not game_state.get('tomato_staged', False) and
                    not game_state.get('tomato_at_chopping', False)):
                    return "pickup(tomato)"
                # If we have tomato, place it at recipe-appropriate destination
                elif game_state.get('tomato_hand') == 'agent':
                    # Choose destination based on recipe requirements
                    if (self._needs_chopped_tomato() and 
                        "place(tomato, chopping_station)" in relevant_actions):
                        return "place(tomato, chopping_station)"
                    elif "place(tomato, staging_station)" in relevant_actions:
                        return "place(tomato, staging_station)"
                    else:
                        return "place(tomato)"  # Fallback to generic place
        
        # Dish workflow: pickup → place
        if "pickup(dish)" in relevant_actions and "place(dish)" in relevant_actions:
            # If we don't have dish and it's not staged, do pickup first
            if (game_state.get('dish_hand') == 'none' and 
                not game_state.get('dish_staged', False)):
                return "pickup(dish)"
            # If we have dish, place it
            elif game_state.get('dish_hand') == 'agent':
                return "place(dish)"
        
        # Handle non-chopping workflows when holding ingredients
        if game_state.get('onion_hand') == 'agent':
            if "place(onion)" in relevant_actions:
                return "place(onion)"
        
        if game_state.get('tomato_hand') == 'agent':
            if "place(tomato)" in relevant_actions:
                return "place(tomato)"
        
        # Single chopped ingredient pickups (when not part of full chopping sequence)
        if (game_state.get('onion_at_chopping', False) and 
            game_state.get('onion_chopped', False) and 
            game_state.get('onion_hand') == 'none' and
            "pickup(chopped_onion)" in relevant_actions and
            "place(chopped_onion)" not in relevant_actions):
            return "pickup(chopped_onion)"
            
        if (game_state.get('tomato_at_chopping', False) and 
            game_state.get('tomato_chopped', False) and 
            game_state.get('tomato_hand') == 'none' and
            "pickup(chopped_tomato)" in relevant_actions and
            "place(chopped_tomato)" not in relevant_actions):
            return "pickup(chopped_tomato)"
        
        # Single pickup actions (when not part of a sequence)
        single_pickups = [
            ("pickup(onion)", lambda: (self._needs_onion() and
                                     game_state.get('onion_hand') == 'none' and 
                                     not game_state.get('onion_staged', False) and 
                                     "place(onion)" not in relevant_actions)),  # Not part of sequence
            ("pickup(tomato)", lambda: (self._needs_tomato() and
                                      game_state.get('tomato_hand') == 'none' and 
                                      not game_state.get('tomato_staged', False) and
                                      "place(tomato)" not in relevant_actions)),  # Not part of sequence  
            ("pickup(dish)", lambda: (self._needs_dish() and
                                    game_state.get('dish_hand') == 'none' and
                                    not game_state.get('dish_staged', False) and
                                    "place(dish)" not in relevant_actions)),  # Not part of sequence
        ]
        
        for action, condition in single_pickups:
            if action in relevant_actions and condition():
                return action
        
        # Default: NOOP
        return "NOOP"
        
    def _needs_onion(self) -> bool:
        """Check if this recipe needs onion"""
        return "onion" in self.recipe_type.value
        
    def _needs_tomato(self) -> bool:
        """Check if this recipe needs tomato"""
        return "tomato" in self.recipe_type.value
        
    def _needs_chopped_onion(self) -> bool:
        """Check if this recipe needs chopped onion"""
        return "onion_chopped" in self.recipe_type.value
        
    def _needs_chopped_tomato(self) -> bool:
        """Check if this recipe needs chopped tomato"""
        return "tomato_chopped" in self.recipe_type.value
        
    def _needs_dish(self) -> bool:
        """Check if this recipe needs a dish (all recipes do)"""
        return True  # All soup recipes need a dish


def create_selector_for_task(task_title: str) -> SecondaryActionSelector:
    """
    Factory function to create the appropriate SecondaryActionSelector 
    for a given task title.
    
    Args:
        task_title: Task title like "Serving Onion Soup", "Serving Chopped Onion Soup", etc.
        
    Returns:
        SecondaryActionSelector configured for the task
    """
    # Use the same parsing logic as the coordinated action agent
    recipe_key = parse_recipe_components(task_title)
    
    # Map recipe key to RecipeType enum
    recipe_type_map = {
        "onion_raw": RecipeType.ONION_RAW,
        "onion_chopped": RecipeType.ONION_CHOPPED,
        "tomato_raw": RecipeType.TOMATO_RAW,
        "tomato_chopped": RecipeType.TOMATO_CHOPPED,
        "onion_raw_tomato_raw": RecipeType.ONION_RAW_TOMATO_RAW,
        "onion_chopped_tomato_raw": RecipeType.ONION_CHOPPED_TOMATO_RAW,
        "onion_raw_tomato_chopped": RecipeType.ONION_RAW_TOMATO_CHOPPED,
        "onion_chopped_tomato_chopped": RecipeType.ONION_CHOPPED_TOMATO_CHOPPED,
    }
    
    recipe_type = recipe_type_map.get(recipe_key, RecipeType.ONION_RAW)
    return SecondaryActionSelector(recipe_type)


# Convenience functions for easy integration

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
    # NEW: Smart secondary action selection without complex selectors
    if predicted_primary_action:
        return _smart_select_secondary_action(game_state, predicted_primary_action, task_title)
    
    # Fallback to old system
    selector = create_selector_for_task(task_title)
    return selector.select_secondary_action(game_state, predicted_primary_action)


def _smart_select_secondary_action(game_state: Dict, predicted_primary_action: str, task_title: str) -> str:
    """
    Smart secondary action selection that considers current state and human intent.
    This fixes the issue where robot keeps trying to pickup when it should place.
    Now also considers what's already in the pot to avoid redundant actions.
    """
    
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
        if game_state.get('onion_hand') == 'agent':
            # Robot has onion - check if it's chopped or raw
            if game_state.get('onion_chopped', False):
                # Robot has chopped onion, should stage it for human
                return "place(chopped_onion)"  # Use chopped_onion for proper item type
            else:
                # Robot has raw onion, should place it at chopping station
                return "place(onion, chopping_station)"
        elif game_state.get('onion_hand') == 'none' and game_state.get('onion_at_chopping', False) and game_state.get('onion_chopped', False):
            # Onion is chopped and ready, robot should pick it up
            return "pickup(chopped_onion)"
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
        if game_state.get('tomato_hand') == 'agent':
            # Robot has tomato - check if it's chopped or raw
            if game_state.get('tomato_chopped', False):
                # Robot has chopped tomato, should stage it for human
                return "place(chopped_tomato)"  # Use chopped_tomato for proper item type
            else:
                # Robot has raw tomato, should place it at chopping station
                return "place(tomato, chopping_station)"
        elif game_state.get('tomato_hand') == 'none' and game_state.get('tomato_at_chopping', False) and game_state.get('tomato_chopped', False):
            # Tomato is chopped and ready, robot should pick it up
            return "pickup(chopped_tomato)"
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


def parse_robot_task_to_action(robot_task: str) -> str:
    """
    Parse a robot task from plan secondary actions into actual secondary action.
    
    Args:
        robot_task: Robot task string like "Robot Fetch onion", "Stage dish"
        
    Returns:
        Secondary action string like "pickup(onion)", "place(dish)"
    """
    return ROBOT_TASK_TO_ACTION.get(robot_task, "NOOP")


def parse_plan_secondary_actions(secondary_plan: List[List[str]]) -> List[str]:
    """
    Parse plan secondary actions into actual secondary action options.
    
    Args:
        secondary_plan: List of lists of robot tasks from plan
                       e.g., [['Robot Fetch onion', 'Stage onion']]
        
    Returns:
        List of possible secondary actions
    """
    all_actions = []
    
    for step_tasks in secondary_plan:
        for task in step_tasks:
            action = parse_robot_task_to_action(task)
            if action not in all_actions:
                all_actions.append(action)
    
    # Always include NOOP as an option
    if "NOOP" not in all_actions:
        all_actions.append("NOOP")
        
    return all_actions


def get_all_recipe_types() -> list:
    """Get all available recipe types"""
    return [recipe_type.value for recipe_type in RecipeType]


def get_all_primary_actions() -> list:
    """Get all available primary actions"""
    return [action.value for action in PrimaryAction]


# For debugging and analysis
def analyze_decision_process(game_state: Dict, task_title: str, 
                           predicted_primary_action: Optional[str] = None) -> Dict:
    """
    Analyze the decision process for debugging purposes.
    
    Args:
        game_state: Current game state
        task_title: Task title
        predicted_primary_action: Predicted primary action
        
    Returns:
        Dict containing analysis information
    """
    selector = create_selector_for_task(task_title)
    
    # Get relevant actions for the primary action
    relevant_actions = selector._get_relevant_secondary_actions(predicted_primary_action)
    
    # Get selected action
    selected_action = selector.select_secondary_action(game_state, predicted_primary_action)
    
    # Get recipe requirements
    requirements = {
        "needs_onion": selector._needs_onion(),
        "needs_tomato": selector._needs_tomato(),
        "needs_chopped_onion": selector._needs_chopped_onion(),
        "needs_chopped_tomato": selector._needs_chopped_tomato(),
    }
    
    return {
        "task_title": task_title,
        "recipe_type": selector.recipe_type.value,
        "predicted_primary_action": predicted_primary_action,
        "relevant_secondary_actions": relevant_actions,
        "selected_action": selected_action,
        "game_state": game_state,
        "recipe_requirements": requirements,
    }