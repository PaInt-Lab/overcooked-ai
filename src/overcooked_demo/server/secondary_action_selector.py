"""
Secondary Action Selection Framework

This module provides algorithmic secondary action selection to replace LLM-based 
secondary action decisions. It implements hardcoded decision rules for consistent 
and fast secondary action selection.
"""

from typing import Dict, Optional
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
    Algorithmic secondary action selector with hardcoded decision rules.
    Based on the coordinated action agent's fallback system but optimized for speed.
    """
    
    def __init__(self, recipe_type: RecipeType):
        self.recipe_type = recipe_type
        
    def select_secondary_action(self, game_state: Dict, 
                               primary_action: Optional[str] = None) -> str:
        """
        Select the appropriate secondary action based on current game state.
        Uses hardcoded priority-based action selection for maximum speed.
        
        Args:
            game_state: Current game state (from agent.summarize_state())
            primary_action: The selected primary action (optional, for future use)
            
        Returns:
            String representing the best secondary action
        """
        # Priority-based action selection (based on coordinated agent fallback system)
        
        # 1. If we have soup in hand, serve it
        if game_state.get('soup_hand') == 'agent':
            return 'place(soup)'
        
        # 2. If soup is ready and staged, pick it up
        if (game_state.get('soup_ready', False) and 
            game_state.get('soup_staged', False) and 
            game_state.get('soup_hand') == 'none'):
            return 'pickup(soup)'
        
        # 3. If we have chopped onion, place it at staging
        if (game_state.get('onion_hand') == 'agent' and 
            game_state.get('onion_chopped', False)):
            return 'place(chopped_onion)'
        
        # 4. If we have chopped tomato, place it at staging
        if (game_state.get('tomato_hand') == 'agent' and 
            game_state.get('tomato_chopped', False)):
            return 'place(chopped_tomato)'
        
        # 5. If we have raw onion, chop it (for chopped recipes)
        if (game_state.get('onion_hand') == 'agent' and 
            not game_state.get('onion_chopped', False) and
            self._needs_chopped_onion()):
            return 'place(onion, chopping_station)'
        
        # 6. If we have raw tomato, chop it (for chopped recipes)
        if (game_state.get('tomato_hand') == 'agent' and 
            not game_state.get('tomato_chopped', False) and
            self._needs_chopped_tomato()):
            return 'place(tomato, chopping_station)'
        
        # 7. If we have raw onion and recipe doesn't need chopping, place it directly
        if (game_state.get('onion_hand') == 'agent' and 
            not self._needs_chopped_onion()):
            return 'place(onion)'
        
        # 8. If we have raw tomato and recipe doesn't need chopping, place it directly
        if (game_state.get('tomato_hand') == 'agent' and 
            not self._needs_chopped_tomato()):
            return 'place(tomato)'
        
        # 9. If onion is at chopping and chopped, pick it up
        if (game_state.get('onion_at_chopping', False) and 
            game_state.get('onion_chopped', False) and 
            game_state.get('onion_hand') == 'none'):
            return 'pickup(chopped_onion)'
        
        # 10. If tomato is at chopping and chopped, pick it up
        if (game_state.get('tomato_at_chopping', False) and 
            game_state.get('tomato_chopped', False) and 
            game_state.get('tomato_hand') == 'none'):
            return 'pickup(chopped_tomato)'
        
        # 11. If we have dish, place it at staging
        if game_state.get('dish_hand') == 'agent':
            return 'place(dish)'
        
        # 12. If soup is cooking and we don't have dish, get dish
        if (game_state.get('soup_cooking', False) and 
            game_state.get('dish_hand') == 'none'):
            return 'pickup(dish)'
        
        # 13. If we need onion and don't have it, get onion
        if (self._needs_onion() and
            game_state.get('onion_hand') == 'none' and 
            not game_state.get('onion_staged', False) and 
            not game_state.get('onion_at_chopping', False) and 
            not game_state.get('onion_in_pot', False)):
            return 'pickup(onion)'
        
        # 14. If we need tomato and don't have it, get tomato
        if (self._needs_tomato() and
            game_state.get('tomato_hand') == 'none' and 
            not game_state.get('tomato_staged', False) and 
            not game_state.get('tomato_at_chopping', False) and 
            not game_state.get('tomato_in_pot', False)):
            return 'pickup(tomato)'
        
        # Default: no action needed
        return 'NOOP'
        
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
                          primary_action: Optional[str] = None) -> str:
    """
    Convenience function to select secondary action for a given task.
    
    Args:
        game_state: Current game state dict (from agent.summarize_state())
        task_title: Task title to determine recipe type
        primary_action: Selected primary action (optional)
        
    Returns:
        String representation of the secondary action
    """
    selector = create_selector_for_task(task_title)
    return selector.select_secondary_action(game_state, primary_action)


def get_all_recipe_types() -> list:
    """Get all available recipe types"""
    return [recipe_type.value for recipe_type in RecipeType]


# For debugging and analysis
def analyze_decision_process(game_state: Dict, task_title: str) -> Dict:
    """
    Analyze the decision process for debugging purposes.
    
    Args:
        game_state: Current game state
        task_title: Task title
        
    Returns:
        Dict containing analysis information
    """
    selector = create_selector_for_task(task_title)
    
    # Get selected action
    selected_action = selector.select_secondary_action(game_state)
    
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
        "game_state": game_state,
        "selected_action": selected_action,
        "recipe_requirements": requirements,
    }