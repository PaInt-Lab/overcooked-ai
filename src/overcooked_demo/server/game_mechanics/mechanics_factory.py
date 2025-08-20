"""
Factory for selecting the appropriate game mechanics based on recipe type.
Integrates with the existing model selection system.
"""

from typing import Optional
from .base_mechanics import BaseMechanics
from .onion_raw_mechanics import OnionRawMechanics
from .onion_chopped_mechanics import OnionChoppedMechanics
from .tomato_raw_mechanics import TomatoRawMechanics
from .tomato_chopped_mechanics import TomatoChoppedMechanics
from .onion_raw_tomato_raw_mechanics import OnionRawTomatoRawMechanics
from .onion_chopped_tomato_raw_mechanics import OnionChoppedTomatoRawMechanics
from .onion_raw_tomato_chopped_mechanics import OnionRawTomatoChoppedMechanics
from .onion_chopped_tomato_chopped_mechanics import OnionChoppedTomatoChoppedMechanics


# Recipe key to mechanics class mapping
# This aligns with the OVERCOOKED_MODELS mapping in coordinated_action_predictor.py
RECIPE_MECHANICS = {
    "onion_raw": OnionRawMechanics,
    "onion_chopped": OnionChoppedMechanics,
    "tomato_raw": TomatoRawMechanics,
    "tomato_chopped": TomatoChoppedMechanics,
    "onion_raw_tomato_raw": OnionRawTomatoRawMechanics,
    "onion_chopped_tomato_raw": OnionChoppedTomatoRawMechanics,
    "onion_raw_tomato_chopped": OnionRawTomatoChoppedMechanics,
    "onion_chopped_tomato_chopped": OnionChoppedTomatoChoppedMechanics
}


def get_mechanics_for_recipe(recipe_key: str) -> Optional[BaseMechanics]:
    """
    Get the appropriate game mechanics instance for a given recipe key.
    
    Args:
        recipe_key: Recipe identifier (e.g., "onion_raw", "onion_chopped_tomato_raw")
                   Should match the keys used in OVERCOOKED_MODELS
    
    Returns:
        Instance of the appropriate mechanics class, or None if recipe not found
    """
    mechanics_class = RECIPE_MECHANICS.get(recipe_key)
    if mechanics_class:
        return mechanics_class()
    return None


def get_available_recipes() -> list[str]:
    """
    Get list of all available recipe keys.
    
    Returns:
        List of recipe keys that have associated mechanics
    """
    return list(RECIPE_MECHANICS.keys())


def parse_recipe_components(task_title: str) -> str:
    """
    Parse task title to determine recipe components.
    This function should match the one in coordinated_action_predictor.py
    
    Args:
        task_title: Task title (e.g., "Serving Onion Soup", "Serving Chopped Onion & Tomato Soup")
    
    Returns:
        Recipe key (e.g., "onion_raw", "onion_chopped_tomato_chopped")
    """
    task_lower = task_title.lower()
    
    # Check for ingredients
    has_onion = "onion" in task_lower
    has_tomato = "tomato" in task_lower
    
    # Check for chopped variants
    onion_chopped = "chopped onion" in task_lower
    tomato_chopped = "chopped tomato" in task_lower
    
    # Build recipe key
    recipe_parts = []
    
    if has_onion:
        if onion_chopped:
            recipe_parts.append("onion_chopped")
        else:
            recipe_parts.append("onion_raw")
    
    if has_tomato:
        if tomato_chopped:
            recipe_parts.append("tomato_chopped")
        else:
            recipe_parts.append("tomato_raw")
    
    if recipe_parts:
        return "_".join(recipe_parts)
    
    # Default fallback
    return "onion_raw"


def get_mechanics_for_task(task_title: str) -> BaseMechanics:
    """
    Get game mechanics for a given task title.
    
    Args:
        task_title: Task title (e.g., "Serving Onion Soup")
    
    Returns:
        Instance of the appropriate mechanics class
        Falls back to onion_raw mechanics if no match found
    """
    recipe_key = parse_recipe_components(task_title)
    mechanics = get_mechanics_for_recipe(recipe_key)
    
    if mechanics is None:
        print(f"Warning: No mechanics found for recipe '{recipe_key}', falling back to onion_raw")
        mechanics = OnionRawMechanics()
    
    return mechanics


def get_mechanics_prompt_for_task(task_title: str) -> str:
    """
    Get the complete game mechanics prompt for a given task.
    This replaces the old OVERCOOKED_GAME_MECHANICS string.
    
    Args:
        task_title: Task title (e.g., "Serving Onion Soup")
    
    Returns:
        Complete game mechanics prompt string
    """
    mechanics = get_mechanics_for_task(task_title)
    return mechanics.get_mechanics_prompt()


# For debugging and testing
if __name__ == "__main__":
    # Test all recipe types
    test_tasks = [
        "Serving Onion Soup",
        "Serving Chopped Onion Soup", 
        "Serving Tomato Soup",
        "Serving Chopped Tomato Soup",
        "Serving Onion & Tomato Soup",
        "Serving Chopped Onion & Tomato Soup",
        "Serving Onion & Chopped Tomato Soup",
        "Serving Chopped Onion & Chopped Tomato Soup"
    ]
    
    print("Testing mechanics factory:")
    for task in test_tasks:
        recipe_key = parse_recipe_components(task)
        mechanics = get_mechanics_for_task(task)
        print(f"Task: {task}")
        print(f"  Recipe key: {recipe_key}")
        print(f"  Mechanics: {mechanics.__class__.__name__}")
        print(f"  Recipe name: {mechanics.get_recipe_name()}")
        print(f"  Ingredients: {mechanics.get_required_ingredients()}")
        print()