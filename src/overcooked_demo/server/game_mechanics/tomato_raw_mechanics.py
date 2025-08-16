"""
Game mechanics for Raw Tomato Soup recipe.
"""

from .base_mechanics import BaseMechanics
from typing import Dict, List


class TomatoRawMechanics(BaseMechanics):
    """Mechanics for raw tomato soup (tomato goes directly into pot without chopping)"""
    
    def get_recipe_name(self) -> str:
        return "Raw Tomato Soup"
    
    def get_required_ingredients(self) -> List[str]:
        return ["tomato"]
    
    def requires_chopping(self, ingredient: str) -> bool:
        return False  # Raw tomato recipe - no chopping needed
    
    def get_state_variables(self) -> Dict[str, str]:
        base_vars = self.get_base_state_variables()
        recipe_vars = {
            # Tomato states
            'tomato_hand': 'Who is holding the tomato {none, agent, partner}',
            'tomato_staged': 'Is tomato staged/placed somewhere accessible',
            'tomato_in_pot': 'Is tomato placed in cooking pot (raw)',
        }
        return {**base_vars, **recipe_vars}
    
    def get_valid_action_sequences(self) -> List[str]:
        base_sequences = self.get_base_action_sequences()
        recipe_sequences = [
            "FetchTomato → StageTomato → [Human: PlaceTomatoInPot] → [Human: TurnStoveOn] → WaitForSoupToCook → soup_ready=true"
        ]
        return recipe_sequences + base_sequences
    
    def get_transition_rules(self) -> List[str]:
        base_rules = self.get_base_cooking_rules()
        recipe_rules = [
            "FetchTomato: tomato_hand=none → tomato_hand=agent (from dispenser)",
            "StageTomato: tomato_hand=agent → tomato_hand=none, tomato_staged=true (to staging)",
            "PlaceInPot: tomato_hand=partner → tomato_in_pot=true (human action)",
            "TurnStoveOn: tomato_in_pot=true → soup_cooking=true (human action)",
            "Reset: soup_served=true → tomato_in_pot=false (reset for next recipe)"
        ]
        return recipe_rules + base_rules
    
    # Secondary actions are now handled automatically by the optimized coordination system