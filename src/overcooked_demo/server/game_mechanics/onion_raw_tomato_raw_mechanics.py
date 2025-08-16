"""
Game mechanics for Raw Onion & Raw Tomato Soup recipe.
"""

from .base_mechanics import BaseMechanics
from typing import Dict, List


class OnionRawTomatoRawMechanics(BaseMechanics):
    """Mechanics for raw onion & raw tomato soup (both ingredients go directly into pot without chopping)"""
    
    def get_recipe_name(self) -> str:
        return "Raw Onion & Raw Tomato Soup"
    
    def get_required_ingredients(self) -> List[str]:
        return ["onion", "tomato"]
    
    def requires_chopping(self, ingredient: str) -> bool:
        return False  # Both ingredients are raw - no chopping needed
    
    def get_state_variables(self) -> Dict[str, str]:
        base_vars = self.get_base_state_variables()
        recipe_vars = {
            # Onion states
            'onion_hand': 'Who is holding the onion {none, agent, partner}',
            'onion_staged': 'Is onion staged/placed somewhere accessible',
            'onion_in_pot': 'Is onion placed in cooking pot (raw)',
            
            # Tomato states
            'tomato_hand': 'Who is holding the tomato {none, agent, partner}',
            'tomato_staged': 'Is tomato staged/placed somewhere accessible',
            'tomato_in_pot': 'Is tomato placed in cooking pot (raw)',
        }
        return {**base_vars, **recipe_vars}
    
    def get_valid_action_sequences(self) -> List[str]:
        base_sequences = self.get_base_action_sequences()
        recipe_sequences = [
            "FetchOnion → StageOnion → [Human: PlaceOnionInPot] → FetchTomato → StageTomato → [Human: PlaceTomatoInPot] → [Human: TurnStoveOn] → WaitForSoupToCook → soup_ready=true"
        ]
        return recipe_sequences + base_sequences
    
    def get_transition_rules(self) -> List[str]:
        base_rules = self.get_base_cooking_rules()
        recipe_rules = [
            "FetchOnion: onion_hand=none → onion_hand=agent (from dispenser)",
            "StageOnion: onion_hand=agent → onion_hand=none, onion_staged=true (to staging)",
            "PlaceOnionInPot: onion_hand=partner → onion_in_pot=true (human action)",
            "FetchTomato: tomato_hand=none → tomato_hand=agent (from dispenser)",
            "StageTomato: tomato_hand=agent → tomato_hand=none, tomato_staged=true (to staging)",
            "PlaceTomatoInPot: tomato_hand=partner → tomato_in_pot=true (human action)",
            "TurnStoveOn: onion_in_pot=true AND tomato_in_pot=true → soup_cooking=true (human action)",
            "Reset: soup_served=true → onion_in_pot=false, tomato_in_pot=false (reset for next recipe)"
        ]
        return recipe_rules + base_rules
    
    # Secondary actions are now handled automatically by the optimized coordination system