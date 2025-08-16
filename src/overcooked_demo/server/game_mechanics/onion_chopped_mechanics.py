"""
Game mechanics for Chopped Onion Soup recipe.
"""

from .base_mechanics import BaseMechanics
from typing import Dict, List


class OnionChoppedMechanics(BaseMechanics):
    """Mechanics for chopped onion soup (onion must be chopped before going into pot)"""
    
    def get_recipe_name(self) -> str:
        return "Chopped Onion Soup"
    
    def get_required_ingredients(self) -> List[str]:
        return ["onion"]
    
    def requires_chopping(self, ingredient: str) -> bool:
        return ingredient == "onion"
    
    def get_state_variables(self) -> Dict[str, str]:
        base_vars = self.get_base_state_variables()
        recipe_vars = {
            # Onion states
            'onion_hand': 'Who is holding the onion {none, agent, partner}',
            'onion_staged': 'Is onion staged/placed somewhere accessible',
            'onion_at_chopping': 'Is onion at chopping station',
            'onion_chopped': 'Is the onion chopped (flag set when placed at chopping station)',
            'onion_in_pot': 'Is onion placed in cooking pot (chopped)',
        }
        return {**base_vars, **recipe_vars}
    
    def get_valid_action_sequences(self) -> List[str]:
        base_sequences = self.get_base_action_sequences()
        recipe_sequences = [
            "FetchOnion → PlaceOnionAtChopping → [Human: FetchChoppedOnion] → [Human: PlaceChoppedOnionInPot] → [Human: TurnStoveOn] → WaitForSoupToCook → soup_ready=true"
        ]
        return recipe_sequences + base_sequences
    
    def get_transition_rules(self) -> List[str]:
        base_rules = self.get_base_cooking_rules()
        recipe_rules = [
            "FetchOnion: onion_hand=none → onion_hand=agent (from dispenser)",
            "PlaceOnionAtChopping: onion_hand=agent → onion_hand=none, onion_at_chopping=true, onion_chopped=true (auto-chops)",
            "FetchChoppedOnion: onion_chopped=true, onion_hand=none → onion_hand=partner (human action)",
            "PlaceChoppedOnionInPot: onion_hand=partner, onion_chopped=true → onion_in_pot=true (human action)",
            "TurnStoveOn: onion_in_pot=true → soup_cooking=true (human action)",
            "Reset: soup_served=true → onion_chopped=false, onion_at_chopping=false, onion_in_pot=false (reset for next recipe)"
        ]
        return recipe_rules + base_rules
    
    # Secondary actions are now handled automatically by the optimized coordination system