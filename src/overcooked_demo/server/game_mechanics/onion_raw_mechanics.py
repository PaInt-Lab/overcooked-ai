"""
Game mechanics for Raw Onion Soup recipe.
"""

from .base_mechanics import BaseMechanics
from typing import Dict, List


class OnionRawMechanics(BaseMechanics):
    """Mechanics for raw onion soup (onion goes directly into pot without chopping)"""
    
    def get_recipe_name(self) -> str:
        return "Raw Onion Soup"
    
    def get_required_ingredients(self) -> List[str]:
        return ["onion"]
    
    def requires_chopping(self, ingredient: str) -> bool:
        return False  # Raw onion recipe - no chopping needed
    
    def get_state_variables(self) -> Dict[str, str]:
        base_vars = self.get_base_state_variables()
        recipe_vars = {
            # Onion states
            'onion_hand': 'Who is holding the onion {none, agent, partner}',
            'onion_staged': 'Is onion staged/placed somewhere accessible',
            'onion_in_pot': 'Is onion placed in cooking pot (raw)',
        }
        return {**base_vars, **recipe_vars}
    
    def get_valid_action_sequences(self) -> List[str]:
        base_sequences = self.get_base_action_sequences()
        recipe_sequences = [
            "FetchOnion → StageOnion → [Human: PlaceOnionInPot] → [Human: TurnStoveOn] → WaitForSoupToCook → soup_ready=true"
        ]
        return recipe_sequences + base_sequences
    
    def get_transition_rules(self) -> List[str]:
        base_rules = self.get_base_cooking_rules()
        recipe_rules = [
            "FetchOnion: onion_hand=none → onion_hand=agent (from dispenser)",
            "StageOnion: onion_hand=agent → onion_hand=none, onion_staged=true (to staging)",
            "PlaceInPot: onion_hand=partner → onion_in_pot=true (human action)",
            "TurnStoveOn: onion_in_pot=true → soup_cooking=true (human action)",
            "Reset: soup_served=true → onion_in_pot=false (reset for next recipe)"
        ]
        return recipe_rules + base_rules
    
    def get_secondary_actions(self) -> List[str]:
        base_actions = self.get_base_secondary_actions()
        recipe_actions = [
            "pickup(onion): Pick up onion from dispenser",
            "place(onion, staging_station): Place onion at staging station"
        ]
        return recipe_actions + base_actions
    
    def get_decision_rules(self) -> List[str]:
        base_rules = self.get_base_decision_rules()
        recipe_rules = [
            "**Choose pickup(onion) when:**",
            "- onion_hand=\"none\" AND onion_staged=false AND onion_in_pot=false AND soup_staged=false AND soup_hand=\"none\"",
            "- (Need to fetch onion for processing)",
            "",
            "**Choose place(onion, staging_station) when:**",
            "- onion_hand=\"agent\" AND onion_staged=false",
            "- (Agent is holding onion and needs to stage it for human to use)",
            ""
        ]
        return recipe_rules + base_rules