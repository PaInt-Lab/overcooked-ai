"""
Game mechanics for Raw Onion & Chopped Tomato Soup recipe.
"""

from .base_mechanics import BaseMechanics
from typing import Dict, List


class OnionRawTomatoChoppedMechanics(BaseMechanics):
    """Mechanics for raw onion & chopped tomato soup (onion is raw, tomato must be chopped)"""
    
    def get_recipe_name(self) -> str:
        return "Raw Onion & Chopped Tomato Soup"
    
    def get_required_ingredients(self) -> List[str]:
        return ["onion", "tomato"]
    
    def requires_chopping(self, ingredient: str) -> bool:
        return ingredient == "tomato"  # Only tomato needs chopping
    
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
            'tomato_at_chopping': 'Is tomato at chopping station',
            'tomato_chopped': 'Is the tomato chopped (flag set when placed at chopping station)',
            'tomato_in_pot': 'Is tomato placed in cooking pot (chopped)',
        }
        return {**base_vars, **recipe_vars}
    
    def get_valid_action_sequences(self) -> List[str]:
        base_sequences = self.get_base_action_sequences()
        recipe_sequences = [
            "FetchOnion → StageOnion → [Human: PlaceOnionInPot] → FetchTomato → PlaceTomatoAtChopping → [Human: FetchChoppedTomato] → [Human: PlaceChoppedTomatoInPot] → [Human: TurnStoveOn] → WaitForSoupToCook → soup_ready=true"
        ]
        return recipe_sequences + base_sequences
    
    def get_transition_rules(self) -> List[str]:
        base_rules = self.get_base_cooking_rules()
        recipe_rules = [
            "FetchOnion: onion_hand=none → onion_hand=agent (from dispenser)",
            "StageOnion: onion_hand=agent → onion_hand=none, onion_staged=true (to staging)",
            "PlaceOnionInPot: onion_hand=partner → onion_in_pot=true (human action)",
            "FetchTomato: tomato_hand=none → tomato_hand=agent (from dispenser)",
            "PlaceTomatoAtChopping: tomato_hand=agent → tomato_hand=none, tomato_at_chopping=true, tomato_chopped=true (auto-chops)",
            "FetchChoppedTomato: tomato_chopped=true, tomato_hand=none → tomato_hand=partner (human action)",
            "PlaceChoppedTomatoInPot: tomato_hand=partner, tomato_chopped=true → tomato_in_pot=true (human action)",
            "TurnStoveOn: onion_in_pot=true AND tomato_in_pot=true → soup_cooking=true (human action)",
            "Reset: soup_served=true → onion_in_pot=false, tomato_chopped=false, tomato_at_chopping=false, tomato_in_pot=false (reset for next recipe)"
        ]
        return recipe_rules + base_rules
    
    def get_secondary_actions(self) -> List[str]:
        base_actions = self.get_base_secondary_actions()
        recipe_actions = [
            "pickup(onion): Pick up onion from dispenser",
            "pickup(tomato): Pick up tomato from dispenser",
            "pickup(chopped_tomato): Pick up chopped tomato from chopping station",
            "place(onion, staging_station): Place onion at staging station",
            "place(tomato, chopping_station): Place tomato at chopping station (auto-chops)",
            "place(chopped_tomato): Place chopped tomato at staging station"
        ]
        return recipe_actions + base_actions
    
    def get_decision_rules(self) -> List[str]:
        base_rules = self.get_base_decision_rules()
        recipe_rules = [
            "**Choose pickup(onion) when:**",
            "- onion_hand=\"none\" AND onion_staged=false AND onion_in_pot=false",
            "- (Need to fetch onion for processing)",
            "",
            "**Choose place(onion, staging_station) when:**",
            "- onion_hand=\"agent\" AND onion_staged=false",
            "- (Agent is holding onion and needs to stage it for human to use)",
            "",
            "**Choose pickup(tomato) when:**",
            "- tomato_hand=\"none\" AND tomato_at_chopping=false AND tomato_in_pot=false AND onion_in_pot=true",
            "- (Need to fetch tomato for chopping after onion is in pot)",
            "",
            "**Choose place(tomato, chopping_station) when:**",
            "- tomato_hand=\"agent\" AND tomato_chopped=false",
            "- (Agent is holding tomato and needs to chop it)",
            "",
            "**Choose pickup(chopped_tomato) when:**",
            "- tomato_chopped=true AND tomato_at_chopping=true AND tomato_hand=\"none\"",
            "- (Chopped tomato is ready at chopping station)",
            "",
            "**Choose place(chopped_tomato) when:**",
            "- tomato_chopped=true AND tomato_hand=\"agent\" AND tomato_staged=false",
            "- (Agent is holding chopped tomato and needs to stage it)",
            ""
        ]
        return recipe_rules + base_rules