"""
Game mechanics for Chopped Onion & Chopped Tomato Soup recipe.
"""

from .base_mechanics import BaseMechanics
from typing import Dict, List


class OnionChoppedTomatoChoppedMechanics(BaseMechanics):
    """Mechanics for chopped onion & chopped tomato soup (both ingredients must be chopped)"""
    
    def get_recipe_name(self) -> str:
        return "Chopped Onion & Chopped Tomato Soup"
    
    def get_required_ingredients(self) -> List[str]:
        return ["onion", "tomato"]
    
    def requires_chopping(self, ingredient: str) -> bool:
        return True  # Both ingredients need chopping
    
    def get_state_variables(self) -> Dict[str, str]:
        base_vars = self.get_base_state_variables()
        recipe_vars = {
            # Onion states
            'onion_hand': 'Who is holding the onion {none, agent, partner}',
            'onion_staged': 'Is onion staged/placed somewhere accessible',
            'onion_at_chopping': 'Is onion at chopping station',
            'onion_chopped': 'Is the onion chopped (flag set when placed at chopping station)',
            'onion_in_pot': 'Is onion placed in cooking pot (chopped)',
            
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
            "FetchOnion → PlaceOnionAtChopping → [Human: FetchChoppedOnion] → [Human: PlaceChoppedOnionInPot] → FetchTomato → PlaceTomatoAtChopping → [Human: FetchChoppedTomato] → [Human: PlaceChoppedTomatoInPot] → [Human: TurnStoveOn] → WaitForSoupToCook → soup_ready=true"
        ]
        return recipe_sequences + base_sequences
    
    def get_transition_rules(self) -> List[str]:
        base_rules = self.get_base_cooking_rules()
        recipe_rules = [
            "FetchOnion: onion_hand=none → onion_hand=agent (from dispenser)",
            "PlaceOnionAtChopping: onion_hand=agent → onion_hand=none, onion_at_chopping=true, onion_chopped=true (auto-chops)",
            "FetchChoppedOnion: onion_chopped=true, onion_hand=none → onion_hand=partner (human action)",
            "PlaceChoppedOnionInPot: onion_hand=partner, onion_chopped=true → onion_in_pot=true (human action)",
            "FetchTomato: tomato_hand=none → tomato_hand=agent (from dispenser)",
            "PlaceTomatoAtChopping: tomato_hand=agent → tomato_hand=none, tomato_at_chopping=true, tomato_chopped=true (auto-chops)",
            "FetchChoppedTomato: tomato_chopped=true, tomato_hand=none → tomato_hand=partner (human action)",
            "PlaceChoppedTomatoInPot: tomato_hand=partner, tomato_chopped=true → tomato_in_pot=true (human action)",
            "TurnStoveOn: onion_in_pot=true AND tomato_in_pot=true → soup_cooking=true (human action)",
            "Reset: soup_served=true → onion_chopped=false, onion_at_chopping=false, onion_in_pot=false, tomato_chopped=false, tomato_at_chopping=false, tomato_in_pot=false (reset for next recipe)"
        ]
        return recipe_rules + base_rules
    
    def get_secondary_actions(self) -> List[str]:
        base_actions = self.get_base_secondary_actions()
        recipe_actions = [
            "pickup(onion): Pick up onion from dispenser",
            "pickup(chopped_onion): Pick up chopped onion from chopping station",
            "pickup(tomato): Pick up tomato from dispenser",
            "pickup(chopped_tomato): Pick up chopped tomato from chopping station",
            "place(onion, chopping_station): Place onion at chopping station (auto-chops)",
            "place(chopped_onion): Place chopped onion at staging station",
            "place(tomato, chopping_station): Place tomato at chopping station (auto-chops)",
            "place(chopped_tomato): Place chopped tomato at staging station"
        ]
        return recipe_actions + base_actions
    
    def get_decision_rules(self) -> List[str]:
        base_rules = self.get_base_decision_rules()
        recipe_rules = [
            "**Choose pickup(onion) when:**",
            "- onion_hand=\"none\" AND onion_at_chopping=false AND onion_in_pot=false",
            "- (Need to fetch onion for chopping)",
            "",
            "**Choose place(onion, chopping_station) when:**",
            "- onion_hand=\"agent\" AND onion_chopped=false",
            "- (Agent is holding onion and needs to chop it)",
            "",
            "**Choose pickup(chopped_onion) when:**",
            "- onion_chopped=true AND onion_at_chopping=true AND onion_hand=\"none\"",
            "- (Chopped onion is ready at chopping station)",
            "",
            "**Choose place(chopped_onion) when:**",
            "- onion_chopped=true AND onion_hand=\"agent\" AND onion_staged=false",
            "- (Agent is holding chopped onion and needs to stage it)",
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