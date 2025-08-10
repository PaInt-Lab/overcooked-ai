"""
Game mechanics for Chopped Tomato Soup recipe.
"""

from .base_mechanics import BaseMechanics
from typing import Dict, List


class TomatoChoppedMechanics(BaseMechanics):
    """Mechanics for chopped tomato soup (tomato must be chopped before going into pot)"""
    
    def get_recipe_name(self) -> str:
        return "Chopped Tomato Soup"
    
    def get_required_ingredients(self) -> List[str]:
        return ["tomato"]
    
    def requires_chopping(self, ingredient: str) -> bool:
        return ingredient == "tomato"
    
    def get_state_variables(self) -> Dict[str, str]:
        base_vars = self.get_base_state_variables()
        recipe_vars = {
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
            "FetchTomato → PlaceTomatoAtChopping → [Human: FetchChoppedTomato] → [Human: PlaceChoppedTomatoInPot] → [Human: TurnStoveOn] → WaitForSoupToCook → soup_ready=true"
        ]
        return recipe_sequences + base_sequences
    
    def get_transition_rules(self) -> List[str]:
        base_rules = self.get_base_cooking_rules()
        recipe_rules = [
            "FetchTomato: tomato_hand=none → tomato_hand=agent (from dispenser)",
            "PlaceTomatoAtChopping: tomato_hand=agent → tomato_hand=none, tomato_at_chopping=true, tomato_chopped=true (auto-chops)",
            "FetchChoppedTomato: tomato_chopped=true, tomato_hand=none → tomato_hand=partner (human action)",
            "PlaceChoppedTomatoInPot: tomato_hand=partner, tomato_chopped=true → tomato_in_pot=true (human action)",
            "TurnStoveOn: tomato_in_pot=true → soup_cooking=true (human action)",
            "Reset: soup_served=true → tomato_chopped=false, tomato_at_chopping=false, tomato_in_pot=false (reset for next recipe)"
        ]
        return recipe_rules + base_rules
    
    def get_secondary_actions(self) -> List[str]:
        base_actions = self.get_base_secondary_actions()
        recipe_actions = [
            "pickup(tomato): Pick up tomato from dispenser",
            "pickup(chopped_tomato): Pick up chopped tomato from chopping station",
            "place(tomato, chopping_station): Place tomato at chopping station (auto-chops)",
            "place(chopped_tomato): Place chopped tomato at staging station"
        ]
        return recipe_actions + base_actions
    
    def get_decision_rules(self) -> List[str]:
        base_rules = self.get_base_decision_rules()
        recipe_rules = [
            "**Choose pickup(tomato) when:**",
            "- tomato_hand=\"none\" AND tomato_staged=false AND tomato_at_chopping=false AND tomato_in_pot=false AND soup_staged=false AND soup_hand=\"none\"",
            "- (Need to fetch tomato for processing)",
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