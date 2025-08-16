"""
Base game mechanics for all Overcooked recipe types.
Contains common functionality and defines the interface for recipe-specific mechanics.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple


class BaseMechanics(ABC):
    """
    Base class for all game mechanics.
    Defines common state variables and abstract methods for recipe-specific implementations.
    """
    
    def __init__(self):
        self.recipe_name = self.get_recipe_name()
        
    @abstractmethod
    def get_recipe_name(self) -> str:
        """Return the name of this recipe type"""
        pass
    
    @abstractmethod
    def get_required_ingredients(self) -> List[str]:
        """Return list of ingredients needed for this recipe"""
        pass
    
    @abstractmethod
    def requires_chopping(self, ingredient: str) -> bool:
        """Return whether the ingredient needs to be chopped for this recipe"""
        pass
    
    def get_base_state_variables(self) -> Dict[str, str]:
        """
        Common state variables shared by all recipes.
        Individual mechanics can extend this with recipe-specific variables.
        """
        return {
            # Cooking states
            'soup_cooking': 'Is soup actively cooking (ticker >= 1)',
            'soup_ready': 'Is soup ready to serve',
            'soup_in_pot_not_cooking': 'Is soup in pot but not cooking (ticker = -1)',
            'soup_hand': 'Who is holding the soup {none, agent, partner}',
            'soup_staged': 'Is soup staged/placed somewhere accessible',
            'soup_served': 'Is soup delivered to serving station',
            
            # Dish states
            'dish_hand': 'Who is holding the dish {none, agent, partner}',
            'dish_staged': 'Is dish staged/placed somewhere accessible',
        }
    
    @abstractmethod
    def get_state_variables(self) -> Dict[str, str]:
        """
        Get all state variables for this recipe type.
        Should include base variables plus recipe-specific ones.
        """
        pass
    
    def get_base_cooking_rules(self) -> List[str]:
        """Common cooking transition rules"""
        return [
            "Cooking states are mutually exclusive: soup_cooking, soup_ready, and soup_in_pot_not_cooking cannot all be true simultaneously",
            "Ready: soup_cooking=true → soup_ready=true (automatic)",
            "FetchSoup: soup_ready=true, soup_hand=none → soup_hand=agent",
            "StageSoup: soup_hand=agent → soup_hand=none, soup_staged=true",
            "FetchDish: dish_hand=none → dish_hand=agent",
            "StageDish: dish_hand=agent → dish_hand=none, dish_staged=true",
            "ServeSoup: soup_staged=true → soup_hand=agent → soup_served=true"
        ]
    
    @abstractmethod
    def get_transition_rules(self) -> List[str]:
        """
        Get all transition rules for this recipe type.
        Should include base rules plus recipe-specific ones.
        """
        pass
    
    def get_base_action_sequences(self) -> List[str]:
        """Common action sequences"""
        return [
            "FetchDish → StageDish → FetchSoup → StageSoup → ServeSoup → soup_served=true"
        ]
    
    @abstractmethod  
    def get_valid_action_sequences(self) -> List[str]:
        """
        Get valid action sequences for this recipe type.
        Should include base sequences plus recipe-specific ones.
        """
        pass
    
    # Note: Secondary actions are now handled automatically by the optimized coordination system
    # No need for individual mechanics to define secondary actions or decision rules
    
    def get_mechanics_prompt(self) -> str:
        """
        Generate the complete game mechanics prompt for this recipe type.
        This replaces the old OVERCOOKED_GAME_MECHANICS string.
        """
        prompt_sections = [
            f"## OVERCOOKED GAME MECHANICS (MDP Knowledge) - {self.recipe_name.upper()}",
            "",
            "### State Variables:"
        ]
        
        # Add state variables
        for var, description in self.get_state_variables().items():
            prompt_sections.append(f"- {var} - {description}")
        
        prompt_sections.extend([
            "",
            "### Valid Action Sequences:"
        ])
        
        # Add action sequences
        for i, sequence in enumerate(self.get_valid_action_sequences(), 1):
            prompt_sections.append(f"{i}. {sequence}")
        
        prompt_sections.extend([
            "",
            "### Transition Rules:"
        ])
        
        # Add transition rules
        for rule in self.get_transition_rules():
            if rule.startswith("-"):
                prompt_sections.append(rule)
            else:
                prompt_sections.append(f"- {rule}")
        
        prompt_sections.extend([
            "",
            "## TASK EXECUTION",
            "",
            "Each call you receive has this structure:",
            "",
            "STATE SUMMARY:",
            "<one or two sentences describing what the robot and human hold, what's on staging counters, pots, etc., in plain English>",
            "",
            "PLAN:",
            "",
            "1. secondary: [<labels>], primary: [<labels>]",
            "2. secondary: [<labels>], primary: [<labels>]",
            "   ...",
            "   N) secondary: [<labels>], primary: [<labels>]",
            "",
            "Your job:",
            "",
            "1. **Analyze current state** against the MDP knowledge above to understand game mechanics",
            "2. **Identify which plan-step (1...N)** is currently active based on state and progress",
            "3. **From that step's primary list**, choose exactly one of the canonical primary events (reuse the text exactly as given)",
            "",
            "**Note:** The robot will automatically coordinate its secondary actions based on your primary action prediction.",
            "",
            "Return **only** this line (no extra commentary):",
            "",
            "primary: <exact primary event text>"
        ])
        
        return "\n".join(prompt_sections)