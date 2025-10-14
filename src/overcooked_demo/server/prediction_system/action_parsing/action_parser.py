"""Parse and validate actions from LLM responses and robot commands."""

import re
from typing import Tuple, Optional


class ActionParser:
    """Handles parsing of LLM responses and robot actions."""
    
    def __init__(self, tile_checker=None):
        """
        Initialize action parser.
        
        Args:
            tile_checker: Optional TileManager instance for state-aware location detection
        """
        self.tile_checker = tile_checker
    
    def parse_primary_action(self, response: str) -> str:
        """
        Parse the LLM response to extract only the human primary action prediction.
        Expected format:
          Primary: <primary_action_name>
        """
        # Default value
        predicted_human_action = "NOOP"
        
        # Parse primary action prediction
        primary_match = re.search(r'Primary:\s*(.+?)(?:\n|$)', 
                                 response, re.IGNORECASE)
        if primary_match:
            predicted_human_action = primary_match.group(1).strip()
        
        return predicted_human_action
    
    def parse_robot_action(self, robot_action: str, game_state: Optional[dict] = None, 
                          state = None) -> Tuple[str, any]:
        """
        Parse the robot action to get function name and item with state-aware location detection.
        Expected format: pickup(onion) or place(onion, chopping_station) or NOOP
        
        For pickup actions, uses game state to determine the appropriate location:
        - pickup(onion) checks onion_at_chopping, onion_at_sink, counter_tile, or defaults to dispenser
        - pickup(tomato) checks tomato_at_chopping, tomato_at_sink, counter_tile, or defaults to dispenser
        """
        # Check for explicit NOOP
        if robot_action == "NOOP":
            return "NOOP", None

        # Parse pickup actions with state-aware location detection
        pickup_match = re.search(r'pickup\(([^)]+)\)', robot_action)
        if pickup_match:
            item = pickup_match.group(1).strip()
            
            # State-aware location detection for onion/tomato
            if item == "onion" and game_state:
                location = self._detect_ingredient_location("onion", game_state, state)
                return "pickup", (item, location)
                
            elif item == "tomato" and game_state:
                location = self._detect_ingredient_location("tomato", game_state, state)
                return "pickup", (item, location)
            
            # State-aware location detection for dish
            elif item == "dish" and game_state:
                if state and self.tile_checker and self.tile_checker.is_dish_on_counter(state):
                    # Find the specific counter tile with the dish
                    counter_pos = self.tile_checker.find_dish_on_counter(state)
                    if counter_pos:
                        location = f"counter_tile_{counter_pos[0]}_{counter_pos[1]}"  # Specific position
                    else:
                        location = "counter_tile"  # Fallback to generic
                    return "pickup", (item, location)
                else:
                    return "pickup", item  # Default to dispenser
            elif item == "soup":
                return "pickup", item
                
            # Fallback for onion/tomato/dish without game state
            elif item in ["onion", "tomato", "dish"]:
                return "pickup", item

        # Parse place actions with destination
        place_match = re.search(r'place\(([^,]+),\s*([^)]+)\)', robot_action)
        if place_match:
            item = place_match.group(1).strip()
            destination = place_match.group(2).strip()
            if item in ["onion", "tomato", "dish"] and destination in ["chopping_station", "staging_station", "sink", "salt_station", "pepper_station", "counter_tile"]:
                return "place", (item, destination)

        # Parse place actions without destination (for items with fixed destinations)
        place_simple_match = re.search(r'place\(([^)]+)\)', robot_action)
        if place_simple_match:
            item = place_simple_match.group(1).strip()
            if item in ["dish", "soup"]:
                return "place", (item, "default")

        # final fallback
        return "pickup", "onion"
    
    def _detect_ingredient_location(self, ingredient: str, game_state: dict, state) -> str:
        """
        Detect the location of an ingredient based on game state.
        
        Args:
            ingredient: 'onion' or 'tomato'
            game_state: Dictionary with game state predicates
            state: Full game state object
            
        Returns:
            Location string (e.g., 'chopping_station', 'sink', 'dispenser')
        """
        if game_state.get(f'{ingredient}_at_chopping', False):
            return "chopping_station"
        elif game_state.get(f'{ingredient}_at_sink', False):
            return "sink"
        elif game_state.get(f'{ingredient}_at_salt_station', False):
            return "salt_station"
        elif game_state.get(f'{ingredient}_at_pepper_station', False):
            return "pepper_station"
        elif state and self.tile_checker and self.tile_checker.is_ingredient_on_counter(state, ingredient):
            # Find the specific counter tile with the ingredient
            counter_pos = self.tile_checker.find_ingredient_on_counter(state, ingredient)
            if counter_pos:
                return f"counter_tile_{counter_pos[0]}_{counter_pos[1]}"  # Specific position
            else:
                return "counter_tile"  # Fallback to generic
        else:
            return "dispenser"

