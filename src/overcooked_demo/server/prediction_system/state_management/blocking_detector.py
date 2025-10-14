"""Detects and handles blocking situations."""

from typing import Tuple, List, Optional


class BlockingDetector:
    """Detects when the agent is blocking important tiles and finds safe positions."""
    
    def __init__(self, tile_manager=None, mdp=None):
        """
        Initialize blocking detector.
        
        Args:
            tile_manager: TileManager instance for accessing tile locations
            mdp: MDP instance for terrain access
        """
        self.tile_manager = tile_manager
        self.mdp = mdp
    
    def is_blocking_important_tile(self, my_pos: Tuple[int, int], state) -> bool:
        """
        Check if the agent is currently blocking an important staging tile.
        Returns True if blocking, False otherwise.
        """
        if not self.tile_manager:
            return False
            
        # Get all important tiles that shouldn't be blocked
        important_tiles = set()
        
        # Add all staging tiles
        important_tiles.update(self.tile_manager.onion_staging_tiles)
        important_tiles.update(self.tile_manager.tomato_staging_tiles)
        important_tiles.update(self.tile_manager.dish_staging_tiles)
        important_tiles.update(self.tile_manager.soup_staging_tiles)
        
        # Add all chopping stations
        important_tiles.update(self.tile_manager.onion_chopping_stations)
        important_tiles.update(self.tile_manager.tomato_chopping_stations)
        
        # Add all sink stations
        important_tiles.update(self.tile_manager.sink_stations)
        
        # Add all seasoning stations
        important_tiles.update(self.tile_manager.salt_stations)
        important_tiles.update(self.tile_manager.pepper_stations)
        
        # Add all frontier tiles (adjacent to important locations)
        important_tiles.update([pos for pos, _ in self.tile_manager.ingredient_frontier])
        important_tiles.update([pos for pos, _ in self.tile_manager.onion_frontier])
        important_tiles.update([pos for pos, _ in self.tile_manager.tomato_frontier])
        important_tiles.update([pos for pos, _ in self.tile_manager.stove_frontier])
        important_tiles.update([pos for pos, _ in self.tile_manager.dish_frontier]) 
        important_tiles.update([pos for pos, _ in self.tile_manager.delivery_frontier])
        important_tiles.update([pos for pos, _ in self.tile_manager.onion_staging_frontier])
        important_tiles.update([pos for pos, _ in self.tile_manager.tomato_staging_frontier])
        important_tiles.update([pos for pos, _ in self.tile_manager.dish_staging_frontier])
        important_tiles.update([pos for pos, _ in self.tile_manager.soup_staging_frontier])
        important_tiles.update([pos for pos, _ in self.tile_manager.onion_chopping_frontier])
        important_tiles.update([pos for pos, _ in self.tile_manager.tomato_chopping_frontier])
        important_tiles.update([pos for pos, _ in self.tile_manager.sink_frontier])
        important_tiles.update([pos for pos, _ in self.tile_manager.salt_frontier])
        important_tiles.update([pos for pos, _ in self.tile_manager.pepper_frontier])
        
        # Check if current position is blocking an important tile
        is_blocking = my_pos in important_tiles
        
        return is_blocking
    
    def find_safe_position(self, my_pos: Tuple[int, int], state, agent_index: int, 
                          get_action_plan_func) -> Tuple[Tuple[int, int], List]:
        """
        Find a safe position to move to that doesn't block important tiles.
        Returns (new_position, action_plan) or (my_pos, []) if no safe move found.
        
        Args:
            my_pos: Current agent position
            state: Full game state
            agent_index: Agent index for accessing player data
            get_action_plan_func: Function to generate action plan between positions
        """
        if not self.mdp or not self.tile_manager:
            return my_pos, []
            
        terrain = self.mdp.terrain_mtx
        H, W = len(terrain), len(terrain[0])
        
        # Get all important tiles to avoid
        important_tiles = set()
        important_tiles.update(self.tile_manager.onion_staging_tiles)
        important_tiles.update(self.tile_manager.tomato_staging_tiles)
        important_tiles.update(self.tile_manager.dish_staging_tiles)
        important_tiles.update(self.tile_manager.soup_staging_tiles)
        important_tiles.update(self.tile_manager.onion_chopping_stations)
        important_tiles.update(self.tile_manager.tomato_chopping_stations)
        important_tiles.update(self.tile_manager.sink_stations)
        important_tiles.update(self.tile_manager.salt_stations)
        important_tiles.update(self.tile_manager.pepper_stations)
        important_tiles.update([pos for pos, _ in self.tile_manager.ingredient_frontier])
        important_tiles.update([pos for pos, _ in self.tile_manager.onion_frontier])
        important_tiles.update([pos for pos, _ in self.tile_manager.tomato_frontier])
        important_tiles.update([pos for pos, _ in self.tile_manager.stove_frontier])
        important_tiles.update([pos for pos, _ in self.tile_manager.dish_frontier])
        important_tiles.update([pos for pos, _ in self.tile_manager.delivery_frontier])
        important_tiles.update([pos for pos, _ in self.tile_manager.onion_staging_frontier])
        important_tiles.update([pos for pos, _ in self.tile_manager.tomato_staging_frontier])
        important_tiles.update([pos for pos, _ in self.tile_manager.dish_staging_frontier])
        important_tiles.update([pos for pos, _ in self.tile_manager.soup_staging_frontier])
        important_tiles.update([pos for pos, _ in self.tile_manager.onion_chopping_frontier])
        important_tiles.update([pos for pos, _ in self.tile_manager.tomato_chopping_frontier])
        important_tiles.update([pos for pos, _ in self.tile_manager.sink_frontier])
        important_tiles.update([pos for pos, _ in self.tile_manager.salt_frontier])
        important_tiles.update([pos for pos, _ in self.tile_manager.pepper_frontier])
        
        # Get other player position to avoid blocking them
        other_player_pos = state.player_positions[1 - agent_index]
        
        # Find safe positions within reasonable distance (max 3 steps)
        safe_positions = []
        for distance in range(1, 4):  # Check 1, 2, 3 steps away
            for dc in range(-distance, distance + 1):
                for dr in range(-distance, distance + 1):
                    if abs(dc) + abs(dr) == distance:  # Manhattan distance
                        new_col = my_pos[0] + dc
                        new_row = my_pos[1] + dr
                        
                        # Check bounds
                        if 0 <= new_row < H and 0 <= new_col < W:
                            new_pos = (new_col, new_row)
                            
                            # Check if position is walkable and not important
                            if (terrain[new_row][new_col] == ' ' and 
                                new_pos not in important_tiles and
                                new_pos != other_player_pos):
                                safe_positions.append(new_pos)
            
            # If we found safe positions at this distance, stop searching
            if safe_positions:
                break
        
        # If no safe positions found, stay put
        if not safe_positions:
            return my_pos, []
        
        # Choose the closest safe position
        best_pos = min(safe_positions, key=lambda pos: abs(pos[0] - my_pos[0]) + abs(pos[1] - my_pos[1]))
        
        # Generate action plan to move to safe position
        my_ori = state.to_dict()["players"][agent_index]["orientation"]
        start_pair = (my_pos, tuple(my_ori))
        goal_pair = (best_pos, tuple(my_ori))  # Keep same orientation
        
        action_plan = get_action_plan_func(start_pair, goal_pair)
        
        return best_pos, action_plan

