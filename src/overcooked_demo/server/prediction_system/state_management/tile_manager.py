"""Manages counter tiles, staging stations, and frontiers."""

from typing import List, Tuple, Optional, Set


def compute_frontier(tiles: List[Tuple[int, int]], terrain) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Return the set of walkable tiles adjacent to any tile in 'tiles'."""
    H, W = len(terrain), len(terrain[0])
    frontier = set()
    WALKABLE = {' '}

    for c, r in tiles:
        for dc, dr in [(1,0), (-1,0), (0,1), (0,-1)]:
            nc, nr = c + dc, r + dr
            if (
                0 <= nr < H and 
                0 <= nc < W and 
                terrain[nr][nc] in WALKABLE
            ):
                orient = (-dc, -dr)
                frontier.add(((nc, nr), orient))
    return list(frontier)


class TileManager:
    """Manages all tile-related functionality including counters, staging, and checking."""
    
    def __init__(self):
        # Core tile locations
        self.ingredient_spawns = []
        self.onion_spawns = []
        self.tomato_spawns = []
        self.stove_tiles = []
        self.dish_spawns = []
        self.delivery_tiles = []
        
        # Staging and processing stations
        self.onion_staging_tiles = []
        self.tomato_staging_tiles = []
        self.dish_staging_tiles = []
        self.soup_staging_tiles = []
        self.onion_chopping_stations = []
        self.tomato_chopping_stations = []
        self.sink_stations = []
        self.salt_stations = []
        self.pepper_stations = []
        
        # Counter tiles (for dropping objects)
        self.counter_tiles = []
        
        # Frontiers (walkable tiles adjacent to important locations)
        self.ingredient_frontier = []
        self.onion_frontier = []
        self.tomato_frontier = []
        self.stove_frontier = []
        self.dish_frontier = []
        self.delivery_frontier = []
        self.onion_staging_frontier = []
        self.tomato_staging_frontier = []
        self.dish_staging_frontier = []
        self.soup_staging_frontier = []
        self.onion_chopping_frontier = []
        self.tomato_chopping_frontier = []
        self.sink_frontier = []
        self.salt_frontier = []
        self.pepper_frontier = []
        self.counter_frontier = []
    
    def initialize_from_mdp(self, mdp):
        """Initialize all tile locations from the MDP terrain."""
        terrain = mdp.terrain_mtx
        
        # Extract basic tiles from terrain
        self.ingredient_spawns = [(j, i)
                            for i, row in enumerate(terrain)
                            for j, c in enumerate(row)
                            if c in ('O', 'T')]
        self.onion_spawns = [(j, i)
                            for i, row in enumerate(terrain)
                            for j, c in enumerate(row)
                            if c == 'O']
        self.tomato_spawns = [(j, i)
                            for i, row in enumerate(terrain)
                            for j, c in enumerate(row)
                            if c == 'T']
        self.stove_tiles = [(j, i)
                            for i, row in enumerate(terrain)
                            for j, c in enumerate(row)
                            if c == 'P']
        self.dish_spawns = [(j, i)
                            for i, row in enumerate(terrain)
                            for j, c in enumerate(row)
                            if c == 'D']
        self.delivery_tiles = [(j, i)
                            for i, row in enumerate(terrain)
                            for j, c in enumerate(row)
                            if c == 'S']
        
        # Assign staging stations per stove with directional logic
        H, W = len(terrain), len(terrain[0]) 
        
        for (stove_c, stove_r) in self.stove_tiles:
            adjacent_staging = []
            for dc, dr in [(1,0), (-1,0), (0,1), (0,-1)]:
                nc, nr = stove_c + dc, stove_r + dr
                if 0 <= nr < H and 0 <= nc < W and terrain[nr][nc] == 'X':
                    adjacent_staging.append((nc, nr, dc, dr))
            
            left_stations = [(c, r) for c, r, dc, dr in adjacent_staging if dc == -1]
            right_stations = [(c, r) for c, r, dc, dr in adjacent_staging if dc == 1]  
            top_stations = [(c, r) for c, r, dc, dr in adjacent_staging if dr == -1]   
            bottom_stations = [(c, r) for c, r, dc, dr in adjacent_staging if dr == 1]
            
            if left_stations:
                self.onion_staging_tiles.extend(left_stations)
            elif bottom_stations:
                self.onion_staging_tiles.extend(bottom_stations)
            
            # Default: tomato staging same as onion staging
            self.tomato_staging_tiles = self.onion_staging_tiles.copy()
            self.dish_staging_tiles = self.onion_staging_tiles.copy() # Moving the dish staging tile to agent side
            self.soup_staging_tiles = self.dish_staging_tiles

        # Get layout name for special handling
        layout_name = getattr(mdp, 'layout_name', 'unknown')
        
        # Special handling for counter_circuit layout
        if layout_name == 'counter_circuit':
            # Override tomato staging tile to be at (2,2) for counter_circuit
            self.tomato_staging_tiles = [(2, 2)]
        elif layout_name == 'custom_counter_circuit':
            # Override tomato staging tile to be at (4,3) for custom_counter_circuit
            self.tomato_staging_tiles = [(4, 3)]
        elif layout_name == 'custom_cramped_room':
            # Override staging tiles: left staging at (1,0), right staging at (3,0) - both sides of stove at (2,0)
            self.onion_staging_tiles = [(1, 0)]  # Left side of stove
            self.tomato_staging_tiles = [(3, 0)]  # Right side of stove
        
        # Create chopping stations
        if layout_name == 'cramped_room_tomato':
            # For cramped_room_tomato: use hardcoded chopping station at (0,2)
            self.onion_chopping_stations = [(0, 2)]
            self.tomato_chopping_stations = [(0, 2)]
        elif layout_name == 'custom_cramped_room':
            # For custom_cramped_room: chopping station beneath dish station - dish at (0,1), chopping at (0,2)
            self.onion_chopping_stations = [(0, 2)]
            self.tomato_chopping_stations = [(0, 2)]
        else:
            # Use dynamic detection for other layouts
            for staging_pos in self.onion_staging_tiles:
                staging_c, staging_r = staging_pos
                for dc, dr in [(-1, 0), (0, -1)]:
                    chopping_c, chopping_r = staging_c + dc, staging_r + dr
                    if (0 <= chopping_r < H and 0 <= chopping_c < W and 
                        terrain[chopping_r][chopping_c] == 'X' and
                        (chopping_c, chopping_r) not in self.onion_chopping_stations):
                        self.onion_chopping_stations.append((chopping_c, chopping_r))
                        break
            
            self.tomato_chopping_stations = self.onion_chopping_stations.copy()

        # Create sink stations
        if layout_name == 'counter_circuit':
            # For counter_circuit: sink at (0,2)
            self.sink_stations.append((0, 2))
        elif layout_name == 'cramped_room_tomato':
            # For cramped_room_tomato: sink at (2,3)
            self.sink_stations.append((2, 3))
        elif layout_name == 'custom_counter_circuit':
            # For custom_counter_circuit: sink at (0,3)
            self.sink_stations.append((0, 3))
        elif layout_name == 'custom_cramped_room':
            # For custom_cramped_room: sink station beneath chopping station - chopping at (0,2), sink at (0,3)
            self.sink_stations.append((0, 3))
        
        # Create salt stations
        if layout_name == 'counter_circuit':
            # For counter_circuit: salt at (5,2)
            self.salt_stations.append((5, 2))
        elif layout_name == 'custom_counter_circuit':
            # For custom_counter_circuit: salt at (5,3) - counter tile in row 3
            self.salt_stations.append((5, 3))
        elif layout_name == 'custom_cramped_room':
            # For custom_cramped_room: salt at (4,2) - right side wall counter (layout is 5 columns, so col 4 is right wall)
            self.salt_stations.append((4, 2))
        
        # Create pepper stations
        if layout_name == 'counter_circuit':
            # For counter_circuit: pepper at (6,2)
            self.pepper_stations.append((6, 2))
        elif layout_name == 'custom_counter_circuit':
            # For custom_counter_circuit: pepper at (8,3) - counter tile in row 3
            self.pepper_stations.append((8, 3))
        elif layout_name == 'custom_cramped_room':
            # For custom_cramped_room: pepper at (4,3) - right side wall counter, below salt
            self.pepper_stations.append((4, 3))
        
        # Compute frontiers
        self.ingredient_frontier = compute_frontier(self.ingredient_spawns, terrain)
        self.onion_frontier = compute_frontier(self.onion_spawns, terrain)
        self.tomato_frontier = compute_frontier(self.tomato_spawns, terrain)
        self.stove_frontier = compute_frontier(self.stove_tiles, terrain) 
        self.dish_frontier = compute_frontier(self.dish_spawns, terrain)
        self.delivery_frontier = compute_frontier(self.delivery_tiles, terrain)
        self.onion_staging_frontier = compute_frontier(self.onion_staging_tiles, terrain)
        self.tomato_staging_frontier = compute_frontier(self.tomato_staging_tiles, terrain)
        self.dish_staging_frontier = compute_frontier(self.dish_staging_tiles, terrain)
        self.soup_staging_frontier = compute_frontier(self.soup_staging_tiles, terrain)
        self.onion_chopping_frontier = compute_frontier(self.onion_chopping_stations, terrain)
        self.tomato_chopping_frontier = compute_frontier(self.tomato_chopping_stations, terrain)
        self.sink_frontier = compute_frontier(self.sink_stations, terrain)
        self.salt_frontier = compute_frontier(self.salt_stations, terrain)
        self.pepper_frontier = compute_frontier(self.pepper_stations, terrain)
        
        # Compute counter tile frontier for dropping wrong objects
        self.counter_tiles = self._find_counter_tiles(terrain)
        self.counter_frontier = compute_frontier(self.counter_tiles, terrain)
    
    def _find_counter_tiles(self, terrain) -> List[Tuple[int, int]]:
        """
        Find all counter tiles (X) that are available for dropping objects.
        Excludes important counter tiles used for staging, chopping, and washing stations.
        """
        H, W = len(terrain), len(terrain[0])
        counter_tiles = []
        
        # Get all important counter tile positions to exclude
        important_tiles = set()
        important_tiles.update(self.onion_staging_tiles)
        important_tiles.update(self.tomato_staging_tiles)
        important_tiles.update(self.dish_staging_tiles)
        important_tiles.update(self.soup_staging_tiles)
        important_tiles.update(self.onion_chopping_stations)
        important_tiles.update(self.tomato_chopping_stations)
        important_tiles.update(self.sink_stations)
        important_tiles.update(self.salt_stations)
        important_tiles.update(self.pepper_stations)
        
        for row in range(H):
            for col in range(W):
                if terrain[row][col] == 'X':  # Counter tile
                    pos = (col, row)
                    # Only include if it's not an important counter tile
                    if pos not in important_tiles:
                        counter_tiles.append(pos)
        
        return counter_tiles
    
    def is_ingredient_on_counter(self, state, ingredient: str) -> bool:
        """
        Check if the specified ingredient is available on a counter tile using the full state.
        Excludes important counter tiles (staging, chopping, washing, stove stations).
        """
        safe_counter_tiles = self.counter_tiles
        sd = state.to_dict()
        tile_contents = {}
        
        for obj in sd["objects"]:
            p = tuple(obj["position"])
            name = obj.get("ingredient") or obj.get("name")
            tile_contents.setdefault(p, []).append(name)
        
        # Check if any safe counter tile has the ingredient
        for counter_pos in safe_counter_tiles:
            if ingredient in tile_contents.get(counter_pos, []):
                return True
        
        return False
    
    def find_ingredient_on_counter(self, state, ingredient: str) -> Optional[Tuple[int, int]]:
        """
        Find the specific counter tile position where the ingredient is located.
        Excludes important counter tiles (staging, chopping, washing, stove stations).
        """
        safe_counter_tiles = self.counter_tiles
        sd = state.to_dict()
        tile_contents = {}
        
        for obj in sd["objects"]:
            p = tuple(obj["position"])
            name = obj.get("ingredient") or obj.get("name")
            tile_contents.setdefault(p, []).append(name)
        
        # Find the specific counter tile with the ingredient
        for counter_pos in safe_counter_tiles:
            if ingredient in tile_contents.get(counter_pos, []):
                return counter_pos
        
        return None
    
    def is_dish_on_counter(self, state) -> bool:
        """
        Check if a dish is available on a counter tile using the full state.
        Excludes important counter tiles (staging, chopping, washing, stove stations).
        """
        safe_counter_tiles = self.counter_tiles
        sd = state.to_dict()
        tile_contents = {}
        
        for obj in sd["objects"]:
            p = tuple(obj["position"])
            name = obj.get("ingredient") or obj.get("name")
            tile_contents.setdefault(p, []).append(name)
        
        # Check if any safe counter tile has a dish
        for counter_pos in safe_counter_tiles:
            if "dish" in tile_contents.get(counter_pos, []):
                return True
        
        return False
    
    def find_dish_on_counter(self, state) -> Optional[Tuple[int, int]]:
        """
        Find the specific counter tile position where a dish is located.
        Excludes important counter tiles (staging, chopping, washing, stove stations).
        """
        safe_counter_tiles = self.counter_tiles
        sd = state.to_dict()
        tile_contents = {}
        
        for obj in sd["objects"]:
            p = tuple(obj["position"])
            name = obj.get("ingredient") or obj.get("name")
            tile_contents.setdefault(p, []).append(name)
        
        # Find the specific counter tile with the dish
        for counter_pos in safe_counter_tiles:
            if "dish" in tile_contents.get(counter_pos, []):
                return counter_pos
        
        return None

