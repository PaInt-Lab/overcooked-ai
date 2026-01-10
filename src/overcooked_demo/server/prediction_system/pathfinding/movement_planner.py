"""Movement planning and action execution."""

from typing import List, Optional, Tuple
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.planning.planners import MotionPlanner
from .bfs_planner import bfs_fallback


class MovementPlanner:
    """Handles movement planning and action execution for the agent."""
    
    def __init__(self, mdp=None, planner=None):
        self.mdp = mdp
        self.planner = planner
        
        # Frontier and tile references (will be set by agent)
        self.onion_frontier = []
        self.tomato_frontier = []
        self.dish_frontier = []
        self.soup_staging_frontier = []
        self.onion_chopping_frontier = []
        self.tomato_chopping_frontier = []
        self.onion_staging_frontier = []
        self.tomato_staging_frontier = []
        self.dish_staging_frontier = []
        self.sink_frontier = []
        self.salt_frontier = []
        self.pepper_frontier = []
        self.delivery_frontier = []
        self.counter_frontier = []
    
    def get_action_plan(self, start_pair, goal_pair):
        """Get action plan between two position/orientation pairs."""
        if self.mdp is None:
            return []
        terrain = self.mdp.terrain_mtx

        if self.planner is None:
            start_pos, start_ori = start_pair
            goal_pos, goal_ori = goal_pair
            return bfs_fallback(start_pos, goal_pos, terrain, goal_ori)
        
        try:
            action_plan, _, _ = self.planner.get_plan(start_pair, goal_pair)
            return action_plan
        except KeyError:
            try:
                goal_pos, goal_ori = goal_pair
                action_plan, _, _ = self.planner.action_plan_from_positions(
                    [goal_pos], start_pair, goal_pair
                )
                return action_plan
            except Exception:
                start_pos, start_ori = start_pair
                goal_pos, goal_ori = goal_pair
                action_plan = bfs_fallback(start_pos, goal_pos, terrain, goal_ori)
                return action_plan
    
    def move_to(self, action: str, item_info, start_pos: tuple, start_ori: tuple, destination: str = None):
        """Move to the appropriate location for the given action and item."""
        if action == "pickup":
            # Handle new tuple format (item, location) or legacy format (just item)
            if isinstance(item_info, tuple):
                item, location = item_info
                if location == "chopping_station":
                    choices = self.onion_chopping_frontier if item == "onion" else self.tomato_chopping_frontier
                elif location == "sink":
                    choices = self.sink_frontier
                elif location == "salt_station":
                    choices = self.salt_frontier
                elif location == "pepper_station":
                    choices = self.pepper_frontier
                elif location == "counter_tile":
                    # Use counter tiles for picking up ingredients
                    choices = self.counter_frontier
                elif location.startswith("counter_tile_"):
                    # Use specific counter tile position
                    try:
                        # Parse position from location string (e.g., "counter_tile_3_4")
                        parts = location.split("_")
                        if len(parts) == 4 and parts[0] == "counter" and parts[1] == "tile":
                            x, y = int(parts[2]), int(parts[3])
                            specific_pos = (x, y)
                            # Find frontier tiles adjacent to this specific counter tile
                            from ..state_management.tile_manager import compute_frontier
                            choices = compute_frontier([specific_pos], self.mdp.terrain_mtx)
                        else:
                            choices = self.counter_frontier  # Fallback
                    except (ValueError, IndexError):
                        choices = self.counter_frontier  # Fallback
                elif location == "dispenser":
                    choices = self.onion_frontier if item == "onion" else self.tomato_frontier
                else:
                    choices = None
            else:
                # Legacy format: just item name
                item = item_info
                frontier_map = {
                    "onion": self.onion_frontier,
                    "tomato": self.tomato_frontier,
                    "dish": self.dish_frontier,
                    "soup": self.soup_staging_frontier,
                }
                choices = frontier_map.get(item)
        elif action == "place":
            # Extract item name from tuple format if needed
            if isinstance(item_info, tuple):
                item = item_info[0]  # Extract item from (item, location) tuple
            else:
                item = item_info
                
            if destination == "chopping_station":
                frontier_map = {
                    "onion": self.onion_chopping_frontier,
                    "tomato": self.tomato_chopping_frontier,
                }
                choices = frontier_map.get(item)
            elif destination == "staging_station":
                frontier_map = {
                    "onion": self.onion_staging_frontier,
                    "tomato": self.tomato_staging_frontier,
                    "dish": self.dish_staging_frontier,
                }
                choices = frontier_map.get(item)
            elif destination == "sink":
                frontier_map = {
                    "onion": self.sink_frontier,
                    "tomato": self.sink_frontier,
                }
                choices = frontier_map.get(item)
            elif destination == "salt_station":
                frontier_map = {
                    "onion": self.salt_frontier,
                    "tomato": self.salt_frontier,
                }
                choices = frontier_map.get(item)
            elif destination == "pepper_station":
                frontier_map = {
                    "onion": self.pepper_frontier,
                    "tomato": self.pepper_frontier,
                }
                choices = frontier_map.get(item)
            elif destination == "counter_tile":
                # Use counter tiles for dropping wrong objects
                choices = self.counter_frontier
            elif destination.startswith("counter_tile_"):
                # Use specific counter tile position
                try:
                    # Parse position from location string (e.g., "counter_tile_3_4")
                    parts = destination.split("_")
                    if len(parts) == 4 and parts[0] == "counter" and parts[1] == "tile":
                        x, y = int(parts[2]), int(parts[3])
                        specific_pos = (x, y)
                        # Find frontier tiles adjacent to this specific counter tile
                        from ..state_management.tile_manager import compute_frontier
                        if self.mdp:
                            choices = compute_frontier([specific_pos], self.mdp.terrain_mtx)
                        else:
                            choices = self.counter_frontier  # Fallback
                    else:
                        choices = self.counter_frontier  # Fallback
                except (ValueError, IndexError):
                    choices = self.counter_frontier  # Fallback
            else:
                frontier_map = {
                    "dish": self.dish_staging_frontier,
                    "soup": self.delivery_frontier,
                }
                choices = frontier_map.get(item)
        else:
            return []
        
        if not choices:
            return []
        
        goal = min(choices, key=lambda mo: abs(mo[0][0] - start_pos[0]) + abs(mo[0][1] - start_pos[1]))
        start_pair = (start_pos, tuple(start_ori))
        return self.get_action_plan(start_pair, goal)
    
    def pickup(self, item_info, start_pos, start_ori):
        """Returns an action plan to pick up the specified item (supports both tuple and string format)."""
        action_plan = self.move_to("pickup", item_info, start_pos, start_ori)
        # Only add INTERACT if it's not already in the plan
        if action_plan and Action.INTERACT not in action_plan:
            action_plan.append(Action.INTERACT)
        return action_plan
    
    def place(self, item, start_pos, start_ori, destination: str = "default"):
        """Returns an action plan to place the specified item at the destination."""
        if destination == "default":
            action_plan = self.move_to("place", item, start_pos, start_ori, destination)
        else:
            action_plan = self.move_to("place", item, start_pos, start_ori, destination)
        
        # Only add INTERACT if it's not already in the plan
        if action_plan and Action.INTERACT not in action_plan:
            action_plan.append(Action.INTERACT)
        return action_plan

