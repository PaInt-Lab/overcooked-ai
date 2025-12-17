"""
Confirmation Manager

Manages the human confirmation button system for primary actions.
Handles detection, execution, and clearing of confirmation states.
"""

from dataclasses import dataclass
from typing import Optional, Dict, List
from overcooked_ai_py.mdp.actions import Action


@dataclass
class PendingConfirmation:
    """
    Represents a pending human confirmation for a primary action.

    When an ingredient is placed at a processing station (chop, wash, salt, pepper),
    this dataclass tracks the confirmation state until the human clicks the button.
    """
    action_type: str  # 'chop', 'wash', 'salt', 'pepper'
    ingredient_name: str  # 'onion', 'tomato'
    station_location: tuple  # (x, y) grid position
    placed_by: str  # 'agent' or 'human'
    placed_at_timestep: int
    ingredient_id: str  # unique identifier


class ConfirmationManager:
    """
    Manages the confirmation button system for a game.

    Responsibilities:
    - Detect ingredients at processing stations requiring confirmation
    - Track who placed ingredients (agent or human)
    - Execute confirmed actions (mutate object state)
    - Clear confirmations when ingredients are removed
    - Maintain confirmation history for analysis
    """

    def __init__(self, game):
        """
        Initialize the confirmation manager.

        Args:
            game: Reference to the OvercookedGame instance
        """
        self.game = game
        self.pending_confirmation: Optional[PendingConfirmation] = None
        self.last_placement: Dict[str, dict] = {}

    def detect_ingredient_at_station(self) -> Optional[PendingConfirmation]:
        """
        Detect if a processable ingredient is at a processing station.

        Returns:
            PendingConfirmation if ingredient needs confirmation, None otherwise
        """
        # Skip if confirmation already pending
        if self.pending_confirmation is not None:
            return None

        # Get station locations from agent's tile manager
        # The agent has references to all station positions from MDP
        agent = self._get_llm_agent()
        if not agent:
            return None

        chop_stations = getattr(agent, 'onion_chopping_stations', []) + getattr(agent, 'tomato_chopping_stations', [])
        sink_stations = getattr(agent, 'sink_stations', [])
        salt_stations = getattr(agent, 'salt_stations', [])
        pepper_stations = getattr(agent, 'pepper_stations', [])

        # Check chopping stations
        for station_pos in chop_stations:
            if self.game.state.has_object(station_pos):
                obj = self.game.state.get_object(station_pos)
                if obj.name in ['onion', 'tomato']:
                    if not hasattr(obj, 'state') or obj.state != 'chopped':
                        return self._create_pending_confirmation(
                            'chop', obj.name, station_pos, id(obj)
                        )

        # Check sink stations (washing)
        for station_pos in sink_stations:
            if self.game.state.has_object(station_pos):
                obj = self.game.state.get_object(station_pos)
                if obj.name in ['onion', 'tomato']:
                    if not hasattr(obj, 'state') or obj.state != 'washed':
                        return self._create_pending_confirmation(
                            'wash', obj.name, station_pos, id(obj)
                        )

        # Check salt stations
        for station_pos in salt_stations:
            if self.game.state.has_object(station_pos):
                obj = self.game.state.get_object(station_pos)
                if obj.name in ['onion', 'tomato']:
                    # Check if not already salted
                    if not (hasattr(obj, 'properties') and 'salted' in obj.properties):
                        return self._create_pending_confirmation(
                            'salt', obj.name, station_pos, id(obj)
                        )

        # Check pepper stations
        for station_pos in pepper_stations:
            if self.game.state.has_object(station_pos):
                obj = self.game.state.get_object(station_pos)
                if obj.name in ['onion', 'tomato']:
                    # Check if not already peppered
                    if not (hasattr(obj, 'properties') and 'peppered' in obj.properties):
                        return self._create_pending_confirmation(
                            'pepper', obj.name, station_pos, id(obj)
                        )

        return None

    def _create_pending_confirmation(self, action_type: str, ingredient: str,
                                     pos: tuple, obj_id: int) -> PendingConfirmation:
        """
        Helper to create PendingConfirmation with placement tracking.

        Args:
            action_type: Type of action ('chop', 'wash', 'salt', 'pepper')
            ingredient: Ingredient name ('onion', 'tomato')
            pos: Station position (x, y)
            obj_id: Object ID for tracking

        Returns:
            PendingConfirmation instance
        """
        # Determine who placed this ingredient
        placed_by = 'unknown'
        if pos in self.last_placement:
            placed_by = self.last_placement[pos].get('player', 'unknown')

        return PendingConfirmation(
            action_type=action_type,
            ingredient_name=ingredient,
            station_location=pos,
            placed_by=placed_by,
            placed_at_timestep=self.game.curr_tick,
            ingredient_id=str(obj_id)
        )

    def clear_if_ingredient_removed(self) -> bool:
        """
        Check if confirmation should be cleared because ingredient was removed.

        Returns:
            True if confirmation should be cleared, False otherwise
        """
        if not self.pending_confirmation:
            return False

        # Check if object still exists at station
        if not self.game.state.has_object(self.pending_confirmation.station_location):
            return True

        obj = self.game.state.get_object(self.pending_confirmation.station_location)

        # Check if it's the right type of ingredient
        if obj.name != self.pending_confirmation.ingredient_name:
            # Different ingredient type - clear confirmation
            return True

        # Already processed - clear confirmation
        if self.pending_confirmation.action_type == 'chop':
            if hasattr(obj, 'state') and obj.state == 'chopped':
                return True
        elif self.pending_confirmation.action_type == 'wash':
            if hasattr(obj, 'state') and obj.state == 'washed':
                return True
        elif self.pending_confirmation.action_type == 'salt':
            if hasattr(obj, 'properties') and 'salted' in obj.properties:
                return True
        elif self.pending_confirmation.action_type == 'pepper':
            if hasattr(obj, 'properties') and 'peppered' in obj.properties:
                return True

        # Ingredient is still there and not processed - keep confirmation
        return False

    def execute_confirmation_action(self):
        """Execute the confirmed processing action by mutating object state."""
        if not self.pending_confirmation:
            return

        # Check if object still exists at station
        if not self.game.state.has_object(self.pending_confirmation.station_location):
            print("[ConfirmationManager] Confirmation cancelled - ingredient removed")
            return

        obj = self.game.state.get_object(self.pending_confirmation.station_location)
        if obj.name != self.pending_confirmation.ingredient_name:
            print("[ConfirmationManager] Confirmation cancelled - ingredient changed")
            return

        # Apply transformation based on action type
        # This mutates the object state, which StateSummarizer will detect next tick
        if self.pending_confirmation.action_type == 'chop':
            obj.state = 'chopped'
            print(f"[ConfirmationManager] Chopped {obj.name} at {self.pending_confirmation.station_location}")
        elif self.pending_confirmation.action_type == 'wash':
            obj.state = 'washed'
            print(f"[ConfirmationManager] Washed {obj.name} at {self.pending_confirmation.station_location}")
        elif self.pending_confirmation.action_type == 'salt':
            if not hasattr(obj, 'properties'):
                obj.properties = []
            if 'salted' not in obj.properties:
                obj.properties.append('salted')
            print(f"[ConfirmationManager] Salted {obj.name} at {self.pending_confirmation.station_location}")
        elif self.pending_confirmation.action_type == 'pepper':
            if not hasattr(obj, 'properties'):
                obj.properties = []
            if 'peppered' not in obj.properties:
                obj.properties.append('peppered')
            print(f"[ConfirmationManager] Peppered {obj.name} at {self.pending_confirmation.station_location}")

        # Mark confirmation as completed and queue UI dismissal
        if getattr(self.game, "pending_confirmation_event", None) is None:
            # Only emit a dismissal if one is not already queued
            self.game.pending_confirmation_event = {"type": "dismissed"}
        self.pending_confirmation = None

    def track_placement(self, player_idx: int, action, state, curr_tick: int, human_players: set):
        """
        Track ingredient placement for confirmation system.

        Args:
            player_idx: Index of player performing action
            action: Action being performed
            state: Current game state
            curr_tick: Current game tick
            human_players: Set of human player IDs
        """
        if action == Action.INTERACT:
            player_pos = state.players[player_idx].position
            # Check if placing ingredient at station
            if state.has_object(player_pos):
                obj_at_pos = state.get_object(player_pos)
                player_id = self.game.players[player_idx]
                self.last_placement[player_pos] = {
                    'player': 'human' if player_id in human_players else 'agent',
                    'timestep': curr_tick,
                    'object_id': str(id(obj_at_pos))
                }

    def _get_llm_agent(self):
        """Get the LLM agent from game's NPC policies."""
        for policy in self.game.npc_policies.values():
            # Check if it's the CoordinatedActionPredictorAgent
            if hasattr(policy, 'onion_chopping_stations'):
                return policy
        return None

    def reset(self):
        """Reset the confirmation manager to initial state."""
        self.pending_confirmation = None
        self.last_placement.clear()
