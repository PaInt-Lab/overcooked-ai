"""
Primary Action Sequence Tracker

Tracks the sequence of primary actions during soup creation and serving.
Handles deduplication of consecutive identical actions.
"""

import time
from typing import List, Dict, Optional

# State to action mapping
STATE_ACTION_MAP = {
    'onion_chopped': 'Chop Onion',
    'onion_washed': 'Wash Onion',
    'onion_staged': 'Stage Onion',
    'onion_in_pot': 'Place Onion in Pot',
    'tomato_chopped': 'Chop Tomato',
    'tomato_washed': 'Wash Tomato',
    'tomato_staged': 'Stage Tomato',
    'tomato_in_pot': 'Place Tomato in Pot'
}

# Additional action mappings for human grab actions (when human picks up staged items)
HUMAN_GRAB_ACTION_MAP = {
    'onion_hand': 'Human Grab Onion',  # When human picks up staged onion
    'tomato_hand': 'Human Grab Tomato'  # When human picks up staged tomato
}


class ActionTracker:
    """
    Tracks the current sequence of primary actions for plan adaptation.
    
    Features:
    - Deduplication of consecutive identical actions
    - Timestamp tracking for each action
    """
    
    def __init__(self):
        """Initialize an empty action tracker."""
        self.current_sequence: List[Dict] = []
        self.last_action: Optional[str] = None
        self.sequence_start_time: Optional[float] = None
        self.recipe_type: Optional[str] = None
        
        # Mode tracking
        self.sequential_mode: bool = False
        
        # State recording flags
        self.onion_chopped_recorded: bool = False
        self.onion_washed_recorded: bool = False
        self.onion_staged_recorded: bool = False
        self.onion_in_pot_recorded: bool = False
        self.tomato_chopped_recorded: bool = False
        self.tomato_washed_recorded: bool = False
        self.tomato_staged_recorded: bool = False
        self.tomato_in_pot_recorded: bool = False
        
        # Human grab action flags
        self.human_grab_onion_recorded: bool = False
        self.human_grab_tomato_recorded: bool = False
    
    def record_action(self, action: str, game_state: dict) -> None:
        """
        Record a primary action based on current mode (state-based or sequential).
        
        Args:
            action: The predicted primary action string
            game_state: Current game state dictionary
        """
        # Don't record actions when soup is already served
        if game_state.get('soup_served', False):
            return
        
        # Check if we should switch to sequential mode
        if (game_state.get('onion_in_pot', False) and 
            game_state.get('tomato_in_pot', False)):
            self.sequential_mode = True
        
        if not self.sequential_mode:
            # Phase 1: State-based recording
            self._record_state_achievements(game_state)
        else:
            # Phase 2: Sequential recording
            self._record_sequential_action(action)
    
    def _record_state_achievements(self, game_state: dict) -> None:
        """
        Record actions based on state achievements (Phase 1).
        
        Args:
            game_state: Current game state dictionary
        """
        # Initialize sequence start time if this is the first action
        if not self.sequence_start_time:
            self.sequence_start_time = time.time()
        
        # Check each state and record if first time becoming true
        for state_key, action_string in STATE_ACTION_MAP.items():
            if game_state.get(state_key, False):
                # Check if we haven't recorded this state yet
                recorded_flag = f"{state_key}_recorded"
                if not getattr(self, recorded_flag, False):
                    # Record the action with timestamp
                    action_record = {
                        'action': action_string,
                        'timestamp': time.time()
                    }
                    
                    self.current_sequence.append(action_record)
                    setattr(self, recorded_flag, True)
                    self.last_action = action_string
        
        # Check for human grab actions (when human picks up staged items)
        for hand_key, action_string in HUMAN_GRAB_ACTION_MAP.items():
            if game_state.get(hand_key) == "partner":  # Human is holding the item
                # Check if we haven't recorded this grab action yet
                recorded_flag = f"human_grab_{hand_key.split('_')[0]}_recorded"
                if not getattr(self, recorded_flag, False):
                    # Record the action with timestamp
                    action_record = {
                        'action': action_string,
                        'timestamp': time.time()
                    }
                    
                    self.current_sequence.append(action_record)
                    setattr(self, recorded_flag, True)
                    self.last_action = action_string
    
    def _record_sequential_action(self, action: str) -> None:
        """
        Record actions sequentially with deduplication (Phase 2).
        
        Args:
            action: The predicted primary action string
        """
        # Only record if different from last action (deduplication)
        if action != self.last_action:
            # Initialize sequence start time if this is the first action
            if not self.sequence_start_time:
                self.sequence_start_time = time.time()
            
            # Record the action with timestamp
            action_record = {
                'action': action,
                'timestamp': time.time()
            }
            
            self.current_sequence.append(action_record)
            self.last_action = action
    
    def get_current_sequence(self) -> List[str]:
        """
        Get the current sequence of actions (just the action names).
        
        Returns:
            List of action strings in sequence order
        """
        return [step['action'] for step in self.current_sequence]
    
    def get_current_sequence_with_timestamps(self) -> List[Dict]:
        """
        Get the current sequence with timestamps.
        
        Returns:
            List of action records with timestamps
        """
        return self.current_sequence.copy()
    
    def is_empty(self) -> bool:
        """Check if the tracker has no recorded actions."""
        return len(self.current_sequence) == 0
    
    def get_sequence_duration(self) -> float:
        """
        Get the duration of the current sequence.
        
        Returns:
            Duration in seconds, or 0 if no sequence
        """
        if not self.sequence_start_time or not self.current_sequence:
            return 0.0
        return time.time() - self.sequence_start_time
    
    def reset(self) -> None:
        """Reset the tracker to empty state."""
        self.current_sequence.clear()
        self.last_action = None
        self.sequence_start_time = None
        self.recipe_type = None
        
        # Reset mode tracking
        self.sequential_mode = False
        
        # Reset state recording flags
        self.onion_chopped_recorded = False
        self.onion_washed_recorded = False
        self.onion_staged_recorded = False
        self.onion_in_pot_recorded = False
        self.tomato_chopped_recorded = False
        self.tomato_washed_recorded = False
        self.tomato_staged_recorded = False
        self.tomato_in_pot_recorded = False
        
        # Reset human grab action flags
        self.human_grab_onion_recorded = False
        self.human_grab_tomato_recorded = False
    
    def set_recipe_type(self, recipe_type: str) -> None:
        """Set the recipe type for the current sequence."""
        self.recipe_type = recipe_type
    



