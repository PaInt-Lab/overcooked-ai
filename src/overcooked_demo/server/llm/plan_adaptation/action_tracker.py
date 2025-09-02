"""
Primary Action Sequence Tracker

Tracks the sequence of primary actions during soup creation and serving.
Handles deduplication of consecutive identical actions.
"""

import time
from typing import List, Dict, Optional


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
    
    def record_action(self, action: str) -> None:
        """
        Record a primary action if it's different from the last one.
        
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
    
    def set_recipe_type(self, recipe_type: str) -> None:
        """Set the recipe type for the current sequence."""
        self.recipe_type = recipe_type
    



