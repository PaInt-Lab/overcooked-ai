"""
Plan Repository for Successful Primary Action Sequences

Stores and manages successful primary action sequences for plan adaptation.
Provides access to historical plans for LLM prompt enhancement.
"""

import time
from typing import List, Dict, Optional


class PlanRepository:
    """
    Repository for storing successful primary action sequences.
    
    Features:
    - Store successful plans with incremental sequence IDs
    - Retrieve all plans for LLM prompt enhancement
    - Session-based storage (no persistence)
    """
    
    def __init__(self):
        """Initialize an empty plan repository."""
        self.successful_plans: List[Dict] = []
        self.next_sequence_id: int = 1
    
    def add_successful_plan(self, plan: Dict) -> None:
        """
        Add a successful plan to the repository.
        
        Args:
            plan: Plan dictionary with actions, recipe_type, etc.
        """
        # Ensure the plan has a sequence ID
        if 'sequence_id' not in plan:
            plan['sequence_id'] = self.next_sequence_id
            self.next_sequence_id += 1
        
        # Add completion timestamp if not present
        if 'completed_at' not in plan:
            plan['completed_at'] = time.time()
        
        # Store the plan
        self.successful_plans.append(plan)
    
    def get_all_plans(self) -> List[Dict]:
        """
        Get all successful plans stored in the repository.
        
        Returns:
            List of all successful plans
        """
        return self.successful_plans.copy()
    
    def get_plans_by_recipe_type(self, recipe_type: str) -> List[Dict]:
        """
        Get plans filtered by recipe type.
        
        Args:
            recipe_type: Recipe type to filter by
            
        Returns:
            List of plans matching the recipe type
        """
        return [plan for plan in self.successful_plans if plan.get('recipe_type') == recipe_type]
    
    def get_recent_plans(self, count: int) -> List[Dict]:
        """
        Get the most recent successful plans.
        
        Args:
            count: Number of recent plans to return
            
        Returns:
            List of most recent plans (up to count)
        """
        return self.successful_plans[-count:] if self.successful_plans else []
    
    def get_plan_count(self) -> int:
        """Get the total number of successful plans stored."""
        return len(self.successful_plans)
    
    def is_empty(self) -> bool:
        """Check if the repository has no plans."""
        return len(self.successful_plans) == 0
    
    def reset(self) -> None:
        """Reset the repository to empty state."""
        self.successful_plans.clear()
        self.next_sequence_id = 1
    



