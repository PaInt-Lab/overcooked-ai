"""
Action Coordination System

This package handles human-robot action coordination:
- Secondary action selection based on human predictions
- Action mapping and coordination logic
"""

from .secondary_action_selector import select_secondary_action, PrimaryAction

__all__ = [
    'select_secondary_action',
    'PrimaryAction'
]
