"""
Game Mechanics System for Overcooked AI

This module provides task-specific game mechanics for different recipe types.
Each recipe type has its own specialized mechanics while sharing common functionality.
"""

from .base_mechanics import BaseMechanics
from .mechanics_factory import get_mechanics_for_recipe, get_mechanics_for_task

__all__ = ['BaseMechanics', 'get_mechanics_for_recipe', 'get_mechanics_for_task']