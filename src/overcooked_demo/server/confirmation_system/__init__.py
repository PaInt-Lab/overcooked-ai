"""
Confirmation System Module

Handles human confirmation button system for primary actions (chop, wash, salt, pepper).
This module enables measurement of Human TES (Trajectory Efficiency Score) by requiring
explicit human confirmation for processing actions.
"""

from .confirmation_manager import ConfirmationManager, PendingConfirmation

__all__ = ['ConfirmationManager', 'PendingConfirmation']
