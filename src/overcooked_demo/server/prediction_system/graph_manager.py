"""
Global singleton manager for the complete state graph.

This module manages the lifecycle of the complete state graph, ensuring it's
loaded once at server startup and shared across all agents. This prevents
blocking during game creation and improves performance.

The singleton pattern with thread-safe locking ensures that even if multiple
agents try to access the graph simultaneously, only one instance is created.
"""

import threading
from .complete_state_graph import CompleteStateGraphGenerator

# Global singleton instance of the graph (loaded once when first accessed)
_GLOBAL_GRAPH_INSTANCE = None
_GLOBAL_GRAPH_LOCK = threading.Lock()


def get_global_graph():
    """
    Get or create the global singleton graph instance.
    
    This function ensures the graph is loaded only once across all agents.
    Subsequent calls return the cached instance immediately.
    
    Thread-safe: Multiple simultaneous calls will block until the graph
    is loaded, then all will receive the same instance.
    
    Returns:
        CompleteRecipeGraph: The global shared state graph instance
    """
    global _GLOBAL_GRAPH_INSTANCE
    
    with _GLOBAL_GRAPH_LOCK:
        if _GLOBAL_GRAPH_INSTANCE is None:
            print("Loading global complete state graph (one-time initialization)...")
            generator = CompleteStateGraphGenerator()
            _GLOBAL_GRAPH_INSTANCE = generator.get_or_generate_graph()
            print("Global graph ready! All agents will share this instance.")
        return _GLOBAL_GRAPH_INSTANCE


def reset_global_graph():
    """
    Reset the global graph instance (primarily for testing).
    
    WARNING: This should only be used in test scenarios. Resetting during
    production use could cause inconsistencies across active agents.
    """
    global _GLOBAL_GRAPH_INSTANCE
    
    with _GLOBAL_GRAPH_LOCK:
        _GLOBAL_GRAPH_INSTANCE = None
        print("Global graph instance reset.")

