"""
Ollama Infrastructure

This package handles local LLM integration using Ollama:
- Ollama client for local model communication
- Request routing and orchestration
"""

from .ollama_client import query_ollama

__all__ = [
    'query_ollama'
]
