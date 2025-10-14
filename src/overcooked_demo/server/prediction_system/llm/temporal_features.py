"""Temporal features for time-aware action prediction."""

from datetime import datetime


def get_temporal_temperature() -> float:
    """Get temperature setting based on time of day."""
    hour = datetime.now().hour
    
    if 6 <= hour < 12:  # Morning - more deterministic
        return 0.1
    elif 12 <= hour < 18:  # Afternoon - balanced
        return 0.3
    elif 18 <= hour < 24:  # Evening - more creative
        return 0.5
    else:  # Late night - very deterministic for quick decisions
        return 0.0


def get_temporal_context() -> str:
    """Generate real-time temporal context for LLM prompts."""
    now = datetime.now()
    hour = now.hour
    weekday = now.weekday()  # 0=Monday, 6=Sunday
    
    # Day context
    if weekday < 5:  # Weekday
        day_context = "It's a weekday - humans typically prefer efficient, structured approaches"
    else:  # Weekend  
        day_context = "It's the weekend - humans may be more experimental and collaborative"
    
    # Time context
    if 6 <= hour < 12:
        time_context = "It's morning - humans tend to be more methodical and prefer step-by-step approaches"
    elif 12 <= hour < 18:
        time_context = "It's afternoon - humans may be more efficient and prefer faster workflows"
    elif 18 <= hour < 24:
        time_context = "It's evening - humans might be more relaxed and collaborative"
    else:
        time_context = "It's late night - humans may be more focused on quick completion"
    
    return f"{day_context}. {time_context}"

