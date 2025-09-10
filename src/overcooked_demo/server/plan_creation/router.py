import re
from typing import List

from .subtask_creator import SubtaskAgent

def _parse_numbered_list(raw_text: str) -> List[str]:
    """
    Extracts lines from a numbered (or bulleted) list returned by the LLM.
    Strips leading numbers or bullets, returns a list of clean steps.
    Example input:
        "1. Get onions
         2. Peel onions
         3. Boil onions
         …"
    Returns: ["Get onions", "Peel onions", "Boil onions", …]
    """
    lines = raw_text.splitlines()
    parsed = []
    for line in lines:
        # Remove leading numbers, bullets, or whitespace (e.g. "1. ", "- ")
        clean = re.sub(r"^\s*[\-\d\.\)\>]+\s*", "", line).strip()
        if clean:
            parsed.append(clean)
    return parsed

def route_generate_subtasks(
    task_name: str,
    existing_subtasks: List[str],
    notes: str
) -> List[str]:
    """
    1. Build prompt via SubtaskAgent.generate_response(...)
    2. Parse the returned text into a list of subtasks
    3. Return List[str] of cleaned subtasks
    """
    # Call SubtaskAgent to get raw LLM response
    agent = SubtaskAgent()
    raw = agent.generate_response(
        task_name=task_name,
        existing_subtasks=existing_subtasks,
        notes=notes,
        context=""  # No context needed for one-time subtask generation
    )

    # Parse the raw LLM output into a Python list
    subtasks = _parse_numbered_list(raw)

    return subtasks
