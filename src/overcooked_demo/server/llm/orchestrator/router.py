import re
from typing import List

from llm.agents.subtask_creator import SubtaskAgent
from llm.memory.vector_memory import VectorMemory

# Initialize a singleton VectorMemory
memory = VectorMemory()

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
    1. Retrieve context from memory (based on task_name)
    2. Build prompt via SubtaskAgent.generate_response(...)
    3. Parse the returned text into a list of subtasks
    4. Add (task_name + returned text) to memory for future context
    5. Return List[str] of cleaned subtasks
    """
    # 1) Get context from memory (if any)
    context = memory.get_context(task_name) if task_name else ""

    # 2) Call SubtaskAgent to get raw LLM response
    agent = SubtaskAgent()
    raw = agent.generate_response(
        task_name=task_name,
        existing_subtasks=existing_subtasks,
        notes=notes,
        context=context
    )

    # 3) Parse the raw LLM output into a Python list
    subtasks = _parse_numbered_list(raw)

    # 4) Store this interaction in memory
    memory.add_interaction(task_name, raw)

    return subtasks
