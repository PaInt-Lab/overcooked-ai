from typing import List, Optional
from ollama_infrastructure import query_ollama

class SubtaskAgent():
    """
    Builds a prompt asking the LLM to generate a numbered list of subtasks
    (plus optional comments) given:
      - task_name
      - any existing subtasks the user already defined
      - notes
      - context (from memory)
    """

    def generate_response(
        self,
        task_name: str,
        existing_subtasks: Optional[List[str]] = None,
        notes: Optional[str] = None,
        context: Optional[str] = None
    ) -> str:
        # Build a clear prompt template
        prompt_lines = [
            f"Task: {task_name}",
        ]

        if existing_subtasks:
            # Join existing subtasks on commas or new lines
            joined = "\n".join(f"- {step}" for step in existing_subtasks)
            prompt_lines.append("Existing subtasks (if any):")
            prompt_lines.append(joined)

        if notes:
            prompt_lines.append(f"Notes: {notes}")

        if context:
            prompt_lines.append("Memory from past tasks:")
            prompt_lines.append(context)

        prompt_lines.append(
            "Please generate a clean, numbered list of subtasks to complete this task. "
            "You may include brief comments after each step."
        )

        full_prompt = "\n\n".join(prompt_lines)

        response = query_ollama("subtask_creator", full_prompt)
        return response
