import json
from typing import List, Dict, Optional
from llm.ollama.ollama_client import query_ollama

def classify_subtasks(
        subtasks: List[str]) -> List[Dict[str, str]]:
    """
    Calls Ollama fine-tuned classifier to tag each subtask.
    Expects the model to return JSON:
    { "tagged_subtasks": [
        {"action": "...", "type": "primary"},
        ...
        ]
    }
    """
    prompt_lines = [
        "Here is an ordered list of subtasks.  ",
        "Classify each one as “primary” or “secondary” and return ONLY JSON with key tagged_subtasks:",
        "{",
        '  "tagged_subtasks": [',
    ]
    
    for step in subtasks:
        prompt_lines.append(f' {{"action": "{step}", "type": ""}},')

    prompt_lines.append(" ]")
    prompt_lines.append("}")

    raw_input = json.dumps({"subtasks": subtasks})
    print(f"Classifying subtasks: {raw_input}")

    response = query_ollama("overcooked_task_classifier", raw_input)
    print(f"Response from classifier: {response}")

    data = json.loads(response)
    return data["tagged_subtasks"]

def group_events(tagged: list[dict]) -> list[dict]:
    """
    Implements Algorithm 1 grouping:
      - collects consecutive secondary into s_list
      - pairs with next primary
      - merges into same event if s_list matches previous
    Returns [{"secondary": [...], "primary": [...]}, …]
    """
    events = []
    curr_s = []
    prev_s_set = None

    for item in tagged:
        if item["type"] == "secondary":
            curr_s.append(item["action"])
        else:  # primary
            s_list = curr_s or ["NOOP"]
            s_set = set(s_list)

            if events and (s_set == prev_s_set or s_set.issubset(prev_s_set)):
                # merge into last event
                events[-1]["primary"].append(item["action"])
                events[-1]["secondary"].append(s_list)
            else:
                # new event
                events.append({
                    "secondary": [s_list],
                    "primary": [item["action"]]
                })
                prev_s_set = s_set

            curr_s = []
    
    if curr_s:
        events.append({
            "secondary": [curr_s],
            "primary": ["NOOP"]
        })

    return events

