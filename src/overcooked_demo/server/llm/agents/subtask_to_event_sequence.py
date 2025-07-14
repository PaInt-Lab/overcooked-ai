import json
from typing import List, Dict, Optional
from llm.ollama.ollama_client import query_ollama
import copy

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
    # print(f"Classifying subtasks: {raw_input}")

    response = query_ollama("task_tagger", raw_input)

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




def normalize_events(raw_events: list[dict]) -> list[dict]:
    """
    Batch-normalize every primary and secondary label in raw_events via the LLM.
    Returns a new event list with the exact same structure, but all labels
    replaced by their canonical versions (or NOOP for secondary).
    """

    # 1) Flatten the primaries and secondaries into two lists, preserving order
    prim_labels = []
    sec_labels  = []
    for ev in raw_events:
        prim_labels.extend(ev["primary"])
        for sec_list in ev["secondary"]:
            sec_labels.extend(sec_list)

    # 2) Call the normalizer
    payload = json.dumps({
      "primary_labels":   prim_labels,
      "secondary_labels": sec_labels
    })
    resp = query_ollama("task_normalizer", payload)
    out = json.loads(resp)

    prim_norm = out.get("primary_normalized", [])
    sec_norm  = out.get("secondary_normalized", [])

    # 3) Rebuild events by consuming from those two arrays in order
    normalized = []
    p_idx = 0
    s_idx = 0
    for ev in raw_events:
        n_prims = []
        for _ in ev["primary"]:
            # safety bounds check
            if p_idx < len(prim_norm):
                n_prims.append(prim_norm[p_idx])
            else:
                n_prims.append("NOOP")
            p_idx += 1

        n_secs = []
        for sec_list in ev["secondary"]:
            this_sec_list = []
            for _ in sec_list:
                if s_idx < len(sec_norm):
                    this_sec_list.append(sec_norm[s_idx])
                else:
                    this_sec_list.append("NOOP")
                s_idx += 1
            n_secs.append(this_sec_list)

        normalized.append({
            "primary":   n_prims,
            "secondary": n_secs
        })

    return normalized