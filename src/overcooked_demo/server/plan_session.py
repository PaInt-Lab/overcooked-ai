from uuid import uuid4

class PlanSession:
    def __init__(self, events, task_title: str = ""):
        self.events = events
        self.task_title = task_title  # Store task title

PLAN_STORE: dict[str, PlanSession] = {}