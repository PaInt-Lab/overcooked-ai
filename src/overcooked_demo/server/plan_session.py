from uuid import uuid4

class PlanSession:
    def __init__(self, events, task_title: str = ""):
        self.events = events
        self.task_title = task_title  # Store task title
        self.idx = 0

    def current(self):
        return self.events[self.idx]
    
    def advance(self):
        self.idx += 1

PLAN_STORE: dict[str, PlanSession] = {}