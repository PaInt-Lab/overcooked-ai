from uuid import uuid4

class PlanSession:
    def __init__(self, events):
        self.events = events
        self.idx = 0

    def current(self):
        return self.events[self.idx]
    
    def advance(self):
        self.idx += 1

PLAN_STORE: dict[str, PlanSession] = {}