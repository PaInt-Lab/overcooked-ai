from uuid import uuid4

class PlanSession:
    def __init__(self, events, task_title: str = "", plan_time: str = "", plan_day: str = ""):
        self.events = events
        self.task_title = task_title  # Store task title
        self.plan_time = plan_time  # Store time of plan (HH:MM format)
        self.plan_day = plan_day  # Store day of week

PLAN_STORE: dict[str, PlanSession] = {}