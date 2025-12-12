"""Human interaction channel handling preferences, step advice, and uncertainty."""

from typing import List, Dict, Optional, Tuple, Callable


class HumanInteractionChannel:
    """
    Manages human guidance:
    - Always-open inbox of messages (PREF vs STEP)
    - Pending step advice with staleness/expiry
    - Global preferences
    - Uncertainty prompting budget and waiting state
    """

    def __init__(
        self,
        prompt_callback: Optional[Callable[[dict, List[str]], None]] = None,
        uncertainty_threshold: float = 0.6,
        prompt_timeout: int = 2,
        prompt_budget: int = 3,
        advice_fresh_turns: int = 3,
    ):
        self.global_preferences: List[Dict] = []
        self.pending_step_advice: Optional[Dict] = None
        self.live_channel_inbox: List[Tuple[str, Optional[dict]]] = []

        self.waiting_for_human: bool = False
        self.waiting_turn_started: int = 0

        self.prompt_timeout = prompt_timeout
        self.prompt_budget = prompt_budget
        self.prompts_sent = 0
        self.uncertainty_threshold = uncertainty_threshold
        self.advice_fresh_turns = advice_fresh_turns

        self.prompt_callback = prompt_callback

    # --------------------
    # Inbox + classification
    # --------------------
    def enqueue_message(self, message: str, state_snapshot: Optional[dict] = None):
        self.live_channel_inbox.append((message, state_snapshot))

    def process_inbox(self, turn: int, current_state: Optional[dict]):
        """Process queued human messages and expire stale advice."""
        if self.live_channel_inbox:
            for message, snapshot in self.live_channel_inbox:
                classification = self._classify_human_message(message)
                if classification == "preference":
                    self._store_global_preference(message, snapshot, turn)
                else:
                    self._store_step_advice(message, snapshot, turn)
            self.live_channel_inbox.clear()
        self._expire_stale_step_advice(turn, current_state)

    def _classify_human_message(self, message: str) -> str:
        """Classify message as 'preference' (PREF) or 'step' (STEP) using prefixes + heuristics."""
        lower = message.lower()
        if lower.startswith("pref:"):
            return "preference"
        if lower.startswith("step:"):
            return "step"
        if "always" in lower or "never" in lower:
            return "preference"
        return "step"

    def _store_global_preference(self, message: str, snapshot: Optional[dict], turn: int):
        clean_text = message.replace("PREF:", "").replace("pref:", "").strip()
        pref_entry = {"text": clean_text, "turn": turn, "state": snapshot or {}}
        self.global_preferences.append(pref_entry)
        self.global_preferences = self.global_preferences[-5:]

    def _store_step_advice(self, message: str, snapshot: Optional[dict], turn: int):
        clean_text = message.replace("STEP:", "").replace("step:", "").strip()
        self.pending_step_advice = {
            "text": clean_text,
            "turn": turn,
            "state": snapshot or {},
            "expires_at": turn + self.advice_fresh_turns,
        }

    # --------------------
    # Advice lifecycle
    # --------------------
    def consume_fresh_step_advice(self, turn: int, current_state: Optional[dict]) -> Optional[str]:
        """Return pending step advice if fresh; consume it."""
        if not self.pending_step_advice:
            return None
        if self._is_advice_stale(self.pending_step_advice, turn, current_state):
            self.pending_step_advice = None
            return None
        advice_text = self.pending_step_advice["text"]
        self.pending_step_advice = None
        return advice_text

    def _expire_stale_step_advice(self, turn: int, current_state: Optional[dict]):
        if self.pending_step_advice and self._is_advice_stale(self.pending_step_advice, turn, current_state):
            self.pending_step_advice = None

    def _is_advice_stale(self, advice: Dict, turn: int, current_state: Optional[dict]) -> bool:
        if turn > advice.get("expires_at", turn):
            return True
        saved_state = advice.get("state") or {}
        current = current_state or {}
        keys_to_check = [
            "onion_in_pot",
            "tomato_in_pot",
            "soup_ready",
            "soup_cooking",
            "onion_hand",
            "tomato_hand",
            "dish_hand",
            "soup_hand",
        ]
        for k in keys_to_check:
            if saved_state.get(k) != current.get(k):
                return True
        return False

    # --------------------
    # Prompting + uncertainty
    # --------------------
    def compute_uncertainty_from_certainty(self, certainty_score: Optional[float]) -> float:
        """
        Convert LLM-reported certainty (0-1) to uncertainty (0-1).
        If certainty is None, default to high uncertainty to stay cautious.
        """
        if certainty_score is None:
            return 0.8
        return max(0.0, min(1.0, 1.0 - certainty_score))

    def should_prompt(self, uncertainty: float) -> bool:
        return (
            uncertainty >= self.uncertainty_threshold
            and self.prompts_sent < self.prompt_budget
            and not self.waiting_for_human
        )

    def start_waiting(self, turn: int):
        self.waiting_for_human = True
        self.waiting_turn_started = turn
        self.prompts_sent += 1

    def waiting_timed_out(self, turn: int) -> bool:
        return self.waiting_for_human and (turn - self.waiting_turn_started) > self.prompt_timeout

    def clear_waiting(self):
        self.waiting_for_human = False

    def emit_prompt(self, state: dict, available_actions: List[str]):
        if self.prompt_callback:
            self.prompt_callback(state, available_actions)
        else:
            print(f"[HUMAN PROMPT] Uncertain now. State: {state}")
            print(f"Options: {available_actions}")
            print("Reply with STEP: <next action> or PREF: <preference>.")

    # --------------------
    # Formatting helpers
    # --------------------
    def format_preferences(self) -> str:
        if not self.global_preferences:
            return ""
        return "; ".join([p["text"] for p in self.global_preferences])

    def format_step_advice(self, turn: int, current_state: Optional[dict]) -> str:
        if self.pending_step_advice and not self._is_advice_stale(self.pending_step_advice, turn, current_state):
            return self.pending_step_advice["text"]
        return ""


