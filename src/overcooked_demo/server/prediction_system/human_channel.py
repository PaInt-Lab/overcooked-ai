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
        advice_fresh_turns: int = 6,
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
        self.advice_fresh_turns = max(advice_fresh_turns, 1)

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
        if clean_text.lower() == "clear":
            self.pending_step_advice = None
            print("[HUMAN ADVICE] Cleared by user request (STEP: clear)")
            return
        self.pending_step_advice = {
            "text": clean_text,
            "turn": turn,
            "state": snapshot or {},
            # No TTL drop: rely on task completion or explicit clear
            "expires_at": None,
        }
        print(f"[HUMAN ADVICE] Stored step advice: '{clean_text}' (turn {turn})")

    # --------------------
    # Advice lifecycle
    # --------------------
    def consume_fresh_step_advice(self, turn: int, current_state: Optional[dict]) -> Optional[str]:
        """Return pending step advice if fresh; consume it (used when waiting)."""
        if not self.pending_step_advice:
            return None
        if self._is_advice_stale(self.pending_step_advice, turn, current_state):
            self.pending_step_advice = None
            return None
        advice_text = self.pending_step_advice["text"]
        self.pending_step_advice = None
        return advice_text

    def peek_step_advice(self, turn: int, current_state: Optional[dict]) -> Optional[str]:
        """Return pending step advice if fresh; do not consume (for prompt context)."""
        if not self.pending_step_advice:
            return None
        if self._is_advice_stale(self.pending_step_advice, turn, current_state):
            self.pending_step_advice = None
            return None
        print(f"[HUMAN ADVICE] Using step advice in prompt: '{self.pending_step_advice['text']}'")
        return self.pending_step_advice["text"]

    def _expire_stale_step_advice(self, turn: int, current_state: Optional[dict]):
        if self.pending_step_advice and self._is_advice_stale(self.pending_step_advice, turn, current_state):
            print(f"[HUMAN ADVICE] Dropped stale advice: '{self.pending_step_advice['text']}'")
            self.pending_step_advice = None

    def _is_advice_stale(self, advice: Dict, turn: int, current_state: Optional[dict]) -> bool:
        expires_at = advice.get("expires_at", None)
        if expires_at is not None and turn > expires_at:
            return True
        saved_state = advice.get("state") or {}
        current = current_state or {}
        if self._advice_completed(advice.get("text", ""), saved_state, current):
            return True
        # Clear if soup was served
        if current.get("soup_served", False) and not saved_state.get("soup_served", False):
            return True
        return False

    # --------------------
    # Advice completion heuristics
    # --------------------
    def _advice_completed(self, text: str, prev: dict, cur: dict) -> bool:
        t = text.lower()
        ing = self._ingredient_mentions(t)
        if not ing:
            # If no ingredient mentioned, require TTL expiry or explicit clear
            return False
        # pickup/grab
        if any(v in t for v in ["pick up", "pickup", "grab"]):
            for item in ing:
                hand_key = f"{item}_hand"
                if cur.get(hand_key) in ("agent", "partner"):
                    return True
        # processing
        if "wash" in t:
            if ("onion" in ing and cur.get("onion_washed")) or ("tomato" in ing and cur.get("tomato_washed")):
                return True
        if "chop" in t:
            if ("onion" in ing and cur.get("onion_chopped")) or ("tomato" in ing and cur.get("tomato_chopped")):
                return True
        if "salt" in t:
            if ("onion" in ing and cur.get("onion_salted")) or ("tomato" in ing and cur.get("tomato_salted")):
                return True
        if "pepper" in t:
            if ("onion" in ing and cur.get("onion_peppered")) or ("tomato" in ing and cur.get("tomato_peppered")):
                return True
        # staging
        if any(v in t for v in ["stage", "place", "set"]):
            if ("onion" in ing and cur.get("onion_staged")) or ("tomato" in ing and cur.get("tomato_staged")):
                return True
            if "dish" in ing and cur.get("dish_staged"):
                return True
            if "soup" in ing and cur.get("soup_staged"):
                return True
        # pot / cooking
        if "pot" in t or "put in pot" in t:
            if ("onion" in ing and cur.get("onion_in_pot")) or ("tomato" in ing and cur.get("tomato_in_pot")):
                return True
        if "turn" in t and "stove" in t:
            if cur.get("soup_cooking") and not prev.get("soup_cooking"):
                return True
        if "wait" in t and "cook" in t:
            if cur.get("soup_ready") and not prev.get("soup_ready"):
                return True
        # serving
        if any(v in t for v in ["pour", "serve", "deliver"]):
            if cur.get("soup_served"):
                return True
        return False

    def _ingredient_mentions(self, text: str) -> set:
        ing = set()
        if "onion" in text:
            ing.add("onion")
        if "tomato" in text:
            ing.add("tomato")
        if "dish" in text:
            ing.add("dish")
        if "soup" in text:
            ing.add("soup")
        return ing

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


