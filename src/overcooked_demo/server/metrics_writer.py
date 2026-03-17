"""
MetricsWriter — appends per-turn and per-game rows to CSV files.

Two files are managed:
  metrics/turns.csv  — one row per LLM call
  metrics/games.csv  — one row per game (completed or not)
"""

import csv
import os
import threading
from datetime import datetime, timezone


TURNS_COLUMNS = [
    "game_id", "turn", "timestamp",
    "api_latency_ms", "turn_total_ms",
    "prompt_tokens", "completion_tokens", "total_tokens",
    "memoryless", "exceeded_window",
]

GAMES_COLUMNS = [
    "game_id", "timestamp", "recipe_type",
    "plan_day", "plan_time",
    "duration_s", "num_actions",
    "memoryless", "plans_in_repo_at_start",
    "completed", "pref_count", "step_count",
]


class MetricsWriter:
    def __init__(self, metrics_dir: str = None):
        if metrics_dir is None:
            server_root = os.path.dirname(os.path.abspath(__file__))
            metrics_dir = os.path.join(server_root, "metrics")
        self.metrics_dir = metrics_dir
        os.makedirs(metrics_dir, exist_ok=True)

        self.turns_path = os.path.join(metrics_dir, "turns.csv")
        self.games_path = os.path.join(metrics_dir, "games.csv")
        self._lock = threading.Lock()

        self._ensure_header(self.turns_path, TURNS_COLUMNS)
        self._ensure_header(self.games_path, GAMES_COLUMNS)

    def _ensure_header(self, path: str, columns: list) -> None:
        if not os.path.isfile(path):
            with open(path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(columns)

    def write_turn_row(
        self,
        game_id: str,
        turn: int,
        api_latency_ms: float,
        turn_total_ms: float,
        prompt_tokens,
        completion_tokens,
        total_tokens,
        memoryless: bool,
        exceeded_window: bool,
    ) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        row = [
            game_id, turn, timestamp,
            round(api_latency_ms, 1),
            round(turn_total_ms, 1),
            prompt_tokens, completion_tokens, total_tokens,
            memoryless, exceeded_window,
        ]
        with self._lock:
            with open(self.turns_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(row)

    def write_game_row(
        self,
        game_id: str,
        recipe_type: str,
        plan_day,
        plan_time,
        duration_s: float,
        num_actions: int,
        memoryless: bool,
        plans_in_repo_at_start: int,
        completed: bool,
        pref_count: int,
        step_count: int,
    ) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        row = [
            game_id, timestamp, recipe_type,
            plan_day, plan_time,
            round(duration_s, 2), num_actions,
            memoryless, plans_in_repo_at_start,
            completed, pref_count, step_count,
        ]
        with self._lock:
            with open(self.games_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(row)
