"""
Plan Repository for Successful Primary Action Sequences

Stores and manages successful primary action sequences for plan adaptation.
Provides access to historical plans for LLM prompt enhancement.
Plans are persisted to disk under server/plans/{recipe_type}/plan_{id}.json
and reloaded automatically on startup.
"""

import os
import json
import tempfile
import threading
import time
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple


RECIPE_SUBDIRS = ("onion", "tomato", "onion_tomato", "unknown", "preloaded")


class PlanRepository:
    """
    Repository for storing successful primary action sequences.

    Features:
    - Store successful plans with incremental sequence IDs
    - Persist plans to disk (server/plans/{recipe_type}/plan_{id}.json)
    - Reload all plans from disk on startup
    - Retrieve plans for LLM prompt enhancement with temporal scoring
    """

    def __init__(self, plans_dir: str = None):
        """Initialize repository, creating disk dirs and loading existing plans."""
        self._lock = threading.Lock()

        if plans_dir is None:
            # __file__ is .../server/plan_adaptation/plan_repository.py
            # dirname twice => .../server
            server_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            plans_dir = os.path.join(server_root, "plans")

        self.plans_dir = plans_dir
        self._ensure_dirs()

        self.successful_plans: List[Dict] = []
        self.next_sequence_id: int = 1
        self._load_from_disk()

    # ------------------------------------------------------------------
    # Disk helpers
    # ------------------------------------------------------------------

    def _ensure_dirs(self) -> None:
        """Create plans_dir and all recipe subfolders if they don't exist."""
        for subdir in RECIPE_SUBDIRS:
            os.makedirs(os.path.join(self.plans_dir, subdir), exist_ok=True)

    def _load_from_disk(self) -> None:
        """Load all plan JSON files from disk into memory. Called once at init."""
        loaded = []
        for subdir in RECIPE_SUBDIRS:
            folder = os.path.join(self.plans_dir, subdir)
            if not os.path.isdir(folder):
                continue
            for filename in os.listdir(folder):
                if not filename.endswith(".json"):
                    continue
                filepath = os.path.join(folder, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        plan = json.load(f)
                    loaded.append(plan)
                except Exception as e:
                    print(f"[PlanRepository] WARNING: skipping corrupt plan file {filepath}: {e}")

        loaded.sort(key=lambda p: p.get("sequence_id", 0))
        self.successful_plans = loaded

        if loaded:
            self.next_sequence_id = max(p.get("sequence_id", 0) for p in loaded) + 1
            print(f"[PlanRepository] Loaded {len(loaded)} plans from disk. "
                  f"Next sequence_id={self.next_sequence_id}")
        else:
            self.next_sequence_id = 1

    def _save_plan_to_disk(self, plan: Dict) -> None:
        """
        Write a single plan dict to disk as JSON using an atomic write.
        Caller must hold self._lock.
        """
        recipe_type = plan.get("recipe_type", "unknown")
        subdir = recipe_type if recipe_type in RECIPE_SUBDIRS else "unknown"
        folder = os.path.join(self.plans_dir, subdir)
        os.makedirs(folder, exist_ok=True)

        seq_id = plan.get("sequence_id", self.next_sequence_id)
        target_path = os.path.join(folder, f"plan_{seq_id}.json")

        tmp_fd, tmp_path = tempfile.mkstemp(dir=folder, suffix=".tmp")
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(plan, f, indent=2, default=str)
            os.replace(tmp_path, target_path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_successful_plan(self, plan: Dict) -> None:
        """
        Add a successful plan to the repository and persist it to disk.

        Args:
            plan: Plan dictionary with actions, recipe_type, etc.
        """
        with self._lock:
            if "sequence_id" not in plan:
                plan["sequence_id"] = self.next_sequence_id
                self.next_sequence_id += 1

            if "completed_at" not in plan:
                plan["completed_at"] = datetime.now(timezone.utc).isoformat()

            self.successful_plans.append(plan)
            self._save_plan_to_disk(plan)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_all_plans(self) -> List[Dict]:
        """Get all successful plans stored in the repository."""
        with self._lock:
            return list(self.successful_plans)

    def get_plans_by_recipe_type(self, recipe_type: str) -> List[Dict]:
        """Get plans filtered by recipe type."""
        with self._lock:
            return [p for p in self.successful_plans if p.get("recipe_type") == recipe_type]

    def get_recent_plans(self, count: int) -> List[Dict]:
        """Get the most recent successful plans."""
        with self._lock:
            return list(self.successful_plans[-count:]) if self.successful_plans else []

    def get_most_recent_plan(self) -> Optional[Dict]:
        """Get the most recent successful plan."""
        with self._lock:
            return self.successful_plans[-1] if self.successful_plans else None

    def get_plans_sorted_by_recency(self) -> List[Dict]:
        """Get all successful plans sorted by recency (most recent first)."""
        with self._lock:
            all_plans = list(self.successful_plans)
        if not all_plans:
            return []
        return sorted(all_plans, key=lambda x: x.get("completed_at", 0), reverse=True)

    # ------------------------------------------------------------------
    # Temporal helpers
    # ------------------------------------------------------------------

    def _parse_hour(self, plan_time: Optional[str]) -> Optional[int]:
        """Parse hour from HH:MM time string."""
        if not plan_time:
            return None
        try:
            hour_str = str(plan_time).strip().split(":")[0]
            hour = int(hour_str)
            return hour if 0 <= hour <= 23 else None
        except (ValueError, TypeError, AttributeError):
            return None

    def _daypart_from_hour(self, hour: Optional[int]) -> Optional[str]:
        """Bucket hour into morning/afternoon/evening/night."""
        if hour is None:
            return None
        if 5 <= hour <= 11:
            return "morning"
        if 12 <= hour <= 16:
            return "afternoon"
        if 17 <= hour <= 20:
            return "evening"
        return "night"

    def _weekpart_from_day(self, plan_day: Optional[str]) -> Optional[str]:
        """Bucket day into weekday/weekend."""
        if not plan_day:
            return None
        day = str(plan_day).strip().lower()
        if day in {"saturday", "sunday"}:
            return "weekend"
        return "weekday"

    def _get_temporal_bucket(self, plan_day: Optional[str], plan_time: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
        """Return (daypart, weekpart) temporal bucket."""
        hour = self._parse_hour(plan_time)
        daypart = self._daypart_from_hour(hour)
        weekpart = self._weekpart_from_day(plan_day)
        return daypart, weekpart

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_relevant_plans(
        self,
        recipe_type: Optional[str],
        plan_day: Optional[str],
        plan_time: Optional[str],
        top_k: int = 6
    ) -> List[Dict]:
        """
        Get plans relevant to recipe and temporal context.

        Relevance is based on:
        - Recipe type match (hard filter when provided)
        - Temporal similarity (daypart + weekday/weekend)
        - Recency (tiebreaker)
        """
        with self._lock:
            all_plans = list(self.successful_plans)

        plans = all_plans
        if recipe_type and recipe_type != "unknown":
            plans = [plan for plan in all_plans if plan.get("recipe_type") == recipe_type]

        if not plans:
            return []

        target_daypart, target_weekpart = self._get_temporal_bucket(plan_day, plan_time)

        def score(plan: Dict) -> Tuple[int, str]:
            plan_daypart, plan_weekpart = self._get_temporal_bucket(
                plan.get("plan_day"),
                plan.get("plan_time")
            )
            temporal_score = 0
            if target_daypart and plan_daypart and target_daypart == plan_daypart:
                temporal_score += 2
            if target_weekpart and plan_weekpart and target_weekpart == plan_weekpart:
                temporal_score += 1
            return temporal_score, plan.get("completed_at", "")

        ranked = sorted(plans, key=score, reverse=True)
        if top_k and top_k > 0:
            if len(ranked) <= top_k:
                if recipe_type and recipe_type != "unknown" and len(ranked) < top_k:
                    existing_ids = {plan.get("sequence_id") for plan in ranked}
                    backfill = [
                        plan for plan in sorted(
                            all_plans,
                            key=lambda x: x.get("completed_at", ""),
                            reverse=True
                        )
                        if plan.get("sequence_id") not in existing_ids
                    ]
                    remaining = top_k - len(ranked)
                    return ranked + backfill[:remaining]
                return ranked
            temporal_matches = []
            non_temporal = []
            for plan in ranked:
                plan_daypart, plan_weekpart = self._get_temporal_bucket(
                    plan.get("plan_day"),
                    plan.get("plan_time")
                )
                if (target_daypart and plan_daypart == target_daypart) or (
                    target_weekpart and plan_weekpart == target_weekpart
                ):
                    temporal_matches.append(plan)
                else:
                    non_temporal.append(plan)
            remaining = top_k - len(temporal_matches)
            if remaining <= 0:
                return temporal_matches[:top_k]
            non_temporal_sorted = sorted(
                non_temporal,
                key=lambda x: x.get("completed_at", ""),
                reverse=True
            )
            combined = temporal_matches + non_temporal_sorted[:remaining]
            if len(combined) < top_k and recipe_type and recipe_type != "unknown":
                existing_ids = {plan.get("sequence_id") for plan in combined}
                backfill = [
                    plan for plan in sorted(
                        all_plans,
                        key=lambda x: x.get("completed_at", ""),
                        reverse=True
                    )
                    if plan.get("sequence_id") not in existing_ids
                ]
                return combined + backfill[: top_k - len(combined)]
            return combined
        return ranked

    # ------------------------------------------------------------------
    # Counts / state
    # ------------------------------------------------------------------

    def get_plan_count(self) -> int:
        """Get the total number of successful plans stored."""
        with self._lock:
            return len(self.successful_plans)

    def is_empty(self) -> bool:
        """Check if the repository has no plans."""
        with self._lock:
            return len(self.successful_plans) == 0

    def reset(self) -> None:
        """Reset the repository to empty state and delete all plan files from disk."""
        with self._lock:
            self.successful_plans.clear()
            self.next_sequence_id = 1
            for subdir in RECIPE_SUBDIRS:
                folder = os.path.join(self.plans_dir, subdir)
                if not os.path.isdir(folder):
                    continue
                for filename in os.listdir(folder):
                    if filename.endswith(".json"):
                        try:
                            os.unlink(os.path.join(folder, filename))
                        except OSError as e:
                            print(f"[PlanRepository] WARNING: could not delete {filename}: {e}")
