"""
Coordinated Action Predictor Agent.

This agent uses coordinated state graph navigation for goal-directed behavior.
It predicts human behavior and coordinates accordingly, balancing goal progress
with coordination quality while adapting to human actions dynamically.
"""

from typing import List, Dict, Optional
from overcooked_ai_py.agents.agent import Agent
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.planning.planners import MotionPlanner
from overcooked_ai_py.mdp.actions import Action

from plan_creation import PLAN_STORE
from action_coordination import select_secondary_action
from plan_adaptation import ActionTracker, PlanRepository

# Import our new modular components
from .llm import query_openai
from .pathfinding import MovementPlanner
from .action_parsing import ActionParser
from .state_management import StateSummarizer, TileManager, BlockingDetector
from .complete_state_graph import CompleteRecipeState
from .graph_manager import get_global_graph


class CoordinatedActionPredictorAgent(Agent):
    """
    An agent that uses coordinated state graph navigation for goal-directed behavior.
    
    This agent:
    1. Uses a state graph to navigate toward the goal
    2. Predicts human behavior and coordinates accordingly
    3. Balances goal progress with coordination quality
    4. Adapts to human actions dynamically
    """
    
    def __init__(self):
        super().__init__()
        self.mdp = None
        self.planner = None
        self.agent_index = None
        self.last_summary = None
        
        # Initialize modular components
        self.tile_manager = TileManager()
        self.state_summarizer = StateSummarizer()
        self.movement_planner = MovementPlanner()
        self.action_parser = ActionParser()
        self.blocking_detector = BlockingDetector()
        
        # Link components
        self.state_summarizer.tile_manager = self.tile_manager
        self.action_parser.tile_checker = self.tile_manager
        self.blocking_detector.tile_manager = self.tile_manager
        
        # Complete state graph with washing capabilities
        # Use global singleton instance (loaded once across all agents)
        self.complete_state_graph = None
        
        # Task title for coordination
        self.task_title = None
        
        # Plan time and day for temporal context
        self.plan_time = None
        self.plan_day = None
        
        # Use simple vanilla model
        from .llm.openai_client import DEFAULT_MODEL
        self.selected_model = DEFAULT_MODEL
        
        # Plan adaptation system
        self.action_tracker = ActionTracker()
        self.plan_repository = PlanRepository()
        self.soup_served_flag = False
        
        # Plan session
        self.plan = None
        self.primary_tasks = []

    def _initialize_complete_state_graph(self):
        """Initialize the complete state graph using global singleton"""
        if self.complete_state_graph is None:
            # Get the global shared instance (loaded once for all agents)
            self.complete_state_graph = get_global_graph()
    
    def get_available_primary_actions(self, state, info):
        """Get available primary actions from complete state graph"""
        # Initialize complete state graph if needed
        self._initialize_complete_state_graph()
        
        # Get current state summary
        state_summary = self.summarize_state(state, info)
        
        # Convert to CompleteRecipeState
        complete_state = CompleteRecipeState.from_game_state(state_summary)
        
        # Find corresponding node in graph
        node_id = self.complete_state_graph.get_node_for_state(complete_state)
        
        if node_id:
            # Get available actions from state graph
            available_actions = self.complete_state_graph.get_possible_actions(node_id)
            return available_actions
        else:
            # Fallback to default actions if state not found
            return ["NOOP"]
    
    def set_agent_index(self, agent_index: int):
        super().set_agent_index(agent_index)
        self.agent_index = agent_index

    def set_plan(self, session_id: str):
        """Attach the full PlanSession to this agent and filter for primary tasks only."""
        self.plan = PLAN_STORE[session_id]
        
        # Filter plan to extract only primary tasks for LLM context
        self.primary_tasks = self._extract_primary_tasks_from_plan()
        
        # Set task title from plan for coordination
        if hasattr(self.plan, 'task_title'):
            self.task_title = self.plan.task_title
        
        # Set plan time and day for temporal context
        if hasattr(self.plan, 'plan_time'):
            self.plan_time = self.plan.plan_time
        if hasattr(self.plan, 'plan_day'):
            self.plan_day = self.plan.plan_day
    
    def load_preloaded_plans(self, preloaded_plans: List[Dict]):
        """
        Load pre-loaded plans into the plan repository.
        
        Args:
            preloaded_plans: List of plan dictionaries with structure:
                {
                    'id': int,
                    'subtasks': List[str],
                    'planTime': str (HH:MM format),
                    'planDay': str (day of week)
                }
        """
        print(f"Loading {len(preloaded_plans)} pre-loaded plans into repository...")
        for plan_data in preloaded_plans:
            successful_plan = {
                'actions': plan_data.get('subtasks', []),
                'duration': 0.0,  # Unknown duration for pre-loaded plans
                'recipe_type': 'preloaded',
                'plan_time': plan_data.get('planTime', ''),
                'plan_day': plan_data.get('planDay', ''),
                'preloaded': True  # Flag to distinguish from actual successful plans
            }
            self.plan_repository.add_successful_plan(successful_plan)
            print(f"  Loaded Plan #{plan_data.get('id', '?')}: {len(successful_plan['actions'])} actions ({successful_plan['plan_day']} at {successful_plan['plan_time']})")
        
        print(f"Plan repository now has {self.plan_repository.get_plan_count()} total plans")
    
    def _extract_primary_tasks_from_plan(self) -> List[str]:
        """
        Extract only primary tasks from the full plan.
        This filters out secondary tasks so the LLM only sees human actions.
        The plan structure has arrays of primary actions per step.
        """
        if not self.plan or not hasattr(self.plan, 'events'):
            return []
        
        # Handle the actual plan structure with arrays of primary actions
        primary_tasks = []
        for event in self.plan.events:
            # Extract primary actions from the 'primary' field (which is an array)
            if 'primary' in event and isinstance(event['primary'], list):
                for primary_action in event['primary']:
                    # Skip NOOP actions as they're not meaningful for LLM context
                    if primary_action != 'NOOP':
                        primary_tasks.append(primary_action)
        
        return primary_tasks

    def set_mdp(self, mdp: OvercookedGridworld):
        super().set_mdp(mdp)
        self.mdp = mdp
        
        # Initialize tile manager from MDP
        self.tile_manager.initialize_from_mdp(mdp)
        
        # Expose station positions at agent level for game.py to access
        self.onion_chopping_stations = self.tile_manager.onion_chopping_stations
        self.tomato_chopping_stations = self.tile_manager.tomato_chopping_stations
        self.onion_staging_tiles = self.tile_manager.onion_staging_tiles
        self.tomato_staging_tiles = self.tile_manager.tomato_staging_tiles
        self.sink_stations = self.tile_manager.sink_stations
        self.salt_stations = self.tile_manager.salt_stations
        self.pepper_stations = self.tile_manager.pepper_stations
        
        # Update movement planner and blocking detector with MDP
        self.movement_planner.mdp = mdp
        self.blocking_detector.mdp = mdp
        
        # Copy frontiers to movement planner
        self.movement_planner.onion_frontier = self.tile_manager.onion_frontier
        self.movement_planner.tomato_frontier = self.tile_manager.tomato_frontier
        self.movement_planner.dish_frontier = self.tile_manager.dish_frontier
        self.movement_planner.soup_staging_frontier = self.tile_manager.soup_staging_frontier
        self.movement_planner.onion_chopping_frontier = self.tile_manager.onion_chopping_frontier
        self.movement_planner.tomato_chopping_frontier = self.tile_manager.tomato_chopping_frontier
        self.movement_planner.onion_staging_frontier = self.tile_manager.onion_staging_frontier
        self.movement_planner.tomato_staging_frontier = self.tile_manager.tomato_staging_frontier
        self.movement_planner.dish_staging_frontier = self.tile_manager.dish_staging_frontier
        self.movement_planner.sink_frontier = self.tile_manager.sink_frontier
        self.movement_planner.salt_frontier = self.tile_manager.salt_frontier
        self.movement_planner.pepper_frontier = self.tile_manager.pepper_frontier
        self.movement_planner.delivery_frontier = self.tile_manager.delivery_frontier
        self.movement_planner.counter_frontier = self.tile_manager.counter_frontier
        
        # Create motion planner
        my_goals = {
            'ingredient': self.tile_manager.ingredient_spawns,
            'pot': self.tile_manager.stove_tiles,
            'dish': self.tile_manager.dish_spawns,
            'delivery': self.tile_manager.delivery_tiles
        }
        self.planner = MotionPlanner(mdp, counter_goals=my_goals)
        self.movement_planner.planner = self.planner

    def summarize_state(self, state, info):
        """Extract state predicates for the LLM (delegates to StateSummarizer)"""
        return self.state_summarizer.summarize_state(state, self.agent_index, info)

    def action(self, state):
        """Main action selection using optimized coordination system."""
        assert self.agent_index is not None, "agent_index is None in action!"
        
        # Get current state summary (last_info contains event_infos from game.py)
        self.last_summary = self.summarize_state(state, getattr(self, 'last_info', {}))
        
        # Get available primary actions from our complete state graph with washing
        try:
            available_primary_actions = self.get_available_primary_actions(state, {})
        except Exception as e:
            available_primary_actions = []
        
        # FALLBACK: If no actions available from complete state graph, provide basic actions
        if not available_primary_actions:
            print("WARNING: No actions from complete state graph, using fallback actions")
            available_primary_actions = [
                "Wash Onion",                  # Robot washes raw onion at sink
                "Chop Onion",                  # Robot chops washed onion at chopping station
                "Stage Onion",                 # Robot stages processed onion for human
                "Human Grab Onion",            # Human grabs processed onion
                "Place Onion in Pot",          # Human places onion in cooking pot
                
                "Wash Tomato",                 # Robot washes raw tomato at sink
                "Chop Tomato",                 # Robot chops washed tomato at chopping station
                "Stage Tomato",                # Robot stages processed tomato for human
                "Human Grab Tomato",           # Human grabs processed tomato
                "Place Tomato in Pot",         # Human places tomato in cooking pot
                
                "Turn Stove On",               # Human starts cooking when both ingredients in pot
                "Wait For Ingredients to Cook", # System state - cooking in progress
                "Human Grab Dish",             # Human grabs clean dish for serving
                "Pour Soup",                   # Human pours ready soup into dish
                "Human Stage Soup",            # Human stages soup for serving
                "Wait For Robot To Serve Soup"  # Robot serves the soup (robot action)
            ]
        
        # Build plan text for LLM
        plan_text = self._build_plan_text()
        
        # Determine which plan to use: original plan first, then all successful plans with priority
        plan_to_use = self._get_plan_to_use(plan_text)

        # Build temporal context if available
        temporal_context = ""
        if self.plan_time and self.plan_day:
            temporal_context = f"{self.plan_day} and {self.plan_time}"
        
        # **Training Data Approach:**
        # - Past successful sequences are provided as training data, not rigid plans to follow
        # - The agent should consider patterns from training data but adapt to the current state
        # - Time/day matching helps identify relevant patterns (human preferences vary by time)
        # - If no training data exists, the agent makes decisions based solely on the current state
        # - Flexibility is encouraged - use training data as guidance, not strict instructions

        # Select the action that best aligns with the current state, informed by training data when available.

        # Build prompt for LLM
        prompt = f"""
        You are helping a human cook soup. Select the best next action based on the current state and available training data.

        CURRENT TIME & DAY: 
        {temporal_context}

        CURRENT STATE:
        {self.last_summary}

        TRAINING DATA:
        {plan_to_use}

        AVAILABLE ACTIONS:
        {available_primary_actions}

        DECISION PROCESS:
        1. FIRST, carefully review ALL training data to identify patterns:
        - Read through each sequence completely - don't just look at the first action
        - Identify what the human tends to prefer: do they favor onion or tomato overall?
        - Look for common themes: do most sequences process one ingredient before the other?

        2. Analyze the current state:
        - What has been completed so far?
        - What items are currently being held?

        3. Determine the next action:
        - If at the START: look at the overall preference pattern from training data - what does the human usually do first?
        - If mid-sequence: continue in a way that's consistent with how similar states progressed in training data
        - Verify the action exists in AVAILABLE ACTIONS

        4. Select the action:
        - Choose the action that aligns with the human's demonstrated preferences from training data
        - When training data shows a clear preference (e.g., most sequences favor one ingredient), respect that preference
        - Adapt to current state but stay consistent with learned human behavior patterns

        RETURN FORMAT:
        Primary: <action_name>
        """
        # Display essential information for testing
        self._print_debug_info(available_primary_actions)
        
        # Call LLM to get predictions (using fixed temperature for consistency)
        response = query_openai(prompt, self.selected_model, temperature=0.3)
        
        # Parse the response - only need primary action now
        predicted_human_action = self.action_parser.parse_primary_action(response)
        
        # Track primary action for plan adaptation
        self._track_and_adapt_plan(predicted_human_action)
        
        # Get robot action using our smart coordination system
        try:
            robot_action = select_secondary_action(self.last_summary, self.task_title, predicted_human_action)
        except Exception as e:
            robot_action = "NOOP"
        
        print(f"LLM PREDICTION: {predicted_human_action}")
        print(f"ROBOT ACTION: {robot_action}")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        
        # Execute the robot action
        return self._execute_robot_action(robot_action, state, response, 
                                         predicted_human_action, available_primary_actions)

    def _build_plan_text(self) -> str:
        """Build plan text from primary tasks with time/day context."""
        if hasattr(self, 'primary_tasks') and self.primary_tasks:
            # Add time/day header if available
            header = ""
            if self.plan_time and self.plan_day:
                header = f"Original User Plan (for {self.plan_day} at {self.plan_time}):\n"
            else:
                header = "Original User Plan:\n"
            
            plan_lines = []
            for idx, primary_task in enumerate(self.primary_tasks):
                plan_lines.append(f"{idx+1}) {primary_task}")
            return header + "\n".join(plan_lines)
        else:
            return "No primary tasks available"
    
    def _get_plan_to_use(self, plan_text: str) -> str:
        """Build training data text from plan repository and current session."""
        training_lines = []
        sequence_counter = 1
        
        # Include the original user plan first as initial training data
        if hasattr(self, 'primary_tasks') and self.primary_tasks:
            # Build time/day info if available
            time_info = ""
            if self.plan_time and self.plan_day:
                time_info = f" ({self.plan_day} at {self.plan_time})"
            
            # Format as training sequence
            sequence_label = f"SEQUENCE #{sequence_counter}{time_info}"
            actions_str = " → ".join(self.primary_tasks)
            training_lines.append(f"{sequence_label}:\n{actions_str}")
            sequence_counter += 1
        
        # Add all successful sequences from repository
        if not self.plan_repository.is_empty():
            all_plans = self.plan_repository.get_plans_sorted_by_recency()
            
            for plan in all_plans:
                # Build time/day info if available
                time_info = ""
                if plan.get('plan_time') and plan.get('plan_day'):
                    time_info = f" ({plan['plan_day']} at {plan['plan_time']})"
                
                # Format as training sequence
                sequence_label = f"SEQUENCE #{sequence_counter}{time_info}"
                
                actions_str = " → ".join(plan['actions'])
                training_lines.append(f"{sequence_label}:\n{actions_str}")
                sequence_counter += 1
        
        # Return combined training data or indicate none available
        if training_lines:
            return "\n\n".join(training_lines)
        else:
            return "No training data available - make decisions based on current state"
    
    def _print_debug_info(self, available_primary_actions):
        """Print debug information."""
        print(f"CURRENT STATE: {self.last_summary}")
        print(f"SUPPLYING {len(available_primary_actions)} ACTIONS TO LLM: {available_primary_actions}")

    def _infer_recipe_type(self) -> str:
        """Infer recipe type from current game state."""
        if not self.last_summary:
            return 'unknown'

        recipe_components = []

        # Check for onion usage
        if (self.last_summary.get('onion_in_pot') or
            self.last_summary.get('onion_chopped') or
            self.last_summary.get('onion_washed') or
            self.last_summary.get('onion_staged')):
            recipe_components.append('onion')

        # Check for tomato usage
        if (self.last_summary.get('tomato_in_pot') or
            self.last_summary.get('tomato_chopped') or
            self.last_summary.get('tomato_washed') or
            self.last_summary.get('tomato_staged')):
            recipe_components.append('tomato')

        return '_'.join(recipe_components) if recipe_components else 'unknown'

    def _track_and_adapt_plan(self, predicted_human_action: str):
        """Track actions and adapt plan based on success."""
        # Track primary action for plan adaptation
        self.action_tracker.record_action(predicted_human_action, self.last_summary)
        
        # Check if soup was served (success detection)
        if self.last_summary and self.last_summary.get('soup_served', False):
            # Finalize and store the successful sequence
            if not self.action_tracker.is_empty():
                # Infer recipe type dynamically from actual state
                recipe_type = self._infer_recipe_type()

                successful_plan = {
                    'actions': self.action_tracker.get_current_sequence(),
                    'duration': self.action_tracker.get_sequence_duration(),
                    'recipe_type': recipe_type,
                    'plan_time': self.plan_time,  # Store time when plan was executed
                    'plan_day': self.plan_day  # Store day when plan was executed
                }

                # Print detailed plan debug output BEFORE saving
                print("\n" + "="*80)
                print("PLAN COMPLETED - DEBUG OUTPUT")
                print("="*80)
                print(f"Soup served by: {self.last_summary.get('soup_delivered_by', 'unknown')}")
                print(f"Recipe type: {recipe_type}")
                print(f"Duration: {successful_plan['duration']:.1f}s")
                print(f"Plan time: {self.plan_time}")
                print(f"Plan day: {self.plan_day}")
                print(f"Number of actions: {len(successful_plan['actions'])}")
                print(f"\nAction sequence:")
                for i, action in enumerate(successful_plan['actions'], 1):
                    print(f"  {i}. {action}")
                print(f"\nFull plan object:")
                import json
                print(json.dumps(successful_plan, indent=2, default=str))
                print("="*80 + "\n")

                # Now save to repository
                self.plan_repository.add_successful_plan(successful_plan)
                print(f"✓ PLAN SAVED: Repository now has {self.plan_repository.get_plan_count()} total plans")
                
                # Reset tracker for next sequence
                self.action_tracker.reset()
                print("PLAN TRACKING: Reset tracker for new sequence")
                
                # Reset the soup served flag and processing states after processing
                self.soup_served_flag = False
                self.state_summarizer.reset_processing_states()
    
    def _execute_robot_action(self, robot_action: str, state, response: str,
                             predicted_human_action: str, available_primary_actions: List) -> tuple:
        """Execute the robot action and return the appropriate move."""
        my_pos = state.player_positions[self.agent_index]
        my_ori = state.to_dict()["players"][self.agent_index]["orientation"]
        
        try:
            # Parse the robot action
            func_name, item_info = self.action_parser.parse_robot_action(
                robot_action, self.last_summary, state
            )
            
            if func_name == "NOOP":
                return self._handle_noop(my_pos, state, response, predicted_human_action, 
                                        robot_action, available_primary_actions)
            
            # Execute the compound action
            if func_name == "pickup":
                action_plan = self.movement_planner.pickup(item_info, my_pos, my_ori)
            elif func_name == "place":
                if isinstance(item_info, tuple):
                    item_to_place, destination = item_info
                    action_plan = self.movement_planner.place(item_to_place, my_pos, my_ori, destination)
                else:
                    action_plan = self.movement_planner.place(item_info, my_pos, my_ori, "default")
            else:
                # Fallback to simple movement
                action_plan = [Action.STAY]

            # Return first action from the plan
            if action_plan:
                move = action_plan[0] if action_plan else Action.STAY
                
                return move, {
                    "predicted_human_action": predicted_human_action,
                    "robot_action": robot_action,
                    "llm_response": response,
                    "available_primary_actions": available_primary_actions,
                    "function_call": f"{func_name}({item_info})",
                    "action_plan": action_plan
                }
            else:
                return Action.STAY, {
                    "predicted_human_action": predicted_human_action,
                    "robot_action": robot_action,
                    "llm_response": response,
                    "available_primary_actions": available_primary_actions,
                    "function_call": f"{func_name}({item_info})",
                    "action_plan": []
                }
        
        except Exception as e:
            return Action.STAY, {
                "predicted_human_action": predicted_human_action,
                "robot_action": robot_action,
                "llm_response": response,
                "available_primary_actions": available_primary_actions,
                "reasoning": "Action execution failed, staying in place"
            }
    
    def _handle_noop(self, my_pos, state, response, predicted_human_action, 
                    robot_action, available_primary_actions):
        """Handle NOOP action with blocking prevention."""
        # Check if the agent is blocking important tiles
        if self.blocking_detector.is_blocking_important_tile(my_pos, state):
            # If blocking, find a safe position to move to
            safe_pos, action_plan = self.blocking_detector.find_safe_position(
                my_pos, state, self.agent_index, self.movement_planner.get_action_plan
            )
            if action_plan:
                # Return first action from the plan
                move = action_plan[0] if action_plan else Action.STAY
                return move, {
                    "predicted_human_action": predicted_human_action,
                    "robot_action": robot_action,
                    "llm_response": response,
                    "available_primary_actions": available_primary_actions,
                    "blocking_prevention": True,
                    "action_plan": action_plan
                }
        
        # If not blocking or no safe move found, stay put
        return Action.STAY, {
            "predicted_human_action": predicted_human_action,
            "robot_action": robot_action,
            "llm_response": response,
            "available_primary_actions": available_primary_actions,
            "blocking_prevention": False,
            "action_plan": []
        }

    def actions(self, states, agent_indices):
        return [self.action(s) for s in states]
