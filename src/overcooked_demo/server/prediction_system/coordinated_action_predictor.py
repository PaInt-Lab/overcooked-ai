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
from .llm import query_openai, get_temporal_temperature, get_temporal_context
from .pathfinding import MovementPlanner
from .action_parsing import ActionParser
from .state_management import StateSummarizer, TileManager, BlockingDetector
from .complete_state_graph import CompleteStateGraphGenerator, CompleteRecipeState


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
        self.complete_state_graph_generator = None
        self.complete_state_graph = None
        
        # Task title for coordination
        self.task_title = None
        
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
        """Initialize the complete state graph"""
        if self.complete_state_graph_generator is None:
            print("Initializing complete state graph...")
            self.complete_state_graph_generator = CompleteStateGraphGenerator()
            # Use cached version for fast loading
            self.complete_state_graph = self.complete_state_graph_generator.get_or_generate_graph()
            print("Complete state graph ready for action selection")
    
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
        
        # Get current state summary
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

        # Build prompt for LLM
        prompt = f"""
        You are helping a human cook soup. Follow the user's plan step-by-step in the correct sequence.

        TEMPORAL CONTEXT:
        {get_temporal_context()}

        CURRENT STATE:
        {self.last_summary}

        AVAILABLE PLANS (most recent first):
        {plan_to_use}

        AVAILABLE ACTIONS:
        {available_primary_actions}

        **CRITICAL: Follow the plan sequence step-by-step!**
        - DO NOT skip to the next ingredient if you're currently holding an item that is involved in the current steps for the plan.
        - If the plan's next step involves an item that someone is holding, there are higher odds that that is the next correct action!
        - When multiple plans are available, PRIORITIZE the most recent plan as it better represents current human preferences
        - Older plans can be used as backup guidance if the most recent plan doesn't fit the current state
        - The plans are very important and are designed to be followed in order!
        - Don't skip ahead to later steps
        - Only choose actions that are both AVAILABLE and the NEXT LOGICAL STEP in the plan

        Select the action that best aligns with the most recent plan and current state.

        Return only this line:
        Primary: <action_name>
        """
        
        # Display essential information for testing
        self._print_debug_info(available_primary_actions)
        
        # Call LLM to get predictions
        response = query_openai(prompt, self.selected_model, get_temporal_temperature())
        
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
        """Build plan text from primary tasks."""
        if hasattr(self, 'primary_tasks') and self.primary_tasks:
            plan_lines = []
            for idx, primary_task in enumerate(self.primary_tasks):
                plan_lines.append(f"{idx+1}) {primary_task}")
            return "\n".join(plan_lines)
        else:
            return "No primary tasks available"
    
    def _get_plan_to_use(self, plan_text: str) -> str:
        """Determine which plan to use based on plan repository."""
        if not self.plan_repository.is_empty():
            # Use all successful plans with priority indicators (most recent first)
            all_plans = self.plan_repository.get_plans_sorted_by_recency()
            plan_lines = []
            
            for i, plan in enumerate(all_plans):
                if i == 0:
                    # Most recent plan - better represents current human preferences
                    priority_label = "MOST RECENT PLAN (prioritize this - represents current human preferences)"
                else:
                    # Older plans - use as backup reference
                    priority_label = f"OLDER PLAN #{i+1} (use as backup reference)"
                
                actions_str = " → ".join(plan['actions'])
                plan_lines.append(f"{priority_label}:\n{actions_str}")
            
            return "\n\n".join(plan_lines)
        else:
            # Use original user plan for first time
            return f"USER PLAN (follow in order):\n{plan_text}"
    
    def _print_debug_info(self, available_primary_actions):
        """Print debug information."""
        print(f"CURRENT STATE: {self.last_summary}")
        print(f"SUPPLYING {len(available_primary_actions)} ACTIONS TO LLM: {available_primary_actions}")
        
        # Show plan adaptation info
        if not self.plan_repository.is_empty():
            total_plans = self.plan_repository.get_plan_count()
            print(f"PLAN ADAPTATION: Using {total_plans} successful plans (most recent prioritized)")
            # Show a summary of all plans sorted by recency
            all_plans = self.plan_repository.get_plans_sorted_by_recency()
            for i, plan in enumerate(all_plans):
                priority = "MOST RECENT" if i == 0 else f"OLDER #{i+1}"
                print(f"  {priority}: {' → '.join(plan['actions'][:3])}{'...' if len(plan['actions']) > 3 else ''}")
        else:
            print("PLAN ADAPTATION: Using original user plan (no successful plans yet)")
    
    def _track_and_adapt_plan(self, predicted_human_action: str):
        """Track actions and adapt plan based on success."""
        # Track primary action for plan adaptation
        self.action_tracker.record_action(predicted_human_action, self.last_summary)
        print(f"PLAN TRACKING: Recorded action '{predicted_human_action}' (sequence length: {len(self.action_tracker.get_current_sequence())})")
        
        # Check if soup was served (success detection)
        if self.last_summary and self.last_summary.get('soup_served', False):
            # Finalize and store the successful sequence
            if not self.action_tracker.is_empty():
                successful_plan = {
                    'actions': self.action_tracker.get_current_sequence(),
                    'duration': self.action_tracker.get_sequence_duration(),
                    'recipe_type': 'onion_washed_chopped_tomato_washed_chopped'  # Fixed recipe type for now
                }
                self.plan_repository.add_successful_plan(successful_plan)
                
                # Print detailed plan summary
                print(f"PLAN SUCCESS: Soup served by {self.last_summary.get('soup_delivered_by', 'unknown')}!")
                print(f"SAVED PLAN: {len(successful_plan['actions'])} actions over {successful_plan['duration']:.1f}s")
                print(f"PLAN SEQUENCE: {' → '.join(successful_plan['actions'])}")
                print(f"PLAN REPOSITORY: Now has {self.plan_repository.get_plan_count()} total plans")
                
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
