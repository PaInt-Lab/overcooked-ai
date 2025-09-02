import json
import os
import pickle
import random
from abc import ABC, abstractmethod
from queue import Empty, Full, LifoQueue, Queue
from threading import Lock, Thread
from time import time

import ray
from utils import DOCKER_VOLUME, create_dirs

from llm.agents.action_predictor import ActionPredictorAgent
from llm.agents.coordinated_action_predictor import CoordinatedActionPredictorAgent
from human_aware_rl.rllib.rllib import load_agent
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.planning.planners import (
    NO_COUNTERS_PARAMS,
    MotionPlanner,
)

# Relative path to where all static pre-trained agents are stored on server
AGENT_DIR = None

# Maximum allowable game time (in seconds)
MAX_GAME_TIME = None


def _configure(max_game_time, agent_dir):
    global AGENT_DIR, MAX_GAME_TIME
    MAX_GAME_TIME = max_game_time
    AGENT_DIR = agent_dir


def fix_bc_path(path):
    """
    Loading a PPO agent trained with a BC agent requires loading the BC model as well when restoring the trainer, even though the BC model is not used in game
    For now the solution is to include the saved BC model and fix the relative path to the model in the config.pkl file
    """

    import dill

    # the path is the agents/Rllib.*/agent directory
    agent_path = os.path.dirname(path)
    with open(os.path.join(agent_path, "config.pkl"), "rb") as f:
        data = dill.load(f)
    bc_model_dir = data["bc_params"]["bc_config"]["model_dir"]
    last_dir = os.path.basename(bc_model_dir)
    bc_model_dir = os.path.join(agent_path, "bc_params", last_dir)
    data["bc_params"]["bc_config"]["model_dir"] = bc_model_dir
    with open(os.path.join(agent_path, "config.pkl"), "wb") as f:
        dill.dump(data, f)


class Game(ABC):

    """
    Class representing a game object. Coordinates the simultaneous actions of arbitrary
    number of players. Override this base class in order to use.

    Players can post actions to a `pending_actions` queue, and driver code can call `tick` to apply these actions.


    It should be noted that most operations in this class are not on their own thread safe. Thus, client code should
    acquire `self.lock` before making any modifications to the instance.

    One important exception to the above rule is `enqueue_actions` which is thread safe out of the box
    """

    # Possible TODO: create a static list of IDs used by the class so far to verify id uniqueness
    # This would need to be serialized, however, which might cause too great a performance hit to
    # be worth it

    EMPTY = "EMPTY"

    class Status:
        DONE = "done"
        ACTIVE = "active"
        RESET = "reset"
        INACTIVE = "inactive"
        ERROR = "error"

    def __init__(self, *args, **kwargs):
        """
        players (list): List of IDs of players currently in the game
        spectators (set): Collection of IDs of players that are not allowed to enqueue actions but are currently watching the game
        id (int):   Unique identifier for this game
        pending_actions List[(Queue)]: Buffer of (player_id, action) pairs have submitted that haven't been commited yet
        lock (Lock):    Used to serialize updates to the game state
        is_active(bool): Whether the game is currently being played or not
        """
        self.players = []
        self.spectators = set()
        self.pending_actions = []
        self.id = kwargs.get("id", id(self))
        self.lock = Lock()
        self._is_active = False

    @abstractmethod
    def is_full(self):
        """
        Returns whether there is room for additional players to join or not
        """
        pass

    @abstractmethod
    def apply_action(self, player_idx, action):
        """
        Updates the game state by applying a single (player_idx, action) tuple. Subclasses should try to override this method
        if possible
        """
        pass

    @abstractmethod
    def is_finished(self):
        """
        Returns whether the game has concluded or not
        """
        pass

    def is_ready(self):
        """
        Returns whether the game can be started. Defaults to having enough players
        """
        return self.is_full()

    @property
    def is_active(self):
        """
        Whether the game is currently being played
        """
        return self._is_active

    @property
    def reset_timeout(self):
        """
        Number of milliseconds to pause game on reset
        """
        return 3000

    def apply_actions(self):
        """
        Updates the game state by applying each of the pending actions in the buffer. Is called by the tick method. Subclasses
        should override this method if joint actions are necessary. If actions can be serialized, overriding `apply_action` is
        preferred
        """
        for i in range(len(self.players)):
            try:
                # Check if pending_actions[i] is actually a Queue (not EMPTY string)
                if hasattr(self.pending_actions[i], 'get'):
                    while True:
                        action = self.pending_actions[i].get(block=False)
                        self.apply_action(i, action)
            except Empty:
                pass

    def activate(self):
        """
        Activates the game to let server know real-time updates should start. Provides little functionality but useful as
        a check for debugging
        """
        self._is_active = True

    def deactivate(self):
        """
        Deactives the game such that subsequent calls to `tick` will be no-ops. Used to handle case where game ends but
        there is still a buffer of client pings to handle
        """
        self._is_active = False

    def reset(self):
        """
        Restarts the game while keeping all active players by resetting game stats and temporarily disabling `tick`
        """
        if not self.is_active:
            raise ValueError("Inactive Games cannot be reset")
        if self.is_finished():
            return self.Status.DONE
        self.deactivate()
        self.activate()
        return self.Status.RESET

    def needs_reset(self):
        """
        Returns whether the game should be reset on the next call to `tick`
        """
        return False

    def tick(self):
        """
        Updates the game state by applying each of the pending actions. This is done so that players cannot directly modify
        the game state, offering an additional level of safety and thread security.

        One can think of "enqueue_action" like calling "git add" and "tick" like calling "git commit"

        Subclasses should try to override `apply_actions` if possible. Only override this method if necessary
        """
        if not self.is_active:
            return self.Status.INACTIVE
        if self.needs_reset():
            self.reset()
            return self.Status.RESET

        self.apply_actions()
        return self.Status.DONE if self.is_finished() else self.Status.ACTIVE

    def enqueue_action(self, player_id, action):
        """
        Add (player_id, action) pair to the pending action queue, without modifying underlying game state

        Note: This function IS thread safe
        """
        if not self.is_active:
            # Could run into issues with is_active not being thread safe
            return
        if player_id not in self.players:
            # Only players actively in game are allowed to enqueue actions
            return
        try:
            player_idx = self.players.index(player_id)
            self.pending_actions[player_idx].put(action)
        except Full:
            pass

    def get_state(self):
        """
        Return a JSON compatible serialized state of the game. Note that this should be as minimalistic as possible
        as the size of the game state will be the most important factor in game performance. This is sent to the client
        every frame update.
        """
        return {"players": self.players}

    def to_json(self):
        """
        Return a JSON compatible serialized state of the game. Contains all information about the game, does not need to
        be minimalistic. This is sent to the client only once, upon game creation
        """
        return self.get_state()

    def is_empty(self):
        """
        Return whether it is safe to garbage collect this game instance
        """
        return not self.num_players

    def add_player(self, player_id, idx=None, buff_size=-1):
        """
        Add player_id to the game
        """
        if self.is_full():
            raise ValueError("Cannot add players to full game")
        if self.is_active:
            raise ValueError("Cannot add players to active games")
        if not idx and self.EMPTY in self.players:
            idx = self.players.index(self.EMPTY)
        elif not idx:
            idx = len(self.players)

        padding = max(0, idx - len(self.players) + 1)
        for _ in range(padding):
            self.players.append(self.EMPTY)
            self.pending_actions.append(self.EMPTY)

        self.players[idx] = player_id
        self.pending_actions[idx] = Queue(maxsize=buff_size)

    def add_spectator(self, spectator_id):
        """
        Add spectator_id to list of spectators for this game
        """
        if spectator_id in self.players:
            raise ValueError("Cannot spectate and play at same time")
        self.spectators.add(spectator_id)

    def remove_player(self, player_id):
        """
        Remove player_id from the game
        """
        try:
            idx = self.players.index(player_id)
            self.players[idx] = self.EMPTY
            self.pending_actions[idx] = self.EMPTY
        except ValueError:
            return False
        else:
            return True

    def remove_spectator(self, spectator_id):
        """
        Removes spectator_id if they are in list of spectators. Returns True if spectator successfully removed, False otherwise
        """
        try:
            self.spectators.remove(spectator_id)
        except ValueError:
            return False
        else:
            return True

    def clear_pending_actions(self):
        """
        Remove all queued actions for all players
        """
        for i, player in enumerate(self.players):
            if player != self.EMPTY:
                queue = self.pending_actions[i]
                queue.queue.clear()

    @property
    def num_players(self):
        return len([player for player in self.players if player != self.EMPTY])

    def get_data(self):
        """
        Return any game metadata to server driver.
        """
        return {}


class DummyGame(Game):

    """
    Standin class used to test basic server logic
    """

    def __init__(self, **kwargs):
        super(DummyGame, self).__init__(**kwargs)
        self.counter = 0

    def is_full(self):
        return self.num_players == 2

    def apply_action(self, idx, action):
        pass

    def apply_actions(self):
        self.counter += 1

    def is_finished(self):
        return self.counter >= 100

    def get_state(self):
        state = super(DummyGame, self).get_state()
        state["count"] = self.counter
        return state


class DummyInteractiveGame(Game):

    """
    Standing class used to test interactive components of the server logic
    """

    def __init__(self, **kwargs):
        super(DummyInteractiveGame, self).__init__(**kwargs)
        self.max_players = int(
            kwargs.get("playerZero", "human") == "human"
        ) + int(kwargs.get("playerOne", "human") == "human")
        self.max_count = kwargs.get("max_count", 30)
        self.counter = 0
        self.counts = [0] * self.max_players

    def is_full(self):
        return self.num_players == self.max_players

    def is_finished(self):
        return max(self.counts) >= self.max_count

    def apply_action(self, player_idx, action):
        if action.upper() == Direction.NORTH:
            self.counts[player_idx] += 1
        if action.upper() == Direction.SOUTH:
            self.counts[player_idx] -= 1

    def apply_actions(self):
        super(DummyInteractiveGame, self).apply_actions()
        self.counter += 1

    def get_state(self):
        state = super(DummyInteractiveGame, self).get_state()
        state["count"] = self.counter
        for i in range(self.num_players):
            state["player_{}_count".format(i)] = self.counts[i]
        return state


class OvercookedGame(Game):
    """
    Class for bridging the gap between Overcooked_Env and the Game interface

    Instance variable:
        - max_players (int): Maximum number of players that can be in the game at once
        - mdp (OvercookedGridworld): Controls the underlying Overcooked game logic
        - score (int): Current reward acheived by all players
        - max_time (int): Number of seconds the game should last
        - npc_policies (dict): Maps user_id to policy (Agent) for each AI player
        - npc_state_queues (dict): Mapping of NPC user_ids to LIFO queues for the policy to process
        - curr_tick (int): How many times the game server has called this instance's `tick` method
        - ticker_per_ai_action (int): How many frames should pass in between NPC policy forward passes.
            Note that this is a lower bound; if the policy is computationally expensive the actual frames
            per forward pass can be higher
        - action_to_overcooked_action (dict): Maps action names returned by client to action names used by OvercookedGridworld
            Note that this is an instance variable and not a static variable for efficiency reasons
        - human_players (set(str)): Collection of all player IDs that correspond to humans
        - npc_players (set(str)): Collection of all player IDs that correspond to AI
        - randomized (boolean): Whether the order of the layouts should be randomized

    Methods:
        - npc_policy_consumer: Background process that asynchronously computes NPC policy forward passes. One thread
            spawned for each NPC
        - _curr_game_over: Determines whether the game on the current mdp has ended
    """

    def __init__(
        self,
        layouts=["cramped_room"],
        mdp_params={},
        num_players=2,
        gameTime=30,
        playerZero="human",
        playerOne="human",
        showPotential=False,
        randomized=False,
        ticks_per_ai_action=1,
        plan_session_id=None,
        **kwargs
    ):
        super(OvercookedGame, self).__init__(**kwargs)
        self.show_potential = showPotential
        self.mdp_params = mdp_params
        self.layouts = layouts
        self.max_players = int(num_players)
        self.mdp = None
        self.mp = None
        self.score = 0
        self.phi = 0
        self.max_time = min(int(gameTime), MAX_GAME_TIME)
        self.npc_policies = {}
        self.npc_state_queues = {}
        self.action_to_overcooked_action = {
            "STAY": Action.STAY,
            "UP": Direction.NORTH,
            "DOWN": Direction.SOUTH,
            "LEFT": Direction.WEST,
            "RIGHT": Direction.EAST,
            "SPACE": Action.INTERACT,
        }
        self.ticks_per_ai_action = ticks_per_ai_action
        self.curr_tick = 0
        self.human_players = set()
        self.npc_players = set()
        self.plan_session_id = plan_session_id

        if randomized:
            random.shuffle(self.layouts)

        if playerZero != "human":
            player_zero_id = playerZero + "_0"
            self.add_player(player_zero_id, idx=0, buff_size=1, is_human=False)
            self.npc_policies[player_zero_id] = self.get_policy(
                playerZero, idx=0
            )
            self.npc_state_queues[player_zero_id] = LifoQueue()

        if playerOne != "human":
            player_one_id = playerOne + "_1"
            self.add_player(player_one_id, idx=1, buff_size=1, is_human=False)
            self.npc_policies[player_one_id] = self.get_policy(
                playerOne, idx=1
            )
            self.npc_state_queues[player_one_id] = LifoQueue()
        # Always kill ray after loading agent, otherwise, ray will crash once process exits
        # Only kill ray after loading both agents to avoid having to restart ray during loading
        if ray.is_initialized():
            ray.shutdown()

        if kwargs["dataCollection"]:
            self.write_data = True
            self.write_config = kwargs["collection_config"]
        else:
            self.write_data = False

        self.trajectory = []

    def _curr_game_over(self):
        return time() - self.start_time >= self.max_time

    def needs_reset(self):
        return self._curr_game_over() and not self.is_finished()

    def add_player(self, player_id, idx=None, buff_size=-1, is_human=True):
        super(OvercookedGame, self).add_player(
            player_id, idx=idx, buff_size=buff_size
        )
        if is_human:
            self.human_players.add(player_id)
        else:
            self.npc_players.add(player_id)

    def remove_player(self, player_id):
        removed = super(OvercookedGame, self).remove_player(player_id)
        if removed:
            if player_id in self.human_players:
                self.human_players.remove(player_id)
            elif player_id in self.npc_players:
                self.npc_players.remove(player_id)
            else:
                raise ValueError("Inconsistent state")

    def npc_policy_consumer(self, policy_id):
        queue = self.npc_state_queues[policy_id]
        policy = self.npc_policies[policy_id]
        while self._is_active:
            state = queue.get()
            npc_action, _ = policy.action(state)
            super(OvercookedGame, self).enqueue_action(policy_id, npc_action)

    def is_full(self):
        return self.num_players >= self.max_players

    def is_finished(self):
        val = not self.layouts and self._curr_game_over()
        return val

    def is_empty(self):
        """
        Game is considered safe to scrap if there are no active players or if there are no humans (spectating or playing)
        """
        return (
            super(OvercookedGame, self).is_empty()
            or not self.spectators
            and not self.human_players
        )

    def is_ready(self):
        """
        Game is ready to be activated if there are a sufficient number of players and at least one human (spectator or player)
        """
        return super(OvercookedGame, self).is_ready() and not self.is_empty()

    def apply_action(self, player_idx, action):
        """Apply a single human action immediately for real-time movement"""
        if player_idx >= len(self.players):
            return
        
        player_id = self.players[player_idx]
        if player_id not in self.human_players:
            # Only process human actions here
            return
            
        # Action is already an overcooked action (converted in enqueue_action)
        overcooked_action = action
        
        # Create a joint action with this human action and STAY for others
        joint_action = [Action.STAY] * len(self.players)
        joint_action[player_idx] = overcooked_action
        
        # Apply the action to the current state
        prev_state = self.state
        self.state, info = self.mdp.get_state_transition(prev_state, joint_action)
        
        # Update score if there was a reward
        curr_reward = sum(info["sparse_reward_by_agent"])
        self.score += curr_reward
        
        # Pass info to agents so they can detect soup delivery events
        for agent in self.npc_policies.values():
            agent.last_info = info
            
            # Check for soup delivery and set flag if detected
            if info and info.get("event_infos", {}).get("soup_delivery", [False, False]):
                soup_delivery = info.get("event_infos", {}).get("soup_delivery", [False, False])
                if any(soup_delivery):
                    agent.soup_served_flag = True
                    # Determine who delivered the soup
                    if soup_delivery[0]:  # Agent (robot) delivered
                        agent.soup_delivered_by = "agent"
                    elif soup_delivery[1]:  # Partner (human) delivered
                        agent.soup_delivered_by = "partner"
                    print(f"DEBUG: Soup delivery detected in game.py! Setting flag. soup_delivery={soup_delivery}, delivered_by={agent.soup_delivered_by}")
        
        # Log the transition for trajectory
        transition = {
            "state": json.dumps(prev_state.to_dict()),
            "joint_action": json.dumps(joint_action),
            "reward": curr_reward,
            "time_left": max(self.max_time - (time() - self.start_time), 0),
            "score": self.score,
            "time_elapsed": time() - self.start_time,
            "cur_gameloop": self.curr_tick,
            "layout": json.dumps(self.mdp.terrain_mtx),
            "layout_name": self.curr_layout,
            "trial_id": str(self.start_time),
            "player_0_id": self.players[0],
            "player_1_id": self.players[1],
            "player_0_is_human": self.players[0] in self.human_players,
            "player_1_is_human": self.players[1] in self.human_players,
        }
        self.trajectory.append(transition)

    def apply_actions(self):
        # Process human actions immediately (real-time movement)
        for i in range(len(self.players)):
            if self.players[i] in self.human_players:
                try:
                    # Check if pending_actions[i] is actually a Queue (not EMPTY string)
                    if hasattr(self.pending_actions[i], 'get'):
                        # Process all pending human actions immediately
                        while True:
                            action = self.pending_actions[i].get(block=False)
                            self.apply_action(i, action)
                except Empty:
                    pass  # No more human actions to process
        
        # Check if there are any agents in the game
        if not self.npc_players:
            # Human-only game, no agent actions to process
            return None, None, None
        
        # Process agent actions when ready (blocking)
        joint_action = [Action.STAY] * len(self.players)
        agent_actions_processed = False
        
        for i in range(len(self.players)):
            if self.players[i] not in self.human_players:
                try:
                    # Check if pending_actions[i] is actually a Queue (not EMPTY string)
                    if hasattr(self.pending_actions[i], 'get'):
                        # Block on agent actions to ensure they get to do one action per state
                        joint_action[i] = self.pending_actions[i].get(block=True, timeout=0.1)
                        agent_actions_processed = True
                except Empty:
                    # Agent didn't respond in time, stay in place
                    pass

        # Only apply agent actions if at least one agent provided an action
        if agent_actions_processed:
            # Apply overcooked game logic to get state transition
            prev_state = self.state
            self.state, info = self.mdp.get_state_transition(prev_state, joint_action)

            for agent in self.npc_policies.values():
                agent.last_info = info

            if self.show_potential:
                self.phi = self.mdp.potential_function(prev_state, self.mp, gamma=0.99)

            # Send next state to all background consumers if needed
            if self.curr_tick % self.ticks_per_ai_action == 0:
                for npc_id in self.npc_policies:
                    self.npc_state_queues[npc_id].put(self.state, block=False)

            # Update score based on soup deliveries that might have occurred
            curr_reward = sum(info["sparse_reward_by_agent"])
            self.score += curr_reward

            # Log the transition for trajectory
            transition = {
                "state": json.dumps(prev_state.to_dict()),
                "joint_action": json.dumps(joint_action),
                "reward": curr_reward,
                "time_left": max(self.max_time - (time() - self.start_time), 0),
                "score": self.score,
                "time_elapsed": time() - self.start_time,
                "cur_gameloop": self.curr_tick,
                "layout": json.dumps(self.mdp.terrain_mtx),
                "layout_name": self.curr_layout,
                "trial_id": str(self.start_time),
                "player_0_id": self.players[0],
                "player_1_id": self.players[1],
                "player_0_is_human": self.players[0] in self.human_players,
                "player_1_is_human": self.players[1] in self.human_players,
            }
            self.trajectory.append(transition)

            return prev_state, joint_action, info
        
        return None, None, None

    def enqueue_action(self, player_id, action):
        overcooked_action = self.action_to_overcooked_action[action]
        
        # For human players, apply the action immediately for real-time movement
        if player_id in self.human_players:
            player_idx = self.players.index(player_id)
            self.apply_action(player_idx, overcooked_action)
        else:
            # For agents, queue the action normally
            super(OvercookedGame, self).enqueue_action(player_id, overcooked_action)

    def reset(self):
        status = super(OvercookedGame, self).reset()
        if status == self.Status.RESET:
            # Hacky way of making sure game timer doesn't "start" until after reset timeout has passed
            self.start_time += self.reset_timeout / 1000

    def tick(self):
        self.curr_tick += 1
        return super(OvercookedGame, self).tick()

    def activate(self):
        super(OvercookedGame, self).activate()

        # Sanity check at start of each game
        if not self.npc_players.union(self.human_players) == set(self.players):
            raise ValueError("Inconsistent State")

        self.curr_layout = self.layouts.pop()
        self.mdp = OvercookedGridworld.from_layout_name(
            self.curr_layout, **self.mdp_params
        )
        # Set layout name on MDP for agent access
        self.mdp.layout_name = self.curr_layout
        

        if self.show_potential:
            self.mp = MotionPlanner.from_pickle_or_compute(
                self.mdp, counter_goals=NO_COUNTERS_PARAMS
            )
        self.state = self.mdp.get_standard_start_state()
        if self.show_potential:
            self.phi = self.mdp.potential_function(
                self.state, self.mp, gamma=0.99
            )
        self.start_time = time()
        self.curr_tick = 0
        self.score = 0
        self.threads = []
        for npc_policy in self.npc_policies:
            self.npc_policies[npc_policy].reset()
            # Set agent index after reset to ensure it is not None
            agent = self.npc_policies[npc_policy]
            if hasattr(agent, 'set_agent_index'):
                if npc_policy.endswith('_0'):
                    agent.set_agent_index(0)
                elif npc_policy.endswith('_1'):
                    agent.set_agent_index(1)
            # Set MDP after reset to ensure it is not cleared
            if hasattr(agent, 'set_mdp'):
                agent.set_mdp(self.mdp)
            self.npc_state_queues[npc_policy].put(self.state)
            t = Thread(target=self.npc_policy_consumer, args=(npc_policy,))
            self.threads.append(t)
            t.start()

    def deactivate(self):
        super(OvercookedGame, self).deactivate()
        # Ensure the background consumers do not hang
        for npc_policy in self.npc_policies:
            self.npc_state_queues[npc_policy].put(self.state)

        # Wait for all background threads to exit
        for t in self.threads:
            t.join()

        # Clear all action queues
        self.clear_pending_actions()

    def get_state(self):
        state_dict = {}
        state_dict["potential"] = self.phi if self.show_potential else None
        state_dict["state"] = self.state.to_dict()
        state_dict["score"] = self.score
        state_dict["time_left"] = max(
            self.max_time - (time() - self.start_time), 0
        )
        return state_dict

    def to_json(self):
        obj_dict = {}
        obj_dict["state"] = self.get_state() if self._is_active else None
        
        # Create a COPY of terrain for graphics with special tile markers
        if self._is_active and self.mdp:
            # Start with original terrain (keeps game logic intact)
            original_terrain = self.mdp.terrain_mtx
            graphics_terrain = [row[:] for row in original_terrain]  # Deep copy
            H, W = len(graphics_terrain), len(graphics_terrain[0])
            
            # Find stove tiles to identify staging and chopping positions
            stove_tiles = [(j, i) for i, row in enumerate(graphics_terrain) for j, c in enumerate(row) if c == 'P']
            
            # For each stove, find adjacent counter tiles that can be staging areas
            staging_positions = []
            for (stove_c, stove_r) in stove_tiles:
                adjacent_counters = []
                for dc, dr in [(1,0), (-1,0), (0,1), (0,-1)]:
                    nc, nr = stove_c + dc, stove_r + dr
                    if 0 <= nr < H and 0 <= nc < W and graphics_terrain[nr][nc] == 'X':
                        adjacent_counters.append((nc, nr, dc, dr))
                
                # Prioritize left tiles, then bottom tiles as staging areas
                left_stations = [(c, r) for c, r, dc, dr in adjacent_counters if dc == -1]
                bottom_stations = [(c, r) for c, r, dc, dr in adjacent_counters if dr == 1]
                
                if left_stations:
                    staging_positions.extend(left_stations)
                elif bottom_stations:
                    staging_positions.extend(bottom_stations)
            
            # Mark staging tiles as 'G' in the COPY
            for (x, y) in staging_positions:
                graphics_terrain[y][x] = 'G'
            
            # Get chopping stations from agent (for layouts with hardcoded positions) or use dynamic detection
            chopping_positions = []
            agent_chopping_positions = []
            for npc_policy in self.npc_policies.values():
                if hasattr(npc_policy, 'onion_chopping_stations') and npc_policy.onion_chopping_stations:
                    agent_chopping_positions.extend(npc_policy.onion_chopping_stations)
                    break  # Only need one agent's chopping positions
            
            if agent_chopping_positions:
                # Use agent's hardcoded positions
                chopping_positions = agent_chopping_positions
            else:
                # Use dynamic detection for layouts without hardcoded positions
                for staging_pos in staging_positions:
                    staging_c, staging_r = staging_pos
                    for dc, dr in [(-1, 0), (0, -1)]:  # Check left and up from staging
                        chopping_c, chopping_r = staging_c + dc, staging_r + dr
                        if (0 <= chopping_r < H and 0 <= chopping_c < W and 
                            graphics_terrain[chopping_r][chopping_c] == 'X' and
                            (chopping_c, chopping_r) not in chopping_positions and
                            (chopping_c, chopping_r) not in staging_positions):
                            chopping_positions.append((chopping_c, chopping_r))
                            break
            
            # Mark chopping tiles as 'C' in the COPY
            for (x, y) in chopping_positions:
                graphics_terrain[y][x] = 'C'
            
            # Add sink stations if using LLM agent with sink support
            sink_positions = []
            for npc_policy in self.npc_policies.values():
                if hasattr(npc_policy, 'sink_stations') and npc_policy.sink_stations:
                    sink_positions.extend(npc_policy.sink_stations)
                    break  # Only need one agent's sink positions
            
            # Mark sink tiles as 'W' in the COPY
            for (x, y) in sink_positions:
                if 0 <= y < H and 0 <= x < W:
                    graphics_terrain[y][x] = 'W'
            
            # Send the MODIFIED terrain copy to graphics
            obj_dict["terrain"] = graphics_terrain
        else:
            obj_dict["terrain"] = None
        
        return obj_dict

    def get_policy(self, npc_id, idx=0):
        if npc_id == "overcooked_llm":
            assert idx is not None, "Agent index must not be None for LLM agent!"
            agent = CoordinatedActionPredictorAgent()
            agent.set_agent_index(idx)
            plan_id = getattr(self, "plan_session_id", None)
            if plan_id:
                agent.set_plan(plan_id)
            return agent
        if npc_id.lower().startswith("rllib"):
            try:
                # Loading rllib agents requires additional helpers
                fpath = os.path.join(AGENT_DIR, npc_id, "agent")
                fix_bc_path(fpath)
                agent = load_agent(fpath, agent_index=idx)
                return agent
            except Exception as e:
                raise IOError(
                    "Error loading Rllib Agent\n{}".format(e.__repr__())
                )
        else:
            try:
                fpath = os.path.join(AGENT_DIR, npc_id, "agent.pickle")
                with open(fpath, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                raise IOError("Error loading agent\n{}".format(e.__repr__()))

    def get_data(self):
        """
        Returns and then clears the accumulated trajectory
        """
        data = {
            "uid": str(time()),
            "trajectory": self.trajectory,
        }
        self.trajectory = []
        # if we want to store the data and there is data to store
        if self.write_data and len(data["trajectory"]) > 0:
            configs = self.write_config
            # create necessary dirs
            data_path = create_dirs(configs, self.curr_layout)
            # the 3-layer-directory structure should be able to uniquely define any experiment
            with open(os.path.join(data_path, "result.pkl"), "wb") as f:
                pickle.dump(data, f)
        return data


class OvercookedTutorial(OvercookedGame):

    """
    Wrapper on OvercookedGame that includes additional data for tutorial mechanics, most notably the introduction of tutorial "phases"

    Instance Variables:
        - curr_phase (int): Indicates what tutorial phase we are currently on
        - phase_two_score (float): The exact sparse reward the user must obtain to advance past phase 2
    """

    def __init__(
        self,
        layouts=["tutorial_0"],
        mdp_params={},
        playerZero="human",
        playerOne="AI",
        phaseTwoScore=15,
        **kwargs
    ):
        super(OvercookedTutorial, self).__init__(
            layouts=layouts,
            mdp_params=mdp_params,
            playerZero=playerZero,
            playerOne=playerOne,
            showPotential=False,
            **kwargs
        )
        self.phase_two_score = phaseTwoScore
        self.phase_two_finished = False
        self.max_time = 0
        self.max_players = 2
        self.ticks_per_ai_action = 1
        self.curr_phase = 0
        # we don't collect tutorial data
        self.write_data = False

    @property
    def reset_timeout(self):
        return 1

    def needs_reset(self):
        if self.curr_phase == 0:
            return self.score > 0
        elif self.curr_phase == 1:
            return self.score > 0
        elif self.curr_phase == 2:
            return self.phase_two_finished
        return False

    def is_finished(self):
        return not self.layouts and self.score >= float("inf")

    def reset(self):
        super(OvercookedTutorial, self).reset()
        self.curr_phase += 1

    def get_policy(self, *args, **kwargs):
        return TutorialAI()

    def apply_actions(self):
        """
        Apply regular MDP logic with retroactive score adjustment tutorial purposes
        """
        _, _, info = super(OvercookedTutorial, self).apply_actions()

        human_reward, ai_reward = info["sparse_reward_by_agent"]

        # We only want to keep track of the human's score in the tutorial
        self.score -= ai_reward

        # Phase two requires a specific reward to complete
        if self.curr_phase == 2:
            self.score = 0
            if human_reward == self.phase_two_score:
                self.phase_two_finished = True


class DummyOvercookedGame(OvercookedGame):
    """
    Class that hardcodes the AI to be random. Used for debugging
    """

    def __init__(self, layouts=["cramped_room"], **kwargs):
        super(DummyOvercookedGame, self).__init__(layouts, **kwargs)

    def get_policy(self, *args, **kwargs):
        return DummyAI()


class DummyAI:
    """
    Randomly samples actions. Used for debugging
    """

    def action(self, state):
        [action] = random.sample(
            [
                Action.STAY,
                Direction.NORTH,
                Direction.SOUTH,
                Direction.WEST,
                Direction.EAST,
                Action.INTERACT,
            ],
            1,
        )
        return action, None

    def reset(self):
        pass


class DummyComputeAI(DummyAI):
    """
    Performs simulated compute before randomly sampling actions. Used for debugging
    """

    def __init__(self, compute_unit_iters=1e5):
        """
        compute_unit_iters (int): Number of for loop cycles in one "unit" of compute. Number of
                                    units performed each time is randomly sampled
        """
        super(DummyComputeAI, self).__init__()
        self.compute_unit_iters = int(compute_unit_iters)

    def action(self, state):
        # Randomly sample amount of time to busy wait
        iters = random.randint(1, 10) * self.compute_unit_iters

        # Actually compute something (can't sleep) to avoid scheduling optimizations
        val = 0
        for i in range(iters):
            # Avoid branch prediction optimizations
            if i % 2 == 0:
                val += 1
            else:
                val += 2

        # Return randomly sampled action
        return super(DummyComputeAI, self).action(state)


class StayAI:
    """
    Always returns "stay" action. Used for debugging
    """

    def action(self, state):
        return Action.STAY, None

    def reset(self):
        pass


class TutorialAI:
    COOK_SOUP_LOOP = [
        # Grab first onion
        Direction.WEST,
        Direction.WEST,
        Direction.WEST,
        Action.INTERACT,
        # Place onion in pot
        Direction.EAST,
        Direction.NORTH,
        Action.INTERACT,
        # Grab second onion
        Direction.WEST,
        Action.INTERACT,
        # Place onion in pot
        Direction.EAST,
        Direction.NORTH,
        Action.INTERACT,
        # Grab third onion
        Direction.WEST,
        Action.INTERACT,
        # Place onion in pot
        Direction.EAST,
        Direction.NORTH,
        Action.INTERACT,
        # Cook soup
        Action.INTERACT,
        # Grab plate
        Direction.EAST,
        Direction.SOUTH,
        Action.INTERACT,
        Direction.WEST,
        Direction.NORTH,
        # Deliver soup
        Action.INTERACT,
        Direction.EAST,
        Direction.EAST,
        Direction.EAST,
        Action.INTERACT,
        Direction.WEST,
    ]

    COOK_SOUP_COOP_LOOP = [
        # Grab first onion
        Direction.WEST,
        Direction.WEST,
        Direction.WEST,
        Action.INTERACT,
        # Place onion in pot
        Direction.EAST,
        Direction.SOUTH,
        Action.INTERACT,
        # Move to start so this loops
        Direction.EAST,
        Direction.EAST,
        # Pause to make cooperation more real time
        Action.STAY,
        Action.STAY,
        Action.STAY,
        Action.STAY,
        Action.STAY,
        Action.STAY,
        Action.STAY,
        Action.STAY,
        Action.STAY,
    ]

    def __init__(self):
        self.curr_phase = -1
        self.curr_tick = -1

    def action(self, state):
        self.curr_tick += 1
        if self.curr_phase == 0:
            return (
                self.COOK_SOUP_LOOP[self.curr_tick % len(self.COOK_SOUP_LOOP)],
                None,
            )
        elif self.curr_phase == 2:
            return (
                self.COOK_SOUP_COOP_LOOP[
                    self.curr_tick % len(self.COOK_SOUP_COOP_LOOP)
                ],
                None,
            )
        return Action.STAY, None

    def reset(self):
        self.curr_tick = -1
        self.curr_phase += 1
