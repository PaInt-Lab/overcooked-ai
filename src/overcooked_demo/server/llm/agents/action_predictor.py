from collections import deque
import json
import re
from overcooked_ai_py.agents.agent import Agent
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.planning.planners import MotionPlanner, NO_COUNTERS_PARAMS
from overcooked_ai_py.mdp.actions import Action, Direction
from llm.ollama.ollama_client import query_ollama
from plan_session import PLAN_STORE
import os
from openai import OpenAI
from llm.memory.vector_memory import VectorMemory

OVERCOOKED_GAME_MECHANICS = """
## OVERCOOKED GAME MECHANICS (MDP Knowledge)

### State Variables:
- onion_hand in {none, agent, partner} - Who is holding the onion
- onion_staged in {true, false} - Is onion staged/placed somewhere accessible
- onion_at_chopping in {true, false} - Is onion at chopping station
- onion_chopped in {true, false} - Is the onion chopped (flag set when placed at chopping station)
- tomato_hand in {none, agent, partner} - Who is holding the tomato
- tomato_staged in {true, false} - Is tomato staged/placed somewhere accessible
- tomato_at_chopping in {true, false} - Is tomato at chopping station
- tomato_chopped in {true, false} - Is the tomato chopped (flag set when placed at chopping station)
- onion_in_pot in {true, false} - Is onion placed in cooking pot (raw or chopped)
- tomato_in_pot in {true, false} - Is tomato placed in cooking pot (raw or chopped)
- soup_cooking in {true, false} - Is soup actively cooking (ticker >= 1)
- soup_ready in {true, false} - Is soup ready to serve
- soup_hand in {none, agent, partner} - Who is holding the soup
- soup_staged in {true, false} - Is soup staged/placed somewhere accessible
- soup_in_pot_not_cooking in {true, false} - Is soup in pot but not cooking (ticker = -1)
- dish_hand in {none, agent, partner} - Who is holding the dish
- dish_staged in {true, false} - Is dish staged/placed somewhere accessible
- soup_served in {true, false} - Is soup delivered to serving station

Note: Cooking states are mutually exclusive: soup_cooking, soup_ready, and soup_in_pot_not_cooking cannot all be true simultaneously.

### Valid Action Sequences:
1. FetchOnion → [StageOnion OR PlaceOnionAtChopping] → FetchTomato → [StageTomato OR PlaceTomatoAtChopping] → TurnStoveOn → WaitForSoupToCook → soup_ready=true
2. FetchDish → StageDish → FetchSoup → StageSoup → ServeSoup → soup_served=true

### Transition Rules:
- FetchOnion: onion_hand=none → onion_hand=agent (from dispenser)
- StageOnion: onion_hand=agent → onion_hand=none, onion_staged=true (to staging)
- PlaceOnionAtChopping: onion_hand=agent → onion_hand=none, onion_at_chopping=true, onion_chopped=true (auto-chops)
- FetchTomato: tomato_hand=none → tomato_hand=agent (from dispenser)
- StageTomato: tomato_hand=agent → tomato_hand=none, tomato_staged=true (to staging)
- PlaceTomatoAtChopping: tomato_hand=agent → tomato_hand=none, tomato_at_chopping=true, tomato_chopped=true (auto-chops)
- PlaceInPot: onion_hand=partner OR tomato_hand=partner → onion_in_pot=true OR tomato_in_pot=true
- Cooking & TurnStoveOn: soup_in_pot_not_cooking=true → soup_cooking=true (automatic)
- Ready: soup_cooking=true → soup_ready=true (automatic)
- FetchSoup: soup_ready=true, soup_hand=none → soup_hand=agent
- StageSoup: soup_hand=agent → soup_hand=none, soup_staged=true
- FetchDish: dish_hand=none → dish_hand=agent
- StageDish: dish_hand=agent → dish_hand=none, dish_staged=true
- ServeSoup: soup_staged=true → soup_hand=agent → soup_served=true → onion_chopped=false, tomato_chopped=false (reset flags)

### Preconditions:
- Can only place onion in pot if holding onion (raw or chopped)
- Can only place tomato in pot if holding tomato (raw or chopped)
- Can only fetch soup if soup_ready=true
- Can only serve soup if soup_staged=true
- Both ingredients must be in pot before cooking can begin
- Chopping happens automatically when ingredients are placed at chopping stations
- Chopped flags are reset when soup is served

### Secondary Actions:
- pickup(onion): Pick up onion from dispenser
- pickup(tomato): Pick up tomato from dispenser
- pickup(chopped_onion): Pick up chopped onion from chopping station
- pickup(chopped_tomato): Pick up chopped tomato from chopping station
- pickup(dish): Pick up dish from dispenser
- pickup(soup): Pick up soup from staging
- place(onion, chopping_station): Place onion at chopping station (auto-chops)
- place(onion, staging_station): Place onion at staging station
- place(tomato, chopping_station): Place tomato at chopping station (auto-chops)
- place(tomato, staging_station): Place tomato at staging station
- place(chopped_onion): Place chopped onion at staging station
- place(chopped_tomato): Place chopped tomato at staging station
- place(dish): Place dish at staging station
- place(soup): Place soup at serving station
- NOOP: No secondary action needed

## TASK EXECUTION

Each call you receive has this structure:

STATE SUMMARY:
<one or two sentences describing what the robot and human hold, what's on staging counters, pots, etc., in plain English>

PLAN:

1. secondary: [<labels>], primary: [<labels>]
2. secondary: [<labels>], primary: [<labels>]
   ...
   N) secondary: [<labels>], primary: [<labels>]

Your job:

1. **Analyze current state** against the MDP knowledge above to understand game mechanics
2. **Identify which plan-step (1...N)** is currently active based on state and progress
3. **From that step's primary list**, choose exactly one of the canonical primary events (reuse the text exactly as given)
4. **From the same step's secondary list**, choose exactly one of:
   • pickup(onion)  
   • pickup(tomato)
   • pickup(dish)
   • pickup(soup)
   • place(onion, chopping_station)
   • place(onion, staging_station)
   • place(tomato, chopping_station)
   • place(tomato, staging_station)
   • place(dish)
   • place(soup)
   • NOOP

**EXACT DECISION RULES for secondary actions:**

**Choose pickup(onion) when:**
- onion_hand="none" AND onion_staged=false AND onion_at_chopping=false AND onion_in_pot=false AND soup_staged=false AND soup_hand=none
- (Need to fetch onion for processing)

**Choose place(onion, chopping_station) when:**
- onion_hand="agent" AND onion_chopped=false
- (Agent is holding onion and plan calls for chopping)

**Choose place(onion, staging_station) when:**
- onion_hand="agent" AND onion_staged=false
- (Agent is holding onion and plan does not call for chopping)

**Choose pickup(tomato) when:**
- tomato_hand="none" AND tomato_staged=false AND tomato_at_chopping=false AND tomato_in_pot=false AND soup_staged=false AND soup_hand=none
- (Need to fetch tomato for processing)

**Choose place(tomato, chopping_station) when:**
- tomato_hand="agent" AND tomato_chopped=false
- (Agent is holding tomato and plan calls for chopping)

**Choose place(tomato, staging_station) when:**
- tomato_hand="agent" AND tomato_staged=false
- (Agent is holding tomato and plan does not call for chopping)

**Choose pickup(chopped_onion) when:**
- onion_chopped=true AND onion_at_chopping=true AND onion_hand="none"
- (Chopped onion is ready at chopping station)

**Choose place(chopped_onion) when:**
- onion_chopped=true AND onion_hand="agent" AND onion_staged=false
- (Agent is holding chopped onion and needs to stage it)

**Choose pickup(chopped_tomato) when:**
- tomato_chopped=true AND tomato_at_chopping=true AND tomato_hand="none"
- (Chopped tomato is ready at chopping station)

**Choose place(chopped_tomato) when:**
- tomato_chopped=true AND tomato_hand="agent" AND tomato_staged=false
- (Agent is holding chopped tomato and needs to stage it)

**Choose pickup(dish) when:**
- dish_staged=false AND soup_cooking=true
- (Need dish ready when soup is cooking)

**Choose place(dish) when:**
- dish_hand="agent" AND dish_staged=false
- (Agent is holding dish and needs to stage it)

**Choose pickup(soup) when:**
- soup_hand="none" AND soup_staged=true
- (Soup is staged and ready for serving)

**Choose place(soup) when:**
- soup_hand="agent" AND soup_served=false
- (Agent is holding soup and needs to serve it)

**Choose NOOP when:**
- All required items are already staged or in progress
- Waiting for cooking to complete (soup_cooking=true) AND dish_staged=true
- Waiting for partner to complete their action
- No immediate action needed based on current plan step

Return **only** these two lines (no extra commentary):

primary: <exact primary event text>  
secondary: <one of pickup(onion|tomato|dish|soup) or place(onion|tomato|dish|soup) or NOOP>
"""


OVERCOOKED_MODEL = "ft:gpt-4o-mini-2024-07-18:personal:ap-onion-tomato-chopped:BysMfU2m"

def get_model_for_task(task_title: str) -> str:
    """
    Select the appropriate model based on the task title.
    Returns the model name to use for the API call.
    """
    # Always use the default OVERCOOKED_MODEL for now
    # This avoids issues with specialized models that may not be built
    return OVERCOOKED_MODEL

def serialize_state(state, mdp) -> str:
    """
    Convert OvercookedState and MDP into a compact JSON string for the LLM prompt.
    """
    state_info = {
        'terrain': mdp.terrain_mtx,
        'state': state.to_dict()
    }
    return json.dumps(state_info)


def _bfs_fallback(start, goal, terrain, goal_orientation=None):
    """
    Return a list of (delta_col, delta_row) moves to walk from start to goal on terrain,
    ignoring orientation. Guaranteed to find a path if one exists.
    Always adds goal orientation at the end if provided.
    """
    H, W = len(terrain), len(terrain[0])
    visited = {start}
    parent = {}
    queue = deque([start])

    # Four cardinal directions: (delta_col, delta_row) - right, left, down, up
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    while queue:
        col, row = queue.popleft()
        if (col, row) == goal:
            # Reconstruct path of (dc, dr) tuples
            path = []
            cur = goal
            while cur != start:
                prev = parent[cur]
                dc = cur[0] - prev[0]
                dr = cur[1] - prev[1]
                path.append((dc, dr))
                cur = prev
            path = list(reversed(path))  
            
            # Always add goal orientation if provided (agent can't move through objects anyway)
            if goal_orientation is not None and path[-1] != goal_orientation:
                if goal_orientation == (1, 0): 
                    path.append((1, 0))
                elif goal_orientation == (-1, 0):  
                    path.append((-1, 0))
                elif goal_orientation == (0, 1):  
                    path.append((0, 1))
                elif goal_orientation == (0, -1):  
                    path.append((0, -1))

            
            # Add INTERACT at the very end
            path.append(Action.INTERACT)
            
            return path

        for dc, dr in directions:
            new_col = col + dc
            new_row = row + dr
            if (
                0 <= new_row < H and
                0 <= new_col < W and
                terrain[new_row][new_col] != 'X' and
                (new_col, new_row) not in visited
            ):
                visited.add((new_col, new_row))
                parent[(new_col, new_row)] = (col, row)
                queue.append((new_col, new_row))

    return []

def query_openai(prompt: str, model: str = OVERCOOKED_MODEL, temperature: float = 0.0) -> str:
    """
    Query the OpenAI API with the given prompt and return the response text.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set.")
   
    # Initialize the client with the API key
    client = OpenAI(api_key=api_key)
   
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=256,
        )
       
        content = response.choices[0].message.content
        if content is not None:
            return content.strip()
        return ""
       
    except Exception as e:
        print(f"Error querying OpenAI: {e}")
        return ""


class ActionPredictorAgent(Agent):
    """
    An agent that uses LLM-based action prediction with blocking prevention.
    
    This agent:
    1. Uses an LLM to predict high-level actions based on game state and plan
    2. Converts high-level actions into low-level movement commands
    3. Prevents blocking important staging tiles when doing NOOP actions
    4. Automatically moves to safe positions when blocking is detected
    
    Important tiles that are protected from blocking:
    - Staging tiles for onions, dishes, and soup
    - Frontier tiles adjacent to important locations (stoves, ingredient spawns, etc.)
    - Delivery tiles and their adjacent positions
    """
    def __init__(self):
        super().__init__()
        self.mdp = None
        self.planner = None
        self.ingredient_spawns = []
        self.onion_spawns = []
        self.tomato_spawns = []
        self.stove_tiles = []
        self.dish_spawns = []
        self.delivery_tiles = []
        self.staging_tiles = []
        self.onion_chopping_stations = []
        self.tomato_chopping_stations = []
        self.last_info = None
        self.cleaned_terrain = None
        self.last_summary = None
        self.agent_index = None
        self.memory = VectorMemory()  # Initialize vector memory
        # Persistent chopped flags that survive across state summaries
        self.onion_chopped = False
        self.tomato_chopped = False

    def set_agent_index(self, agent_index: int):
        super().set_agent_index(agent_index)
        self.agent_index = agent_index  # Make sure this is set!

    TERRAIN_MAPPING = {
        "X": "Wall",
        "P": "Stove",
        "O": "Onions",
        "T": "Tomatoes",  
        "D": "DishSpawn",
        "S": "Serving",
        " ": "Empty"
    }

    def set_mdp(self, mdp: OvercookedGridworld):
        super().set_mdp(mdp)
        self.mdp = mdp
        terrain = mdp.terrain_mtx
        self.ingredient_spawns = [(j, i)
                            for i, row in enumerate(terrain)
                            for j, c in enumerate(row)
                            if c in ('O', 'T')]
        self.onion_spawns = [(j, i)
                            for i, row in enumerate(terrain)
                            for j, c in enumerate(row)
                            if c == 'O']
        self.tomato_spawns = [(j, i)
                            for i, row in enumerate(terrain)
                            for j, c in enumerate(row)
                            if c == 'T']
        self.stove_tiles       = [(j, i) # Need to manipulate this for the bot to place dishes next to stove
                            for i, row in enumerate(terrain)
                            for j, c in enumerate(row)
                            if c == 'P']
        self.dish_spawns       = [(j, i)
                            for i, row in enumerate(terrain)
                            for j, c in enumerate(row)
                            if c == 'D']
        self.delivery_tiles    = [(j, i)
                            for i, row in enumerate(terrain)
                            for j, c in enumerate(row)
                            if c == 'S']
        # Assign staging stations per stove with directional logic
        self.onion_staging_tiles = []
        self.tomato_staging_tiles = []
        self.dish_staging_tiles = []
        H, W = len(terrain), len(terrain[0]) 
        
        for (stove_c, stove_r) in self.stove_tiles:
            # Find all adjacent staging positions for this stove
            adjacent_staging = []
            for dc, dr in [(1,0), (-1,0), (0,1), (0,-1)]:
                nc, nr = stove_c + dc, stove_r + dr
                if 0 <= nr < H and 0 <= nc < W and terrain[nr][nc] == 'X':
                    adjacent_staging.append((nc, nr, dc, dr))
            
            # Categorize by direction relative to stove
            left_stations = [(c, r) for c, r, dc, dr in adjacent_staging if dc == -1]  # left of stove
            right_stations = [(c, r) for c, r, dc, dr in adjacent_staging if dc == 1]  # right of stove  
            top_stations = [(c, r) for c, r, dc, dr in adjacent_staging if dr == -1]   # above stove
            bottom_stations = [(c, r) for c, r, dc, dr in adjacent_staging if dr == 1] # below stove
            
            # Assign onion staging: prefer left, fallback to bottom
            if left_stations:
                self.onion_staging_tiles.extend(left_stations)
            elif bottom_stations:
                self.onion_staging_tiles.extend(bottom_stations)
            
            # Tomato staging spots are the same as onion staging spots
            self.tomato_staging_tiles = self.onion_staging_tiles.copy()
            
            # Assign dish staging: prefer right, fallback to top (if not used by tomato)
            if right_stations:
                self.dish_staging_tiles.extend(right_stations)
            elif top_stations:
                self.dish_staging_tiles.extend(top_stations)

            self.soup_staging_tiles = self.dish_staging_tiles



        # Create chopping stations adjacent to staging tiles
        self.onion_chopping_stations = []
        self.tomato_chopping_stations = []
        
        # Find chopping stations for onion staging tiles (prefer left, fallback to top)
        for staging_pos in self.onion_staging_tiles:
            staging_c, staging_r = staging_pos
            # Try left first, then top
            for dc, dr in [(-1, 0), (0, -1)]:  # left, then top
                chopping_c, chopping_r = staging_c + dc, staging_r + dr
                if (0 <= chopping_r < H and 0 <= chopping_c < W and 
                    terrain[chopping_r][chopping_c] == 'X' and
                    (chopping_c, chopping_r) not in self.onion_chopping_stations):
                    self.onion_chopping_stations.append((chopping_c, chopping_r))
                    break
        
        # Tomato chopping stations are the same as onion chopping stations
        self.tomato_chopping_stations = self.onion_chopping_stations.copy()



        self.ingredient_frontier = self._compute_frontier(self.ingredient_spawns, terrain)
        self.onion_frontier = self._compute_frontier(self.onion_spawns, terrain)
        self.tomato_frontier = self._compute_frontier(self.tomato_spawns, terrain)
        self.stove_frontier      = self._compute_frontier(self.stove_tiles,       terrain) 
        self.dish_frontier       = self._compute_frontier(self.dish_spawns,       terrain)
        self.delivery_frontier   = self._compute_frontier(self.delivery_tiles,    terrain)
        self.onion_staging_frontier = self._compute_frontier(self.onion_staging_tiles, terrain)
        self.tomato_staging_frontier = self._compute_frontier(self.tomato_staging_tiles, terrain)
        self.dish_staging_frontier  = self._compute_frontier(self.dish_staging_tiles,  terrain)
        self.soup_staging_frontier  = self._compute_frontier(self.soup_staging_tiles,  terrain)
        self.onion_chopping_frontier = self._compute_frontier(self.onion_chopping_stations, terrain)
        self.tomato_chopping_frontier = self._compute_frontier(self.tomato_chopping_stations, terrain)
        
        my_goals = {
            'ingredient': self.ingredient_spawns,
            'pot':        self.stove_tiles,
            'dish':       self.dish_spawns,
            'delivery':   self.delivery_tiles
        } 
        
        self.cleaned_terrain = [
            [ self.TERRAIN_MAPPING.get(cell, "Unknown") for cell in row ]
            for row in terrain
        ]

        self.planner = MotionPlanner(mdp, counter_goals=my_goals)

    def set_plan(self, session_id: str):
        """Attach the full PlanSession to this agent."""
        self.plan = PLAN_STORE[session_id]

    def _compute_frontier(self, tiles, terrain):
        """
        Return the set of walkable tiles adjacent to any tile in 'tiles'.
        """
        H, W = len(terrain), len(terrain[0])
        frontier = set()
        WALKABLE = {' '}

        for c, r in tiles:
            for dc, dr in [(1,0), (-1,0), (0,1), (0,-1)]:
                nc, nr = c + dc, r + dr
                if (
                    0 <= nr < H and 
                    0 <= nc < W and 
                    terrain[nr][nc] in WALKABLE
                ):
                    orient = (-dc, -dr)
                    frontier.add(((nc, nr), orient))
        return list(frontier)

    def summarize_state(self, state, info):
        """
        Extract exactly the predicates we need for the LLM based on new MDP:
        - onion_hand: one of "none", "agent", "partner" 
        - onion_staged: is raw onion on any onion‐staging tile?
        - onion_at_chopping: is raw onion at chopping station?
        - onion_chopped: is raw onion chopped?
        - tomato_hand: one of "none", "agent", "partner"
        - tomato_staged: is raw tomato on any staging tile? 
        - tomato_at_chopping: is raw tomato at chopping station?
        - tomato_chopped: is raw tomato chopped?
        - onion_in_pot: is any onion in any stove tile?
        - tomato_in_pot: is any tomato in any stove tile?
        - soup_cooking: is soup actively cooking?
        - soup_ready: is soup ready to serve?
        - soup_in_pot_not_cooking: is soup in pot but not cooking?
        - dish_hand: one of "none", "agent", "partner"
        - dish_staged: is any clean dish on any dish‐staging tile?
        - soup_in_dish: is any soup on any dish‐staging tile?
        - soup_served: did the last transition include a soup_delivery?
        """
        sd = state.to_dict()

        def held_item(player_dict):
            held = player_dict["held_object"]
            if held is None:
                return "none"
            # raw ingredients sometimes come back under "name"
            if held.get("name") in ("onion", "tomato"):
                return held["name"]
            # some objects still use "ingredient"
            if held.get("ingredient") in ("onion", "tomato"):
                return held["ingredient"]
            if held.get("name") in ("dish", "soup"):
                return held["name"]
            return "none"

        # 1) who holds what
        me   = sd["players"][self.agent_index]
        them = sd["players"][1 - (self.agent_index or 0)]
        agent_item   = held_item(me)
        partner_item = held_item(them)

        # Map to new MDP variables
        onion_hand = "none"
        if agent_item == "onion":
            onion_hand = "agent"
        elif partner_item == "onion":
            onion_hand = "partner"

        tomato_hand = "none"
        if agent_item == "tomato":
            tomato_hand = "agent"
        elif partner_item == "tomato":
            tomato_hand = "partner"

        dish_hand = "none"
        if agent_item == "dish":
            dish_hand = "agent"
        elif partner_item == "dish":
            dish_hand = "partner"

        # Check for soup in hands
        soup_hand = "none"
        if agent_item == "soup":
            soup_hand = "agent"
        elif partner_item == "soup":
            soup_hand = "partner"

        # 2) collect what's on every tile and analyze soup states
        tile_contents = {}
        onion_in_pot = False
        tomato_in_pot = False
        soup_cooking = False
        soup_ready = False
        soup_in_pot_not_cooking = False
        
        for obj in sd["objects"]:
            p    = tuple(obj["position"])
            name = obj.get("ingredient") or obj.get("name")
            tile_contents.setdefault(p, []).append(name)
            
            # Special handling for soup objects that contain ingredients
            if obj.get("name") == "soup" and obj.get("_ingredients"):
                for ingredient in obj["_ingredients"]:
                    ing_name = ingredient.get("name")
                    if ing_name:
                        tile_contents.setdefault(p, []).append(ing_name)
                
                # Check if soup is on stove (pot)
                if p in self.stove_tiles:
                    # Check what ingredients are in the pot
                    for ingredient in obj["_ingredients"]:
                        ing_name = ingredient.get("name")
                        if ing_name in ("onion", "chopped_onion"):
                            onion_in_pot = True
                        elif ing_name in ("tomato", "chopped_tomato"):
                            tomato_in_pot = True
                    
                    # Check cooking states
                    cooking_tick = obj.get("cooking_tick", -1)
                    is_cooking = obj.get("is_cooking", False)
                    is_ready = obj.get("is_ready", False)
                    
                    if is_cooking and cooking_tick >= 1:
                        soup_cooking = True
                    elif is_ready:
                        soup_ready = True
                    elif cooking_tick == -1:
                        soup_in_pot_not_cooking = True

        # 3) Check for ingredients at chopping stations and set chopped flags
        onion_staged = any(
            "onion" in tile_contents.get(pos, [])
            for pos in self.onion_staging_tiles
        )
        
        onion_at_chopping = any(
            "onion" in tile_contents.get(pos, [])
            for pos in self.onion_chopping_stations
        )
        
        # Update persistent chopped flag if onion is at chopping station (auto-chops)
        if onion_at_chopping:
            self.onion_chopped = True
        # Note: self.onion_chopped stays True even after pickup until soup is served

        # 4) Separate staging checks for raw and chopped tomatoes
        tomato_staged = any(
            "tomato" in tile_contents.get(pos, [])
            for pos in self.tomato_staging_tiles
        )
        
        tomato_at_chopping = any(
            "tomato" in tile_contents.get(pos, [])
            for pos in self.tomato_chopping_stations
        )
        
        # Update persistent chopped flag if tomato is at chopping station (auto-chops)
        if tomato_at_chopping:
            self.tomato_chopped = True
        # Note: self.tomato_chopped stays True even after pickup until soup is served

        # 5) dish_staged?
        dish_staged = any(
            "dish" in tile_contents.get(pos, [])
            for pos in self.dish_staging_tiles
        )

        # 6) soup_staged?
        soup_staged = any(
            "soup" in tile_contents.get(pos, [])
            for pos in self.soup_staging_tiles
        )

        # 7) soup_served?
        soup_served = False
        if info:
            soup_served = any(
                info.get("event_infos", {})
                    .get("soup_delivery", [False, False])
            )

        # Reset chopped flags when soup is served (end of meal)
        if soup_served:
            self.onion_chopped = False
            self.tomato_chopped = False

        return {
            "onion_hand":               onion_hand,
            "onion_staged":             onion_staged,
            "onion_at_chopping":        onion_at_chopping,
            "onion_chopped":            self.onion_chopped,
            "tomato_hand":              tomato_hand,
            "tomato_staged":            tomato_staged,
            "tomato_at_chopping":       tomato_at_chopping,
            "tomato_chopped":           self.tomato_chopped,
            "onion_in_pot":             onion_in_pot,
            "tomato_in_pot":            tomato_in_pot,
            "soup_cooking":             soup_cooking,
            "soup_ready":               soup_ready,
            "soup_in_pot_not_cooking":  soup_in_pot_not_cooking,
            "dish_hand":                dish_hand,
            "dish_staged":              dish_staged,
            "soup_hand":                soup_hand,
            "soup_staged":              soup_staged,
            "soup_served":              soup_served
        }

    
    def _parse_function_call(self, response):
        """
        Parse the LLM response to extract both primary event and function call.
        Expected format:
          primary: <event description>
          secondary: pickup(onion) or place(onion, chopping_station) or NOOP
        """

        # grab the primary
        primary_match = re.search(r'primary:\s*(.+?)(?:\n|secondary:|$)',
                                  response,
                                  re.IGNORECASE)
        primary_event = primary_match.group(1).strip() if primary_match else "Unknown Event"

        # check for explicit NOOP
        if re.search(r'secondary:\s*noop', response, re.IGNORECASE):
            return primary_event, "NOOP", None

        # Parse pickup actions
        pickup_match = re.search(r'pickup\(([^)]+)\)', response)
        if pickup_match:
            item = pickup_match.group(1).strip()
            if item in ["onion", "tomato", "chopped_onion", "chopped_tomato", "dish", "soup"]:
                return primary_event, "pickup", item

        # Parse place actions with destination
        place_match = re.search(r'place\(([^,]+),\s*([^)]+)\)', response)
        if place_match:
            item = place_match.group(1).strip()
            destination = place_match.group(2).strip()
            if item in ["onion", "tomato"] and destination in ["chopping_station", "staging_station"]:
                return primary_event, "place", (item, destination)

        # Parse place actions without destination (for items with fixed destinations)
        place_simple_match = re.search(r'place\(([^)]+)\)', response)
        if place_simple_match:
            item = place_simple_match.group(1).strip()
            if item in ["chopped_onion", "chopped_tomato", "dish", "soup"]:
                return primary_event, "place", (item, "default")

        # final fallback
        return primary_event, "pickup", "onion"
    
    def _is_blocking_important_tile(self, my_pos: tuple, state) -> bool:
        """
        Check if the agent is currently blocking an important staging tile.
        Returns True if blocking, False otherwise.
        """
        # Get all important tiles that shouldn't be blocked
        important_tiles = set()
        
        # Add all staging tiles
        important_tiles.update(self.onion_staging_tiles)
        important_tiles.update(self.tomato_staging_tiles)
        important_tiles.update(self.dish_staging_tiles)
        important_tiles.update(self.soup_staging_tiles)
        
        # Add all chopping stations
        important_tiles.update(self.onion_chopping_stations)
        important_tiles.update(self.tomato_chopping_stations)
        
        # Add all frontier tiles (adjacent to important locations)
        important_tiles.update([pos for pos, _ in self.ingredient_frontier])
        important_tiles.update([pos for pos, _ in self.onion_frontier])
        important_tiles.update([pos for pos, _ in self.tomato_frontier])
        important_tiles.update([pos for pos, _ in self.stove_frontier])
        important_tiles.update([pos for pos, _ in self.dish_frontier])
        important_tiles.update([pos for pos, _ in self.delivery_frontier])
        important_tiles.update([pos for pos, _ in self.onion_staging_frontier])
        important_tiles.update([pos for pos, _ in self.tomato_staging_frontier])
        important_tiles.update([pos for pos, _ in self.dish_staging_frontier])
        important_tiles.update([pos for pos, _ in self.soup_staging_frontier])
        important_tiles.update([pos for pos, _ in self.onion_chopping_frontier])
        important_tiles.update([pos for pos, _ in self.tomato_chopping_frontier])
        
        # Check if current position is blocking an important tile
        is_blocking = my_pos in important_tiles
        
        return is_blocking

    def _find_safe_position(self, my_pos: tuple, state) -> tuple:
        """
        Find a safe position to move to that doesn't block important tiles.
        Returns (new_position, action_plan) or (my_pos, []) if no safe move found.
        """
        if not self.mdp:
            return my_pos, []
            
        terrain = self.mdp.terrain_mtx
        H, W = len(terrain), len(terrain[0])
        
        # Get all important tiles to avoid
        important_tiles = set()
        important_tiles.update(self.onion_staging_tiles)
        important_tiles.update(self.tomato_staging_tiles)
        important_tiles.update(self.dish_staging_tiles)
        important_tiles.update(self.soup_staging_tiles)
        important_tiles.update(self.onion_chopping_stations)
        important_tiles.update(self.tomato_chopping_stations)
        important_tiles.update([pos for pos, _ in self.ingredient_frontier])
        important_tiles.update([pos for pos, _ in self.onion_frontier])
        important_tiles.update([pos for pos, _ in self.tomato_frontier])
        important_tiles.update([pos for pos, _ in self.stove_frontier])
        important_tiles.update([pos for pos, _ in self.dish_frontier])
        important_tiles.update([pos for pos, _ in self.delivery_frontier])
        important_tiles.update([pos for pos, _ in self.onion_staging_frontier])
        important_tiles.update([pos for pos, _ in self.tomato_staging_frontier])
        important_tiles.update([pos for pos, _ in self.dish_staging_frontier])
        important_tiles.update([pos for pos, _ in self.soup_staging_frontier])
        important_tiles.update([pos for pos, _ in self.onion_chopping_frontier])
        important_tiles.update([pos for pos, _ in self.tomato_chopping_frontier])
        
        # Get other player position to avoid blocking them
        other_player_pos = state.player_positions[1 - self.agent_index]
        
        # Find safe positions within reasonable distance (max 3 steps)
        safe_positions = []
        for distance in range(1, 4):  # Check 1, 2, 3 steps away
            for dc in range(-distance, distance + 1):
                for dr in range(-distance, distance + 1):
                    if abs(dc) + abs(dr) == distance:  # Manhattan distance
                        new_col = my_pos[0] + dc
                        new_row = my_pos[1] + dr
                        
                        # Check bounds
                        if 0 <= new_row < H and 0 <= new_col < W:
                            new_pos = (new_col, new_row)
                            
                            # Check if position is walkable and not important
                            if (terrain[new_row][new_col] == ' ' and 
                                new_pos not in important_tiles and
                                new_pos != other_player_pos):
                                safe_positions.append(new_pos)
            
            # If we found safe positions at this distance, stop searching
            if safe_positions:
                break
        
        # If no safe positions found, stay put
        if not safe_positions:
            return my_pos, []
        
        # Choose the closest safe position
        best_pos = min(safe_positions, key=lambda pos: abs(pos[0] - my_pos[0]) + abs(pos[1] - my_pos[1]))
        
        # Generate action plan to move to safe position
        my_ori = state.to_dict()["players"][self.agent_index]["orientation"]
        start_pair = (my_pos, tuple(my_ori))
        goal_pair = (best_pos, tuple(my_ori))  # Keep same orientation
        
        action_plan = self._get_action_plan(start_pair, goal_pair)
        
        return best_pos, action_plan

    def _get_action_plan(self, start_pair, goal_pair):
        """
        Get action plan between two position/orientation pairs.
        Returns the action plan using the motion planner with BFS fallback.
        """

        if self.mdp is None:
            return []
        terrain = self.mdp.terrain_mtx

        if self.planner is None:
            # Fallback to BFS if no planner available
            start_pos, start_ori = start_pair
            goal_pos, goal_ori = goal_pair
            return _bfs_fallback(start_pos, goal_pos, terrain, goal_ori)
        
        try:
            # Plan A: orientation‐specific
            action_plan, _, _ = self.planner.get_plan(start_pair, goal_pair)
            # print(f"Plan get_plan: {action_plan}")
            return action_plan
        except KeyError:
            try:
                # Plan B: orientation‐agnostic  
                goal_pos, goal_ori = goal_pair
                action_plan, _, _ = self.planner.action_plan_from_positions(

                    [goal_pos], start_pair, goal_pair
                )
                # print(f"Plan action_plan_from_positions: {action_plan}")
                return action_plan
            except Exception:
                # Plan C: guaranteed BFS fallback
                start_pos, start_ori = start_pair
                goal_pos, goal_ori = goal_pair
                action_plan = _bfs_fallback(start_pos, goal_pos, terrain, goal_ori)
                # print(f"Plan BFS fallback: {action_plan}")
                return action_plan
            
    def _get_frontier_for_action(self, action: str, item: str, destination: str = None):
        """Get the appropriate frontier for a given action and item."""
        if action == "pickup":
            # Pickup actions - always from specific locations
            frontier_map = {
                "onion": self.onion_frontier,  # From onion dispenser
                "tomato": self.tomato_frontier,  # From tomato dispenser
                "chopped_onion": self.onion_chopping_frontier,  # From chopping station
                "chopped_tomato": self.tomato_chopping_frontier,  # From chopping station
                "dish": self.dish_frontier,  # From dish dispenser
                "soup": self.soup_staging_frontier,  # From soup staging
            }
            return frontier_map.get(item)
            
        elif action == "place":
            # Place actions - simplified destination logic
            if destination == "chopping_station":
                # Place at chopping station (only for onions and tomatoes)
                frontier_map = {
                    "onion": self.onion_chopping_frontier,
                    "tomato": self.tomato_chopping_frontier,
                }
                return frontier_map.get(item)
            elif destination == "staging_station":
                # Place at appropriate staging station (only for onions and tomatoes)
                frontier_map = {
                    "onion": self.onion_staging_frontier,
                    "tomato": self.tomato_staging_frontier,
                }
                return frontier_map.get(item)
            else:
                # Default destinations for other items (no destination needed)
                frontier_map = {
                    "chopped_onion": self.onion_staging_frontier,  # Always to onion staging
                    "chopped_tomato": self.tomato_staging_frontier,  # Always to tomato staging
                    "dish": self.dish_staging_frontier,  # Always to dish staging
                    "soup": self.delivery_frontier,  # Always to serving station
                }
                return frontier_map.get(item)
            
        else:
            return None

    def _find_nearest_goal(self, choices, my_pos: tuple, my_ori: tuple) -> tuple:
        """Find the nearest goal from a list of choices."""
        if not choices:
            return (my_pos, tuple(my_ori))

        def sort_key(mo):
            (c, r), _ = mo
            # primary: Manhattan distance
            dist = abs(c - my_pos[0]) + abs(r - my_pos[1])
            # secondary: prefer smaller col, then smaller row
            return (dist, c, r)

        goal_pos, goal_orient = min(choices, key=sort_key)
        return (goal_pos, goal_orient)

    def _move_to(self, action: str, item: str, start_pos: tuple, start_ori: tuple, destination: str = None):
        """Move to the appropriate location for the given action and item."""
        choices = self._get_frontier_for_action(action, item, destination)
        if choices is None:
            return []
        
        goal = self._find_nearest_goal(choices, start_pos, start_ori)
        start_pair = (start_pos, tuple(start_ori))
        return self._get_action_plan(start_pair, goal)

    def _drop_item_at_proper_location(self, item_to_drop: str, start_pos: tuple, start_ori: tuple):
        """
        Drop the specified item at its proper staging location.
        Returns action plan to drop the item.
        """
        # Map items to their proper staging locations
        staging_map = {
            "onion": self.onion_staging_frontier,
            "tomato": self.tomato_staging_frontier,
            "dish": self.dish_staging_frontier,
            "soup": self.soup_staging_frontier
        }
        
        choices = staging_map.get(item_to_drop, [])
        if not choices:
            return []
        
        goal = self._find_nearest_goal(choices, start_pos, start_ori)
        start_pair = (start_pos, tuple(start_ori))
        drop_plan = self._get_action_plan(start_pair, goal)
        # Add INTERACT action to drop the item
        drop_plan.append(Action.INTERACT)
        return drop_plan

    def PickUp(self, item, start_pos, start_ori):
        """Returns an action plan to pick up the specified item."""
        action_plan = self._move_to("pickup", item, start_pos, start_ori)
        # Add INTERACT action to pick up the item
        if action_plan:
            action_plan.append(Action.INTERACT)
        return action_plan

    def Place(self, item, start_pos, start_ori, destination: str = "default"):
        """Returns an action plan to place the specified item at the destination."""
        # For onions and tomatoes, destination is required
        # For other items, destination is ignored and uses default behavior
        if destination == "default":
            # Use default destination based on item type
            action_plan = self._move_to("place", item, start_pos, start_ori, destination)
        else:
            # Use specified destination (for onions and tomatoes)
            action_plan = self._move_to("place", item, start_pos, start_ori, destination)
        
        # Add INTERACT action to place the item
        if action_plan:
            action_plan.append(Action.INTERACT)
        return action_plan



    def action(self, state):
        assert self.agent_index is not None, "agent_index is None in action!"
        if not hasattr(self, "plan") or self.plan is None:
            raise RuntimeError("No plan set for ActionPredictorAgent! Did you forget to call set_plan()?")
        info = getattr(self, "last_info", {})       
        self.last_summary = self.summarize_state(state, self.last_info)
        print(f"Current position: {state.player_positions[self.agent_index]}")
        print(f"State summary: {self.last_summary}")

        # Retrieve similar past experiences from memory
        # task_title = getattr(self.plan, 'task_title', 'Unknown Task')
        # game_context = ""
        # try:
        #     game_context = self.memory.get_game_context(
        #         current_state=self.last_summary, 
        #         task_title=task_title, 
        #         k=3
        #     )
        # except Exception as e:
        #     print(f"Memory retrieval failed: {e}")
        #     game_context = ""

        plan_lines = []
        for idx, ev in enumerate(self.plan.events):
            sec = ev["secondary"]
            prim = ev["primary"]
            plan_lines.append(f"{idx+1}) secondary: {sec}, primary: {prim}")
        plan_text = "\n".join(plan_lines)

        # Include game context in the prompt if available
        # context_section = ""
        # if game_context:
        #     context_section = f"SIMILAR PAST EXPERIENCES:\n{game_context}\n\n"


        prompt = (
        f"{OVERCOOKED_GAME_MECHANICS}\n\n"
        # f"{context_section}"
        f"STATE SUMMARY:\n{self.last_summary}\n\n"
        f"PLAN:\n{plan_text}\n\n"
        "Based on the current state and plan, determine what the robot should do."
        "Respond in this exact format:\n"
        "primary: <description of the primary event from the plan>\n"
        "secondary: pickup(onion) or pickup(tomato) or pickup(dish) or pickup(soup) or place(onion, chopping_station) or place(onion, staging_station) or place(tomato, chopping_station) or place(tomato, staging_station) or place(dish) or place(soup) or NOOP\n"
        "Choose the appropriate primary event and secondary action based on the current state and plan."
    )

        # Select the appropriate model based on task title
        task_title = getattr(self.plan, 'task_title', 'Unknown Task')
        model_name = get_model_for_task(task_title)

        response = query_openai(prompt, model=model_name)

        # Parse the function call from LLM response
        primary_event, func_name, item = self._parse_function_call(response)
        print(f"Primary event: {primary_event}")
        print(f"Secondary action: {func_name}({item})")

        # Get current position and orientation
        my_pos = state.player_positions[self.agent_index]
        my_ori = state.to_dict()["players"][self.agent_index]["orientation"]

        if func_name == "NOOP":
            # Check if the agent is blocking important tiles
            if self._is_blocking_important_tile(my_pos, state):
                # If blocking, find a safe position to move to
                safe_pos, action_plan = self._find_safe_position(my_pos, state)
                if action_plan:
                    # Return first action from the plan to move to safe position
                    move = action_plan[0] if action_plan else Action.STAY
                    return move, {
                        "primary_event": primary_event,
                        "function_call": "NOOP_MOVE_TO_SAFE",
                        "action_plan": action_plan,
                        "response": response,
                        "blocking_prevention": True
                    }
                else:
                    # If no safe move found, stay put
                    return Action.STAY, {
                        "primary_event": primary_event,
                        "function_call": "NOOP",
                        "action_plan": [],
                        "response": response,
                        "blocking_prevention": False
                    }
            else:
                # If not blocking, stay put
                return Action.STAY, {
                    "primary_event": primary_event,
                    "function_call": "NOOP",
                    "action_plan": [],
                    "response": response,
                    "blocking_prevention": False
                }

        # Execute the compound action
        if func_name == "pickup":
            action_plan = self.PickUp(item, my_pos, my_ori)
        elif func_name == "place":
            item_to_place, destination = item
            action_plan = self.Place(item_to_place, my_pos, my_ori, destination)
        else:
            # Fallback to simple movement
            action_plan = [Action.STAY]

        # Return first action from the plan
        move = action_plan[0] if action_plan else Action.STAY
        print(f"Next Move: {move}")

        print("\n\n")
        
        # Store in memory for future reference
        task_title = getattr(self.plan, 'task_title', 'Unknown Task')
        self.memory.add_game_memory(
            state_summary=self.last_summary,
            action_info={
                "primary_event": primary_event,
                "function_call": f"{func_name}({item})"
            },
            task_title=task_title
        )
        
        return move, {
            "primary_event": primary_event,
            "function_call": f"{func_name}({item})",
            "action_plan": action_plan,
            "response": response
        }

    def actions(self, states, agent_indices):
        return [self.action(s) for s in states]


