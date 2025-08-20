OVERCOOKED_GAME_MECHANICS = """
## OVERCOOKED GAME MECHANICS (MDP Knowledge)

### State Variables:
- onion_hand in {none, agent, partner} - Who is holding the onion
- onion_staged in {true, false} - Is onion staged/placed somewhere accessible  
- ingredient_in_pot in {true, false} - Is ingredient placed in cooking pot
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
1. FetchOnion -> StageOnion -> PlaceIngredientInPot -> [Human: TurnStoveOn] -> WaitForSoupToCook -> soup_ready=true
2. FetchDish -> StageDish -> FetchSoup -> StageSoup -> ServeSoup -> soup_served=true

### Transition Rules:
- FetchOnion: onion_hand=none -> onion_hand=agent
- StageOnion: onion_hand=agent -> onion_hand=none, onion_staged=true
- PlaceIngredientInPot: onion_hand=partner -> onion_hand=none, ingredient_in_pot=true
- Cooking & TurnStoveOn: soup_in_pot_not_cooking=true -> soup_cooking=true 
- Ready: soup_cooking=true -> soup_ready=true (automatic)
- FetchSoup: soup_ready=true, soup_hand=none -> soup_hand=agent
- StageSoup: soup_hand=agent -> soup_hand=none, soup_staged=true
- FetchDish: dish_hand=none -> dish_hand=agent
- StageDish: dish_hand=agent -> dish_hand=none, dish_staged=true
- ServeSoup: soup_staged=true -> soup_hand=agent -> soup_served=true

### Preconditions:
- Can only place ingredient in pot if holding ingredient
- Can only fetch soup if soup_ready=true
- Can only serve soup if soup_staged=true

### Secondary Actions:
- pickup_and_place(onion): Handles onion acquisition, staging, and pot placement
- pickup_and_place(dish): Handles dish acquisition and staging
- pickup_and_place(soup): Handles soup serving and delivery
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
   • pickup_and_place(onion)  
   • pickup_and_place(dish)  
   • pickup_and_place(soup)  
   • NOOP

**EXACT DECISION RULES for secondary actions:**

**Choose pickup_and_place(onion) when:**
- onion_hand="none" AND onion_staged=false AND ingredient_in_pot=false AND soup_staged=false AND soup_hand=none
- (Need to fetch and stage onion for cooking)

**Choose pickup_and_place(dish) when:**
- dish_staged=false AND soup_cooking=true
- (Need dish ready when soup is cooking)

**Choose pickup_and_place(soup) when:**
- soup_hand="none" AND soup_staged=true
- (Soup is staged and ready for serving)

**Choose NOOP when:**
- All required items are already staged or in progress
- Waiting for cooking to complete (soup_cooking=true) AND dish_staged=true
- Waiting for partner to complete their action
- No immediate action needed based on current plan step

Return **only** these two lines (no extra commentary):

primary: <exact primary event text>  
secondary: <one of pickup_and_place(onion|dish|soup) or NOOP>
"""