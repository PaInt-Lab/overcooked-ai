# Coordinated Navigation System Guide

## Overview

The coordinated navigation system represents a fundamental shift from **plan-following** to **goal-directed navigation** with human coordination. Instead of following a fixed linear plan, the robot now navigates through a state space toward the goal while coordinating with human behavior.

## Key Components

### 1. State Graph (`state_graph.py`)
- **Purpose**: Represents all possible game states as nodes in a graph
- **Features**: 
  - Pre-computed state space with all valid transitions
  - A* path finding for optimal goal-directed navigation
  - Support for robot actions, human actions, and environmental changes

### 2. Coordination System (`coordination_system.py`)
- **Purpose**: Manages human behavior prediction and coordinated action selection
- **Features**:
  - Human behavior prediction based on current state
  - Coordination quality evaluation
  - Conflict risk assessment
  - Balanced action selection (goal progress + coordination)

### 3. Coordinated Action Predictor (`coordinated_action_predictor.py`)
- **Purpose**: New agent that replaces the plan-based action predictor
- **Features**:
  - Goal-directed navigation through state space
  - Human-aware action selection
  - Dynamic adaptation to human behavior
  - Coordination-aware movement planning

## How It Works

### Old System (Plan-Following)
```
1. Receive state → 2. Match to plan step → 3. Execute action → 4. Repeat
```

### New System (Goal-Directed Navigation)
```
1. Receive state → 2. Find current node → 3. Navigate toward goal → 4. Adapt to human → 5. Repeat
```

## Usage

### Basic Usage

```python
from server.llm.agents.coordinated_action_predictor import CoordinatedActionPredictorAgent

# Create the agent
agent = CoordinatedActionPredictorAgent()

# Set up the environment (same as before)
agent.set_mdp(mdp)
agent.set_agent_index(0)

# Get actions (now uses coordinated navigation)
action, info = agent.action(state)
```

### Key Differences from Old System

1. **No Plan Required**: The agent doesn't need a predefined plan
2. **Goal-Oriented**: Always works toward serving soup
3. **Human-Aware**: Predicts and coordinates with human behavior
4. **Adaptive**: Responds to unexpected human actions
5. **Coordinated**: Balances goal progress with coordination quality

## Configuration

### Coordination Weights
You can adjust the balance between goal progress and coordination:

```python
# In coordination_system.py, CoordinatedActionSelector._evaluate_action()
goal_weight = 0.6        # How much to prioritize goal progress
coordination_weight = 0.4 # How much to prioritize coordination
```

### Human Behavior Prediction
The system includes simple human behavior prediction that can be enhanced:

```python
# In coordination_system.py, HumanBehaviorPredictor.predict_human_action()
# Add more sophisticated prediction logic here
```

## Testing

Run the test suite to verify everything works:

```bash
cd src/overcooked_demo
python test_coordinated_navigation.py
```

## Migration from Old System

### Step 1: Replace Agent
```python
# Old
from server.llm.agents.action_predictor import ActionPredictorAgent
agent = ActionPredictorAgent()
agent.set_plan(session_id)  # Required

# New
from server.llm.agents.coordinated_action_predictor import CoordinatedActionPredictorAgent
agent = CoordinatedActionPredictorAgent()
# No plan needed!
```

### Step 2: Update Action Handling
```python
# Old: Returns (move, info) with plan-based info
move, info = agent.action(state)
# info contains: {"primary_event": "...", "function_call": "...", "response": "..."}

# New: Returns (move, info) with coordination info
move, info = agent.action(state)
# info contains: {"action": "...", "coordination_context": CoordinationContext(...)}
```

### Step 3: Handle Coordination Context
```python
# Access coordination information
coordination_context = info["coordination_context"]
print(f"Predicted human action: {coordination_context.predicted_human_action}")
print(f"Coordination quality: {coordination_context.coordination_quality}")
print(f"Conflict risk: {coordination_context.conflict_risk}")
print(f"Goal progress: {coordination_context.goal_progress}")
```

## Advanced Features

### Human Behavior Observation
```python
# Update the system with observed human behavior
agent.coordination_manager.observe_human_action("human_grab_onion")
```

### Coordination Status Monitoring
```python
# Get current coordination status
status = agent.coordination_manager.get_coordination_status()
if status:
    print(f"Current coordination quality: {status.coordination_quality}")
```

### Custom State Graph
```python
# Generate custom state graph with different parameters
from server.state_graph import StateGraphGenerator
generator = StateGraphGenerator()
graph = generator.generate_state_graph()
```

## Performance Considerations

### State Graph Size
- The current implementation generates ~691,200 edges
- Generation happens once at startup
- Memory usage is predictable and manageable

### Path Finding Performance
- Uses A* algorithm for efficient path finding
- Heuristic function can be optimized for better performance
- Paths are cached for repeated queries

### Coordination Overhead
- Human behavior prediction is lightweight
- Coordination evaluation is fast
- No significant performance impact on action selection

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all relative imports are correct
2. **State Graph Generation**: Large state space may take time to generate
3. **Path Finding Failures**: Check if goal state is reachable
4. **Coordination Issues**: Verify human behavior prediction logic

### Debug Information
```python
# Enable debug output
print(f"Current state: {agent.last_summary}")
print(f"Selected action: {action}")
print(f"Coordination context: {coordination_context}")
```

## Future Enhancements

### Planned Improvements
1. **ML-based Human Prediction**: Replace rule-based prediction with ML models
2. **Dynamic Coordination Weights**: Adjust weights based on human behavior patterns
3. **Multi-Goal Support**: Support for different types of goals
4. **Learning Coordination**: Learn from successful coordination patterns

### Extensibility
The system is designed to be easily extensible:
- Add new state variables in `StateGraphGenerator`
- Implement new coordination strategies in `CoordinationEvaluator`
- Enhance human prediction in `HumanBehaviorPredictor`

## Conclusion

The coordinated navigation system provides a more robust, human-aware approach to multi-agent coordination in Overcooked. It moves from rigid plan following to flexible goal-directed behavior that adapts to human actions in real-time.

The system maintains the same interface as the old action predictor while providing much more sophisticated coordination capabilities. This makes it easy to integrate into existing systems while gaining the benefits of coordinated navigation. 