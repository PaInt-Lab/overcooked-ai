# Action Predictor Modelfiles

This directory contains specialized Modelfiles for different soup types in Overcooked AI. Each Modelfile is optimized for a specific soup recipe to improve action prediction accuracy.

## Available Modelfiles

### 1. `Modelfile_onion` - Onion Soup Only
- **Use case**: Maps with only onion ingredients
- **State variables**: Focuses on onion_hand, onion_staged, onion_in_pot
- **Secondary actions**: pickup_and_place(onion), pickup_and_place(dish), pickup_and_place(soup), NOOP
- **Best for**: Simple onion soup recipes (1-3 onions)

### 2. `Modelfile_tomato` - Tomato Soup Only  
- **Use case**: Maps with only tomato ingredients
- **State variables**: Focuses on tomato_hand, tomato_staged, tomato_in_pot
- **Secondary actions**: pickup_and_place(tomato), pickup_and_place(dish), pickup_and_place(soup), NOOP
- **Best for**: Simple tomato soup recipes (1-3 tomatoes)

### 3. `Modelfile_mixed` - Mixed Soups (Onion + Tomato)
- **Use case**: Maps with both onion and tomato ingredients
- **State variables**: Handles both onion_hand/tomato_hand, onion_staged/tomato_staged, onion_in_pot/tomato_in_pot
- **Secondary actions**: pickup_and_place(onion), pickup_and_place(tomato), pickup_and_place(dish), pickup_and_place(soup), NOOP
- **Best for**: Complex mixed soup recipes (e.g., 2 onions + 1 tomato, 1 onion + 2 tomatoes)

### 4. `Modelfile_onion_original` - Legacy
- **Use case**: Original onion-focused Modelfile (kept for reference)
- **Note**: This is the original file, consider using `Modelfile_onion` instead

## Usage

### Building Models

To build a model for a specific soup type:

```bash
# For onion soup maps
ollama create overcooked-action-predictor-onion -f Modelfile_onion

# For tomato soup maps  
ollama create overcooked-action-predictor-tomato -f Modelfile_tomato

# For mixed soup maps
ollama create overcooked-action-predictor-mixed -f Modelfile_mixed
```

### Selecting the Right Model

Choose the Modelfile based on your map's ingredients:

1. **Check the layout file** (e.g., `soup_coordination.layout`) for:
   - `start_all_orders` - what soups are worth points
   - `start_bonus_orders` - bonus soup recipes

2. **Analyze the ingredients**:
   - Only onions → Use `Modelfile_onion`
   - Only tomatoes → Use `Modelfile_tomato` 
   - Both onions and tomatoes → Use `Modelfile_mixed`

### Example Layout Analysis

```json
{
  "start_all_orders": [
    {"ingredients": ["onion", "onion", "tomato"]},  // Mixed soup
    {"ingredients": ["tomato", "tomato", "tomato"]}, // Tomato soup
    {"ingredients": ["onion", "tomato"]},            // Mixed soup
    {"ingredients": ["tomato"]},                     // Tomato soup
    {"ingredients": ["onion"]}                       // Onion soup
  ]
}
```

This layout has mixed soups, so use `Modelfile_mixed`.

## Benefits of Specialized Models

1. **Focused Knowledge**: Each model only knows about relevant ingredients
2. **Cleaner Decision Logic**: No confusion between onion vs tomato mechanics
3. **Better Performance**: Smaller, more focused models can be more accurate
4. **Easier Debugging**: Clear separation of concerns

## Integration with Server

To use these models in your server, you'll need to:

1. Update the action predictor agent to select the appropriate model based on the current layout
2. Modify the model loading logic to use the correct Modelfile
3. Ensure the state summarization includes the right ingredient tracking

## Future Enhancements

- **Dynamic Model Selection**: Automatically choose the right model based on layout analysis
- **Hybrid Models**: Models that can handle multiple soup types but with clear priority rules
- **Recipe-Specific Models**: Even more specialized models for specific recipes (e.g., 2-onion-1-tomato soup)