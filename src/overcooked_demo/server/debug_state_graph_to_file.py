#!/usr/bin/env python3
"""
Comprehensive State Graph Debugging - Output to File
This file tests the state graph and saves all debugging information to a file for analysis.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from state_graph import StateGraphGenerator, StateGraph
import json
from datetime import datetime

def debug_to_file():
    """Run comprehensive debugging and save to file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_file = f"state_graph_debug_{timestamp}.txt"
    
    print(f"🔧 Starting comprehensive state graph debugging...")
    print(f"📄 Output will be saved to: {debug_file}")
    
    with open(debug_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("COMPREHENSIVE STATE GRAPH DEBUGGING REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        # Test 1: State Generation Analysis
        f.write("SECTION 1: STATE GENERATION ANALYSIS\n")
        f.write("-" * 40 + "\n")
        
        generator = StateGraphGenerator()
        
        # Generate states
        f.write("Generating states...\n")
        generator._generate_all_states()
        f.write(f"Generated {len(generator.graph.nodes)} states\n\n")
        
        # Analyze state distribution
        f.write("STATE DISTRIBUTION ANALYSIS:\n")
        state_counts = {
            'total': len(generator.graph.nodes),
            'goal_states': 0,
            'initial_states': 0,
            'cooking_states': 0,
            'soup_ready_states': 0,
            'soup_in_pot_states': 0
        }
        
        for node in generator.graph.nodes.values():
            if node.is_goal_state:
                state_counts['goal_states'] += 1
            if (node.state_summary['onion_hand'] == 'none' and 
                node.state_summary['tomato_hand'] == 'none' and
                node.state_summary['dish_hand'] == 'none' and
                node.state_summary['soup_hand'] == 'none' and
                not node.state_summary['soup_served']):
                state_counts['initial_states'] += 1
            if node.state_summary['soup_cooking']:
                state_counts['cooking_states'] += 1
            if node.state_summary['soup_ready']:
                state_counts['soup_ready_states'] += 1
            if node.state_summary['onion_in_pot'] or node.state_summary['tomato_in_pot']:
                state_counts['soup_in_pot_states'] += 1
        
        for key, count in state_counts.items():
            f.write(f"  {key}: {count}\n")
        f.write("\n")
        
        # Show sample states of each type
        f.write("SAMPLE STATES BY TYPE:\n")
        
        # Initial states
        initial_states = []
        for node in generator.graph.nodes.values():
            if (node.state_summary['onion_hand'] == 'none' and 
                node.state_summary['tomato_hand'] == 'none' and
                node.state_summary['dish_hand'] == 'none' and
                node.state_summary['soup_hand'] == 'none' and
                not node.state_summary['soup_served']):
                initial_states.append(node)
        
        f.write(f"INITIAL STATES (first 5):\n")
        for i, node in enumerate(initial_states[:5]):
            f.write(f"  {i+1}. {node.node_id}: {node.state_summary}\n")
        f.write("\n")
        
        # Goal states
        goal_states = [node for node in generator.graph.nodes.values() if node.is_goal_state]
        f.write(f"GOAL STATES (first 5):\n")
        for i, node in enumerate(goal_states[:5]):
            f.write(f"  {i+1}. {node.node_id}: {node.state_summary}\n")
        f.write("\n")
        
        # Cooking states
        cooking_states = [node for node in generator.graph.nodes.values() if node.state_summary['soup_cooking']]
        f.write(f"COOKING STATES (first 5):\n")
        for i, node in enumerate(cooking_states[:5]):
            f.write(f"  {i+1}. {node.node_id}: {node.state_summary}\n")
        f.write("\n")
        
        # Soup ready states
        soup_ready_states = [node for node in generator.graph.nodes.values() if node.state_summary['soup_ready']]
        f.write(f"SOUP READY STATES (first 5):\n")
        for i, node in enumerate(soup_ready_states[:5]):
            f.write(f"  {i+1}. {node.node_id}: {node.state_summary}\n")
        f.write("\n")
        
        # Test 2: Transition Generation Analysis
        f.write("SECTION 2: TRANSITION GENERATION ANALYSIS\n")
        f.write("-" * 40 + "\n")
        
        f.write("Generating transitions...\n")
        generator._generate_all_transitions()
        
        total_edges = sum(len(edges) for edges in generator.graph.edges.values())
        f.write(f"Generated {total_edges} transitions\n\n")
        
        # Analyze transition distribution
        f.write("TRANSITION ANALYSIS:\n")
        
        # Count transitions by action type
        action_counts = {}
        for edges in generator.graph.edges.values():
            for edge in edges:
                action = edge.action
                action_counts[action] = action_counts.get(action, 0) + 1
        
        f.write("TRANSITIONS BY ACTION TYPE:\n")
        for action, count in sorted(action_counts.items(), key=lambda x: x[1], reverse=True):
            f.write(f"  {action}: {count}\n")
        f.write("\n")
        
        # Show sample transitions from initial states
        if initial_states:
            start_node = initial_states[0]
            f.write(f"SAMPLE TRANSITIONS FROM INITIAL STATE ({start_node.node_id}):\n")
            f.write(f"  Initial state: {start_node.state_summary}\n")
            
            if start_node.node_id in generator.graph.edges:
                edges = generator.graph.edges[start_node.node_id]
                f.write(f"  Has {len(edges)} outgoing transitions:\n")
                for i, edge in enumerate(edges[:10]):  # Show first 10
                    to_node = generator.graph.get_node(edge.to_node)
                    f.write(f"    {i+1}. {edge.action} -> {edge.to_node}\n")
                    f.write(f"       To state: {to_node.state_summary}\n")
            else:
                f.write("  [ERROR] NO OUTGOING TRANSITIONS!\n")
        f.write("\n")
        
        # Test 3: Pathfinding Analysis
        f.write("SECTION 3: PATHFINDING ANALYSIS\n")
        f.write("-" * 40 + "\n")
        
        if not initial_states or not goal_states:
            f.write("[ERROR] Cannot test pathfinding - missing initial or goal states\n")
            return debug_file
        
        start_node = initial_states[0]
        goal_node = goal_states[0]
        
        f.write(f"TESTING PATHFINDING:\n")
        f.write(f"  From: {start_node.node_id}\n")
        f.write(f"  To: {goal_node.node_id}\n")
        f.write(f"  Start state: {start_node.state_summary}\n")
        f.write(f"  Goal state: {goal_node.state_summary}\n\n")
        
        # Test pathfinding
        path = generator.graph.find_path_to_goal(start_node.node_id, goal_node.node_id)
        
        if path:
            f.write(f"[SUCCESS] PATH FOUND!\n")
            f.write(f"  Path length: {len(path)}\n")
            f.write(f"  Path: {path}\n\n")
            
            f.write("DETAILED PATH ANALYSIS:\n")
            for i in range(len(path) - 1):
                current_id = path[i]
                next_id = path[i + 1]
                current_node = generator.graph.get_node(current_id)
                next_node = generator.graph.get_node(next_id)
                
                # Find the action that connects these states
                action = "unknown"
                if current_id in generator.graph.edges:
                    for edge in generator.graph.edges[current_id]:
                        if edge.to_node == next_id:
                            action = edge.action
                            break
                
                f.write(f"  Step {i+1}: {action}\n")
                f.write(f"    From: {current_node.state_summary}\n")
                f.write(f"    To: {next_node.state_summary}\n")
                
                # Show what changed
                changes = []
                for key in current_node.state_summary:
                    if current_node.state_summary[key] != next_node.state_summary[key]:
                        changes.append(f"{key}: {current_node.state_summary[key]} -> {next_node.state_summary[key]}")
                
                if changes:
                    f.write(f"    Changes: {', '.join(changes)}\n")
                f.write("\n")
        else:
            f.write(f"[FAILED] NO PATH FOUND!\n\n")
            
            # Analyze why no path exists
            f.write("ANALYZING WHY NO PATH EXISTS:\n")
            
            # Check goal state connectivity
            incoming_edges = generator.graph.get_edges_to(goal_node.node_id)
            f.write(f"  Goal state has {len(incoming_edges)} incoming edges\n")
            
            if incoming_edges:
                f.write("  Sample incoming actions:\n")
                for i, edge in enumerate(incoming_edges[:10]):
                    from_node = generator.graph.get_node(edge.from_node)
                    f.write(f"    {i+1}. {edge.action} from {edge.from_node}\n")
                    f.write(f"       From state: {from_node.state_summary}\n")
            f.write("\n")
            
            # Check start state connectivity
            start_edges = generator.graph.get_edges_from(start_node.node_id)
            f.write(f"  Start state has {len(start_edges)} outgoing edges\n")
            
            if start_edges:
                f.write("  Sample outgoing actions:\n")
                for i, edge in enumerate(start_edges[:10]):
                    to_node = generator.graph.get_node(edge.to_node)
                    f.write(f"    {i+1}. {edge.action} to {edge.to_node}\n")
                    f.write(f"       To state: {to_node.state_summary}\n")
            f.write("\n")
            
            # Test reachability from start
            f.write("REACHABILITY ANALYSIS FROM START:\n")
            for i, edge in enumerate(start_edges[:5]):
                target_node = generator.graph.get_node(edge.to_node)
                f.write(f"  {i+1}. {edge.action} -> {edge.to_node}\n")
                f.write(f"     State: {target_node.state_summary}\n")
                
                # Test if this can reach goal
                test_path = generator.graph.find_path_to_goal(edge.to_node, goal_node.node_id)
                if test_path:
                    f.write(f"     [SUCCESS] Can reach goal! (path length: {len(test_path)})\n")
                else:
                    f.write(f"     [FAILED] Cannot reach goal\n")
            f.write("\n")
        
        # Test 4: State Graph Connectivity Analysis
        f.write("SECTION 4: STATE GRAPH CONNECTIVITY ANALYSIS\n")
        f.write("-" * 40 + "\n")
        
        # Check for isolated states
        isolated_states = []
        for node_id, node in generator.graph.nodes.items():
            incoming = len(generator.graph.get_edges_to(node_id))
            outgoing = len(generator.graph.get_edges_from(node_id))
            if incoming == 0 and outgoing == 0:
                isolated_states.append(node)
        
        f.write(f"ISOLATED STATES: {len(isolated_states)}\n")
        if isolated_states:
            f.write("Sample isolated states:\n")
            for i, node in enumerate(isolated_states[:5]):
                f.write(f"  {i+1}. {node.node_id}: {node.state_summary}\n")
        f.write("\n")
        
        # Check for states with no incoming edges (unreachable)
        unreachable_states = []
        for node_id, node in generator.graph.nodes.items():
            incoming = len(generator.graph.get_edges_to(node_id))
            if incoming == 0:
                unreachable_states.append(node)
        
        f.write(f"UNREACHABLE STATES: {len(unreachable_states)}\n")
        if unreachable_states:
            f.write("Sample unreachable states:\n")
            for i, node in enumerate(unreachable_states[:5]):
                f.write(f"  {i+1}. {node.node_id}: {node.state_summary}\n")
        f.write("\n")
        
        # Check for states with no outgoing edges (dead ends)
        dead_end_states = []
        for node_id, node in generator.graph.nodes.items():
            outgoing = len(generator.graph.get_edges_from(node_id))
            if outgoing == 0:
                dead_end_states.append(node)
        
        f.write(f"DEAD END STATES: {len(dead_end_states)}\n")
        if dead_end_states:
            f.write("Sample dead end states:\n")
            for i, node in enumerate(dead_end_states[:5]):
                f.write(f"  {i+1}. {node.node_id}: {node.state_summary}\n")
        f.write("\n")
        
        # Test 5: Critical Path Analysis
        f.write("SECTION 5: CRITICAL PATH ANALYSIS\n")
        f.write("-" * 40 + "\n")
        
        # Look for the cooking workflow path
        f.write("LOOKING FOR COOKING WORKFLOW PATH:\n")
        
        # Find states with ingredients in pot
        pot_states = [node for node in generator.graph.nodes.values() 
                     if node.state_summary['onion_in_pot'] or node.state_summary['tomato_in_pot']]
        f.write(f"States with ingredients in pot: {len(pot_states)}\n")
        
        if pot_states:
            f.write("Sample pot states:\n")
            for i, node in enumerate(pot_states[:5]):
                f.write(f"  {i+1}. {node.node_id}: {node.state_summary}\n")
        f.write("\n")
        
        # Find states with soup cooking
        cooking_states = [node for node in generator.graph.nodes.values() 
                         if node.state_summary['soup_cooking']]
        f.write(f"States with soup cooking: {len(cooking_states)}\n")
        
        if cooking_states:
            f.write("Sample cooking states:\n")
            for i, node in enumerate(cooking_states[:5]):
                f.write(f"  {i+1}. {node.node_id}: {node.state_summary}\n")
        f.write("\n")
        
        # Find states with soup ready
        ready_states = [node for node in generator.graph.nodes.values() 
                       if node.state_summary['soup_ready']]
        f.write(f"States with soup ready: {len(ready_states)}\n")
        
        if ready_states:
            f.write("Sample ready states:\n")
            for i, node in enumerate(ready_states[:5]):
                f.write(f"  {i+1}. {node.node_id}: {node.state_summary}\n")
        f.write("\n")
        
        # Test connectivity between these critical states
        if pot_states and cooking_states:
            f.write("TESTING CONNECTIVITY: Pot -> Cooking\n")
            test_path = generator.graph.find_path_to_goal(pot_states[0].node_id, cooking_states[0].node_id)
            if test_path:
                f.write(f"[SUCCESS] Path exists from pot to cooking (length: {len(test_path)})\n")
            else:
                f.write("[FAILED] No path from pot to cooking\n")
        f.write("\n")
        
        if cooking_states and ready_states:
            f.write("TESTING CONNECTIVITY: Cooking -> Ready\n")
            test_path = generator.graph.find_path_to_goal(cooking_states[0].node_id, ready_states[0].node_id)
            if test_path:
                f.write(f"[SUCCESS] Path exists from cooking to ready (length: {len(test_path)})\n")
            else:
                f.write("[FAILED] No path from cooking to ready\n")
        f.write("\n")
        
        if ready_states and goal_states:
            f.write("TESTING CONNECTIVITY: Ready -> Goal\n")
            test_path = generator.graph.find_path_to_goal(ready_states[0].node_id, goal_states[0].node_id)
            if test_path:
                f.write(f"[SUCCESS] Path exists from ready to goal (length: {len(test_path)})\n")
            else:
                f.write("[FAILED] No path from ready to goal\n")
        f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("DEBUGGING COMPLETE\n")
        f.write("=" * 80 + "\n")
    
    print(f"✅ Debugging complete! Results saved to: {debug_file}")
    return debug_file

if __name__ == "__main__":
    debug_file = debug_to_file()
    print(f"\n📄 Open {debug_file} to analyze the results in detail.") 