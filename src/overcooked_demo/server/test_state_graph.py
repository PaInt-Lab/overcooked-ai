#!/usr/bin/env python3
"""
Comprehensive State Graph Debugging Test
This file tests the state graph pathfinding step by step to identify where it fails.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from state_graph import StateGraphGenerator, StateGraph
import json

def print_state_diff(state1, state2, label=""):
    """Print the differences between two states"""
    print(f"   {label}")
    for key in state1:
        if state1[key] != state2[key]:
            print(f"     {key}: {state1[key]} -> {state2[key]}")

def test_state_graph_pathfinding():
    """Test state graph pathfinding with detailed debugging"""
    print("🔧 GENERATING STATE GRAPH FOR TESTING...")
    
    # Generate the state graph
    generator = StateGraphGenerator()
    state_graph = generator.generate_state_graph()
    
    print(f"✅ Generated state graph with {len(state_graph.nodes)} nodes")
    print(f"✅ Generated {sum(len(edges) for edges in state_graph.edges.values())} edges")
    
    # Find initial and goal states
    initial_states = []
    goal_states = []
    
    for node in state_graph.nodes.values():
        # Initial state: nothing in hands, nothing served
        if (node.state_summary['onion_hand'] == 'none' and 
            node.state_summary['tomato_hand'] == 'none' and
            node.state_summary['dish_hand'] == 'none' and
            node.state_summary['soup_hand'] == 'none' and
            not node.state_summary['soup_served']):
            initial_states.append(node)
        
        # Goal state: soup served
        if node.is_goal_state:
            goal_states.append(node)
    
    print(f"\n📊 STATE ANALYSIS:")
    print(f"   Found {len(initial_states)} initial states")
    print(f"   Found {len(goal_states)} goal states")
    
    if not initial_states:
        print("   ❌ NO INITIAL STATES FOUND!")
        return
    
    if not goal_states:
        print("   ❌ NO GOAL STATES FOUND!")
        return
    
    # Test pathfinding with detailed debugging
    start_node = initial_states[0]
    goal_node = goal_states[0]
    
    print(f"\n🎯 TESTING PATHFINDING:")
    print(f"   From: {start_node.node_id}")
    print(f"   To: {goal_node.node_id}")
    print(f"   Start state: {start_node.state_summary}")
    print(f"   Goal state: {goal_node.state_summary}")
    
    # Show the difference between start and goal
    print_state_diff(start_node.state_summary, goal_node.state_summary, "State difference:")
    
    # Test pathfinding with custom debugging
    path = test_pathfinding_with_debug(state_graph, start_node.node_id, goal_node.node_id)
    
    if path:
        print(f"\n✅ PATH FOUND!")
        print(f"   Path length: {len(path)}")
        print(f"   Path: {path}")
        
        # Show the actions needed
        print(f"\n📋 ACTIONS NEEDED:")
        for i in range(len(path) - 1):
            current_id = path[i]
            next_id = path[i + 1]
            edges = state_graph.get_edges_from(current_id)
            for edge in edges:
                if edge.to_node == next_id:
                    current_node = state_graph.get_node(current_id)
                    next_node = state_graph.get_node(next_id)
                    print(f"   Step {i+1}: {edge.action}")
                    print_state_diff(current_node.state_summary, next_node.state_summary, f"     State change:")
                    break
    else:
        print(f"\n❌ NO PATH FOUND!")
        analyze_why_no_path(state_graph, start_node, goal_node)

def test_pathfinding_with_debug(state_graph, start_id, goal_id):
    """Test pathfinding with detailed step-by-step debugging"""
    print(f"\n🔍 DETAILED PATHFINDING DEBUG:")
    
    if start_id not in state_graph.nodes or goal_id not in state_graph.nodes:
        print(f"   ❌ Invalid node IDs: start={start_id}, goal={goal_id}")
        return []
    
    # Priority queue for A*: (f_score, node_id, path)
    open_set = [(0, start_id, [start_id])]
    closed_set = set()
    
    # g_score[node] = cost from start to node
    g_score = {start_id: 0}
    
    iterations = 0
    max_iterations = 1000  # Limit for debugging
    
    print(f"   Starting A* search...")
    print(f"   Max iterations: {max_iterations}")
    
    while open_set and iterations < max_iterations:
        iterations += 1
        f_score, current_id, path = open_set.pop(0)  # Use list instead of heap for debugging
        
        print(f"\n   Iteration {iterations}:")
        print(f"     Current node: {current_id}")
        print(f"     Path so far: {path}")
        print(f"     Open set size: {len(open_set)}")
        print(f"     Closed set size: {len(closed_set)}")
        
        if current_id == goal_id:
            print(f"     ✅ GOAL REACHED!")
            return path
        
        if current_id in closed_set:
            print(f"     ⏭️ Already visited, skipping")
            continue
            
        closed_set.add(current_id)
        
        # Get current node state
        current_node = state_graph.get_node(current_id)
        print(f"     Current state: {current_node.state_summary}")
        
        edges = state_graph.get_edges_from(current_id)
        print(f"     Available edges: {len(edges)}")
        
        for i, edge in enumerate(edges):
            neighbor_id = edge.to_node
            tentative_g_score = g_score[current_id] + edge.weight
            
            print(f"       Edge {i+1}: {edge.action} -> {neighbor_id}")
            
            if neighbor_id not in g_score or tentative_g_score < g_score[neighbor_id]:
                g_score[neighbor_id] = tentative_g_score
                new_path = path + [neighbor_id]
                h_score = state_graph._heuristic(neighbor_id, goal_id)
                f_score = tentative_g_score + h_score
                
                # Get neighbor state for debugging
                neighbor_node = state_graph.get_node(neighbor_id)
                print(f"         Adding to open set (f_score={f_score:.2f}, h_score={h_score:.2f})")
                print(f"         Neighbor state: {neighbor_node.state_summary}")
                
                # Insert in sorted order (simple implementation)
                insert_pos = 0
                for j, (existing_f, _, _) in enumerate(open_set):
                    if f_score < existing_f:
                        insert_pos = j
                        break
                    insert_pos = j + 1
                
                open_set.insert(insert_pos, (f_score, neighbor_id, new_path))
            else:
                print(f"         Skipping (worse path)")
        
        # Show top 3 items in open set
        if open_set:
            print(f"     Top 3 in open set:")
            for j, (f_score, node_id, path) in enumerate(open_set[:3]):
                node = state_graph.get_node(node_id)
                print(f"       {j+1}. {node_id} (f={f_score:.2f}): {node.state_summary}")
    
    if iterations >= max_iterations:
        print(f"   ⚠️ Pathfinding timed out after {iterations} iterations")
    else:
        print(f"   ❌ No path found after {iterations} iterations")
    
    return []

def analyze_why_no_path(state_graph, start_node, goal_node):
    """Analyze why no path exists between start and goal"""
    print(f"\n🔍 ANALYZING WHY NO PATH EXISTS:")
    
    # Check goal state connectivity
    incoming_edges = state_graph.get_edges_to(goal_node.node_id)
    print(f"   Goal state ({goal_node.node_id}) has {len(incoming_edges)} incoming edges")
    if incoming_edges:
        print(f"   Sample incoming actions:")
        for i, edge in enumerate(incoming_edges[:5]):
            from_node = state_graph.get_node(edge.from_node)
            print(f"     {i+1}. {edge.action} from {edge.from_node}")
            print(f"        From state: {from_node.state_summary}")
    
    # Check start state connectivity
    start_edges = state_graph.get_edges_from(start_node.node_id)
    print(f"\n   Start state ({start_node.node_id}) has {len(start_edges)} outgoing edges")
    if start_edges:
        print(f"   Sample outgoing actions:")
        for i, edge in enumerate(start_edges[:5]):
            to_node = state_graph.get_node(edge.to_node)
            print(f"     {i+1}. {edge.action} to {edge.to_node}")
            print(f"        To state: {to_node.state_summary}")
    
    # Test if we can reach any goal-like states from start
    print(f"\n   Testing reachability from start:")
    reachable_states = []
    for edge in start_edges:
        target_node = state_graph.get_node(edge.to_node)
        if target_node:
            reachable_states.append({
                'action': edge.action,
                'node_id': edge.to_node,
                'state': target_node.state_summary
            })
    
    # Show states reachable from start
    for i, reachable in enumerate(reachable_states[:5]):
        print(f"     {i+1}. {reachable['action']} -> {reachable['node_id']}")
        print(f"        State: {reachable['state']}")
        
        # Test if this reachable state can reach the goal
        test_path = state_graph.find_path_to_goal(reachable['node_id'], goal_node.node_id)
        if test_path:
            print(f"        ✅ Can reach goal from here! (path length: {len(test_path)})")
        else:
            print(f"        ❌ Cannot reach goal from here")
    
    # Check for missing transitions
    print(f"\n   Checking for missing transitions:")
    print(f"   Looking for states that should lead to goal...")
    
    # Find states that are one step away from goal
    goal_predecessors = []
    for edge in incoming_edges:
        pred_node = state_graph.get_node(edge.from_node)
        goal_predecessors.append({
            'node_id': edge.from_node,
            'action': edge.action,
            'state': pred_node.state_summary
        })
    
    print(f"   States that can reach goal:")
    for i, pred in enumerate(goal_predecessors[:5]):
        print(f"     {i+1}. {pred['node_id']} via {pred['action']}")
        print(f"        State: {pred['state']}")
        
        # Test if start can reach this predecessor
        test_path = state_graph.find_path_to_goal(start_node.node_id, pred['node_id'])
        if test_path:
            print(f"        ✅ Start can reach this! (path length: {len(test_path)})")
        else:
            print(f"        ❌ Start cannot reach this")

def test_state_generation():
    """Test state generation to see what states are created"""
    print("🔧 TESTING STATE GENERATION...")
    
    generator = StateGraphGenerator()
    
    # Test state generation step by step
    print("   Generating states...")
    generator._generate_all_states()
    print(f"   Generated {len(generator.graph.nodes)} states")
    
    # Show some sample states
    print("   Sample states:")
    for i, (node_id, node) in enumerate(list(generator.graph.nodes.items())[:5]):
        print(f"     {i+1}. {node_id}: {node.state_summary}")
    
    # Test transition generation
    print("   Generating transitions...")
    generator._generate_all_transitions()
    
    total_edges = sum(len(edges) for edges in generator.graph.edges.values())
    print(f"   Generated {total_edges} transitions")
    
    # Show some sample transitions
    print("   Sample transitions:")
    for i, (node_id, edges) in enumerate(list(generator.graph.edges.items())[:3]):
        node = generator.graph.get_node(node_id)
        print(f"     From {node_id}: {node.state_summary}")
        for j, edge in enumerate(edges[:3]):
            to_node = generator.graph.get_node(edge.to_node)
            print(f"       {j+1}. {edge.action} -> {edge.to_node}: {to_node.state_summary}")

if __name__ == "__main__":
    print("🚀 STARTING COMPREHENSIVE STATE GRAPH DEBUGGING")
    print("=" * 60)
    
    # Test 1: State generation
    test_state_generation()
    print("\n" + "=" * 60)
    
    # Test 2: Pathfinding
    test_state_graph_pathfinding()
    print("\n" + "=" * 60)
    
    print("🏁 DEBUGGING COMPLETE") 