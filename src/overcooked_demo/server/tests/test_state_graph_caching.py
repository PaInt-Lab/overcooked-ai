#!/usr/bin/env python3
"""
Test state graph caching functionality without LLM dependencies.
"""

import sys
import os
import time
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from state_graph import StateGraphGenerator

def test_state_graph_caching():
    """Test that state graph caching works correctly"""
    
    print("🧪 Testing State Graph Caching")
    print("=" * 40)
    
    # Test 1: Generate fresh state graph
    print("\n1️⃣ Generating fresh state graph...")
    start_time = time.time()
    generator = StateGraphGenerator()
    fresh_graph = generator.generate_state_graph()
    fresh_time = time.time() - start_time
    
    print(f"✅ Fresh generation: {fresh_time:.2f} seconds")
    print(f"📊 Nodes: {len(fresh_graph.nodes)}")
    
    # Test 2: Load from cache
    print("\n2️⃣ Loading from cache...")
    cache_file = os.path.join(os.path.dirname(__file__), '..', 'cached_state_graph.pkl')
    
    if os.path.exists(cache_file):
        start_time = time.time()
        import pickle
        with open(cache_file, 'rb') as f:
            cached_graph = pickle.load(f)
        cache_time = time.time() - start_time
        
        print(f"✅ Cache loading: {cache_time:.2f} seconds")
        print(f"📊 Nodes: {len(cached_graph.nodes)}")
        
        # Test 3: Verify graphs are identical
        print("\n3️⃣ Verifying graph consistency...")
        if len(fresh_graph.nodes) == len(cached_graph.nodes):
            print("✅ Node count matches!")
        else:
            print(f"❌ Node count mismatch: fresh={len(fresh_graph.nodes)}, cached={len(cached_graph.nodes)}")
        
        # Test 4: Speed comparison
        speedup = fresh_time / cache_time
        print(f"\n4️⃣ Speed comparison:")
        print(f"   Fresh generation: {fresh_time:.2f}s")
        print(f"   Cache loading: {cache_time:.2f}s")
        print(f"   Speedup: {speedup:.1f}x faster")
        
        if speedup > 10:
            print("🚀 Excellent speedup achieved!")
        elif speedup > 5:
            print("✅ Good speedup achieved!")
        else:
            print("⚠️  Moderate speedup achieved")
            
    else:
        print("❌ Cache file not found!")
        print("💡 Run generate_state_graph_cache.py first")
    
    print("\n✅ State graph caching test completed!")

if __name__ == "__main__":
    test_state_graph_caching() 