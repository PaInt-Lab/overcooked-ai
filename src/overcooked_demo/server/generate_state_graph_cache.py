#!/usr/bin/env python3
"""
Pre-generate and cache the state graph for faster game startup.
Run this script once to create the cache file.
"""

import sys
import os
import pickle
import time
from pathlib import Path

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from state_graph import StateGraphGenerator

def generate_and_cache_state_graph():
    """Generate the state graph and cache it to disk"""
    
    cache_file = Path(__file__).parent / "cached_state_graph.pkl"
    
    print("🚀 Generating state graph...")
    start_time = time.time()
    
    # Generate the state graph
    generator = StateGraphGenerator()
    state_graph = generator.generate_state_graph()
    
    generation_time = time.time() - start_time
    print(f"✅ State graph generated in {generation_time:.2f} seconds")
    print(f"📊 Graph contains {len(state_graph.nodes)} nodes")
    
    # Cache to disk
    print("💾 Caching state graph to disk...")
    cache_start = time.time()
    
    with open(cache_file, 'wb') as f:
        pickle.dump(state_graph, f)
    
    cache_time = time.time() - cache_start
    total_time = time.time() - start_time
    
    print(f"✅ State graph cached in {cache_time:.2f} seconds")
    print(f"📁 Cache file: {cache_file}")
    print(f"⏱️  Total time: {total_time:.2f} seconds")
    print(f"📏 Cache file size: {cache_file.stat().st_size / (1024*1024):.1f} MB")

def load_cached_state_graph():
    """Load the cached state graph from disk"""
    cache_file = Path(__file__).parent / "cached_state_graph.pkl"
    
    if not cache_file.exists():
        print("❌ Cache file not found. Run generate_and_cache_state_graph() first.")
        return None
    
    print("📂 Loading cached state graph...")
    start_time = time.time()
    
    with open(cache_file, 'rb') as f:
        state_graph = pickle.load(f)
    
    load_time = time.time() - start_time
    print(f"✅ State graph loaded in {load_time:.2f} seconds")
    print(f"📊 Graph contains {len(state_graph.nodes)} nodes")
    
    return state_graph

if __name__ == "__main__":
    print("🎯 State Graph Cache Generator")
    print("=" * 40)
    
    # Check if cache already exists
    cache_file = Path(__file__).parent / "cached_state_graph.pkl"
    if cache_file.exists():
        print(f"📁 Cache file already exists: {cache_file}")
        print("🔄 Regenerating...")
    
    generate_and_cache_state_graph()
    
    # Test loading
    print("\n🧪 Testing cache loading...")
    loaded_graph = load_cached_state_graph()
    
    if loaded_graph:
        print("✅ Cache test successful!")
    else:
        print("❌ Cache test failed!") 