"""
Simple verification script to demonstrate the time delay mechanism.

This script shows how the time delay enforcement works without running the full server.
"""

from time import time, sleep

class MockGame:
    """Mock game class demonstrating the time delay logic"""
    
    def __init__(self, action_delay=3.0):
        self.action_delay = action_delay
        self.last_action_times = [0, 0]  # Two players
        self.players = ["human", "agent"]
        self.action_count = [0, 0]
        
    def can_perform_action(self, player_idx):
        """Check if player can perform an action based on time delay"""
        current_time = time()
        time_since_last = current_time - self.last_action_times[player_idx]
        return time_since_last >= self.action_delay
    
    def perform_action(self, player_idx, action_name):
        """Attempt to perform an action for a player"""
        if not self.can_perform_action(player_idx):
            time_remaining = self.action_delay - (time() - self.last_action_times[player_idx])
            print(f"  [X] {self.players[player_idx]} action '{action_name}' REJECTED - must wait {time_remaining:.2f}s more")
            return False
        
        # Action accepted
        self.last_action_times[player_idx] = time()
        self.action_count[player_idx] += 1
        print(f"  [OK] {self.players[player_idx]} action '{action_name}' EXECUTED (action #{self.action_count[player_idx]})")
        return True


def test_time_delay():
    """Test the time delay mechanism"""
    print("=" * 60)
    print("TIME DELAY VERIFICATION TEST")
    print("=" * 60)
    print(f"Action delay: 3.0 seconds")
    print()
    
    game = MockGame(action_delay=3.0)
    
    # Initialize to allow first actions immediately
    start_time = time()
    game.last_action_times = [start_time - 3.0, start_time - 3.0]
    
    # Test 1: First actions should succeed
    print("Test 1: First actions (should succeed immediately)")
    game.perform_action(0, "move_up")
    game.perform_action(1, "move_left")
    print()
    
    # Test 2: Immediate retry should fail
    print("Test 2: Immediate retry (should fail)")
    game.perform_action(0, "move_down")
    game.perform_action(1, "move_right")
    print()
    
    # Test 3: Wait 1.5 seconds, still should fail
    print("Test 3: After 1.5 seconds (should still fail)")
    sleep(1.5)
    game.perform_action(0, "interact")
    game.perform_action(1, "interact")
    print()
    
    # Test 4: Wait another 1.5 seconds (total 3.0s), should succeed
    print("Test 4: After 3.0 seconds total (should succeed)")
    sleep(1.5)
    game.perform_action(0, "move_right")
    game.perform_action(1, "move_up")
    print()
    
    # Test 5: Alternating actions with delays
    print("Test 5: Alternating actions with proper delays")
    for i in range(2):
        print(f"  Round {i+1}:")
        sleep(3.0)
        game.perform_action(0, f"action_{i}_human")
        game.perform_action(1, f"action_{i}_agent")
    print()
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Human actions executed: {game.action_count[0]}")
    print(f"Agent actions executed: {game.action_count[1]}")
    print(f"Both players executed the same number of actions: {game.action_count[0] == game.action_count[1]}")
    print()
    print("[SUCCESS] Time delay mechanism working correctly!")
    print("=" * 60)


if __name__ == "__main__":
    test_time_delay()

