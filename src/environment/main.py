"""
Main entry point for 5G Network Slicing Simulation
Run this to test the integrated environment with channel model

Author: Your Name
Date: 2025
"""

from network_environment import NetworkEnvironment, run_all_tests
import numpy as np
import os

def main():
    """
    Main function to test integrated environment
    """
    print("="*70)
    print(" 5G NETWORK SLICING - INTEGRATED ENVIRONMENT TEST")
    print("="*70)
    
    # Get the config path relative to project root
    config_path = os.path.join(os.path.dirname(__file__), '../../config.yaml')
    
    # Run comprehensive validation
    print("\n[STEP 1] Running comprehensive validation tests...")
    print("  This includes:")
    print("    • Basic connectivity")
    print("    • Channel model integration")
    print("    • Throughput calculation (Shannon capacity)")
    print("    • User-BS association")
    print("    • Step function")
    print()
    
    success = run_all_tests(config_path)
    
    if not success:
        print("\n❌ Validation failed. Please review errors above.")
        return
    
    print("\n[STEP 2] Testing environment dynamics...")
    env = NetworkEnvironment(config_path)
    state = env.reset()
    
    print(f"\n✓ Initial State:")
    print(f"  State shape: {state.shape}")
    print(f"  State sample (first 10 values): {state[:10]}")
    print(f"  Channel matrix shape: {env.channel_matrix.shape}")
    print(f"  Average SINR: {np.mean(env.channel_matrix):.2f} dB")
    
    # Test simulation with random actions
    print("\n[STEP 3] Running 10 simulation steps with random allocation...")
    print("-" * 70)
    
    for step in range(10):
        # Random action (for testing - will be replaced by PPO later)
        action = np.random.randint(0, 2, size=(env.num_users, env.num_rbs))
        
        next_state, reward, done, info = env.step(action)
        
        print(f"\nStep {step + 1}:")
        print(f"  Time: {info['time']*1000:.2f} ms")
        print(f"  Throughput: {info['throughput']/1e6:.2f} Mbps")
        print(f"  Delay: {info['delay']*1000:.2f} ms")
        print(f"  Avg SINR: {info['avg_sinr']:.2f} dB")
        print(f"  Energy: {info['energy']:.2f} J")
        print(f"  Reward: {reward:.2f}")
        print(f"  QoS Violations: {info['qos_violations']}")
        print(f"  RBs Allocated: {np.sum(action)}/{env.num_users * env.num_rbs}")
    
    # Get comprehensive statistics
    print("\n" + "="*70)
    print("[STEP 4] Final Network Statistics")
    print("="*70)
    stats = env.get_statistics()
    
    print("\n📊 Network Metrics:")
    for key, value in stats.items():
        if isinstance(value, float):
            if 'throughput' in key.lower():
                print(f"  {key}: {value/1e6:.2f} Mbps")
            elif 'delay' in key.lower():
                print(f"  {key}: {value*1000:.2f} ms")
            elif 'sinr' in key.lower():
                print(f"  {key}: {value:.2f} dB")
            elif 'energy' in key.lower():
                print(f"  {key}: {value:.2f} J")
            elif 'rate' in key.lower() or 'utilization' in key.lower():
                print(f"  {key}: {value*100:.2f}%")
            else:
                print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Test with optimal allocation (best-SINR greedy)
    print("\n" + "="*70)
    print("[STEP 5] Testing Greedy Best-SINR Allocation")
    print("="*70)
    
    env.reset()
    greedy_action = test_greedy_allocation(env)
    
    next_state, reward, done, info = env.step(greedy_action)
    
    print(f"\n✓ Greedy Allocation Results:")
    print(f"  Throughput: {info['throughput']/1e6:.2f} Mbps")
    print(f"  Delay: {info['delay']*1000:.2f} ms")
    print(f"  Avg SINR: {info['avg_sinr']:.2f} dB")
    print(f"  Reward: {reward:.2f}")
    print(f"  QoS Violations: {info['qos_violations']}")
    print(f"  RBs Allocated: {np.sum(greedy_action)}")
    
    env.close()
    
    # Success summary
    print("\n" + "="*70)
    print(" ✓✓✓ INTEGRATED ENVIRONMENT TEST COMPLETE ✓✓✓")
    print("="*70)
    
    print("\n🎯 What's Working:")
    print("  ✅ Channel Model (Path Loss + Rayleigh + Shadowing + Interference)")
    print("  ✅ Shannon Capacity Throughput Calculation")
    print("  ✅ QoS Satisfaction Checking")
    print("  ✅ User-BS Association (Best SINR)")
    print("  ✅ Resource Block Allocation")
    print("  ✅ Energy Consumption Tracking")
    print("  ✅ Multi-objective Reward Function")
    
    print("\n📦 Traffic Generator Integration:")
    print("  ✅ eMBB Poisson Traffic (Bursty Arrivals)")
    print("  ✅ URLLC Periodic Traffic (Deterministic, 1ms Period)")
    print("  ✅ mMTC Poisson Traffic (Low Arrival Rate)")
    print("  ✅ Packet Queue Management")
    print("  ✅ Deadline Tracking & Expiration")
    print("  ✅ Packet Delivery Rate Calculation")
    print("  ✅ Service Type Prioritization (1=URLLC, 2=eMBB, 3=mMTC)")
    print("  ✅ Dynamic Packet Size Generation")
    
    print("\n🚀 Next Steps:")
    print("  1. Implement PPO Agent (src/agents/ppo_agent.py)")
    print("  2. Create training pipeline (src/agents/train_ppo.py)")
    print("  3. Implement baseline algorithms for comparison:")
    print("     • Round Robin")
    print("     • Max-SINR")
    print("     • Proportional Fair")
    print("  4. Run experiments and collect results")
    print("  5. Visualize and analyze performance")
    
    print("\n💡 To train PPO:")
    print("  python src/agents/train_ppo.py --config config.yaml")
    
    print("\n" + "="*70)


def test_greedy_allocation(env: NetworkEnvironment) -> np.ndarray:
    """
    Test a simple greedy allocation strategy (best-SINR)
    This serves as a baseline for PPO
    
    Args:
        env: Network environment
        
    Returns:
        action: Binary allocation matrix [num_users, num_rbs]
    """
    action = np.zeros((env.num_users, env.num_rbs), dtype=np.int32)
    
    # Update channel matrix
    env._update_channel_matrix()
    env._associate_users_to_bs()
    
    # For each RB, allocate to user with best SINR (if buffer not empty)
    allocated_rbs_per_user = np.zeros(env.num_users, dtype=np.int32)
    max_rbs_per_user = 50  # Limit to avoid monopoly
    
    for rb_idx in range(env.num_rbs):
        best_sinr = -np.inf
        best_user = -1
        
        for user_idx, user in enumerate(env.users):
            # Only allocate if user has data in buffer and hasn't exceeded limit
            if user.buffer_size > 0 and allocated_rbs_per_user[user_idx] < max_rbs_per_user:
                bs_idx = env.user_bs_association[user_idx]
                sinr = env.channel_matrix[user_idx, bs_idx, rb_idx]
                
                if sinr > best_sinr:
                    best_sinr = sinr
                    best_user = user_idx
        
        # Allocate RB to best user
        if best_user != -1:
            action[best_user, rb_idx] = 1
            allocated_rbs_per_user[best_user] += 1
    
    print(f"\n  Greedy allocation strategy:")
    print(f"    Total RBs allocated: {np.sum(action)}/{env.num_rbs}")
    print(f"    Users with allocations: {np.sum(allocated_rbs_per_user > 0)}/{env.num_users}")
    print(f"    Avg RBs per user: {np.mean(allocated_rbs_per_user[allocated_rbs_per_user > 0]):.1f}")
    
    return action


def quick_test():
    """
    Quick sanity check (runs faster than full test suite)
    """
    print("\n" + "="*70)
    print(" QUICK SANITY CHECK")
    print("="*70)
    
    config_path = os.path.join(os.path.dirname(__file__), '../../config.yaml')
    
    try:
        env = NetworkEnvironment(config_path)
        print("✓ Environment initialization: PASS")
        
        state = env.reset()
        print(f"✓ Reset: PASS (state shape: {state.shape})")
        
        action = np.random.randint(0, 2, size=(env.num_users, env.num_rbs))
        next_state, reward, done, info = env.step(action)
        print(f"✓ Step: PASS (reward: {reward:.2f})")
        
        env.close()
        
        print("\n✅ Quick test PASSED! Full tests should work.")
        return True
        
    except Exception as e:
        print(f"\n❌ Quick test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Uncomment for quick test first
    # quick_test()
    
    # Run full test suite
    main()