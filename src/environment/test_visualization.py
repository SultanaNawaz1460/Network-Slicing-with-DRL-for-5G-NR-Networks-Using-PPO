"""
Test Pygame Visualization (No PPO Required)
This runs the environment with RANDOM actions to visualize the network
"""

import numpy as np
import sys
import os
# Fix imports when running from src/environment/
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # Go up to 'src'
project_root = os.path.dirname(parent_dir)  # Go up to project root

# Add both to path
sys.path.insert(0, current_dir)  # For direct imports
sys.path.insert(0, parent_dir)   # For 'environment.' imports
sys.path.insert(0, project_root)  # For project-level imports

# Now import directly (no 'environment.' prefix)
from network_environment import NetworkEnvironment
from network_visualizer import NetworkGameVisualizer


def test_random_policy():
    """
    Test visualization with random resource allocation
    This simulates a "dumb" baseline agent
    """
    print("="*70)
    print(" 🎮 5G NETWORK VISUALIZATION - RANDOM POLICY")
    print("="*70)
    print("\n📋 Instructions:")
    print("  - Watch the network operate in real-time")
    print("  - Blue dots = eMBB users (video)")
    print("  - Green dots = URLLC users (critical)")
    print("  - Orange dots = mMTC users (IoT sensors)")
    print("  - Red triangles = Base stations")
    print("  - Lines show connections (color = signal strength)")
    print("\n🎮 Controls:")
    print("  SPACE = Pause/Resume")
    print("  C = Toggle connection lines")
    print("  V = Toggle coverage circles")
    print("  Q/ESC = Quit")
    print("\n" + "="*70)
    
    input("\nPress ENTER to start visualization...")
    
    # Get config path (relative to project root)
    config_path = os.path.join(project_root, 'config.yaml')
    
    # Initialize environment
    print("\n[1/3] Initializing 5G Network Environment...")
    env = NetworkEnvironment(config_path)
    
    # Initialize visualizer with auto-detected screen size
    print("[2/3] Starting Pygame Window...")
    # Auto-detect screen size
    import pygame as pg
    pg.init()
    screen_info = pg.display.Info()
    # Use 90% of available screen, with reasonable minimums
    screen_width = int(screen_info.current_w * 0.90)
    screen_height = int(screen_info.current_h * 0.90)
    # Ensure minimum size
    screen_width = max(screen_width, 1200)
    screen_height = max(screen_height, 800)
    print(f"   Screen Resolution: {screen_width}x{screen_height}")
    pg.quit()
    
    visualizer = NetworkGameVisualizer(env, screen_width=screen_width, screen_height=screen_height)
    
    # Reset environment
    print("[3/3] Resetting Environment...")
    state = env.reset()
    
    print("\n✅ Visualization Ready!")
    print("=" * 70)
    print("🎬 Starting Simulation...\n")
    
    # Simulation loop
    episode = 0
    step = 0
    total_reward = 0.0
    
    running = True
    
    while running:
        # Random action (baseline - no intelligence)
        # Each user gets random RB allocation
        action = np.random.randint(0, 2, size=(env.num_users, env.num_rbs))
        
        # Optional: Make it slightly smarter (allocate based on buffer size)
        # Comment out the line above and uncomment below for "greedy" baseline:
        # action = greedy_allocation(env)
        
        # Step environment
        try:
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            step += 1
            
            # Render the visualization
            visualizer.render(reward=reward)
            
            # Print step info to console (optional)
            if step % 50 == 0:  # Print every 50 steps
                print(f"Step {step}: "
                      f"Throughput={info['throughput']/1e6:.2f} Mbps, "
                      f"Delay={info['delay']*1000:.2f} ms, "
                      f"QoS Violations={info['qos_violations']}, "
                      f"Reward={reward:.2f}")
            
            # Episode done
            if done:
                episode += 1
                print(f"\n📊 Episode {episode} Complete!")
                print(f"  Total Reward: {total_reward:.2f}")
                print(f"  Avg Throughput: {np.mean(env.throughput_history)/1e6:.2f} Mbps")
                print(f"  Avg Delay: {np.mean(env.delay_history)*1000:.2f} ms")
                print(f"  QoS Satisfaction: {env.qos_violations} violations")
                print(f"  Packet Delivery Rate: {info['delivery_rate']*100:.1f}%")
                print(f"\n🔄 Starting new episode...\n")
                
                # Reset for new episode
                state = env.reset()
                total_reward = 0.0
                step = 0
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user")
            running = False
        
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            running = False
    
    # Cleanup
    print("\n🛑 Closing visualization...")
    visualizer.close()
    env.close()
    print("✅ Done!")


def greedy_allocation(env: NetworkEnvironment) -> np.ndarray:
    """
    Simple greedy baseline: Allocate RBs to users with:
    1. Highest priority (URLLC > eMBB > mMTC)
    2. Largest buffer
    3. Best SINR
    
    This is a bit smarter than random but still simple
    """
    action = np.zeros((env.num_users, env.num_rbs), dtype=np.int32)
    
    # Update channel matrix
    env._update_channel_matrix()
    env._associate_users_to_bs()
    
    # Calculate priority score for each user
    user_priorities = []
    for user_idx, user in enumerate(env.users):
        bs_idx = env.user_bs_association[user_idx]
        sinr = env.channel_matrix[user_idx, bs_idx, 0]
        
        # Priority score (higher = more urgent)
        priority_score = (
            user.qos_requirements['priority'] * 10000 +  # Service type priority
            user.buffer_size / 1000 +                     # Buffer urgency
            max(0, sinr)                                  # Channel quality
        )
        
        user_priorities.append((priority_score, user_idx))
    
    # Sort users by priority (highest first)
    user_priorities.sort(reverse=True)
    
    # Allocate RBs
    available_rbs = set(range(env.num_rbs))
    max_rbs_per_user = 50  # Limit to prevent monopoly
    
    for _, user_idx in user_priorities:
        if len(available_rbs) == 0:
            break
        
        user = env.users[user_idx]
        
        # Only allocate if user has data to send
        if user.buffer_size == 0:
            continue
        
        bs_idx = env.user_bs_association[user_idx]
        
        # Find best RBs for this user (highest SINR)
        rb_sinrs = []
        for rb in available_rbs:
            sinr = env.channel_matrix[user_idx, bs_idx, rb]
            rb_sinrs.append((sinr, rb))
        
        rb_sinrs.sort(reverse=True)
        
        # Allocate top RBs (up to limit)
        num_allocated = 0
        for sinr, rb in rb_sinrs:
            if num_allocated >= max_rbs_per_user:
                break
            
            if sinr > -5:  # Only allocate if SINR is reasonable
                action[user_idx, rb] = 1
                available_rbs.remove(rb)
                num_allocated += 1
    
    return action


if __name__ == "__main__":
    test_random_policy()