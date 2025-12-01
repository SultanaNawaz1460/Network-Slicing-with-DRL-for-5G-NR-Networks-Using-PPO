"""
Real-time 5G Network Visualization with Pygame
Looks like a strategy game!
"""

import pygame
import numpy as np
from typing import List
import sys

class NetworkGameVisualizer:
    """
    Game-like visualization for 5G network
    """
    
    def __init__(self, env, screen_width=1400, screen_height=900):
        """
        Initialize Pygame visualizer
        
        Args:
            env: NetworkEnvironment instance
            screen_width: Window width
            screen_height: Window height
        """
        pygame.init()
        
        self.env = env
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Create window
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("5G Network Slicing - PPO Agent in Action")
        
        # Fonts
        self.font_large = pygame.font.Font(None, 36)
        self.font_medium = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 18)
        
        # Colors (modern dark theme)
        self.colors = {
            'background': (20, 20, 30),
            'grid': (40, 40, 50),
            'bs_tower': (255, 50, 50),
            'bs_coverage': (255, 50, 50, 30),  # Transparent red
            'embb': (100, 150, 255),  # Blue
            'urllc': (50, 255, 100),  # Green
            'mmtc': (255, 200, 50),   # Orange
            'connection': (100, 100, 100, 100),  # Gray transparent
            'signal_strong': (0, 255, 0),
            'signal_weak': (255, 0, 0),
            'text': (255, 255, 255),
            'panel': (30, 30, 40),
            'panel_border': (60, 60, 70),
        }
        
        # Scaling factors - adaptive based on screen size
        panel_width = min(400, screen_width // 4)  # Dynamic panel width
        self.scale_x = (screen_width - panel_width - 20) / env.coverage_area[0]
        self.scale_y = (screen_height - 100) / env.coverage_area[1]
        self.offset_x = 30
        self.offset_y = 40
        self.panel_width = panel_width
        
        # Animation state
        self.clock = pygame.time.Clock()
        self.fps = 30
        self.paused = False
        self.show_connections = True
        self.show_coverage = True
        
        # Particle effects for data transmission
        self.particles = []
        
        # Metrics history for mini graphs
        self.history_length = 100  # Number of steps to keep in history
        self.throughput_history = []
        self.delay_history = []
        self.qos_rate_history = []
        
        print("[NetworkGameVisualizer] Initialized")
        print(f"  Resolution: {screen_width}x{screen_height}")
        print(f"  FPS: {self.fps}")
        print("\n🎮 Controls:")
        print("  SPACE: Pause/Resume")
        print("  C: Toggle Connections")
        print("  V: Toggle Coverage Circles")
        print("  Q/ESC: Quit")
    
    def world_to_screen(self, pos):
        """Convert world coordinates to screen coordinates"""
        x = self.offset_x + pos[0] * self.scale_x
        y = self.offset_y + pos[1] * self.scale_y
        return (int(x), int(y))
    
    def draw_grid(self):
        """Draw background grid"""
        grid_spacing = 100  # meters
        
        for x in range(0, int(self.env.coverage_area[0]), grid_spacing):
            screen_x = self.world_to_screen([x, 0])[0]
            pygame.draw.line(
                self.screen, self.colors['grid'],
                (screen_x, self.offset_y),
                (screen_x, self.offset_y + self.env.coverage_area[1] * self.scale_y),
                1
            )
        
        for y in range(0, int(self.env.coverage_area[1]), grid_spacing):
            screen_y = self.world_to_screen([0, y])[1]
            pygame.draw.line(
                self.screen, self.colors['grid'],
                (self.offset_x, screen_y),
                (self.offset_x + self.env.coverage_area[0] * self.scale_x, screen_y),
                1
            )
    
    def draw_base_station(self, bs, selected=False):
        """Draw base station with coverage area"""
        pos_screen = self.world_to_screen(bs.position[:2])
        
        # Coverage circle (transparent)
        if self.show_coverage:
            coverage_radius = int(300 * self.scale_x)  # 300m coverage
            surface = pygame.Surface((coverage_radius*2, coverage_radius*2), pygame.SRCALPHA)
            pygame.draw.circle(
                surface, self.colors['bs_coverage'],
                (coverage_radius, coverage_radius), coverage_radius
            )
            self.screen.blit(surface, (pos_screen[0] - coverage_radius, pos_screen[1] - coverage_radius))
        
        # Tower (triangle)
        tower_height = 30
        tower_width = 20
        points = [
            (pos_screen[0], pos_screen[1] - tower_height),  # Top
            (pos_screen[0] - tower_width//2, pos_screen[1]),  # Bottom left
            (pos_screen[0] + tower_width//2, pos_screen[1])   # Bottom right
        ]
        pygame.draw.polygon(self.screen, self.colors['bs_tower'], points)
        
        if selected:
            pygame.draw.polygon(self.screen, (255, 255, 0), points, 3)
        
        # BS ID
        text = self.font_small.render(f"BS{bs.id}", True, self.colors['text'])
        self.screen.blit(text, (pos_screen[0] - 15, pos_screen[1] + 10))
        
        # Load indicator
        load_bar_width = 40
        load_bar_height = 5
        load_fill = int(bs.current_load * load_bar_width)
        
        pygame.draw.rect(
            self.screen, (50, 50, 50),
            (pos_screen[0] - load_bar_width//2, pos_screen[1] + 25, load_bar_width, load_bar_height)
        )
        
        load_color = (255, 0, 0) if bs.current_load > 0.8 else (0, 255, 0)
        pygame.draw.rect(
            self.screen, load_color,
            (pos_screen[0] - load_bar_width//2, pos_screen[1] + 25, load_fill, load_bar_height)
        )
    
    def draw_user(self, user, user_idx):
        """Draw user with service type color"""
        pos_screen = self.world_to_screen(user.position)
        
        # Service type color
        color_map = {
            'eMBB': self.colors['embb'],
            'URLLC': self.colors['urllc'],
            'mMTC': self.colors['mmtc']
        }
        color = color_map[user.service_type.value]
        
        # User circle size based on buffer load (visualization of queue)
        base_radius = 8
        buffer_factor = min(1.5, 1.0 + (user.buffer_size / 1e6))  # Scale up to 1.5x based on buffer
        radius = int(base_radius * buffer_factor)
        
        pygame.draw.circle(self.screen, color, pos_screen, radius)
        
        # QoS indicator (ring around user) - ENHANCED
        if user.qos_satisfied:
            # Green ring for satisfied QoS
            pygame.draw.circle(self.screen, (0, 255, 0), pos_screen, radius + 3, 2)
        else:
            # Red pulsing ring for violated QoS (draw thicker for visibility)
            pygame.draw.circle(self.screen, (255, 0, 0), pos_screen, radius + 3, 3)
            # Add extra glow effect
            pygame.draw.circle(self.screen, (255, 100, 100), pos_screen, radius + 5, 1)
        
        # Connection line to BS
        if self.show_connections:
            bs_idx = self.env.user_bs_association[user_idx]
            bs = self.env.base_stations[bs_idx]
            bs_pos_screen = self.world_to_screen(bs.position[:2])
            
            # Line thickness based on allocated RBs
            num_rbs = np.sum(self.env.allocated_rbs[user_idx, :])
            thickness = max(1, min(int(num_rbs // 20), 5))
            
            # Line color based on SINR
            sinr = self.env.channel_matrix[user_idx, bs_idx, 0]
            if sinr > 10:
                line_color = self.colors['signal_strong']
            elif sinr > 0:
                line_color = (200, 200, 0)
            else:
                line_color = self.colors['signal_weak']
            
            pygame.draw.line(self.screen, line_color + (100,), pos_screen, bs_pos_screen, thickness)
        
        # Buffer indicator (small bar above user) - now represents queue length
        if user.buffer_size > 0:
            bar_width = 20
            bar_height = 4
            # Scale: normalize to 1MB max for visualization
            fill = min(bar_width, int((user.buffer_size / 1e6) * bar_width))
            
            bar_color = (0, 255, 100) if user.buffer_size < 5e5 else (255, 200, 0) if user.buffer_size < 10e5 else (255, 0, 0)
            
            pygame.draw.rect(
                self.screen, (50, 50, 50),
                (pos_screen[0] - bar_width//2, pos_screen[1] - 18, bar_width, bar_height)
            )
            pygame.draw.rect(
                self.screen, bar_color,
                (pos_screen[0] - bar_width//2, pos_screen[1] - 18, fill, bar_height)
            )
    
    def draw_stats_panel(self, stats, reward):
        """Draw statistics panel on the right"""
        panel_x = self.screen_width - self.panel_width + 10
        panel_y = 20
        panel_width = self.panel_width - 20
        panel_height = self.screen_height - 40
        
        # Panel background
        pygame.draw.rect(
            self.screen, self.colors['panel'],
            (panel_x, panel_y, panel_width, panel_height),
            border_radius=10
        )
        pygame.draw.rect(
            self.screen, self.colors['panel_border'],
            (panel_x, panel_y, panel_width, panel_height),
            3, border_radius=10
        )
        
        # Title
        title = self.font_large.render("Network Stats", True, self.colors['text'])
        self.screen.blit(title, (panel_x + 20, panel_y + 20))
        
        y_offset = 70
        line_spacing = 30
        
        # Stats
        stats_to_display = [
            ("Time", f"{stats['time']*1000:.1f} ms"),
            ("Step", f"{stats['step']}"),
            ("", ""),
            ("=== Performance ===", ""),
            ("Throughput", f"{stats['avg_throughput']/1e6:.2f} Mbps"),
            ("Delay", f"{stats['avg_delay']*1000:.2f} ms"),
            ("Reward", f"{reward:.2f}"),
            ("", ""),
            ("=== QoS ===", ""),
            ("Satisfied", f"{stats['qos_satisfied']}/{len(self.env.users)}"),
            ("Rate", f"{stats['qos_satisfaction_rate']*100:.1f}%"),
            ("", ""),
            ("=== Packets ===", ""),
            ("Generated", f"{stats['packets_generated']}"),
            ("Delivered", f"{stats['packets_delivered']}"),
            ("Dropped", f"{stats['packets_dropped']}"),
            ("Delivery Rate", f"{stats['packet_delivery_rate']*100:.1f}%"),
            ("", ""),
            ("=== Resources ===", ""),
            ("RBs Allocated", f"{stats['total_rbs_allocated']}/{self.env.num_users * self.env.num_rbs}"),
            ("Utilization", f"{stats['resource_utilization']*100:.1f}%"),
            ("Avg BS Load", f"{stats['avg_bs_load']*100:.1f}%"),
            ("", ""),
            ("=== Channel ===", ""),
            ("Avg SINR", f"{stats['avg_sinr']:.2f} dB"),
            ("Min SINR", f"{stats['min_sinr']:.2f} dB"),
            ("Max SINR", f"{stats['max_sinr']:.2f} dB"),
            ("", ""),
            ("=== Energy ===", ""),
            ("Total", f"{stats['total_energy']:.2f} J"),
        ]
        
        for label, value in stats_to_display:
            if label == "":
                continue
            elif label.startswith("==="):
                text = self.font_medium.render(label, True, (255, 200, 0))
            else:
                text = self.font_small.render(f"{label}: {value}", True, self.colors['text'])
            
            self.screen.blit(text, (panel_x + 20, panel_y + y_offset))
            y_offset += line_spacing if not label.startswith("===") else line_spacing * 0.8
    
    def draw_legend(self):
        """Draw legend for service types"""
        legend_x = 50
        legend_y = self.screen_height - 40
        
        services = [
            ("eMBB (Video)", self.colors['embb']),
            ("URLLC (Critical)", self.colors['urllc']),
            ("mMTC (IoT)", self.colors['mmtc'])
        ]
        
        x_offset = 0
        for label, color in services:
            # Circle
            pygame.draw.circle(self.screen, color, (legend_x + x_offset, legend_y), 8)
            # Text
            text = self.font_small.render(label, True, self.colors['text'])
            self.screen.blit(text, (legend_x + x_offset + 15, legend_y - 8))
            x_offset += 180
    
    def draw_mini_graph(self, title, data, x, y, width=120, height=60, max_val=None, color=(0, 255, 100)):
        """Draw a mini line graph for metrics"""
        # Background
        pygame.draw.rect(self.screen, (40, 40, 50), (x, y, width, height))
        pygame.draw.rect(self.screen, (80, 80, 90), (x, y, width, height), 1)
        
        # Title
        text = self.font_small.render(title, True, (200, 200, 200))
        self.screen.blit(text, (x + 5, y + 3))
        
        # Draw graph
        if len(data) > 1:
            if max_val is None:
                max_val = max(data) if data else 1.0
                max_val = max(max_val, 1.0)  # Avoid division by zero
            
            # Draw grid lines
            mid_y = y + height // 2
            pygame.draw.line(self.screen, (60, 60, 70), (x, mid_y), (x + width, mid_y), 1)
            
            # Draw points and line
            rgb_color = tuple(color[:3]) if len(color) >= 3 else color
            
            prev_point = None
            for i, value in enumerate(data[-width:]):
                px = x + 5 + (i / max(1, width - 10)) * (width - 10)
                py = y + height - 15 - (value / max_val) * (height - 20)
                point = (int(px), int(py))
                
                # Draw line segment
                if prev_point is not None:
                    pygame.draw.line(self.screen, rgb_color, prev_point, point, 2)
                
                prev_point = point
            
            # Draw last point as highlight
            if prev_point is not None:
                pygame.draw.circle(self.screen, rgb_color, prev_point, 3)
        
        # Min/max labels
        if data:
            min_val = min(data)
            max_val_actual = max(data)
            min_text = self.font_small.render(f"{min_val:.1f}", True, (100, 100, 100))
            max_text = self.font_small.render(f"{max_val_actual:.1f}", True, (100, 100, 100))
            self.screen.blit(min_text, (x + width - 35, y + height - 12))
            self.screen.blit(max_text, (x + width - 35, y + 15))
    
    def update_metrics_history(self, stats):
        """Update history for mini graphs"""
        self.throughput_history.append(stats.get('avg_throughput', 0) / 1e6)  # Convert to Mbps
        self.delay_history.append(stats.get('avg_delay', 0) * 1000)  # Convert to ms
        
        qos_rate = stats.get('qos_satisfaction_rate', 0)
        self.qos_rate_history.append(qos_rate * 100)  # Convert to percentage
        
        # Keep only recent history
        if len(self.throughput_history) > self.history_length:
            self.throughput_history.pop(0)
            self.delay_history.pop(0)
            self.qos_rate_history.pop(0)
    
    def render(self, reward=0.0):
        """
        Render one frame
        
        Args:
            reward: Current reward value
        """
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_c:
                    self.show_connections = not self.show_connections
                elif event.key == pygame.K_v:
                    self.show_coverage = not self.show_coverage
                elif event.key in [pygame.K_q, pygame.K_ESCAPE]:
                    pygame.quit()
                    sys.exit()
        
        # Clear screen
        self.screen.fill(self.colors['background'])
        
        # Draw grid
        self.draw_grid()
        
        # Draw base stations
        for bs in self.env.base_stations:
            self.draw_base_station(bs)
        
        # Draw users
        for user_idx, user in enumerate(self.env.users):
            self.draw_user(user, user_idx)
        
        # Draw stats panel
        stats = self.env.get_statistics()
        self.draw_stats_panel(stats, reward)
        
        # Update metrics history and draw mini graphs
        self.update_metrics_history(stats)
        
        # Draw mini graphs on the right side with adaptive sizing
        graph_x = self.screen_width - self.panel_width + 10
        graph_y = 250
        graph_width = max(80, (self.panel_width - 30) // 3)
        graph_height = 50
        self.draw_mini_graph("Throughput (Mbps)", self.throughput_history, graph_x, graph_y, 
                           width=graph_width, height=graph_height, color=(0, 255, 100))
        self.draw_mini_graph("Delay (ms)", self.delay_history, graph_x + graph_width + 5, graph_y, 
                           width=graph_width, height=graph_height, color=(255, 150, 0))
        self.draw_mini_graph("QoS Rate (%)", self.qos_rate_history, graph_x + 2*(graph_width + 5), graph_y, 
                           width=graph_width, height=graph_height, max_val=100, color=(100, 200, 255))
        
        # Draw legend
        self.draw_legend()
        
        # Pause indicator
        if self.paused:
            pause_text = self.font_large.render("PAUSED", True, (255, 255, 0))
            self.screen.blit(pause_text, (self.screen_width // 2 - 60, 20))
        
        # Update display
        pygame.display.flip()
        self.clock.tick(self.fps)
    
    def close(self):
        """Clean up"""
        pygame.quit()