"""
User Equipment (UE) Implementation for 5G Network Slicing
Author: Your Name
Date: 2025
"""

import numpy as np
from typing import Dict, Optional, Tuple
from enum import Enum

class ServiceType(Enum):
    """5G Service Types (3GPP TS 22.261)"""
    eMBB = "eMBB"      # Enhanced Mobile Broadband
    URLLC = "URLLC"    # Ultra-Reliable Low-Latency Communications
    mMTC = "mMTC"      # Massive Machine-Type Communications

class MobilityModel(Enum):
    """User mobility models"""
    STATIC = "static"
    RANDOM_WAYPOINT = "RandomWaypoint"
    RANDOM_DIRECTION = "RandomDirection"
    MANHATTAN = "Manhattan"
    HIGHWAY = "Highway"

class User:
    """
    5G UE (User Equipment)
    
    Represents a mobile user with specific service requirements,
    mobility pattern, and traffic characteristics.
    """
    
    def __init__(self,
                 user_id: int,
                 initial_position: np.ndarray,
                 service_type: ServiceType,
                 config: Dict):
        """
        Initialize user
        
        Args:
            user_id: Unique user identifier
            initial_position: [x, y] in meters
            service_type: Type of service (eMBB/URLLC/mMTC)
            config: Configuration dictionary
        """
        self.id = user_id
        self.position = np.array(initial_position, dtype=np.float32)
        self.height = config.get('user_height', 1.5)  # meters
        
        # Service type and QoS requirements
        self.service_type = service_type
        self.qos_requirements = self._get_qos_requirements(service_type, config)
        
        # Mobility
        self.mobility_model = config.get('mobility_model', 'RandomWaypoint')
        self.speed = self._assign_speed(config)  # m/s
        self.velocity = np.array([0.0, 0.0], dtype=np.float32)
        self.waypoint = None
        self.coverage_area = config.get('coverage_area', [1000, 1000])
        
        # Connection state
        self.connected_bs = None
        self.allocated_rbs = []
        self.sinr_db = -np.inf
        self.channel_gain = 0.0
        
        # Traffic state
        self.buffer_size = 0  # bits waiting to be transmitted
        self.packet_queue = []
        self.last_packet_arrival = 0.0
        
        # Statistics
        self.total_throughput = 0.0
        self.total_delay = 0.0
        self.packets_sent = 0
        self.packets_dropped = 0
        self.qos_satisfied = True
        
    def _get_qos_requirements(self, service_type: ServiceType, config: Dict) -> Dict:
        """
        Get QoS requirements from config based on service type
        Based on 3GPP TS 22.261
        """
        services_config = config.get('services', {})
        service_config = services_config.get(service_type.value, {})
        
        return {
            'min_throughput': service_config.get('min_throughput', 1e6),
            'max_latency': service_config.get('max_latency', 0.1),
            'reliability': service_config.get('reliability', 0.95),
            'priority': service_config.get('priority', 2),
            'min_sinr_db': service_config.get('min_sinr_db', 0),
            'packet_size_mean': service_config.get('packet_size_mean', 1500),
            'packet_size_std': service_config.get('packet_size_std', 500),
            'traffic_model': service_config.get('traffic_model', 'poisson'),
            'arrival_rate': service_config.get('arrival_rate', 10)
        }
    
    def _assign_speed(self, config: Dict) -> float:
        """
        Assign speed based on user type (pedestrian vs vehicular)
        """
        speed_dist = config.get('speed_distribution', {'pedestrian': 0.7, 'vehicular': 0.3})
        speeds = config.get('user_speeds', {'pedestrian': 1.4, 'vehicular': 30.0})
        
        if np.random.random() < speed_dist['pedestrian']:
            return speeds['pedestrian']  # 1.4 m/s = 5 km/h
        else:
            return speeds['vehicular']   # 30 m/s = 108 km/h
    
    def update_position(self, dt: float):
        """
        Update user position based on mobility model
        
        Args:
            dt: Time step in seconds
        """
        if self.mobility_model == 'STATIC':
            return
        
        elif self.mobility_model == 'RandomWaypoint':
            self._random_waypoint_move(dt)
        
        elif self.mobility_model == 'RandomDirection':
            self._random_direction_move(dt)
        
        elif self.mobility_model == 'Manhattan':
            self._manhattan_move(dt)
        
        # Enforce boundaries
        self.position = np.clip(self.position, [0, 0], self.coverage_area)
    
    def _random_waypoint_move(self, dt: float):
        """
        Random Waypoint mobility model
        Most common in research literature
        """
        if self.waypoint is None or self._reached_waypoint():
            # Pick new random destination
            self.waypoint = np.random.uniform([0, 0], self.coverage_area)
            
            # Calculate direction
            direction = self.waypoint - self.position
            distance = np.linalg.norm(direction)
            
            if distance > 0:
                self.velocity = (direction / distance) * self.speed
            else:
                self.velocity = np.array([0.0, 0.0])
        
        # Move towards waypoint
        self.position += self.velocity * dt
    
    def _reached_waypoint(self) -> bool:
        """Check if user reached waypoint"""
        if self.waypoint is None:
            return True
        
        distance = np.linalg.norm(self.waypoint - self.position)
        return distance < 1.0  # Within 1 meter
    
    def _random_direction_move(self, dt: float):
        """
        Random Direction mobility model
        User moves in random direction for random time
        """
        if not hasattr(self, 'direction_timer') or self.direction_timer <= 0:
            # Pick new random direction and duration
            angle = np.random.uniform(0, 2 * np.pi)
            self.velocity = self.speed * np.array([np.cos(angle), np.sin(angle)])
            self.direction_timer = np.random.uniform(5, 20)  # 5-20 seconds
        
        self.position += self.velocity * dt
        self.direction_timer -= dt
    
    def _manhattan_move(self, dt: float):
        """
        Manhattan grid mobility (for urban scenarios)
        Users follow street grid pattern
        """
        # Simplified: move along x or y axis
        if not hasattr(self, 'manhattan_direction'):
            self.manhattan_direction = np.random.choice(['x', 'y'])
        
        if self.manhattan_direction == 'x':
            self.velocity = np.array([self.speed, 0.0])
        else:
            self.velocity = np.array([0.0, self.speed])
        
        self.position += self.velocity * dt
        
        # Change direction at intersections (every 100m)
        if np.random.random() < 0.01:  # 1% chance per time step
            self.manhattan_direction = 'x' if self.manhattan_direction == 'y' else 'y'
    
    def generate_traffic(self, dt: float, current_time: float) -> int:
        """
        Generate traffic packets based on service type
        
        Args:
            dt: Time step in seconds
            current_time: Current simulation time
            
        Returns:
            Number of bits generated
        """
        traffic_model = self.qos_requirements['traffic_model']
        arrival_rate = self.qos_requirements['arrival_rate']  # packets/second
        
        bits_generated = 0
        
        if traffic_model == 'poisson':
            # Poisson arrival process (for mMTC)
            num_arrivals = np.random.poisson(arrival_rate * dt)
            for _ in range(num_arrivals):
                packet_size_bytes = max(1, int(np.random.normal(
                    self.qos_requirements['packet_size_mean'],
                    self.qos_requirements['packet_size_std']
                )))
                bits_generated += packet_size_bytes * 8
                self.packet_queue.append({
                    'size': packet_size_bytes * 8,
                    'arrival_time': current_time,
                    'deadline': current_time + self.qos_requirements['max_latency']
                })
        
        elif traffic_model == 'periodic':
            # Periodic arrivals (for URLLC)
            time_since_last = current_time - self.last_packet_arrival
            period = 1.0 / arrival_rate
            
            if time_since_last >= period:
                packet_size_bytes = int(np.random.normal(
                    self.qos_requirements['packet_size_mean'],
                    self.qos_requirements['packet_size_std']
                ))
                bits_generated += packet_size_bytes * 8
                self.packet_queue.append({
                    'size': packet_size_bytes * 8,
                    'arrival_time': current_time,
                    'deadline': current_time + self.qos_requirements['max_latency']
                })
                self.last_packet_arrival = current_time
        
        elif traffic_model == 'bursty':
            # Bursty traffic (for eMBB - video streaming)
            if np.random.random() < 0.1:  # 10% chance of burst
                num_packets = np.random.poisson(5)  # Burst of ~5 packets
                for _ in range(num_packets):
                    packet_size_bytes = int(np.random.normal(
                        self.qos_requirements['packet_size_mean'],
                        self.qos_requirements['packet_size_std']
                    ))
                    bits_generated += packet_size_bytes * 8
                    self.packet_queue.append({
                        'size': packet_size_bytes * 8,
                        'arrival_time': current_time,
                        'deadline': current_time + self.qos_requirements['max_latency']
                    })
        
        self.buffer_size += bits_generated
        return bits_generated
    
    def check_qos_satisfaction(self, achieved_throughput: float, 
                               achieved_latency: float) -> bool:
        """
        Check if QoS requirements are satisfied
        
        Args:
            achieved_throughput: Actual throughput in bps
            achieved_latency: Actual latency in seconds
            
        Returns:
            True if QoS satisfied, False otherwise
        """
        throughput_ok = achieved_throughput >= self.qos_requirements['min_throughput']
        latency_ok = achieved_latency <= self.qos_requirements['max_latency']
        
        self.qos_satisfied = throughput_ok and latency_ok
        return self.qos_satisfied
    
    def get_3d_position(self) -> np.ndarray:
        """Get 3D position [x, y, height]"""
        return np.array([self.position[0], self.position[1], self.height])
    
    def reset(self):
        """Reset user state for new episode"""
        self.buffer_size = 0
        self.packet_queue = []
        self.allocated_rbs = []
        self.total_throughput = 0.0
        self.total_delay = 0.0
        self.packets_sent = 0
        self.packets_dropped = 0
        self.qos_satisfied = True
        self.sinr_db = -np.inf
    
    def __repr__(self) -> str:
        return (f"User(id={self.id}, "
                f"service={self.service_type.value}, "
                f"pos={self.position}, "
                f"speed={self.speed:.1f}m/s)")