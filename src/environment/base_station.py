"""
Base Station (gNB) Implementation for 5G Network Slicing
Author: Your Name
Date: 2025
"""

import numpy as np
from typing import List, Dict, Tuple

class BaseStation:
    """
    5G gNB (Next Generation NodeB)
    
    Represents a base station in the 5G network with realistic parameters
    from 3GPP specifications and industry deployments.
    """
    
    def __init__(self, 
                 bs_id: int, 
                 position: np.ndarray,
                 config: Dict):
        """
        Initialize base station
        
        Args:
            bs_id: Unique base station identifier
            position: [x, y, height] in meters
            config: Configuration dictionary
        """
        self.id = bs_id
        self.position = np.array(position, dtype=np.float32)
        
        # Physical parameters (from 3GPP and industry standards)
        self.antenna_height = config.get('bs_antenna_height', 25.0)  # meters
        self.tx_power_dbm = config.get('bs_tx_power_dbm', 46.0)      # dBm
        self.tx_power_watts = self._dbm_to_watts(self.tx_power_dbm)
        self.antenna_gain_dbi = config.get('bs_antenna_gain_dbi', 18.0)  # dBi
        self.num_antennas = config.get('bs_num_antennas', 64)
        
        # Resource management
        self.num_rbs = config.get('num_rbs', 273)
        self.available_rbs = self.num_rbs
        self.rb_bandwidth = config.get('rb_bandwidth', 360e3)  # 360 kHz
        
        # State tracking
        self.connected_users = []
        self.current_load = 0.0  # Percentage (0-1)
        self.allocated_rbs_matrix = np.zeros(self.num_rbs, dtype=np.int32)
        
        # Statistics
        self.total_throughput = 0.0
        self.served_users_count = 0
        self.energy_consumption = 0.0
        
    def _dbm_to_watts(self, dbm: float) -> float:
        """Convert dBm to Watts"""
        return 10 ** ((dbm - 30) / 10)
    
    def _watts_to_dbm(self, watts: float) -> float:
        """Convert Watts to dBm"""
        return 10 * np.log10(watts) + 30
    
    def get_distance_to(self, user_position: np.ndarray) -> float:
        """
        Calculate 3D Euclidean distance to user
        
        Args:
            user_position: [x, y, height] of user
            
        Returns:
            Distance in meters
        """
        return np.linalg.norm(self.position - user_position)
    
    def get_2d_distance_to(self, user_position: np.ndarray) -> float:
        """
        Calculate 2D distance (ignoring height)
        Useful for horizontal coverage calculations
        """
        return np.linalg.norm(self.position[:2] - user_position[:2])
    
    def allocate_rbs(self, user_id: int, rb_indices: List[int]) -> bool:
        """
        Allocate resource blocks to a user
        
        Args:
            user_id: User identifier
            rb_indices: List of RB indices to allocate
            
        Returns:
            True if allocation successful, False otherwise
        """
        # Check if RBs are available
        if not all(self.allocated_rbs_matrix[idx] == 0 for idx in rb_indices):
            return False
        
        # Allocate
        for idx in rb_indices:
            self.allocated_rbs_matrix[idx] = user_id
        
        self.available_rbs -= len(rb_indices)
        self.current_load = 1.0 - (self.available_rbs / self.num_rbs)
        
        return True
    
    def deallocate_rbs(self, user_id: int) -> int:
        """
        Deallocate all RBs assigned to a user
        
        Args:
            user_id: User identifier
            
        Returns:
            Number of RBs deallocated
        """
        deallocated = np.sum(self.allocated_rbs_matrix == user_id)
        self.allocated_rbs_matrix[self.allocated_rbs_matrix == user_id] = 0
        self.available_rbs += deallocated
        self.current_load = 1.0 - (self.available_rbs / self.num_rbs)
        
        return deallocated
    
    def reset(self):
        """Reset base station state for new episode"""
        self.connected_users = []
        self.available_rbs = self.num_rbs
        self.current_load = 0.0
        self.allocated_rbs_matrix = np.zeros(self.num_rbs, dtype=np.int32)
        self.total_throughput = 0.0
        self.served_users_count = 0
        self.energy_consumption = 0.0
    
    def get_state_vector(self) -> np.ndarray:
        """
        Get base station state as feature vector
        
        Returns:
            State vector: [load, available_rbs, num_connected_users, tx_power, ...]
        """
        return np.array([
            self.current_load,
            self.available_rbs / self.num_rbs,  # Normalized
            len(self.connected_users),
            self.tx_power_dbm / 50.0,  # Normalized (typical max ~50 dBm)
        ], dtype=np.float32)
    
    def compute_energy_consumption(self, dt: float) -> float:
        """
        Compute energy consumption for time interval dt
        
        Simple model: P_total = P_idle + P_tx * load
        
        Args:
            dt: Time interval in seconds
            
        Returns:
            Energy in Joules
        """
        # Power consumption model (simplified)
        P_idle = 100.0  # Watts (idle power)
        P_max = 1000.0  # Watts (max power at full load)
        
        P_total = P_idle + (P_max - P_idle) * self.current_load
        energy = P_total * dt  # Joules
        
        self.energy_consumption += energy
        return energy
    
    def __repr__(self) -> str:
        return (f"BaseStation(id={self.id}, "
                f"pos={self.position}, "
                f"load={self.current_load:.2f}, "
                f"available_rbs={self.available_rbs}/{self.num_rbs})")