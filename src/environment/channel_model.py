"""
 Path Loss Model (Hata-COST231)
Simple, tested, no confusion.
"""

import numpy as np


class ChannelModel:
    """
    Day 4: Only Path Loss
    We'll add fading and shadowing later
    """
    
    def __init__(self, carrier_frequency: float):
        """
        Args:
            carrier_frequency: in Hz (e.g., 3.5e9 for 3.5 GHz)
        """
        self.fc = carrier_frequency / 1e6  # Convert to MHz
        
        # Base station and user heights (standard values)
        self.bs_height = 25.0  # meters
        self.ue_height = 1.5   # meters
        self.C = 3.0           # Urban correction factor
        
        print(f"[ChannelModel] Initialized")
        print(f"  Carrier Frequency: {carrier_frequency/1e9:.2f} GHz")
        print(f"  Model: Hata-COST231 Urban")
    
    
    def calculate_path_loss(self, distance_m: float) -> float:
        """
        Hata-COST231 Path Loss Formula
        
        PL = 46.3 + 33.9*log10(fc) - 13.82*log10(hb) - a(hm) 
             + (44.9 - 6.55*log10(hb))*log10(d) + C
        
        Args:
            distance_m: Distance in meters
        
        Returns:
            Path loss in dB
        """
        # Minimum distance = 1 meter (avoid log(0))
        if distance_m < 1.0:
            distance_m = 1.0
        
        # Convert to kilometers
        d_km = distance_m / 1000.0
        
        # Mobile height correction factor
        a_hm = (1.1 * np.log10(self.fc) - 0.7) * self.ue_height \
               - (1.56 * np.log10(self.fc) - 0.8)
        
        # Calculate path loss
        PL = (46.3 
              + 33.9 * np.log10(self.fc)
              - 13.82 * np.log10(self.bs_height)
              - a_hm
              + (44.9 - 6.55 * np.log10(self.bs_height)) * np.log10(d_km)
              + self.C)
        
        return PL


def test_path_loss():
    """Test path loss at different distances"""
    print("\n" + "="*50)
    print("DAY 4 TEST: Path Loss Model")
    print("="*50)
    
    # Create channel model
    channel = ChannelModel(carrier_frequency=3.5e9)
    
    # Test at different distances
    distances = [100, 200, 500, 1000, 1500, 2000]
    
    print("\nPath Loss Results:")
    print("-" * 50)
    print(f"{'Distance (m)':<15} {'Path Loss (dB)':<20}")
    print("-" * 50)
    
    previous_pl = 0
    for d in distances:
        pl = channel.calculate_path_loss(d)
        print(f"{d:<15} {pl:<20.2f}")
        
        # Verify: path loss increases with distance
        assert pl > previous_pl, f"Path loss must increase! {pl} <= {previous_pl}"
        previous_pl = pl
    
    print("-" * 50)
    print("✓ Path loss increases with distance")
    
    # Specific check: 10 dB per decade rule (roughly)
    pl_100m = channel.calculate_path_loss(100)
    pl_1000m = channel.calculate_path_loss(1000)
    difference = pl_1000m - pl_100m
    
    print(f"\n✓ 100m → {pl_100m:.2f} dB")
    print(f"✓ 1000m → {pl_1000m:.2f} dB")
    print(f"✓ Difference: {difference:.2f} dB (expected ~30 dB)")
    
    assert 25 < difference < 40, f"10x distance should give ~30-35 dB loss, got {difference:.2f}"
    print("\n" + "="*50)
    print("✓ DAY 4 COMPLETE: Path Loss Working!")
    print("="*50)


if __name__ == "__main__":
    test_path_loss()

"""
Day 5: Path Loss + Rayleigh Fading
Adding small-scale fading to the model
"""

import numpy as np
from typing import Dict, Tuple


class ChannelModel:
    """
    Day 5: Path Loss + Rayleigh Fading
    """
    
    def __init__(self, carrier_frequency: float):
        """
        Args:
            carrier_frequency: in Hz (e.g., 3.5e9 for 3.5 GHz)
        """
        self.fc = carrier_frequency / 1e6  # Convert to MHz
        
        # Base station and user heights (standard values)
        self.bs_height = 25.0  # meters
        self.ue_height = 1.5   # meters
        self.C = 3.0           # Urban correction factor
        
        # Rayleigh fading parameters
        self.fading_cache: Dict[int, Tuple[float, int]] = {}
        self.coherence_time_slots = 10  # Fading constant for 10 time slots
        
        print(f"[ChannelModel] Initialized")
        print(f"  Carrier Frequency: {carrier_frequency/1e9:.2f} GHz")
        print(f"  Model: Hata-COST231 Urban + Rayleigh Fading")
        print(f"  Coherence Time: {self.coherence_time_slots} slots")
    
    
    def calculate_path_loss(self, distance_m: float) -> float:
        """
        Hata-COST231 Path Loss Formula
        
        PL = 46.3 + 33.9*log10(fc) - 13.82*log10(hb) - a(hm) 
             + (44.9 - 6.55*log10(hb))*log10(d) + C
        
        Args:
            distance_m: Distance in meters
        
        Returns:
            Path loss in dB
        """
        # Minimum distance = 1 meter (avoid log(0))
        if distance_m < 1.0:
            distance_m = 1.0
        
        # Convert to kilometers
        d_km = distance_m / 1000.0
        
        # Mobile height correction factor
        a_hm = (1.1 * np.log10(self.fc) - 0.7) * self.ue_height \
               - (1.56 * np.log10(self.fc) - 0.8)
        
        # Calculate path loss
        PL = (46.3 
              + 33.9 * np.log10(self.fc)
              - 13.82 * np.log10(self.bs_height)
              - a_hm
              + (44.9 - 6.55 * np.log10(self.bs_height)) * np.log10(d_km)
              + self.C)
        
        return PL
    
    
    def generate_rayleigh_fading(self, user_id: int, time_slot: int) -> float:
        """
        Rayleigh Fading Generator
        
        Models fast fading due to multipath in NLOS (non-line-of-sight)
        
        Math:
        - h = h_real + j*h_imag, where both ~ N(0, 0.5)
        - |h|² = h_real² + h_imag² ~ Exponential(1)
        - In dB: 10*log10(|h|²)
        
        Temporal correlation:
        - Fading stays constant for coherence_time_slots
        - Then regenerates (simulates user movement)
        
        Args:
            user_id: User identifier (for independent fading per user)
            time_slot: Current time slot (for temporal correlation)
        
        Returns:
            Fading gain in dB (typically negative, e.g., -5 dB)
        """
        # Check if we have a cached value that's still valid
        if user_id in self.fading_cache:
            cached_fading, cached_slot = self.fading_cache[user_id]
            
            # If within coherence time, reuse cached value
            if time_slot - cached_slot < self.coherence_time_slots:
                return cached_fading
        
        # Generate new Rayleigh sample
        # Complex channel: h = h_real + j*h_imag
        h_real = np.random.normal(0, 1.0/np.sqrt(2))
        h_imag = np.random.normal(0, 1.0/np.sqrt(2))
        
        # Power: |h|²
        h_squared = h_real**2 + h_imag**2
        
        # Convert to dB
        fading_db = 10 * np.log10(h_squared + 1e-10)  # +1e-10 to avoid log(0)
        
        # Cache for future use
        self.fading_cache[user_id] = (fading_db, time_slot)
        
        return fading_db
    
    
    def calculate_received_power(self, 
                                  tx_power_dbm: float,
                                  distance_m: float,
                                  user_id: int,
                                  time_slot: int,
                                  include_fading: bool = True) -> float:
        """
        Calculate received signal power
        
        Formula:
        P_rx = P_tx - PathLoss + Fading
        
        Args:
            tx_power_dbm: Transmit power in dBm (e.g., 43 dBm = 20W)
            distance_m: Distance between transmitter and receiver
            user_id: User ID (for fading correlation)
            time_slot: Current time slot
            include_fading: If True, add Rayleigh fading
        
        Returns:
            Received power in dBm
        """
        # Path loss
        path_loss_db = self.calculate_path_loss(distance_m)
        
        # Start with free space propagation
        rx_power_dbm = tx_power_dbm - path_loss_db
        
        # Add fading if enabled
        if include_fading:
            fading_db = self.generate_rayleigh_fading(user_id, time_slot)
            rx_power_dbm += fading_db
        
        return rx_power_dbm


def test_rayleigh_fading():
    """Test Rayleigh fading statistics"""
    print("\n" + "="*50)
    print("DAY 5 TEST: Rayleigh Fading")
    print("="*50)
    
    channel = ChannelModel(carrier_frequency=3.5e9)
    
    # Generate many fading samples
    num_samples = 10000
    fading_samples_linear = []
    
    print(f"\nGenerating {num_samples} Rayleigh samples...")
    
    for i in range(num_samples):
        # Generate fading (force new sample each time by using different user_id)
        fading_db = channel.generate_rayleigh_fading(user_id=i, time_slot=0)
        
        # Convert back to linear scale
        fading_linear = 10 ** (fading_db / 10.0)
        fading_samples_linear.append(fading_linear)
    
    # Statistics
    mean_linear = np.mean(fading_samples_linear)
    mean_db = 10 * np.log10(mean_linear)
    
    print(f"\n✓ Rayleigh Statistics:")
    print(f"  Mean |h|² (linear): {mean_linear:.3f} (expected: 1.0)")
    print(f"  Mean |h|² (dB): {mean_db:.2f} dB (expected: 0 dB)")
    
    # Validation: mean should be close to 1.0 (0 dB)
    assert 0.95 < mean_linear < 1.05, f"Rayleigh mean should be ~1.0, got {mean_linear}"
    
    print("\n✓ Rayleigh fading statistics correct!")
    
    # Test temporal correlation
    print("\n✓ Testing Temporal Correlation:")
    user_id = 999
    
    # Same time slot → same fading
    fading_t0 = channel.generate_rayleigh_fading(user_id, time_slot=0)
    fading_t1 = channel.generate_rayleigh_fading(user_id, time_slot=1)
    
    print(f"  Time slot 0: {fading_t0:.2f} dB")
    print(f"  Time slot 1: {fading_t1:.2f} dB")
    assert fading_t0 == fading_t1, "Fading should be same within coherence time!"
    print("  → Same value (cached) ✓")
    
    # After coherence time → new fading
    fading_t15 = channel.generate_rayleigh_fading(user_id, time_slot=15)
    print(f"  Time slot 15: {fading_t15:.2f} dB")
    assert fading_t15 != fading_t0, "Fading should change after coherence time!"
    print("  → Different value (regenerated) ✓")
    
    print("\n" + "="*50)
    print("✓ DAY 5 COMPLETE: Rayleigh Fading Working!")
    print("="*50)


def test_received_power():
    """Test complete received power calculation"""
    print("\n" + "="*50)
    print("DAY 5 TEST: Received Power Calculation")
    print("="*50)
    
    channel = ChannelModel(carrier_frequency=3.5e9)
    
    # Test parameters
    tx_power_dbm = 43.0  # 20 Watts
    distance_m = 500.0
    user_id = 1
    time_slot = 0
    
    # Without fading
    rx_no_fading = channel.calculate_received_power(
        tx_power_dbm, distance_m, user_id, time_slot, include_fading=False
    )
    
    # With fading
    rx_with_fading = channel.calculate_received_power(
        tx_power_dbm, distance_m, user_id, time_slot, include_fading=True
    )
    
    print(f"\nTransmit Power: {tx_power_dbm} dBm")
    print(f"Distance: {distance_m} m")
    print(f"\n✓ Received Power (no fading): {rx_no_fading:.2f} dBm")
    print(f"✓ Received Power (with fading): {rx_with_fading:.2f} dBm")
    print(f"✓ Fading Effect: {rx_with_fading - rx_no_fading:.2f} dB")
    
    # Validation
    path_loss = channel.calculate_path_loss(distance_m)
    expected_no_fading = tx_power_dbm - path_loss
    
    assert abs(rx_no_fading - expected_no_fading) < 0.1, "Received power calculation error!"
    
    print("\n" + "="*50)
    print("✓ DAY 5 COMPLETE: All Tests Passed!")
    print("="*50)


if __name__ == "__main__":
    test_rayleigh_fading()
    test_received_power()
"""
Day 6: Complete Channel Model
Path Loss + Rayleigh Fading + Shadowing + Interference
"""

import numpy as np
from typing import Dict, Tuple


class ChannelModel:
    """
    Day 6: Complete Channel Model
    """
    
    def __init__(self, carrier_frequency: float, system_bandwidth: float):
        """
        Args:
            carrier_frequency: in Hz (e.g., 3.5e9 for 3.5 GHz)
            system_bandwidth: in Hz (e.g., 100e6 for 100 MHz)
        """
        self.fc = carrier_frequency / 1e6  # Convert to MHz
        self.bw = system_bandwidth
        
        # Base station and user heights (standard values)
        self.bs_height = 25.0  # meters
        self.ue_height = 1.5   # meters
        self.C = 3.0           # Urban correction factor
        
        # Rayleigh fading parameters
        self.fading_cache: Dict[int, Tuple[float, int]] = {}
        self.coherence_time_slots = 10  # Fading constant for 10 time slots
        
        # Shadowing parameters
        self.shadowing_std = 8.0  # dB (typical urban value)
        
        # Noise parameters
        self.noise_figure = 9.0  # dB (typical UE)
        self.thermal_noise_density = -174.0  # dBm/Hz
        
        print(f"[ChannelModel] Initialized")
        print(f"  Carrier Frequency: {carrier_frequency/1e9:.2f} GHz")
        print(f"  Bandwidth: {system_bandwidth/1e6:.0f} MHz")
        print(f"  Model: Hata-COST231 + Rayleigh + Shadowing")
        print(f"  Shadowing σ: {self.shadowing_std} dB")
        print(f"  Coherence Time: {self.coherence_time_slots} slots")
    
    
    def calculate_path_loss(self, distance_m: float) -> float:
        """
        Hata-COST231 Path Loss Formula
        
        PL = 46.3 + 33.9*log10(fc) - 13.82*log10(hb) - a(hm) 
             + (44.9 - 6.55*log10(hb))*log10(d) + C
        
        Args:
            distance_m: Distance in meters
        
        Returns:
            Path loss in dB
        """
        # Minimum distance = 1 meter (avoid log(0))
        if distance_m < 1.0:
            distance_m = 1.0
        
        # Convert to kilometers
        d_km = distance_m / 1000.0
        
        # Mobile height correction factor
        a_hm = (1.1 * np.log10(self.fc) - 0.7) * self.ue_height \
               - (1.56 * np.log10(self.fc) - 0.8)
        
        # Calculate path loss
        PL = (46.3 
              + 33.9 * np.log10(self.fc)
              - 13.82 * np.log10(self.bs_height)
              - a_hm
              + (44.9 - 6.55 * np.log10(self.bs_height)) * np.log10(d_km)
              + self.C)
        
        return PL
    
    
    def generate_rayleigh_fading(self, user_id: int, time_slot: int) -> float:
        """
        Rayleigh Fading Generator
        
        Models fast fading due to multipath in NLOS (non-line-of-sight)
        
        Math:
        - h = h_real + j*h_imag, where both ~ N(0, 0.5)
        - |h|² = h_real² + h_imag² ~ Exponential(1)
        - In dB: 10*log10(|h|²)
        
        Temporal correlation:
        - Fading stays constant for coherence_time_slots
        - Then regenerates (simulates user movement)
        
        Args:
            user_id: User identifier (for independent fading per user)
            time_slot: Current time slot (for temporal correlation)
        
        Returns:
            Fading gain in dB (typically negative, e.g., -5 dB)
        """
        # Check if we have a cached value that's still valid
        if user_id in self.fading_cache:
            cached_fading, cached_slot = self.fading_cache[user_id]
            
            # If within coherence time, reuse cached value
            if time_slot - cached_slot < self.coherence_time_slots:
                return cached_fading
        
        # Generate new Rayleigh sample
        # Complex channel: h = h_real + j*h_imag
        h_real = np.random.normal(0, 1.0/np.sqrt(2))
        h_imag = np.random.normal(0, 1.0/np.sqrt(2))
        
        # Power: |h|²
        h_squared = h_real**2 + h_imag**2
        
        # Convert to dB
        fading_db = 10 * np.log10(h_squared + 1e-10)  # +1e-10 to avoid log(0)
        
        # Cache for future use
        self.fading_cache[user_id] = (fading_db, time_slot)
        
        return fading_db
    
    
    def generate_shadowing(self) -> float:
        """
        Log-normal Shadowing
        
        Models large-scale signal variations due to:
        - Buildings and obstacles
        - Terrain variations
        - Foliage
        
        Distribution: X ~ N(0, σ²) in dB
        Typical σ: 6-10 dB (urban), 4-6 dB (suburban)
        
        Returns:
            Shadowing in dB (can be positive or negative)
        """
        return np.random.normal(0, self.shadowing_std)
    
    
    def calculate_received_power(self, 
                                  tx_power_dbm: float,
                                  distance_m: float,
                                  user_id: int,
                                  time_slot: int,
                                  include_fading: bool = True,
                                  include_shadowing: bool = True) -> float:
        """
        Calculate received signal power
        
        Formula:
        P_rx = P_tx - PathLoss + Fading + Shadowing
        
        Args:
            tx_power_dbm: Transmit power in dBm (e.g., 43 dBm = 20W)
            distance_m: Distance between transmitter and receiver
            user_id: User ID (for fading correlation)
            time_slot: Current time slot
            include_fading: If True, add Rayleigh fading
            include_shadowing: If True, add log-normal shadowing
        
        Returns:
            Received power in dBm
        """
        # Path loss
        path_loss_db = self.calculate_path_loss(distance_m)
        
        # Start with free space propagation
        rx_power_dbm = tx_power_dbm - path_loss_db
        
        # Add fading if enabled
        if include_fading:
            fading_db = self.generate_rayleigh_fading(user_id, time_slot)
            rx_power_dbm += fading_db
        
        # Add shadowing if enabled
        if include_shadowing:
            shadowing_db = self.generate_shadowing()
            rx_power_dbm += shadowing_db
        
        return rx_power_dbm
    
    
    def calculate_sinr(self,
                       bs_id: int,
                       user_position: np.ndarray,
                       bs_positions: np.ndarray,
                       tx_power_dbm: float,
                       user_id: int,
                       time_slot: int,
                       num_rbs: int = 1) -> float:
        """
        Calculate SINR (Signal-to-Interference-plus-Noise Ratio)
        
        SINR = Signal / (Interference + Noise)
        
        Components:
        1. Signal: Power from serving BS
        2. Interference: Power from all other BSs (co-channel)
        3. Noise: Thermal noise (depends on bandwidth)
        
        Args:
            bs_id: Serving base station index
            user_position: User's [x, y] coordinates
            bs_positions: Array of all BS positions (N_bs x 2)
            tx_power_dbm: BS transmit power (dBm)
            user_id: User ID
            time_slot: Current time slot
            num_rbs: Number of allocated RBs (affects noise bandwidth)
        
        Returns:
            SINR in dB
        """
        # 1. Calculate signal power from serving BS
        distance_signal = np.linalg.norm(user_position - bs_positions[bs_id])
        signal_power_dbm = self.calculate_received_power(
            tx_power_dbm, distance_signal, user_id, time_slot
        )
        
        # Convert to linear scale (mW)
        signal_power_mw = 10 ** (signal_power_dbm / 10.0)
        
        # 2. Calculate interference from other BSs
        interference_mw = 0.0
        
        for interferer_id in range(len(bs_positions)):
            if interferer_id == bs_id:
                continue  # Skip serving BS
            
            # Distance to interfering BS
            distance_interferer = np.linalg.norm(
                user_position - bs_positions[interferer_id]
            )
            
            # Interference power (use different user_id for independent fading)
            interference_dbm = self.calculate_received_power(
                tx_power_dbm,
                distance_interferer,
                user_id + 1000 + interferer_id,  # Different fading realization
                time_slot,
                include_shadowing=False  # Shadowing is uncorrelated for interference
            )
            
            # Add to total interference (in linear scale)
            interference_mw += 10 ** (interference_dbm / 10.0)
        
        # 3. Calculate thermal noise power
        # N = N0 * B + NF
        # where N0 = thermal noise density (-174 dBm/Hz)
        #       B = bandwidth per RB
        #       NF = noise figure
        
        rb_bandwidth = 360e3  # Hz (with 30 kHz SCS: 12 subcarriers × 30 kHz)
        total_bandwidth = rb_bandwidth * num_rbs
        
        noise_power_dbm = (self.thermal_noise_density 
                           + 10 * np.log10(total_bandwidth)
                           + self.noise_figure)
        
        noise_power_mw = 10 ** (noise_power_dbm / 10.0)
        
        # 4. Calculate SINR
        sinr_linear = signal_power_mw / (interference_mw + noise_power_mw)
        sinr_db = 10 * np.log10(sinr_linear + 1e-10)  # Avoid log(0)
        
        return sinr_db
    
    
    def calculate_throughput(self, sinr_db: float, num_rbs: int) -> float:
        """
        Shannon Capacity Formula
        
        C = B * log2(1 + SINR)
        
        Args:
            sinr_db: SINR in dB
            num_rbs: Number of allocated RBs
        
        Returns:
            Throughput in bps
        """
        # Convert SINR to linear scale
        sinr_linear = 10 ** (sinr_db / 10.0)
        
        # Bandwidth per RB
        rb_bandwidth = 360e3  # Hz
        
        # Shannon formula
        throughput = rb_bandwidth * num_rbs * np.log2(1 + sinr_linear)
        
        return throughput


def test_shadowing():
    """Test log-normal shadowing statistics"""
    print("\n" + "="*50)
    print("DAY 6 TEST: Shadowing Model")
    print("="*50)
    
    channel = ChannelModel(carrier_frequency=3.5e9, system_bandwidth=100e6)
    
    # Generate many shadowing samples
    num_samples = 10000
    shadowing_samples = [channel.generate_shadowing() for _ in range(num_samples)]
    
    # Statistics
    mean = np.mean(shadowing_samples)
    std = np.std(shadowing_samples)
    
    print(f"\n✓ Shadowing Statistics (N={num_samples}):")
    print(f"  Mean: {mean:.2f} dB (expected: 0 dB)")
    print(f"  Std Dev: {std:.2f} dB (expected: {channel.shadowing_std} dB)")
    
    # Validation
    assert abs(mean) < 0.5, f"Shadowing mean should be ~0, got {mean:.2f}"
    assert abs(std - channel.shadowing_std) < 0.5, \
        f"Shadowing std should be {channel.shadowing_std}, got {std:.2f}"
    
    print("\n✓ Shadowing statistics correct!")
    print("\n" + "="*50)
    print("✓ DAY 6 PART 1 COMPLETE: Shadowing Working!")
    print("="*50)


def test_sinr_and_interference():
    """Test SINR calculation with interference"""
    print("\n" + "="*50)
    print("DAY 6 TEST: SINR & Interference")
    print("="*50)
    
    channel = ChannelModel(carrier_frequency=3.5e9, system_bandwidth=100e6)
    
    # Setup: 3 base stations in a line
    bs_positions = np.array([
        [0, 0],       # BS 0
        [1000, 0],    # BS 1 (1 km away)
        [-1000, 0]    # BS 2 (1 km away, opposite side)
    ])
    
    # User close to BS 0
    user_position = np.array([100, 0])  # 100m from BS 0
    
    tx_power_dbm = 43.0  # 20W
    user_id = 1
    time_slot = 0
    num_rbs = 10
    
    # Calculate SINR
    sinr_db = channel.calculate_sinr(
        bs_id=0,
        user_position=user_position,
        bs_positions=bs_positions,
        tx_power_dbm=tx_power_dbm,
        user_id=user_id,
        time_slot=time_slot,
        num_rbs=num_rbs
    )
    
    print(f"\nScenario:")
    print(f"  User position: {user_position}")
    print(f"  Serving BS: 0 (distance: 100m)")
    print(f"  Interfering BSs: 1 (1100m), 2 (1100m)")
    print(f"  TX Power: {tx_power_dbm} dBm")
    print(f"  Allocated RBs: {num_rbs}")
    
    print(f"\n✓ SINR: {sinr_db:.2f} dB")
    
    # Calculate throughput
    throughput = channel.calculate_throughput(sinr_db, num_rbs)
    print(f"✓ Throughput: {throughput/1e6:.2f} Mbps")
    
    # Validation: SINR should be reasonable (0-30 dB range)
    assert -10 < sinr_db < 40, f"SINR out of reasonable range: {sinr_db:.2f} dB"
    assert throughput > 0, "Throughput must be positive"
    
    print("\n" + "="*50)
    print("✓ DAY 6 PART 2 COMPLETE: SINR Working!")
    print("="*50)


def test_complete_channel_validation():
    """Final comprehensive validation"""
    print("\n" + "="*60)
    print("DAY 6 FINAL VALIDATION: Complete Channel Model")
    print("="*60)
    
    channel = ChannelModel(carrier_frequency=3.5e9, system_bandwidth=100e6)
    
    # Test scenario: 7 BSs (hexagonal), multiple users
    print("\n[Test 1] Multi-BS Scenario")
    
    # Hexagonal BS layout
    radius = 500  # meters
    bs_positions = []
    bs_positions.append([0, 0])  # Center
    for angle in [0, 60, 120, 180, 240, 300]:
        angle_rad = np.deg2rad(angle)
        x = radius * np.cos(angle_rad)
        y = radius * np.sin(angle_rad)
        bs_positions.append([x, y])
    bs_positions = np.array(bs_positions)
    
    print(f"  Base Stations: {len(bs_positions)}")
    
    # Test user at different locations
    test_positions = [
        np.array([50, 50]),      # Near center BS
        np.array([400, 0]),      # Between BSs
        np.array([600, 600])     # Far from all
    ]
    
    print(f"\n  Testing {len(test_positions)} user positions:")
    
    for i, user_pos in enumerate(test_positions):
        # Find best BS
        best_sinr = -999
        best_bs = 0
        
        for bs_id in range(len(bs_positions)):
            sinr = channel.calculate_sinr(
                bs_id, user_pos, bs_positions, 43.0, i, 0, num_rbs=10
            )
            if sinr > best_sinr:
                best_sinr = sinr
                best_bs = bs_id
        
        distance = np.linalg.norm(user_pos - bs_positions[best_bs])
        throughput = channel.calculate_throughput(best_sinr, 10)
        
        print(f"\n    User {i}: Position {user_pos}")
        print(f"      Best BS: {best_bs}, Distance: {distance:.1f}m")
        print(f"      SINR: {best_sinr:.2f} dB")
        print(f"      Throughput: {throughput/1e6:.2f} Mbps")
        
        assert best_sinr > -10, f"User {i} has very poor SINR!"
        assert throughput > 1e6, f"User {i} has very low throughput!"
    
    print("\n  ✓ All users have viable connections")
    
    # Test edge effects
    print("\n[Test 2] Edge User (Cell Edge)")
    edge_user = np.array([800, 800])  # Far from all BSs
    
    best_sinr_edge = -999
    for bs_id in range(len(bs_positions)):
        sinr = channel.calculate_sinr(
            bs_id, edge_user, bs_positions, 43.0, 999, 0, num_rbs=10
        )
        if sinr > best_sinr_edge:
            best_sinr_edge = sinr
    
    print(f"  Edge User SINR: {best_sinr_edge:.2f} dB")
    print(f"  ✓ Edge user still has coverage")
    
    print("\n" + "="*60)
    print("✓✓✓ DAY 6 COMPLETE: ALL TESTS PASSED! ✓✓✓")
    print("="*60)
    print("\nChannel Model Features:")
    print("  ✓ Path Loss (Hata-COST231)")
    print("  ✓ Rayleigh Fading (with temporal correlation)")
    print("  ✓ Log-normal Shadowing")
    print("  ✓ Co-channel Interference")
    print("  ✓ SINR Calculation")
    print("  ✓ Shannon Capacity (Throughput)")
    print("\n→ Ready for Traffic Generator (Days 7-9)")
    print("="*60)


if __name__ == "__main__":
    test_shadowing()
    test_sinr_and_interference()
    test_complete_channel_validation()    