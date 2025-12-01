"""
Day 7: Traffic Generator
Realistic packet generation for eMBB, URLLC, mMTC services
"""

import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
from collections import deque


@dataclass
class Packet:
    """
    Data packet representation
    """
    packet_id: int
    user_id: int
    service_type: str
    size_bytes: int
    arrival_time: float  # seconds
    deadline: float      # seconds (arrival_time + latency_requirement)
    priority: int        # 1=highest (URLLC), 2=medium (eMBB), 3=low (mMTC)
    
    # Tracking
    transmission_start_time: float = None
    transmission_end_time: float = None
    dropped: bool = False
    
    def is_expired(self, current_time: float) -> bool:
        """Check if packet missed its deadline"""
        return current_time > self.deadline
    
    def get_delay(self) -> float:
        """Get end-to-end delay (if transmitted)"""
        if self.transmission_end_time is not None:
            return self.transmission_end_time - self.arrival_time
        return None
    
    def is_successful(self) -> bool:
        """Check if packet was delivered on time"""
        if self.dropped or self.transmission_end_time is None:
            return False
        return self.transmission_end_time <= self.deadline


class TrafficGenerator:
    """
    Generates realistic traffic for 5G services
    
    Service Characteristics:
    - eMBB: Variable packet sizes (100 KB - 2 MB), bursty arrivals
    - URLLC: Small packets (100-500 bytes), periodic/deterministic
    - mMTC: Tiny packets (20-100 bytes), Poisson arrivals
    """
    
    def __init__(self, slot_duration: float = 0.5e-3):
        """
        Args:
            slot_duration: Time slot duration in seconds (0.5 ms default)
        """
        self.slot_duration = slot_duration
        self.packet_id_counter = 0
        
        # Traffic parameters for each service type
        self.traffic_params = {
            'embb': {
                'packet_size_mean': 500_000,      # 500 KB (video chunk)
                'packet_size_std': 200_000,       # Variable size
                'arrival_rate': 2.0,              # packets/second (λ for Poisson)
                'latency_requirement': 0.050,     # 50 ms
                'priority': 2,
                'burst_probability': 0.3          # 30% chance of burst
            },
            'urllc': {
                'packet_size_mean': 200,          # 200 bytes (control message)
                'packet_size_std': 100,
                'arrival_rate': None,             # Periodic, not Poisson
                'period': 0.001,                  # 1 ms (deterministic)
                'latency_requirement': 0.001,     # 1 ms (CRITICAL)
                'priority': 1,
                'jitter': 0.0001                  # ±0.1 ms timing variation
            },
            'mmtc': {
                'packet_size_mean': 50,           # 50 bytes (sensor reading)
                'packet_size_std': 30,
                'arrival_rate': 0.1,              # 0.1 packets/second (rare)
                'latency_requirement': 10.0,      # 10 seconds (very relaxed)
                'priority': 3
            }
        }
        
        # User-specific state
        self.user_queues: Dict[int, deque] = {}  # user_id -> packet queue
        self.user_last_urllc_time: Dict[int, float] = {}  # For periodic URLLC
        
        print(f"[TrafficGenerator] Initialized")
        print(f"  Slot Duration: {slot_duration*1000:.2f} ms")
        print(f"  Service Types: eMBB, URLLC, mMTC")
    
    
    def generate_packet_size(self, service_type: str) -> int:
        """
        Generate packet size based on service type
        
        Args:
            service_type: 'embb', 'urllc', or 'mmtc'
        
        Returns:
            Packet size in bytes
        """
        params = self.traffic_params[service_type]
        
        # Generate from normal distribution, clip to positive values
        size = np.random.normal(
            params['packet_size_mean'],
            params['packet_size_std']
        )
        
        # Minimum packet size = 20 bytes (headers)
        size = max(20, int(size))
        
        return size
    
    
    def generate_packets_poisson(self, 
                                  user_id: int,
                                  service_type: str,
                                  current_time: float,
                                  time_window: float) -> List[Packet]:
        """
        Generate packets using Poisson arrival process
        
        Used for: eMBB, mMTC
        
        Poisson Process:
        - Number of arrivals in [t, t+Δt] ~ Poisson(λ·Δt)
        - Inter-arrival times ~ Exponential(λ)
        
        Args:
            user_id: User ID
            service_type: 'embb' or 'mmtc'
            current_time: Current simulation time (seconds)
            time_window: Time window to generate packets for (seconds)
        
        Returns:
            List of generated packets
        """
        params = self.traffic_params[service_type]
        arrival_rate = params['arrival_rate']
        
        packets = []
        
        # Expected number of arrivals in this time window
        lambda_t = arrival_rate * time_window
        
        # Generate number of packets (Poisson distribution)
        num_packets = np.random.poisson(lambda_t)
        
        # For eMBB: handle bursts
        if service_type == 'embb' and np.random.random() < params['burst_probability']:
            # Burst: generate 3-5x more packets
            num_packets = int(num_packets * np.random.uniform(3, 5))
        
        # Generate arrival times uniformly within time window
        if num_packets > 0:
            arrival_times = np.sort(
                np.random.uniform(current_time, current_time + time_window, num_packets)
            )
            
            for arrival_time in arrival_times:
                packet = Packet(
                    packet_id=self.packet_id_counter,
                    user_id=user_id,
                    service_type=service_type,
                    size_bytes=self.generate_packet_size(service_type),
                    arrival_time=arrival_time,
                    deadline=arrival_time + params['latency_requirement'],
                    priority=params['priority']
                )
                packets.append(packet)
                self.packet_id_counter += 1
        
        return packets
    
    
    def generate_packets_periodic(self,
                                   user_id: int,
                                   current_time: float,
                                   time_window: float) -> List[Packet]:
        """
        Generate periodic packets for URLLC
        
        URLLC traffic is deterministic (factory automation, remote surgery)
        - Packets arrive every T seconds (period)
        - Small jitter (±0.1 ms)
        
        Args:
            user_id: User ID
            current_time: Current simulation time (seconds)
            time_window: Time window to generate packets for (seconds)
        
        Returns:
            List of generated packets
        """
        params = self.traffic_params['urllc']
        period = params['period']
        jitter = params['jitter']
        
        packets = []
        
        # Get last URLLC packet time for this user
        if user_id not in self.user_last_urllc_time:
            # First packet: schedule at current_time
            self.user_last_urllc_time[user_id] = current_time - period
        
        last_time = self.user_last_urllc_time[user_id]
        
        # Generate packets at period intervals within time window
        next_time = last_time + period
        
        while next_time <= current_time + time_window:
            # Add small jitter
            arrival_time = next_time + np.random.uniform(-jitter, jitter)
            
            packet = Packet(
                packet_id=self.packet_id_counter,
                user_id=user_id,
                service_type='urllc',
                size_bytes=self.generate_packet_size('urllc'),
                arrival_time=arrival_time,
                deadline=arrival_time + params['latency_requirement'],
                priority=params['priority']
            )
            packets.append(packet)
            self.packet_id_counter += 1
            
            next_time += period
        
        # Update last time
        if packets:
            self.user_last_urllc_time[user_id] = packets[-1].arrival_time
        
        return packets
    
    
    def generate_traffic_for_user(self,
                                   user_id: int,
                                   service_type: str,
                                   current_time: float,
                                   time_window: float = None) -> List[Packet]:
        """
        Generate traffic for a specific user
        
        Args:
            user_id: User ID
            service_type: 'embb', 'urllc', or 'mmtc'
            current_time: Current simulation time (seconds)
            time_window: Time window to generate for (default: 1 slot)
        
        Returns:
            List of generated packets
        """
        if time_window is None:
            time_window = self.slot_duration
        
        if service_type == 'urllc':
            return self.generate_packets_periodic(user_id, current_time, time_window)
        else:
            return self.generate_packets_poisson(
                user_id, service_type, current_time, time_window
            )
    
    
    def add_packets_to_queue(self, user_id: int, packets: List[Packet]):
        """Add packets to user's queue"""
        if user_id not in self.user_queues:
            self.user_queues[user_id] = deque()
        
        for packet in packets:
            self.user_queues[user_id].append(packet)
    
    
    def get_queue_size(self, user_id: int) -> int:
        """Get number of packets in user's queue"""
        if user_id not in self.user_queues:
            return 0
        return len(self.user_queues[user_id])
    
    
    def peek_next_packet(self, user_id: int) -> Packet:
        """Get next packet without removing it"""
        if user_id not in self.user_queues or not self.user_queues[user_id]:
            return None
        return self.user_queues[user_id][0]
    
    
    def pop_packet(self, user_id: int) -> Packet:
        """Remove and return next packet from queue"""
        if user_id not in self.user_queues or not self.user_queues[user_id]:
            return None
        return self.user_queues[user_id].popleft()
    
    
    def drop_expired_packets(self, user_id: int, current_time: float) -> int:
        """
        Drop packets that missed their deadline
        
        Returns:
            Number of dropped packets
        """
        if user_id not in self.user_queues:
            return 0
        
        queue = self.user_queues[user_id]
        dropped_count = 0
        
        # Remove expired packets from front of queue
        while queue and queue[0].is_expired(current_time):
            packet = queue.popleft()
            packet.dropped = True
            dropped_count += 1
        
        return dropped_count


def test_traffic_generator():
    """Test traffic generation for all service types"""
    print("\n" + "="*60)
    print("DAY 7 TEST: Traffic Generator")
    print("="*60)
    
    tg = TrafficGenerator(slot_duration=0.5e-3)
    
    # Test 1: eMBB Traffic (Poisson)
    print("\n[Test 1] eMBB Traffic (Video Streaming)")
    print("-" * 60)
    
    user_id_embb = 1
    current_time = 0.0
    time_window = 1.0  # 1 second
    
    embb_packets = tg.generate_traffic_for_user(
        user_id_embb, 'embb', current_time, time_window
    )
    
    print(f"  Time Window: {time_window} seconds")
    print(f"  Packets Generated: {len(embb_packets)}")
    
    if embb_packets:
        avg_size = np.mean([p.size_bytes for p in embb_packets])
        print(f"  Average Packet Size: {avg_size/1000:.1f} KB")
        print(f"  Total Data: {sum(p.size_bytes for p in embb_packets)/1e6:.2f} MB")
        print(f"  First Packet: {embb_packets[0].size_bytes/1000:.1f} KB, "
              f"Deadline: {embb_packets[0].deadline*1000:.1f} ms")
    
    assert len(embb_packets) > 0, "eMBB should generate packets"
    assert all(p.priority == 2 for p in embb_packets), "eMBB priority should be 2"
    
    # Test 2: URLLC Traffic (Periodic)
    print("\n[Test 2] URLLC Traffic (Critical Control)")
    print("-" * 60)
    
    user_id_urllc = 2
    urllc_packets = tg.generate_traffic_for_user(
        user_id_urllc, 'urllc', current_time, time_window
    )
    
    print(f"  Time Window: {time_window} seconds")
    print(f"  Packets Generated: {len(urllc_packets)}")
    print(f"  Expected: ~{int(time_window / 0.001)} packets (1 ms period)")
    
    if len(urllc_packets) > 1:
        periods = [urllc_packets[i+1].arrival_time - urllc_packets[i].arrival_time 
                   for i in range(len(urllc_packets)-1)]
        avg_period = np.mean(periods)
        print(f"  Average Period: {avg_period*1000:.3f} ms (expected: 1.0 ms)")
        print(f"  Period Std Dev: {np.std(periods)*1000:.3f} ms")
    
    assert len(urllc_packets) > 0, "URLLC should generate packets"
    assert all(p.priority == 1 for p in urllc_packets), "URLLC priority should be 1"
    assert all(p.size_bytes < 1000 for p in urllc_packets), "URLLC packets should be small"
    
    # Test 3: mMTC Traffic (Poisson, low rate)
    print("\n[Test 3] mMTC Traffic (IoT Sensors)")
    print("-" * 60)
    
    user_id_mmtc = 3
    mmtc_packets = tg.generate_traffic_for_user(
        user_id_mmtc, 'mmtc', current_time, time_window
    )
    
    print(f"  Time Window: {time_window} seconds")
    print(f"  Packets Generated: {len(mmtc_packets)}")
    print(f"  Expected: ~0.1 packets/second (rare)")
    
    if mmtc_packets:
        avg_size = np.mean([p.size_bytes for p in mmtc_packets])
        print(f"  Average Packet Size: {avg_size:.1f} bytes")
    
    assert all(p.priority == 3 for p in mmtc_packets), "mMTC priority should be 3"
    
    # Test 4: Queue Management
    print("\n[Test 4] Queue Management")
    print("-" * 60)

    # Generate more packets to ensure we have at least 3
    test_packets = tg.generate_traffic_for_user(user_id_embb, 'embb', current_time, time_window=5.0)
    packets_to_add = test_packets[:min(3, len(test_packets))]

    if len(packets_to_add) < 3:
        # Force generate 3 packets for testing
        packets_to_add = []
        for i in range(3):
            packet = Packet(
                packet_id=tg.packet_id_counter,
                user_id=user_id_embb,
                service_type='embb',
                size_bytes=tg.generate_packet_size('embb'),
                arrival_time=current_time,
                deadline=current_time + 0.050,
                priority=2
            )
            packets_to_add.append(packet)
            tg.packet_id_counter += 1

    tg.add_packets_to_queue(user_id_embb, packets_to_add)
    queue_size = tg.get_queue_size(user_id_embb)
    print(f"  Added {len(packets_to_add)} packets to queue")
    print(f"  Queue Size: {queue_size}")
    assert queue_size == len(packets_to_add), f"Queue should have {len(packets_to_add)} packets"
        
    # Test 5: Deadline Expiry
    print("\n[Test 5] Deadline Expiry")
    print("-" * 60)
    
    # Create packet with short deadline
    test_packet = Packet(
        packet_id=9999,
        user_id=99,
        service_type='urllc',
        size_bytes=200,
        arrival_time=0.0,
        deadline=0.001,  # 1 ms deadline
        priority=1
    )
    
    tg.add_packets_to_queue(99, [test_packet])
    
    # Check at different times
    print(f"  Packet Deadline: {test_packet.deadline*1000:.1f} ms")
    print(f"  Expired at t=0.0005s? {test_packet.is_expired(0.0005)}")
    print(f"  Expired at t=0.0015s? {test_packet.is_expired(0.0015)}")
    
    dropped = tg.drop_expired_packets(99, 0.0015)
    print(f"  Dropped Packets: {dropped}")
    assert dropped == 1, "Should drop 1 expired packet"
    
    print("\n" + "="*60)
    print("✓✓✓ DAY 7 COMPLETE: Traffic Generator Working! ✓✓✓")
    print("="*60)
    print("\nTraffic Generator Features:")
    print("  ✓ eMBB: Poisson arrivals with bursts")
    print("  ✓ URLLC: Periodic packets with jitter")
    print("  ✓ mMTC: Low-rate Poisson arrivals")
    print("  ✓ Packet queues per user")
    print("  ✓ Deadline tracking and expiry")
    print("\n→ Ready for Day 8: Traffic Statistics & Patterns")
    print("="*60)


if __name__ == "__main__":
    test_traffic_generator()