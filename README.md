### Intelligent Conflict Resolution in 5G NR Network Slicing  Using Multi-Agent Deep Reinforcement Learning

## Project Overview

This project implements a comprehensive 5G network slicing simulator with Proximal Policy Optimization (PPO) for intelligent resource allocation across eMBB, URLLC, and mMTC service types.

### Key Features

- **5G FR1 Operation**: 3.5 GHz carrier frequency (3GPP band n78)
- **Correct Numerology**: 30 kHz subcarrier spacing (μ=1)
- **Realistic Parameters**: All parameters justified from 3GPP specifications
- **Three Service Types**: eMBB, URLLC, mMTC with distinct QoS requirements
- **Hexagonal BS Deployment**: Industry-standard base station placement
- **Mobility Models**: Random Waypoint, Random Direction, Manhattan grid
- **Physics-Based Channel Models**: Hata-COST231 path loss + Rayleigh fading

---

## Project Status Dashboard

### ✅ Phase 1: Environment Setup (COMPLETED)

- [x] NetworkEnvironment class with correct 5G parameters
- [x] BaseStation class with resource management
- [x] User class with service differentiation
- [x] Hexagonal BS deployment (1, 3, 7, 19 cell configurations)
- [x] Mobility models (Random Waypoint, Random Direction, Manhattan)
- [x] QoS requirements from 3GPP TS 22.261
- [x] Comprehensive validation test suite
- [x] Network visualization and topology
- [x] Service-aware user generation

### 🔄 Phase 2: Channel Modeling (IN PROGRESS)

- [ ] Hata-COST231 path loss model
- [ ] Distance-dependent signal degradation
- [ ] Rayleigh fading generator
- [ ] Log-normal shadowing
- [ ] SINR calculation per resource block
- [ ] Channel statistics validation
- [ ] Interference modeling (co-channel)

### ⏳ Phase 3: Traffic & QoS Metrics (UPCOMING)

- [ ] Traffic generator (Poisson arrivals)
- [ ] Periodic traffic patterns
- [ ] Bursty traffic modeling
- [ ] Shannon capacity calculation
- [ ] Throughput metrics computation
- [ ] Latency tracking per service
- [ ] QoS compliance verification
- [ ] Network KPI aggregation

### ⏳ Phase 4: PPO Agent Implementation (UPCOMING)

- [ ] PPO policy network architecture
- [ ] Value network for baseline estimation
- [ ] Experience replay buffer
- [ ] Advantage and return calculation
- [ ] Policy gradient updates
- [ ] Hyperparameter tuning
- [ ] Training loop implementation

### ⏳ Phase 5: Action Masking & Optimization (UPCOMING)

- [ ] Valid action filtering per user
- [ ] Resource constraint enforcement
- [ ] Action elimination strategies
- [ ] Distributed resource allocation
- [ ] Convergence analysis

### ⏳ Phase 6: Evaluation & Comparison (UPCOMING)

- [ ] Baseline algorithm comparison (Random, Greedy, Rule-based)
- [ ] Performance metrics benchmarking
- [ ] Visualization of results
- [ ] Statistical significance testing

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Setup

```bash
# Clone repository
git clone <your-repo-url>
cd 5g-network-slicing

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements File

Your `requirements.txt` should include:

```
numpy>=1.21.0
matplotlib>=3.4.0
pyyaml>=5.4.0
scipy>=1.7.0
torch>=1.9.0
tensorboard>=2.6.0
```

---

## Usage

### Verify Environment Setup

```bash
python main.py
```

This will:
1. Initialize the 5G network environment
2. Run comprehensive validation tests
3. Display network statistics and coverage analysis
4. Visualize network topology (if enabled)
5. Print environment readiness confirmation

### Custom Configuration

Edit `config.yaml` to customize:

```yaml
# Network Configuration
network:
  num_base_stations: 7
  num_users: 50
  carrier_frequency: 3.5e9
  num_rbs: 273
  coverage_area: [1000, 1000]
  
  # Service Distribution
  service_distribution:
    eMBB: 0.5    # 50% eMBB users
    URLLC: 0.3   # 30% URLLC users
    mMTC: 0.2    # 20% mMTC users

# Channel Model Configuration (Phase 2)
channel:
  enabled: true
  model_type: "hata_cost231"  # upcoming
  fading_type: "rayleigh"
  shadowing_std: 8.0
  
# Traffic Generation (Phase 3)
traffic:
  enabled: false  # will enable in Phase 3
  model_type: "poisson"
  arrival_rate: 5.0
  
# Visualization
visualization:
  enabled: true
  plot_network_topology: true
  plot_coverage_map: true
```

### Run Single Simulation Loop

```python
from network_environment import NetworkEnvironment
import numpy as np

# Initialize environment
env = NetworkEnvironment('config.yaml')
state = env.reset()

# Simulation parameters
num_steps = 100
actions_list = []

# Run simulation
for step in range(num_steps):
    # Placeholder: Random action (will be PPO agent in Phase 4)
    action = np.random.randint(0, 2, size=(env.num_users, env.num_rbs))
    actions_list.append(action)
    
    # Execute step
    next_state, reward, done, info = env.step(action)
    
    # Log metrics (will be enhanced in Phase 3)
    print(f"Step {step}: Reward = {reward:.4f}")
    
    if done:
        break

# Get final statistics
stats = env.get_statistics()
print("\n=== Simulation Summary ===")
print(f"Total Steps: {step + 1}")
print(f"Average Reward: {np.mean([r for r in stats['rewards']]):.4f}")

env.close()
```

---

## Project Structure

```
5g-network-slicing/
├── config.yaml                      # Configuration file
├── requirements.txt                 # Python dependencies
├── main.py                          # Main entry point
│
├── Core Components/
│   ├── network_environment.py       # ✅ Core environment class
│   ├── base_station.py              # ✅ Base station implementation
│   ├── user.py                      # ✅ User equipment implementation
│   │
│   ├── channel_model.py             # 🔄 Phase 2: Channel models
│   ├── traffic_generator.py         # ⏳ Phase 3: Traffic generation
│   ├── qos_metrics.py               # ⏳ Phase 3: QoS calculations
│   │
│   └── utils/
│       ├── config_parser.py         # Configuration utilities
│       ├── constants.py             # 3GPP constants and parameters
│       └── helpers.py               # Helper functions
│
├── Agents/ (Phase 4-5)
│   ├── ppo_agent.py                 # ⏳ PPO agent implementation
│   ├── networks.py                  # ⏳ Policy and value networks
│   ├── memory.py                    # ⏳ Experience buffer
│   └── action_eliminator.py         # ⏳ Action masking
│
├── Training/ (Phase 4)
│   ├── trainer.py                   # ⏳ Main training loop
│   ├── callbacks.py                 # ⏳ Training callbacks
│   └── logger.py                    # ⏳ Metrics logging
│
├── Baselines/ (Phase 6)
│   ├── random_agent.py              # ⏳ Random allocation
│   ├── greedy_agent.py              # ⏳ Greedy algorithm
│   └── rule_based_agent.py          # ⏳ QoS-based heuristics
│
├── Tests/
│   ├── test_environment.py          # ✅ Environment tests
│   ├── test_channel.py              # 🔄 Phase 2 tests
│   ├── test_qos.py                  # ⏳ Phase 3 tests
│   ├── test_agent.py                # ⏳ Phase 4 tests
│   └── test_integration.py          # ⏳ Full system tests
│
├── Notebooks/ (Analysis & Documentation)
│   ├── 01_environment_analysis.ipynb        # ✅ Completed
│   ├── 02_channel_validation.ipynb          # 🔄 Phase 2
│   ├── 03_traffic_patterns.ipynb            # ⏳ Phase 3
│   ├── 04_training_curves.ipynb             # ⏳ Phase 4
│   └── 05_results_comparison.ipynb          # ⏳ Phase 6
│
├── Results/ (Output folder, auto-created)
│   ├── models/                      # Trained PPO models
│   ├── logs/                        # TensorBoard logs
│   └── plots/                       # Generated visualizations
│
└── README.md                        # This file
```

---

## Technical Specifications

### Network Parameters

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Carrier Frequency | 3.5 GHz | Most common 5G FR1 band (n78) |
| Frequency Range | FR1 | Sub-6 GHz (450 MHz - 6 GHz) |
| Subcarrier Spacing | 30 kHz | Standard for FR1 (μ=1) |
| RB Bandwidth | 360 kHz | 12 subcarriers × 30 kHz |
| System Bandwidth | 100 MHz | Typical FR1 allocation (273 RBs) |
| Slot Duration | 0.5 ms | Enables 1ms URLLC latency |
| BS Transmit Power | 46 dBm | Typical macro cell (43-49 dBm) |
| BS Antenna Height | 25 m | Standard macro cell height |

### Service Type Requirements (3GPP TS 22.261)

#### eMBB: Enhanced Mobile Broadband
- **Target Throughput**: 50-100 Mbps
- **Max Latency**: 50 ms
- **Reliability**: 95%
- **Use Cases**: 4K video, web browsing, social media
- **Traffic Priority**: Medium

#### URLLC: Ultra-Reliable Low-Latency Communications
- **Min Throughput**: 10 Mbps
- **Max Latency**: 1 ms ⚠️ **CRITICAL**
- **Reliability**: 99.999% (five nines)
- **Use Cases**: Industrial automation, autonomous vehicles, remote surgery
- **Traffic Priority**: **HIGHEST**

#### mMTC: Massive Machine-Type Communications
- **Min Throughput**: 100 kbps
- **Max Latency**: 10 seconds
- **Reliability**: 90%
- **Use Cases**: IoT sensors, smart meters, environmental monitoring
- **Traffic Priority**: Lowest

---

## Validation & Testing

### Phase 1 Validation (✅ COMPLETED)

#### Test Suite Included:

1. **Basic Connectivity**
   - All users within coverage range (<1000m)
   - Each user connects to at least one BS

2. **Coverage Overlap**
   - Users see multiple BSs for handover support
   - Average 2+ BSs visible per user

3. **Resource Allocation**
   - Total RBs = num_bs × num_rbs_per_bs
   - RB bandwidth verified as 360 kHz

4. **QoS Configuration**
   - Service types configured from 3GPP specs
   - Priority levels assigned correctly

5. **Time Parameters**
   - Slot duration enables URLLC
   - Numerology correctly configured (30 kHz)

6. **Deployment Validation**
   - Hexagonal layout verified
   - Inter-site distance calculated
   - Interference pattern validated

---

## Next Steps: Phase 2 - Channel Modeling

### Implementation Timeline

**Days 1-2: Path Loss Model**
- Implement Hata-COST231 model
- Distance-dependent attenuation
- Frequency correction factors
- Validation against empirical data

**Days 3-4: Fading & Shadowing**
- Rayleigh fading generator
- Log-normal shadowing implementation
- Spatial correlation modeling
- SINR calculation per RB

**Days 5-6: Validation & Integration**
- Channel statistics validation
- Comparison with theoretical models
- Integration with environment
- Performance profiling

### Key Files to Create/Modify

```python
# channel_model.py - Main implementation
class PathLossModel:
    def hata_cost231(self, distance, frequency)
    
class FadingChannel:
    def rayleigh_fading(self, num_samples, power)
    def log_normal_shadowing(self, distance, std_dev)
    
class ChannelModel:
    def compute_sinr(self, tx_power, path_loss, fading, interference)
```

---

## Critical Design Decisions

### Why 30 kHz SCS (not 15 kHz)?

**Answer**: URLLC requires 1ms latency.

**With 15 kHz SCS**:
- Slot duration = 1 ms
- Only 1 slot available → NOT FEASIBLE

**With 30 kHz SCS**:
- Slot duration = 0.5 ms
- 2 slots available within 1ms budget → **ACHIEVABLE** ✅

### Why FR1 (not FR2 mmWave)?

**Advantages**:
- Coverage: 1-3 km (vs 100-200m for FR2)
- Penetration: Walls, rain attenuation manageable
- Simplicity: No beamforming simulation needed
- Realistic: ~60% of 5G networks use FR1

**FR1 at 3.5 GHz is the correct choice for this research project.**

### Why Hexagonal Deployment?

**Justification**:
- Industry standard (proven optimal by cellular theory)
- Realistic interference patterns
- Reproducible results
- Random placement creates unrealistic scenarios

---

## Troubleshooting

### Issue: Import errors

```bash
# Verify virtual environment
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

### Issue: Tests fail - users too far from BS

**Solution**: Adjust coverage in `config.yaml`

```yaml
network:
  coverage_area: [800, 800]  # Reduce from 1000x1000
  # OR increase BS count
  num_base_stations: 19
```

### Issue: URLLC timing validation fails

**Check configuration**:
```yaml
network:
  subcarrier_spacing: 30e3  # Must be 30 kHz, not 15 kHz
  numerology: 1
```

### Issue: Visualization doesn't display

**Enable and configure**:
```yaml
visualization:
  enabled: true
  plot_network_topology: true
  save_figures: true
  output_dir: "./results"
```

---

## Performance Expectations

### Environment Initialization
- Should complete in < 1 second
- Validation suite: < 5 seconds

### Single Simulation Step
- Expected time: 10-50 ms (depends on num_users × num_rbs)
- Full 100-step episode: 1-5 seconds

### Memory Usage
- Base environment: ~50-100 MB
- With 1000 users: ~500 MB
- Recommendation: Use num_users ≤ 100 for testing

---

## References

### 3GPP Standards
- **TS 22.261**: Service requirements for 5G
- **TS 38.211**: Physical channels and modulation
- **TS 38.214**: Physical layer procedures
- **TR 38.901**: Channel models for NR

### Research Papers
- Schulman et al. (2017): "Proximal Policy Optimization Algorithms"
- [Add your core reference paper here]

### Resources
- 3GPP Specifications: https://www.3gpp.org/
- OpenAI Spinning Up: https://spinningup.openai.com/
- PyTorch Docs: https://pytorch.org/docs/

---

## Contributing & Support

For issues or questions:

1. Check troubleshooting section above
2. Review validation test output
3. Verify config parameters against 3GPP specs
4. Check `tests/` directory for relevant test cases

---

## License

[Add your license - MIT, Apache 2.0, etc.]

---

## Changelog

### Phase 1: Environment Setup ✅
- ✅ Core environment with 5G FR1 parameters
- ✅ Base station and user equipment classes
- ✅ Hexagonal deployment system
- ✅ Comprehensive validation suite
- ✅ Network visualization tools
- ✅ Service-differentiated user generation
- ✅ QoS requirement implementation

### Phase 2: Channel Modeling 🔄 (NEXT)
- 🔄 Hata-COST231 path loss implementation
- 🔄 Rayleigh fading generator
- 🔄 Log-normal shadowing
- 🔄 SINR calculation
- 🔄 Channel statistics validation

### Phase 3: Traffic & QoS (UPCOMING)
- ⏳ Traffic generation (Poisson, periodic, bursty)
- ⏳ Shannon capacity calculation
- ⏳ QoS metrics computation
- ⏳ Network KPI aggregation

### Phase 4: PPO Agent (UPCOMING)
- ⏳ Policy and value networks
- ⏳ PPO training algorithm
- ⏳ Experience replay buffer
- ⏳ Training loop with monitoring

### Phase 5: Action Masking (UPCOMING)
- ⏳ Valid action filtering
- ⏳ Constraint enforcement
- ⏳ Distributed allocation strategies

### Phase 6: Evaluation (UPCOMING)
- ⏳ Baseline comparisons
- ⏳ Performance benchmarking
- ⏳ Statistical analysis
- ⏳ Results visualization

---

**Last Updated**: January 2026  
**Project Phase**: 1/6 Complete ✅  
**Next Milestone**: Phase 2 - Channel Model Implementation  
**Estimated Completion**: [Add your timeline]