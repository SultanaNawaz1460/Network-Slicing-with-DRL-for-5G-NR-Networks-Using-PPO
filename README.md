# Network Slicing with PPO for 5G/6G Networks

## Project Overview
Implementation of Proximal Policy Optimization (PPO) for intelligent resource allocation in 5G/6G network slicing.

**Author**: [Your Name]  
**Duration**: 8 Months (32 Weeks)  
**Algorithm**: Proximal Policy Optimization (PPO)  
**Domain**: 5G/6G Network Slicing  

## Current Status
🚧 **Phase 1: Foundation** (Week 1-8)
- [x] Environment setup
- [ ] Literature review
- [ ] Network simulator
- [ ] Base PPO implementation
- [ ] Action elimination

## Setup Instructions

### 1. Create Conda Environment
```bash
conda create -n network-slicing python=3.11 -y
conda activate network-slicing
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Verify Installation
```bash
python tests/test_setup.py
```

## Project Structure
```
network-slicing-ppo/
├── src/                    # Source code
│   ├── environment/        # Network simulation environment
│   ├── agents/            # PPO agent implementation
│   ├── models/            # Neural network architectures
│   ├── baselines/         # Baseline algorithms (RR, PF, etc.)
│   └── utils/             # Helper functions
├── experiments/           # Training and evaluation scripts
├── tests/                 # Unit tests
├── results/               # Experimental results
│   ├── figures/          # Plots and visualizations
│   ├── tables/           # Result tables
│   └── logs/             # Training logs
├── saved_models/         # Trained model checkpoints
└── configs/              # Configuration files
```

## Next Steps
- [ ] Complete literature review
- [ ] Implement network environment
- [ ] Build PPO agent
- [ ] Test action elimination

## References
- PPO Paper: [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- Project Roadmap: See `FYP RoadMap.pdf`

# 5G Network Slicing with Deep Reinforcement Learning (PPO)

## Project Overview

This project implements a comprehensive 5G network slicing simulator with Proximal Policy Optimization (PPO) for intelligent resource allocation across eMBB, URLLC, and mMTC service types.

### Key Features

- **5G FR1 Operation**: 3.5 GHz carrier frequency (3GPP band n78)
- **Correct Numerology**: 30 kHz subcarrier spacing (μ=1)
- **Realistic Parameters**: All parameters justified from 3GPP specifications
- **Three Service Types**: eMBB, URLLC, mMTC with distinct QoS requirements
- **Hexagonal BS Deployment**: Industry-standard base station placement
- **Mobility Models**: Random Waypoint, Random Direction, Manhattan grid
- **Physics-Based Channel Models**: (Coming in Week 4: Hata-COST231 + Rayleigh fading)

---

## Week 3-4 Deliverable Status

### ✅ Completed (Days 1-3)

- [x] NetworkEnvironment class with correct 5G parameters
- [x] BaseStation class with resource management
- [x] User class with service differentiation
- [x] Hexagonal BS deployment (1, 3, 7, 19 cell configurations)
- [x] Mobility models (Random Waypoint implemented)
- [x] QoS requirements from 3GPP TS 22.261
- [x] Comprehensive validation test suite
- [x] Network visualization

### 🔄 In Progress (Days 4-6)

- [ ] Channel model implementation (Hata-COST231 path loss)
- [ ] Rayleigh fading generator
- [ ] Shadowing and interference modeling
- [ ] Channel statistics validation

### ⏳ Upcoming (Days 7-14)

- [ ] Traffic generator (Poisson, periodic, bursty)
- [ ] Shannon capacity calculation
- [ ] QoS metrics computation
- [ ] Full system integration

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

---

## Usage

### Run Validation Tests
```bash
python main.py
```

This will:
1. Initialize the 5G network environment
2. Run comprehensive validation tests
3. Display network statistics
4. Visualize network topology (if enabled in config)

### Custom Configuration

Edit `config.yaml` to customize:
```yaml
network:
  num_base_stations: 7        # Number of base stations
  num_users: 50               # Number of users
  carrier_frequency: 3.5e9    # Carrier frequency (Hz)
  num_rbs: 273                # Number of resource blocks
  
  # Service distribution
  service_distribution:
    eMBB: 0.5    # 50% eMBB users
    URLLC: 0.3   # 30% URLLC users
    mMTC: 0.2    # 20% mMTC users
```

### Run Single Simulation
```python
from network_environment import NetworkEnvironment
import numpy as np

# Initialize environment
env = NetworkEnvironment('config.yaml')
state = env.reset()

# Run simulation
for step in range(100):
    # Random action (will be replaced by PPO agent)
    action = np.random.randint(0, 2, size=(env.num_users, env.num_rbs))
    
    next_state, reward, done, info = env.step(action)
    
    if done:
        break

# Get statistics
stats = env.get_statistics()
print(stats)

env.close()
```

---

## Project Structure
```
5g-network-slicing/
├── config.yaml                 # Configuration file
├── requirements.txt            # Python dependencies
├── main.py                     # Main entry point
├── network_environment.py      # Core environment class
├── base_station.py             # Base station implementation
├── user.py                     # User equipment implementation
├── channel_model.py            # (Week 4) Channel models
├── traffic_generator.py        # (Week 4) Traffic generation
├── agents/
│   ├── ppo_agent.py           # (Week 5-6) PPO agent
│   └── action_elimination.py  # (Week 7-8) Action masking
├── tests/
│   ├── test_environment.py
│   ├── test_channel.py
│   └── test_qos.py
├── notebooks/
│   ├── 01_environment_analysis.ipynb
│   └── 02_channel_validation.ipynb
└── README.md
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

### Service Types (3GPP TS 22.261)

#### eMBB (Enhanced Mobile Broadband)
- **Min Throughput**: 50 Mbps
- **Max Latency**: 50 ms
- **Reliability**: 95%
- **Use Cases**: 4K video streaming, web browsing, social media

#### URLLC (Ultra-Reliable Low-Latency Communications)
- **Min Throughput**: 10 Mbps
- **Max Latency**: 1 ms ⚠️ CRITICAL
- **Reliability**: 99.999% (five nines)
- **Use Cases**: Industrial automation, autonomous vehicles, remote surgery

#### mMTC (Massive Machine-Type Communications)
- **Min Throughput**: 100 kbps
- **Max Latency**: 10 seconds
- **Reliability**: 90%
- **Use Cases**: IoT sensors, smart meters, environmental monitoring

---

## Validation Tests

The system includes comprehensive validation:

### Test 1: Basic Connectivity
- ✅ All users within coverage range (<1000m)
- ✅ Each user can connect to at least one BS

### Test 2: Coverage Overlap
- ✅ Users see multiple BSs for handover
- ✅ Average 2+ BSs visible per user

### Test 3: Resource Allocation
- ✅ Total RBs = num_bs × num_rbs_per_bs
- ✅ RB bandwidth = 360 kHz (corrected)

### Test 4: QoS Requirements
- ✅ Service types configured from 3GPP specs
- ✅ Priority levels assigned correctly

### Test 5: Time Parameters
- ✅ Slot duration enables URLLC (2+ slots for 1ms)
- ✅ Numerology correctly configured

---

## Critical Design Decisions

### Why 30 kHz SCS (not 15 kHz)?

**Answer**: URLLC requires 1ms latency. With 15 kHz SCS:
- Slot duration = 1 ms
- Only 1 slot available for transmission + processing
- **NOT FEASIBLE**

With 30 kHz SCS:
- Slot duration = 0.5 ms
- 2 slots available within 1ms budget
- **ACHIEVABLE** ✅

### Why FR1 (not FR2 mmWave)?

**Answer**: FR2 offers higher bandwidth but:
- Coverage: Only 100-200m (vs 1-3km for FR1)
- Penetration: Blocked by walls, rain
- Complexity: Requires beamforming simulation
- Deployment: <5% of networks

**FR1 at 3.5 GHz is the RIGHT choice for this project.**

### Why Hexagonal BS Deployment?

**Answer**: 
- Industry standard (proven optimal by cellular theory)
- Realistic interference patterns
- Reproducible results
- Random placement creates unrealistic scenarios

---

## Known Limitations (To Be Addressed)

### Current (Week 3-4)
- ⚠️ Channel model not yet implemented (placeholder)
- ⚠️ Traffic generation simplified
- ⚠️ Shannon capacity calculation pending
- ⚠️ Reward function is placeholder

### Future Scope
- Beamforming (beyond this project)
- HARQ retransmissions (optional extension)
- Core network slicing (out of scope)
- Multi-numerology support (extension)

---

## Troubleshooting

### Issue: Import errors
```bash
# Make sure you're in the virtual environment
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue: Tests fail - users too far from BS

**Solution**: Reduce `coverage_area` or increase `num_base_stations` in `config.yaml`
```yaml
network:
  coverage_area: [800, 800]  # Reduce from 1000x1000
  # OR
  num_base_stations: 19      # Increase from 7
```

### Issue: URLLC timing test fails

**Check**: Ensure `subcarrier_spacing: 30e3` (not 15e3) in config

### Issue: Visualization doesn't show

**Solution**: Enable in config and ensure matplotlib backend works
```yaml
visualization:
  enabled: true
  plot_network_topology: true
```

---

## Next Steps (Week 4: Days 4-6)

### Channel Model Implementation

We will implement:

1. **Path Loss Model (Hata-COST231)**
   - Urban/Suburban/Rural environments
   - Distance-dependent signal degradation
   - Frequency-dependent losses

2. **Rayleigh Fading**
   - Small-scale fading (multipath)
   - Exponential distribution
   - Time-varying channel

3. **Log-Normal Shadowing**
   - Slow fading from obstacles
   - 8 dB standard deviation
   - Spatially correlated

4. **Interference Modeling**
   - Co-channel interference from neighboring cells
   - SINR calculation per RB
   - Validation against theoretical models

---

## References

### 3GPP Specifications
- **TS 22.261**: Service requirements for 5G
- **TS 38.211**: Physical channels and modulation
- **TS 38.214**: Physical layer procedures for data
- **TR 38.901**: Channel model for frequency spectrum above 6 GHz

### Research Papers
- Schulman et al. (2017): "Proximal Policy Optimization Algorithms"
- Original project paper: [Add your reference paper here]

### Online Resources
- 3GPP Specifications: https://www.3gpp.org/specifications
- OpenAI Spinning Up: https://spinningup.openai.com/
- PyTorch Documentation: https://pytorch.org/docs/

---

## Contributing

This is a research project. For questions or issues:

1. Check troubleshooting section
2. Review validation test output
3. Verify config parameters against 3GPP specs
4. Contact: your.email@example.com

---

## License

[Add your license here - e.g., MIT, Apache 2.0]

---

## Acknowledgments

- 3GPP for comprehensive 5G specifications
- OpenAI for PPO algorithm and Spinning Up resources
- Research community for network slicing contributions

---

## Changelog

### Week 3-4 (Current)
- ✅ Initial environment setup
- ✅ Correct 5G FR1 parameters (30 kHz SCS, 360 kHz RB)
- ✅ Base station and user classes
- ✅ Hexagonal deployment
- ✅ Comprehensive validation suite
- ✅ Network visualization

### Week 4 (Upcoming)
- 🔄 Channel model implementation
- 🔄 Traffic generator
- 🔄 Shannon capacity
- 🔄 Full integration

---

**Last Updated**: November 2025  
**Project Status**: Week 3-4 Complete ✅  
**Next Milestone**: Channel Model Implementation (Week 4)