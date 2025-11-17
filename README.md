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