# AGNI_V2
# AGNI V2 — Reinforcement Learning–Based Compiler Optimization for XLA  

AGNI V2 is a reinforcement learning–driven compiler optimization framework designed to outperform Google’s XLA (Accelerated Linear Algebra) by learning how to generate more efficient low-level code for different hardware targets.  

Instead of relying solely on hand-engineered heuristics or static graph optimizations, AGNI V2 trains a learning agent to understand how XLA optimizes computation graphs — and to discover even better strategies in terms of **execution time**, **memory usage**, and **hardware utilization**.  

---

## Project Overview  

XLA is a domain-specific JIT compiler that optimizes TensorFlow and JAX computation graphs.  
AGNI V2 builds on this idea — but uses **reinforcement learning** to learn optimization actions dynamically.  

The system analyzes the transformations applied by XLA on linear regression (LR) and other computation graph workloads, learns a **policy** that maps input graph + hardware features → optimization decisions, and produces optimized intermediate code that reduces latency and memory footprint.  

AGNI V2 ultimately aims to generalize optimization behavior across hardware (e.g., Intel, ARM, Raspberry Pi) while retaining interpretability and control.  

---

## Key Features  

- **Reinforcement Learning Optimization Layer**  
  Learns a policy to optimize tensor computation graphs based on reward feedback from execution metrics (e.g., runtime, FLOPs, memory).  

- **Hardware-Aware Policy Network**  
  Models hardware features (CPU type, core count, cache size, etc.) as part of the state to learn hardware-specific reward shaping.  

- **Supervised Bootstrapping from XLA**  
  Starts from supervised training on data collected from XLA-optimized and non-optimized executions, then refines via RL.  

- **Graph-Level Reward System**  
  Rewards are assigned based on execution improvements compared to baseline (no-XLA, XLA).  

- **Cross-Hardware Generalization**  
  The learned agent can adapt to new target architectures (ARM, Intel, etc.) through transfer or fine-tuning.  

---

##Architecture  

```
[ Input: TensorFlow Model ]  
        ↓  
[ Extract Computation Graph + Features ]  
        ↓  
[ Feature Encoder (model type, op count, input size, hardware specs) ]  
        ↓  
[ Policy Network (RL Agent) ]  
        ↓  
[ Action: Select Optimization Transformation ]  
        ↓  
[ Execute + Measure Runtime / Memory ]  
        ↓  
[ Reward: Performance Gain vs XLA / Baseline ]  
        ↓  
[ Policy Update ]
```

AGNI V2 can use various reinforcement algorithms:  
- **PPO (Proximal Policy Optimization)**  
- **DDPG (Deep Deterministic Policy Gradient)**  
- **A3C / Advantage Actor–Critic**  

---

##  Getting Started  

### Prerequisites  
- Python 3.9+  
- TensorFlow or PyTorch  
- Required libraries:  
  ```bash
  pip install -r requirements.txt
  ```
  (Includes: `tensorflow`, `torch`, `gym`, `networkx`, `numpy`, `scikit-learn`, `matplotlib`, etc.)

### Installation  
```bash
git clone https://github.com/AishwaryaTalapuru/AGNI_V2_new.git
cd AGNI_V2_new
pip install -r requirements.txt
```

### Quick Start  
```bash
# Generate dataset from XLA and non-XLA runs
python scripts/collect_data.py --model linear_regression --hardware intel_i3

# Train the RL agent
python train_agent.py --config configs/rl_config.yaml

# Evaluate on new models
python evaluate.py --input my_model.pb --hardware raspberry_pi
```

---

##  Datasets  

The dataset used for AGNI V2 includes:
- Model types: linear regression, CNNs, RNNs  
- Features: input tensor size, op count, FLOPs, memory usage, hardware specs  
- Labels: runtime with and without XLA  

AGNI V2 uses this data to learn XLA’s transformation function — and then to outperform it.  

---

##  Reward Function  

The reward structure encourages improvements in both **execution time** and **space efficiency**:

```
R = α * (T_baseline - T_agent) / T_baseline  +  
    β * (M_baseline - M_agent) / M_baseline
```

Where:
- `T_*` = time
- `M_*` = memory
- `α`, `β` = tunable coefficients (hardware-dependent)

---

## Research Goals  

AGNI V2 explores the following research questions:  
1. Can an RL model learn optimization actions comparable or superior to XLA’s heuristics?  
2. Can optimization strategies learned on one hardware generalize to others?  
3. Can reward structures adapt dynamically to hardware constraints (e.g., memory-limited edge devices)?  

---

##  Directory Structure  

```
AGNI_V2_new/
├── data/                    # Datasets of XLA and non-XLA runs
├── models/                  # Saved RL and baseline models
├── scripts/
│   ├── collect_data.py      # Collect timing/memory data
│   ├── feature_extraction.py
│   ├── graph_encoder.py
│   ├── train_agent.py
│   └── evaluate.py
├── configs/                 # RL, training, and reward configs
├── utils/                   # Helper functions
├── README.md
└── requirements.txt
```

---

##  Experiments  

- Hardware targets: Intel Core i3-12100, ARM (Raspberry Pi 4), and x86 servers  
- Benchmarks: compare average runtime, speedup ratio, and memory use  
- Baselines: TensorFlow + XLA, TensorFlow (no XLA), AGNI V2  

---

##  Contributing  

Contributions are welcome!  
1. Fork the repo  
2. Create a feature branch (`feature/reward-scheduler`)  
3. Commit changes  
4. Submit a pull request  
 

---

*AGNI V2 — Learning to Outperform XLA.*

