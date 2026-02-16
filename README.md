# Split DNN Joint Optimization System for Accuracy and Communication

## Project Overview

This project implements a joint optimization system for accuracy and communication in Split DNN scenarios, based on the HyperRepVGG architecture. It achieves efficient edge-cloud collaborative inference deployment through progressive pruning and throughput optimization.

### Key Features

- **Modular Pipeline Design**: 5 independent steps that can run separately or in batch
- **Dual-Mode Support**: Pseudo mode (local simulation) and Real mode (actual Server-Client communication)
- **Accuracy Optimization**: Progressive pruning based on HyperRepVGG, maintaining accuracy while reducing computation
- **Throughput Optimization**: LUT-based γ/λ gradient descent optimization to maximize end-to-end throughput
- **Multi-Domain Support**: Supports 10 different domains with Non-IID data distribution

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Modular Optimization Pipeline                │
├─────────────────────────────────────────────────────────────┤
│  Step 1: LUT Measurement                                      │
│    ├─ Server: Measure latency of all blocks in 14 γ configs  │
│    ├─ Client: Measure latency of all blocks in 14 γ configs  │
│    └─ Output: hardware_lut_complete.json                      │
├─────────────────────────────────────────────────────────────┤
│  Step 2: Initial Throughput Optimization                      │
│    ├─ LUT-based gradient descent optimization                │
│    ├─ Objective: Maximize throughput (compute + comm)        │
│    └─ Output: initial_throughput_config.json                  │
├─────────────────────────────────────────────────────────────┤
│  Step 3 (Joint): Accuracy-Throughput Joint Optimization│
│    ├─ Goal-programming based joint optimization (Eq. 7)      │
│    ├─ Optimize: α (pruning) + γ (branch) + λ (split point)  │
│    ├─ Roofline enhancement: Intensity matching (Eq. 14)      │
│    ├─ Accuracy constraint: A(u) ≥ A_ref - ε - ξ_A            │
│    └─ Output: domain_X_joint.json (α, γ, λ, accuracy)        │
├─────────────────────────────────────────────────────────────┤
│  Step 4: Results Visualization                                │
│    ├─ Generate accuracy-throughput curves                    │
│    ├─ Compare performance across domains                     │
│    └─ Output: Visualization charts (PDF/PNG)                 │
└─────────────────────────────────────────────────────────────┘
```

## Preprocessing Steps (Optional)

Before running the main pipeline, you can optionally run the following preprocessing steps to generate custom hyper-weights and importance rankings:

### Code 1: Train and Record Hyper-Weights

**Function**: Train HyperRepVGG hyper-weights for each domain and record optimal weight configurations.

```bash
python code1_train_and_record_weights.py
```

**Outputs**:
- `task1_trained_weights_{MODEL_NAME}/trained_hyper_weights.json`: Trained hyper-weights
- `task1_trained_weights_{MODEL_NAME}/training_results.json`: Training history and accuracy results
- `task1_trained_weights_{MODEL_NAME}/hyper_weights_heatmap.png`: Weight heatmap

**Note**: This step is optional. Pre-computed weight files are provided. Only run when retraining or using a different model.

### Code 2: Calculate FLOPs and Importance Rankings

**Function**: Calculate FLOPs for each hyper-branch and compute importance rankings based on weight/FLOPs ratio to guide pruning strategy.

```bash
python code2_calculate_flops_importance.py
```

**Outputs**:
- `task2_importance_rankings_{MODEL_NAME}/importance_rankings.json`: Importance rankings (low to high)
- `task2_importance_rankings_{MODEL_NAME}/importance_scores.json`: Importance scores
- `task2_importance_rankings_{MODEL_NAME}/branch_flops.json`: Branch FLOPs statistics
- `task2_importance_rankings_{MODEL_NAME}/importance_heatmap.png`: Importance heatmap

**Note**: This step is optional. Pre-computed importance rankings are provided. Importance formula: `importance = |weight| / (FLOPs / 1e9)`. Lower scores indicate branches more likely to be pruned.

---

## Quick Start

### Requirements

- Python 3.8+
- PyTorch 1.10+
- CUDA 11.0+ (recommended)
- Other dependencies in `requirements.txt`

### Installation

```bash
pip install -r requirements.txt
```

### Configuration

Edit `config.py` to modify key parameters:

```python
# Model configuration
MODEL_NAME = 'RepVGG-B3'
WEIGHT_PATH_SERVER = '/path/to/RepVGG-B3-200epochs-train.pth'

# Network configuration
SERVER_IP = '127.0.0.1'
PORT_MAIN = 6006

# Optimization parameters
MAX_PRUNING_ROUNDS = 10
THROUGHPUT_OPTIM_STEPS = 800
```

## Running Modes

### Mode 1: Complete Pipeline - Pseudo Mode (Recommended)

Run the complete pipeline using Pseudo mode without real Client hardware:

```bash
# Run all steps (Domains 0-9)
bash run_pipeline.sh

# Or run manually step by step
python step1_measure_lut.py --mode merge
python step2_initial_throughput_opt.py
for i in {0..9}; do
    python step_3_joint.py --domain_id $i --mode pseudo
done
python step5_visualize_results.py
```

### Mode 2: Real Hardware Validation - Real Mode

For final validation and real hardware testing, requires Server-Client communication:

#### Step 1: Establish SSH Tunnel

Execute from Jetson TX2 or relay machine:

```bash
ssh -CNg \
    -L 6006:127.0.0.1:6006 \
    -R 6008:127.0.0.1:6008 \
    root@your-server.com -p PORT
```

#### Step 2: Start Client Worker (on Jetson TX2)

```bash
python client_worker.py --server_ip 127.0.0.1 --port 6006
```

#### Step 3: Start Joint Optimization (on Server)

```bash
# Run Step 3 in Real mode
python step_3_joint.py --domain_id 0 --mode real
```

### Mode 3: Individual Step Execution

Each step can run independently:

```bash
# Step 1: LUT Measurement
python step1_measure_lut.py --mode server  # Server-side measurement
python step1_measure_lut.py --mode client  # Client-side measurement (on Jetson)
python step1_measure_lut.py --mode merge   # Merge results

# Step 2: Initial Throughput Optimization
python step2_initial_throughput_opt.py --lut_file results/hardware_lut_complete.json

# Step 3 (Joint): Accuracy-Throughput Joint Optimization
python step_3_joint.py --domain_id 0 --mode pseudo  # or --mode real

# Step 4: Visualization
python step5_visualize_results.py
```

## File Structure

```
.
├── config.py                          # Global configuration
├── requirements.txt                   # Python dependencies
├── run_pipeline.sh                    # Complete pipeline script
│
├── Preprocessing Scripts 
│   ├── code1_train_and_record_weights.py  # Train hyper-weights
│   └── code2_calculate_flops_importance.py # Calculate importance rankings
│
├── Core Model Files
│   ├── hyper_repvgg.py               # HyperRepVGG model definition
│   ├── repvgg.py                     # RepVGG base model
│   └── se_block.py                   # SE attention module
│
├── Pipeline Steps
│   ├── step1_measure_lut.py          # Step 1: LUT measurement
│   ├── step2_initial_throughput_opt.py  # Step 2: Initial throughput optimization
│   ├── step_3_joint.py               # Step 3: Joint optimization 
│   └── step5_visualize_results.py    # Step 4: Results visualization
│
├── Legacy Pipeline 
│   ├── step3_accuracy_opt.py         # (Legacy) Accuracy optimization
│   ├── step4_final_throughput_opt.py # (Legacy) Throughput optimization
│   └── server_main.py                # (Legacy) Server main program
│
├── Optimization Modules
│   ├── progressive_pruning.py        # Progressive pruning logic
│   ├── throughput_optimizer.py       # Throughput optimizer
│   ├── throughput_optimizer_limit.py # Constrained throughput optimizer
│   ├── gamma_configurator.py         # γ configurator
│   └── gamma_mask_builder.py         # γ mask builder
│
├── Network Communication
│   ├── network_utils.py              # Network communication utilities
│   ├── split_trainer.py              # Server-Client collaborative training
│   ├── server_main.py                # Server main program (legacy)
│   ├── server_main_throughput.py     # Server throughput optimization version
│   └── client_worker.py              # Client Worker program
│
├── Data Processing
│   ├── domain_dataset.py             # Domain dataset loader
│   ├── domain_transforms.py          # Data augmentation
│   └── create_noniid_partition.py    # Non-IID data partitioning
│
├── Performance Measurement
│   ├── lut_manager.py                # LUT measurement and management
│   └── roofline_measure.py           # Roofline model measurement
│
├── Pre-computed Data (Generated by Code 1 and Code 2)
│   ├── task1_trained_weights_RepVGG-B3/
│   │   ├── trained_hyper_weights.json    # Pre-trained hyper-weights (Code 1 output)
│   │   ├── training_results.json         # Training results (Code 1 output)
│   │   └── hyper_weights_heatmap.png     # Weight heatmap (Code 1 output)
│   │
│   ├── task2_importance_rankings_RepVGG-B3/
│   │   ├── importance_rankings.json      # Branch importance rankings (Code 2 output)
│   │   ├── importance_scores.json        # Importance scores (Code 2 output)
│   │   ├── branch_flops.json             # Branch FLOPs statistics (Code 2 output)
│   │   └── importance_heatmap.png        # Importance heatmap (Code 2 output)
│   │
│   └── results/
│       ├── hardware_lut_complete.json    # Complete hardware LUT
│       └── initial_throughput_config.json # Initial throughput config
│
└── Output Results (Generated after execution)
    ├── joint_optimization/               # Joint optimization results
    │   ├── domain_0_joint.json
    │   ├── domain_1_joint.json
    │   └── ...
    └── visualization/                    # Visualization outputs
        ├── accuracy_throughput_curve.pdf
        └── domain_comparison.png
```
