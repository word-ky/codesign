import torch
import json
import os
import argparse
from datetime import datetime
from config import (
    NUM_BLOCKS, MODEL_NAME, WEIGHT_PATH_SERVER,
    THROUGHPUT_OPTIM_STEPS
)

from hyper_repvgg import create_hyper_repvgg
from lut_manager import LUTManager
from throughput_optimizer import ThroughputOptimizer, MaskManager
from progressive_pruning import get_alpha_dict

def create_initial_alpha_dict(model):
    alpha_dict = {}
    for stage_name in ['stage1', 'stage2', 'stage3', 'stage4']:
        stage = getattr(model, stage_name)
        for block_idx, block in enumerate(stage):
            if hasattr(block, 'hyper_dense'):
                block_name = f"{stage_name}.{block_idx}"
                alpha_dict[block_name] = {
                    'hyper_dense': 1.0,
                    'hyper_1x1': 1.0,
                    'hyper_identity': 1.0 if block.rbr_identity is not None else 0.0
                }
    
    return alpha_dict

def run_initial_throughput_optimization(args):
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    print("="*80)
    print("Step 2: Initial Throughput Optimization")
    print("="*80)
    print(f"Device: {device}")
    print(f"LUT file: {args.lut_file}")
    print(f"Optimization steps: {args.steps}")
    print()
    
    print("[1/5] Loading LUT...")
    if not os.path.exists(args.lut_file):
        raise FileNotFoundError(f"LUT file not found: {args.lut_file}")
    
    lut_manager = LUTManager(args.lut_file)
    print(f"✓ Loaded LUT: {args.lut_file}")
    
    print("\n[2/5] Creating model structure...")
    if os.path.exists(WEIGHT_PATH_SERVER):
        model = create_hyper_repvgg(WEIGHT_PATH_SERVER, model_name=MODEL_NAME)
    else:
        print(f"  Warning: Weight file not found at {WEIGHT_PATH_SERVER}")
        print(f"  Creating model structure without weights...")
        model = create_hyper_repvgg(None, model_name=MODEL_NAME)
    
    model = model.to(device)
    model.eval()
    print(f"✓ Model created: {MODEL_NAME}")
    
    print("\n[3/5] Creating initial alpha configuration (all 1.0)...")
    initial_alpha = create_initial_alpha_dict(model)
    print(f"✓ Created initial alpha for {len(initial_alpha)} blocks")
    
    print("\n[4/5] Setting up mask manager...")
    mask_manager = MaskManager(model, device)
    
    print(f"\n[5/5] Running optimization ({args.steps} steps)...")
    print("-" * 80)
    
    optimizer = ThroughputOptimizer(lut_manager, mask_manager, device)
    optimizer.update_mask(initial_alpha)
    history = optimizer.optimize(n_steps=args.steps, verbose=True)
    
    print("-" * 80)
    
    optimal_gamma = optimizer.get_optimal_gamma()
    optimal_lambda = optimizer.get_optimal_lambda()
    final_loss = optimizer.compute_total_loss()
    
    gamma_distribution = {}
    for gamma_idx in range(14):
        count = sum(1 for g in optimal_gamma.values() if g == gamma_idx)
        if count > 0:
            gamma_distribution[gamma_idx] = count
    
    print("\n" + "="*80)
    print("Optimization Results")
    print("="*80)
    print(f"Optimal split point (λ): {optimal_lambda}")
    print(f"Final total loss: {final_loss['total'].item():.4f}")
    print(f"\nLoss components:")
    print(f"  Roofline Client: {final_loss['roofline_client']:.4f}")
    print(f"  Roofline Server: {final_loss['roofline_server']:.4f}")
    print(f"  Time Server:     {final_loss['time_server']:.4f}")
    print(f"  Time Client:     {final_loss['time_client']:.4f}")
    print(f"  Transmission:    {final_loss['trans']:.4f}")
    
    print(f"\nGamma distribution:")
    for gamma_idx in sorted(gamma_distribution.keys()):
        from gamma_configurator import GammaConfigurator
        config_name = GammaConfigurator.get_gamma_name(gamma_idx)
        count = gamma_distribution[gamma_idx]
        print(f"  γ={gamma_idx:2d} ({config_name:12s}): {count:3d} blocks")
    
    print(f"\nSample gamma configs:")
    for i, (block_name, gamma_idx) in enumerate(list(optimal_gamma.items())[:5]):
        from gamma_configurator import GammaConfigurator
        config_name = GammaConfigurator.get_gamma_name(gamma_idx)
        print(f"  {block_name}: γ={gamma_idx} ({config_name})")
    
    output_data = {
        'metadata': {
            'step': 2,
            'description': 'Initial throughput optimization',
            'model': MODEL_NAME,
            'lut_file': args.lut_file,
            'optimization_steps': args.steps,
            'timestamp': str(datetime.now())
        },
        'initial_config': {
            'alpha': 'all 1.0 (no pruning)'
        },
        'optimal_gamma': optimal_gamma,
        'optimal_lambda': optimal_lambda,
        'final_loss': {
            'total': final_loss['total'].item(),
            'roofline_client': final_loss['roofline_client'],
            'roofline_server': final_loss['roofline_server'],
            'time_server': final_loss['time_server'],
            'time_client': final_loss['time_client'],
            'trans': final_loss['trans']
        },
        'gamma_distribution': gamma_distribution,
        'optimization_history': history
    }
    
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✓ Results saved to: {args.output}")
    print("="*80)

def main():
    parser = argparse.ArgumentParser(
        description='Step 2: Initial Throughput Optimization'
    )
    parser.add_argument(
        '--lut_file',
        type=str,
        default='results/hardware_lut_complete.json',
        help='Path to complete LUT file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/initial_throughput_config.json',
        help='Output file path'
    )
    parser.add_argument(
        '--steps',
        type=int,
        default=THROUGHPUT_OPTIM_STEPS,
        help='Number of optimization steps'
    )
    parser.add_argument(
        '--cpu',
        action='store_true',
        help='Use CPU instead of GPU'
    )
    
    args = parser.parse_args()
    
    run_initial_throughput_optimization(args)

if __name__ == '__main__':
    main()