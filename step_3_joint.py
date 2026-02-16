import torch
import json
import os
import argparse
from datetime import datetime
from config import (
    MODEL_NAME, WEIGHT_PATH_SERVER, NUM_DOMAINS,
    DATA_PATH, PARTITION_INFO_PATH,
    TRAINING_RESULTS_FILE, IMPORTANCE_RANKINGS_FILE, BRANCH_FLOPS_FILE,
    MAX_PRUNING_ROUNDS, CANDIDATE_POOL_SIZE, EPOCHS_PER_TEST,
    THRESHOLD_PERCENT, BATCH_SIZE, THROUGHPUT_OPTIM_STEPS
)

from hyper_repvgg import create_hyper_repvgg
from progressive_pruning import progressive_pruning
from split_trainer import LocalTrainer
from gamma_configurator import GammaConfigurator
from lut_manager import LUTManager
from throughput_optimizer import ThroughputOptimizer
from gamma_mask_builder import GammaMaskBuilder

def freeze_bn_stats(model):
    import torch.nn as nn
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
    return model

def create_domain_dataloaders(data_path, partition_info, domain_id, batch_size=BATCH_SIZE):
    try:
        from domain_dataset import create_domain_dataloaders as _create_loaders
        return _create_loaders(data_path, partition_info, domain_id, batch_size)
    except ImportError:
        print(f"  [Warning] domain_dataset not found, using simplified loader")
        import torchvision.transforms as transforms
        from torchvision.datasets import ImageFolder
        from torch.utils.data import DataLoader, Subset
        
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        dataset = ImageFolder(data_path, transform=transform)
        
        if 'domain_indices' in partition_info:
            indices = partition_info['domain_indices'].get(str(domain_id), list(range(len(dataset))))
        else:
            total = len(dataset)
            per_domain = total // NUM_DOMAINS
            start = domain_id * per_domain
            end = start + per_domain if domain_id < NUM_DOMAINS - 1 else total
            indices = list(range(start, end))
        
        n_train = int(len(indices) * 0.8)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        return train_loader, val_loader

def optimize_gamma_for_fixed_lambda(lut_manager, mask_tensor, device,
                                     fixed_lambda, n_steps, verbose=False):
    class SimplifiedMaskManager:
        def __init__(self, mask_tensor, device):
            self.device = device
            self.precomputed_mask = mask_tensor
            self.block_info = {}
        def compute_mask(self, alpha_dict):
            return {}

    mask_manager = SimplifiedMaskManager(mask_tensor, device)

    optimizer = ThroughputOptimizer(
        lut_manager,
        mask_manager,
        device=device,
        fixed_lambda=fixed_lambda
    )

    optimizer.current_mask = mask_tensor

    if verbose:
        print(f"    Optimizing gamma for lambda={fixed_lambda} ({n_steps} steps)...")

    history = optimizer.optimize(n_steps=n_steps, verbose=False)

    optimal_gamma = optimizer.get_optimal_gamma()
    final_loss = optimizer.compute_total_loss()

    gamma_distribution = {}
    for gamma_idx in range(14):
        count = sum(1 for g in optimal_gamma.values() if g == gamma_idx)
        if count > 0:
            gamma_distribution[gamma_idx] = count

    return {
        'optimal_gamma': optimal_gamma,
        'final_loss': {
            'total': final_loss['total'].item(),
            'roofline_client': final_loss['roofline_client'],
            'roofline_server': final_loss['roofline_server'],
            'time_server': final_loss['time_server'],
            'time_client': final_loss['time_client'],
            'trans': final_loss['trans']
        },
        'gamma_distribution': gamma_distribution,
        'history': history
    }

def evaluate_split_configuration(lut_manager, gamma_config, split_point):
    lut = lut_manager.lut
    block_names = list(gamma_config.keys())

    server_blocks = block_names[:split_point]
    client_blocks = block_names[split_point:]

    T_server = lut['T_server']
    T_client = lut['T_client']
    Comm_size = lut['Comm_size']
    bandwidth = lut['bandwidth']['bandwidth_mbps']
    latency = lut['bandwidth']['latency_ms']

    server_time = 0.0
    for block_name in server_blocks:
        gamma_idx = gamma_config[block_name]
        t = T_server[block_name][gamma_idx]
        if t > 0:
            server_time += t

    client_time = 0.0
    for block_name in client_blocks:
        gamma_idx = gamma_config[block_name]
        t = T_client[block_name][gamma_idx]
        if t > 0:
            client_time += t

    if split_point > 0 and split_point < len(block_names):
        split_block = block_names[split_point - 1]
        comm_data_mb = Comm_size[split_block][1]
        comm_time = (comm_data_mb * 8 / bandwidth * 1000) + latency
    else:
        comm_time = 0.0

    total_time = server_time + client_time + comm_time

    return {
        'server_time_ms': server_time,
        'client_time_ms': client_time,
        'comm_time_ms': comm_time,
        'total_time_ms': total_time,
        'throughput_fps': 1000.0 / total_time if total_time > 0 else 0
    }

def run_grid_search_optimization(lut_manager, mask_tensor, device, 
                                  min_split, max_split, steps, verbose=True):
    if verbose:
        print("\n[Grid Search] Optimizing split point and gamma configuration...")
        print("=" * 80)

    all_results = []
    best_result = None
    best_total_time = float('inf')

    for split_point in range(min_split, max_split + 1):
        if verbose:
            print(f"\n[Split {split_point}/{max_split}]")

        opt_result = optimize_gamma_for_fixed_lambda(
            lut_manager,
            mask_tensor,
            device,
            fixed_lambda=split_point,
            n_steps=steps,
            verbose=True
        )

        perf = evaluate_split_configuration(
            lut_manager,
            opt_result['optimal_gamma'],
            split_point
        )

        result = {
            'split_point': split_point,
            'optimal_gamma': opt_result['optimal_gamma'],
            'gamma_distribution': opt_result['gamma_distribution'],
            'final_loss': opt_result['final_loss'],
            'performance': perf
        }

        all_results.append(result)

        if perf['total_time_ms'] < best_total_time:
            best_total_time = perf['total_time_ms']
            best_result = result

        if verbose:
            print(f"  Loss: {opt_result['final_loss']['total']:.2f}")
            print(f"  Performance: Server={perf['server_time_ms']:.2f}ms, "
                  f"Client={perf['client_time_ms']:.2f}ms, "
                  f"Comm={perf['comm_time_ms']:.2f}ms")
            print(f"  Total: {perf['total_time_ms']:.2f}ms ({perf['throughput_fps']:.2f} FPS)")

    if verbose:
        print("\n" + "=" * 80)
        print("\n[Grid Search Results Summary]")
        print("=" * 80)

        sorted_results = sorted(all_results, key=lambda x: x['performance']['total_time_ms'])

        print("\nTop 5 Best Configurations:")
        print("-" * 80)
        for i, res in enumerate(sorted_results[:5], 1):
            marker = " [OPTIMAL]" if i == 1 else ""
            perf = res['performance']
            print(f"{i}. Split={res['split_point']:2d}: "
                  f"Total={perf['total_time_ms']:7.2f}ms "
                  f"(S:{perf['server_time_ms']:5.2f}, C:{perf['client_time_ms']:6.2f}, "
                  f"Comm:{perf['comm_time_ms']:6.2f}){marker}")

        print("\n" + "-" * 80)
        print("Optimal Configuration Details:")
        print("-" * 80)
        opt_perf = best_result['performance']
        print(f"Optimal Split Point:    λ = {best_result['split_point']}")
        print(f"Server latency:         {opt_perf['server_time_ms']:.2f} ms")
        print(f"Client latency:         {opt_perf['client_time_ms']:.2f} ms")
        print(f"Communication latency:  {opt_perf['comm_time_ms']:.2f} ms")
        print(f"Total latency:          {opt_perf['total_time_ms']:.2f} ms")
        print(f"Throughput:             {opt_perf['throughput_fps']:.2f} FPS")

        print(f"\nGamma distribution:")
        for gamma_idx in sorted(best_result['gamma_distribution'].keys()):
            config_name = GammaConfigurator.get_gamma_name(gamma_idx)
            count = best_result['gamma_distribution'][gamma_idx]
            print(f"  γ={gamma_idx:2d} ({config_name:12s}): {count:3d} blocks")

    return best_result, all_results

def run_joint_optimization(args):
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    print("="*80)
    print(f"Step 3 (Joint): Accuracy + Throughput Optimization (Domain {args.domain_id})")
    print("="*80)
    print(f"Device: {device}")
    print(f"Mode: {args.mode}")
    print(f"LUT file: {args.lut_file}")
    print()
    
    print("[Stage 1/2] ACCURACY OPTIMIZATION (Progressive Pruning)")
    print("="*80)
    
    print("\n[1/7] Loading pre-computed data...")
    
    with open(TRAINING_RESULTS_FILE, 'r') as f:
        training_results = json.load(f)
    
    with open(IMPORTANCE_RANKINGS_FILE, 'r') as f:
        importance_rankings = json.load(f)
    
    with open(BRANCH_FLOPS_FILE, 'r') as f:
        branch_flops = json.load(f)
    
    with open(PARTITION_INFO_PATH, 'r') as f:
        partition_info = json.load(f)
    
    domain_key = f'domain_{args.domain_id}'
    domain_training = training_results[domain_key]
    initial_acc = domain_training['initial_acc']
    best_acc = domain_training['best_acc']
    domain_ranking = importance_rankings[domain_key]
    
    print(f"  Domain {args.domain_id} - Initial acc: {initial_acc:.2f}%, Best acc: {best_acc:.2f}%")
    
    print(f"\n[2/7] Creating data loaders for domain {args.domain_id}...")
    train_loader, val_loader = create_domain_dataloaders(
        DATA_PATH, partition_info, args.domain_id, BATCH_SIZE
    )
    print(f"  ✓ Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    print("\n[3/7] Creating model...")
    model = create_hyper_repvgg(WEIGHT_PATH_SERVER, model_name=MODEL_NAME)
    freeze_bn_stats(model)
    model = model.to(device)
    print(f"  ✓ Model created: {MODEL_NAME}")
    
    print(f"\n[4/7] Creating trainer (mode={args.mode})...")
    
    if args.mode == 'real':
        print("  [Real mode] Connecting to Client...")
        from network_utils import ServerConnection
        from split_trainer import ServerClientTrainer
        from config import PORT_MAIN
        
        server_conn = ServerConnection(PORT_MAIN)
        conn = server_conn.start()
        initial_lambda = 10
        trainer = ServerClientTrainer(conn, model, initial_lambda, device)
        print(f"  ✓ Connected to Client, initial split point = {initial_lambda}")
        
    elif args.mode == 'pseudo':
        print("  [Pseudo mode] Using local split simulation...")
        trainer = LocalTrainer(model, device)
        print(f"  ✓ Local trainer created")
        
    else:
        raise ValueError(f"Unknown mode: {args.mode}. Use 'real' or 'pseudo'")
    
    print(f"\n[5/7] Running progressive pruning...")
    print("-" * 80)
    
    pruning_result = progressive_pruning(
        model=model,
        trainer=trainer,
        train_loader=train_loader,
        val_loader=val_loader,
        importance_ranking=domain_ranking,
        initial_acc=initial_acc,
        best_acc=best_acc,
        branch_flops=branch_flops,
        domain_id=args.domain_id,
        max_rounds=args.max_rounds,
        candidate_pool_size=args.candidate_pool_size,
        epochs_per_test=args.epochs_per_test,
        threshold_percent=args.threshold_percent,
        device=device
    )
    
    print("-" * 80)
    print(f"\n✓ Progressive pruning completed!")
    print(f"  Final accuracy: {pruning_result['final_accuracy']:.2f}%")
    print(f"  Total pruned: {len(pruning_result['final_state']['pruned_weights'])} branches")
    
    print("\n" + "="*80)
    print("[Stage 2/2] THROUGHPUT OPTIMIZATION (Grid Search)")
    print("="*80)
    
    print("\n[6/7] Loading LUT and building gamma mask...")
    if not os.path.exists(args.lut_file):
        raise FileNotFoundError(f"LUT file not found: {args.lut_file}")
    
    lut_manager = LUTManager(args.lut_file)
    print(f"  ✓ Loaded LUT: {args.lut_file}")
    
    pruned_weights = pruning_result['final_state']['pruned_weights']
    mask_builder = GammaMaskBuilder(model, device)
    mask_tensor = mask_builder.build_mask(pruned_weights)
    
    mask_builder.print_mask_summary(mask_tensor, pruned_weights)
    
    print(f"\n[7/7] Running grid search optimization...")
    
    best_result, all_results = run_grid_search_optimization(
        lut_manager,
        mask_tensor,
        device,
        min_split=args.min_split,
        max_split=args.max_split,
        steps=args.steps,
        verbose=True
    )
    
    print("\n" + "="*80)
    print("JOINT OPTIMIZATION COMPLETED")
    print("="*80)
    
    print("\n[Saving Results]")
    
    final_result = {
        'metadata': {
            'step': 'step_3_joint',
            'description': 'Joint accuracy and throughput optimization',
            'domain_id': args.domain_id,
            'mode': args.mode,
            'model': MODEL_NAME,
            'timestamp': str(datetime.now())
        },
        'stage1_accuracy': {
            'initial_accuracy': initial_acc,
            'final_accuracy': pruning_result['final_accuracy'],
            'pruned_weights': pruning_result['final_state']['pruned_weights'],
            'pruning_rounds': pruning_result['pruning_rounds'],
            'total_pruned': len(pruning_result['final_state']['pruned_weights'])
        },
        'stage2_throughput': {
            'optimal_split_point': best_result['split_point'],
            'optimal_gamma': best_result['optimal_gamma'],
            'gamma_distribution': best_result['gamma_distribution'],
            'performance': best_result['performance'],
            'all_split_results': [
                {
                    'split_point': r['split_point'],
                    'total_time_ms': r['performance']['total_time_ms'],
                    'throughput_fps': r['performance']['throughput_fps']
                }
                for r in all_results
            ]
        },
        'final_configuration': {
            'split_point': best_result['split_point'],
            'gamma_config': best_result['optimal_gamma'],
            'accuracy': pruning_result['final_accuracy'],
            'throughput_fps': best_result['performance']['throughput_fps'],
            'latency_ms': best_result['performance']['total_time_ms']
        }
    }
    
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    with open(args.output, 'w') as f:
        json.dump(final_result, f, indent=2)
    
    print(f"✓ Result saved to: {args.output}")
    
    model_output = args.output.replace('.json', '_model.pt')
    torch.save(pruning_result['final_model_state'], model_output)
    print(f"✓ Model state saved to: {model_output}")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Domain ID:              {args.domain_id}")
    print(f"Final Accuracy:         {pruning_result['final_accuracy']:.2f}%")
    print(f"Optimal Split Point:    λ = {best_result['split_point']}")
    print(f"Throughput:             {best_result['performance']['throughput_fps']:.2f} FPS")
    print(f"Total Latency:          {best_result['performance']['total_time_ms']:.2f} ms")
    print(f"Total Pruned Branches:  {len(pruning_result['final_state']['pruned_weights'])}")
    print("="*80)
    
    return final_result

def main():
    parser = argparse.ArgumentParser(description='Step 3 (Joint): Accuracy + Throughput Optimization')
    parser.add_argument('--domain_id', type=int, required=True, help='Domain ID (0-9)')
    parser.add_argument('--mode', type=str, default='pseudo', choices=['pseudo', 'real'],
                        help='Running mode: pseudo (local) or real (Server-Client)')
    parser.add_argument('--lut_file', type=str, default='results/hardware_lut_complete.json',
                        help='Path to LUT file')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path')
    parser.add_argument('--max_rounds', type=int, default=MAX_PRUNING_ROUNDS,
                        help='Max pruning rounds')
    parser.add_argument('--candidate_pool_size', type=int, default=CANDIDATE_POOL_SIZE,
                        help='Candidate pool size for pruning')
    parser.add_argument('--epochs_per_test', type=int, default=EPOCHS_PER_TEST,
                        help='Epochs per test')
    parser.add_argument('--threshold_percent', type=float, default=THRESHOLD_PERCENT,
                        help='Accuracy threshold percent')
    parser.add_argument('--min_split', type=int, default=8,
                        help='Minimum split point for grid search')
    parser.add_argument('--max_split', type=int, default=15,
                        help='Maximum split point for grid search')
    parser.add_argument('--steps', type=int, default=THROUGHPUT_OPTIM_STEPS,
                        help='Optimization steps per split point')
    parser.add_argument('--cpu', action='store_true',
                        help='Use CPU instead of GPU')
    
    args = parser.parse_args()
    
    if args.output is None:
        args.output = f'results/joint_optimization/domain_{args.domain_id}_joint.json'
    
    run_joint_optimization(args)

if __name__ == '__main__':
    main()
