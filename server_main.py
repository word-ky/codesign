import torch
import torch.nn as nn
import json
import os
import argparse
import copy
from datetime import datetime
from collections import OrderedDict
import numpy as np

from config import (
    CMD, MODEL_NAME, WEIGHT_PATH_SERVER, NUM_DOMAINS,
    DATA_PATH, PARTITION_INFO_PATH,
    TRAINING_RESULTS_FILE, IMPORTANCE_RANKINGS_FILE, BRANCH_FLOPS_FILE,
    OUTPUT_DIR, LUT_OUTPUT_FILE, PORT_MAIN,
    FIXED_SPLIT_POINT, THROUGHPUT_OPTIM_STEPS,
    BATCH_SIZE
)

from network_utils import ServerConnection, send_message, recv_message
from lut_manager import LUTManager, LUTMeasurer, build_complete_lut, measure_communication_bandwidth
from throughput_optimizer import ThroughputOptimizer, MaskManager
from split_trainer import ServerClientTrainer, LocalTrainer
from progressive_pruning import progressive_pruning, get_alpha_dict

def freeze_bn_stats(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
    return model

def load_pretrained_data():
    print("\n" + "="*80)
    print("Loading Pre-computed Results")
    print("="*80)

    with open(TRAINING_RESULTS_FILE, 'r') as f:
        training_results = json.load(f)
    print(f"✓ Loaded training results: {TRAINING_RESULTS_FILE}")
    
    with open(IMPORTANCE_RANKINGS_FILE, 'r') as f:
        importance_rankings = json.load(f)
    print(f"✓ Loaded importance rankings: {IMPORTANCE_RANKINGS_FILE}")
    
    with open(BRANCH_FLOPS_FILE, 'r') as f:
        branch_flops = json.load(f)
    print(f"✓ Loaded branch FLOPs: {BRANCH_FLOPS_FILE}")
    
    with open(PARTITION_INFO_PATH, 'r') as f:
        partition_info = json.load(f)
    print(f"✓ Loaded partition info: {PARTITION_INFO_PATH}")
    
    return training_results, importance_rankings, branch_flops, partition_info

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

def epoch0_lut_measurement(conn, model, device='cuda'):
    print("\n" + "="*80)
    print("Epoch 0: LUT Measurement")
    print("="*80)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("\n[Phase 1] Measuring Server-side latency...")
    server_measurer = LUTMeasurer(model, device=device)
    server_results = server_measurer.measure_all_blocks()
    model_info = {
        'block_input_shapes': server_measurer.block_input_shapes,
        'block_output_shapes': server_measurer.block_output_shapes
    }
    
    server_results_formatted = {}
    for block_name, gamma_dict in server_results.items():
        server_results_formatted[block_name] = {
            str(k): v for k, v in gamma_dict.items()
        }
    
    server_lut_file = f"{OUTPUT_DIR}/lut_server_full.json"
    with open(server_lut_file, 'w') as f:
        json.dump({
            'metadata': {
                'device': 'server',
                'gpu_name': torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU',
                'build_time': str(datetime.now())
            },
            'T_server': server_results_formatted
        }, f, indent=2)
    print(f"✓ Saved: {server_lut_file}")
    
    print("\n[Phase 2] Requesting Client-side measurement...")
    send_message(conn, CMD.LUT_MEASURE, {'model_name': MODEL_NAME})
    
    cmd, client_data = recv_message(conn)
    assert cmd == CMD.LUT_RESULT, f"Expected LUT_RESULT, got {cmd}"
    
    client_results = client_data['T_client']
    
    client_lut_file = f"{OUTPUT_DIR}/lut_client_full.json"
    with open(client_lut_file, 'w') as f:
        json.dump({
            'metadata': client_data.get('metadata', {}),
            'T_client': client_results
        }, f, indent=2)
    print(f"✓ Saved: {client_lut_file}")
    
    print("\n[Phase 3] Measuring communication bandwidth...")
    comm_model = measure_communication_bandwidth(conn)
    
    comm_file = f"{OUTPUT_DIR}/lut_comm_model.json"
    with open(comm_file, 'w') as f:
        json.dump(comm_model, f, indent=2)
    print(f"✓ Saved: {comm_file}")
    
    print("\n[Phase 4] Building complete LUT...")
    
    client_results_formatted = {}
    for block_name, gamma_dict in client_results.items():
        client_results_formatted[block_name] = {
            str(k): v for k, v in gamma_dict.items()
        }
    
    complete_lut = build_complete_lut(
        server_results_formatted,
        client_results_formatted,
        comm_model,
        model_info=model_info
    )
    
    complete_lut['metadata']['server_device'] = torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'
    complete_lut['metadata']['client_device'] = client_data.get('metadata', {}).get('device_name', 'Unknown')
    
    with open(LUT_OUTPUT_FILE, 'w') as f:
        json.dump(complete_lut, f, indent=2)
    print(f"✓ Saved complete LUT: {LUT_OUTPUT_FILE}")
    
    lut_manager = LUTManager(LUT_OUTPUT_FILE)
    
    print("\n" + "="*80)
    print("✓ Epoch 0 Complete: LUT measurement finished")
    print("="*80)
    
    return lut_manager

def run_joint_optimization(args):
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    print(f"\n[Main] Device: {device}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/accuracy", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/throughput", exist_ok=True)
    
    training_results, importance_rankings, branch_flops, partition_info = load_pretrained_data()
    
    print("\n[Main] Creating model...")
    from hyper_repvgg import create_hyper_repvgg
    base_model = create_hyper_repvgg(WEIGHT_PATH_SERVER, model_name=MODEL_NAME)
    freeze_bn_stats(base_model)
    
    if args.local:
        print("\n[Main] Running in LOCAL mode (no Client)")
        conn = None
    else:
        print(f"\n[Main] Waiting for Client connection on port {PORT_MAIN}...")
        server_conn = ServerConnection(PORT_MAIN)
        conn = server_conn.start()
    
    if args.skip_lut and os.path.exists(LUT_OUTPUT_FILE):
        print(f"\n[Main] Loading existing LUT: {LUT_OUTPUT_FILE}")
        lut_manager = LUTManager(LUT_OUTPUT_FILE)
    else:
        if args.local:

            print("\n[Main] Local mode: Loading existing LUT...")
            if os.path.exists(LUT_OUTPUT_FILE):
                lut_manager = LUTManager(LUT_OUTPUT_FILE)
            else:
                raise FileNotFoundError(f"LUT file not found: {LUT_OUTPUT_FILE}. Run with Client first.")
        else:
            lut_manager = epoch0_lut_measurement(conn, base_model, device)
    
    mask_manager = MaskManager(base_model, device)
    
    throughput_optimizer = ThroughputOptimizer(lut_manager, mask_manager, device)
    
    if args.local:
        trainer = LocalTrainer(base_model, device)
    else:
        trainer = ServerClientTrainer(conn, base_model, FIXED_SPLIT_POINT, device)
    
    all_results = {}
    
    for domain_id in range(NUM_DOMAINS):
        domain_key = f'domain_{domain_id}'
        
        print(f"\n{'#'*80}")
        print(f"# Domain {domain_id}/{NUM_DOMAINS - 1}")
        print(f"{'#'*80}")
        
        if conn:
            send_message(conn, CMD.DOMAIN_START, {'domain_id': domain_id})
        
        domain_training = training_results[domain_key]
        initial_acc = domain_training['initial_acc']
        best_acc = domain_training['best_acc']
        domain_ranking = importance_rankings[domain_key]
        
        print(f"\n[Domain {domain_id}] Initial Acc: {initial_acc:.2f}%, Best Acc: {best_acc:.2f}%")
        
        print(f"\n[Domain {domain_id}] Creating data loaders...")
        train_loader, val_loader = create_domain_dataloaders(
            DATA_PATH, partition_info, domain_id, BATCH_SIZE
        )

        print(f"\n[Debug] Checking val_loader labels...")
        all_labels = []
        for i, (images, labels) in enumerate(val_loader):
            all_labels.extend(labels.numpy())
            if i >= 10:
                break
        
        all_labels = np.array(all_labels)
        unique_labels = np.unique(all_labels)
        print(f"  Total samples checked: {len(all_labels)}")
        print(f"  Unique labels: {unique_labels[:20]}")
        print(f"  Min: {all_labels.min()}, Max: {all_labels.max()}")
        print(f"  Expected: 100 classes (0-99 for domain 0)")

        print(f"\n[Domain {domain_id}] Creating fresh model (hyper=1.0)...")
        model = create_hyper_repvgg(WEIGHT_PATH_SERVER, model_name=MODEL_NAME)
        freeze_bn_stats(model)
        model = model.to(device)
        
        trainer.model = model
        
        print(f"\n[Domain {domain_id}] Phase 1: Throughput Optimization")
        print("-" * 60)
        
        alpha_dict = get_alpha_dict(model)
        throughput_optimizer.update_mask(alpha_dict)
        
        throughput_history = throughput_optimizer.optimize(
            n_steps=THROUGHPUT_OPTIM_STEPS,
            verbose=True
        )
        
        optimal_gamma = throughput_optimizer.get_optimal_gamma()
        optimal_lambda = throughput_optimizer.get_optimal_lambda()
        throughput_constant = throughput_optimizer.get_throughput_constant()
        
        print(f"\n[Phase 1 Results]")
        print(f"  Optimal λ (split point): {optimal_lambda}")
        print(f"  Throughput constant: {throughput_constant:.4f}")
        print(f"  Sample γ configs:")
        for block_name in list(optimal_gamma.keys())[:5]:
            print(f"    {block_name}: γ={optimal_gamma[block_name]}")
        
        throughput_result = {
            'domain_id': domain_id,
            'optimal_gamma': optimal_gamma,
            'optimal_lambda': optimal_lambda,
            'throughput_constant': throughput_constant,
            'history': throughput_history
        }
        
        with open(f"{OUTPUT_DIR}/throughput/domain_{domain_id}_throughput.json", 'w') as f:
            json.dump(throughput_result, f, indent=2)
        
        print(f"\n[Domain {domain_id}] Phase 2: Accuracy Optimization (Progressive Pruning)")
        print("-" * 60)
        
        accuracy_result = progressive_pruning(
            model=model,
            trainer=trainer,
            train_loader=train_loader,
            val_loader=val_loader,
            importance_ranking=domain_ranking,
            initial_acc=initial_acc,
            best_acc=best_acc,
            branch_flops=branch_flops,
            domain_id=domain_id,
            device=device
        )
        
        accuracy_result_save = {k: v for k, v in accuracy_result.items() 
                                if k != 'final_model_state'}
        
        with open(f"{OUTPUT_DIR}/accuracy/domain_{domain_id}_accuracy.json", 'w') as f:
            json.dump(accuracy_result_save, f, indent=2)
        
        torch.save(
            accuracy_result['final_model_state'],
            f"{OUTPUT_DIR}/accuracy/domain_{domain_id}_model.pt"
        )
        
        print(f"\n[Domain {domain_id}] Phase 3: Updating state")
        
        final_alpha_dict = accuracy_result['final_alpha_dict']
        lut_manager.update_for_pruned_alpha(final_alpha_dict)
        
        all_results[domain_key] = {
            'throughput': throughput_result,
            'accuracy': accuracy_result_save
        }
        
        if conn:
            send_message(conn, CMD.DOMAIN_END, {'domain_id': domain_id})
        
        print(f"\n✓ Domain {domain_id} complete!")
    
    print(f"\n{'='*80}")
    print("Saving Final Results")
    print("="*80)
    
    summary = {
        'metadata': {
            'model': MODEL_NAME,
            'num_domains': NUM_DOMAINS,
            'build_time': str(datetime.now())
        },
        'results': {}
    }
    
    for domain_key, result in all_results.items():
        summary['results'][domain_key] = {
            'initial_acc': result['accuracy']['initial_acc'],
            'best_acc': result['accuracy']['best_acc'],
            'final_acc': result['accuracy']['final_state']['final_acc'],
            'flops_reduction': result['accuracy']['final_state']['flops_reduction_percent'],
            'num_pruned': result['accuracy']['final_state']['num_pruned'],
            'optimal_lambda': result['throughput']['optimal_lambda'],
            'throughput_constant': result['throughput']['throughput_constant']
        }
    
    with open(f"{OUTPUT_DIR}/summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Saved: {OUTPUT_DIR}/summary.json")
    
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print("="*80)
    print(f"{'Domain':<10} {'Initial':>10} {'Best':>10} {'Final':>10} {'FLOPs↓':>10} {'Pruned':>8}")
    print("-" * 70)
    
    for domain_key in sorted(summary['results'].keys()):
        r = summary['results'][domain_key]
        domain_id = domain_key.split('_')[1]
        print(f"D{domain_id:<9} {r['initial_acc']:>9.2f}% {r['best_acc']:>9.2f}% "
              f"{r['final_acc']:>9.2f}% {r['flops_reduction']:>9.1f}% {r['num_pruned']:>8}")
    
    print("="*80)
    
    if conn:
        server_conn.close()
    
    print(f"\n✓ Joint optimization complete!")
    print(f"✓ All results saved to: {OUTPUT_DIR}/")

def main():
    parser = argparse.ArgumentParser(description='Joint Optimization - Server Main')
    parser.add_argument('--port', type=int, default=PORT_MAIN,
                       help='Port for Client connection')
    parser.add_argument('--local', action='store_true',
                       help='Run in local mode (no Client)')
    parser.add_argument('--skip_lut', action='store_true',
                       help='Skip LUT measurement if file exists')
    parser.add_argument('--cpu', action='store_true',
                       help='Use CPU instead of GPU')
    
    args = parser.parse_args()
    
    run_joint_optimization(args)

if __name__ == '__main__':
    main()