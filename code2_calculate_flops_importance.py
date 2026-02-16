import torch
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import OrderedDict
from hyper_repvgg import create_hyper_repvgg, HyperRepVGGBlock

def calculate_conv_flops(in_channels, out_channels, kernel_size, 
                        input_h, input_w, groups=1):
    return 2 * in_channels * out_channels * (kernel_size ** 2) * input_h * input_w / groups
def calculate_batchnorm_flops(channels, input_h, input_w):
    return 4 * channels * input_h * input_w
def get_feature_map_size(stage_name, block_idx):
    stage_sizes = {
        'stage1': (112, 112),
        'stage2': (56, 56),
        'stage3': (28, 28),
        'stage4': (14, 14)
    }
    return stage_sizes[stage_name]
def calculate_branch_flops(model):
    flops_dict = OrderedDict()
    branch_info = OrderedDict()
    
    for stage_name in ['stage1', 'stage2', 'stage3', 'stage4']:
        stage = getattr(model, stage_name)
        h, w = get_feature_map_size(stage_name, 0)
        
        for block_idx, module in enumerate(stage):
            if isinstance(module, HyperRepVGGBlock):

                if hasattr(module.rbr_dense, 'conv'):
                    dense_conv = module.rbr_dense.conv
                elif hasattr(module.rbr_dense, 'rbr_dense'):
                    dense_conv = module.rbr_dense.rbr_dense
                else:

                    dense_conv = module.rbr_dense
                
                in_channels = dense_conv.in_channels
                out_channels = dense_conv.out_channels
                groups = dense_conv.groups
                
                dense_flops = calculate_conv_flops(
                    in_channels, out_channels, 3, h, w, groups
                )
                flops_dict[f'{stage_name}.{block_idx}.hyper_dense'] = dense_flops
                branch_info[f'{stage_name}.{block_idx}.hyper_dense'] = 'dense'
                
                if module.rbr_identity is not None:

                    identity_flops = calculate_batchnorm_flops(
                        out_channels, h, w
                    )
                    branch_info[f'{stage_name}.{block_idx}.hyper_identity'] = 'physical_identity'
                else:

                    identity_flops = 0
                    branch_info[f'{stage_name}.{block_idx}.hyper_identity'] = 'virtual_identity'
                
                flops_dict[f'{stage_name}.{block_idx}.hyper_identity'] = identity_flops
                
                has_1x1 = hasattr(module, 'rbr_1x1') and module.rbr_1x1 is not None
                
                if has_1x1:

                    onexone_flops = calculate_conv_flops(
                        in_channels, out_channels, 1, h, w, groups
                    )
                else:

                    onexone_flops = 0.0
                
                branch_info[f'{stage_name}.{block_idx}.hyper_1x1'] = '1x1'
                flops_dict[f'{stage_name}.{block_idx}.hyper_1x1'] = onexone_flops
    
    return flops_dict, branch_info

def calculate_importance(trained_weights_file, flops_dict, branch_info, save_dir):

    with open(trained_weights_file, 'r') as f:
        all_hyper_weights = json.load(f)
    
    os.makedirs(save_dir, exist_ok=True)
    
    all_importance = {}
    all_rankings = {}
    all_filtered = {}
    
    for domain_key, weights in all_hyper_weights.items():
        importance_scores = {}
        filtered_branches = []
        
        for weight_name, weight_value in weights.items():

            if weight_name not in branch_info:
                print(f"⚠️  Warning: {weight_name} not in branch_info, skipping")
                continue
            
            branch_type = branch_info[weight_name]
            
            if branch_type == 'virtual_identity':
                filtered_branches.append({
                    'name': weight_name,
                    'type': branch_type,
                    'weight': weight_value,
                    'reason': 'Virtual layer (no actual module)'
                })
                continue
            
            if weight_name in flops_dict:
                flops = flops_dict[weight_name]
            else:
                print(f"⚠️  Warning: {weight_name} not in FLOPS dict")
                continue
            
            if flops == 0:

                importance = 1e10
            else:

                importance = abs(weight_value) / (flops / 1e9)
            
            importance_scores[weight_name] = importance
        
        sorted_items = sorted(importance_scores.items(), key=lambda x: x[1])
        
        all_importance[domain_key] = importance_scores
        all_rankings[domain_key] = [name for name, _ in sorted_items]
        all_filtered[domain_key] = filtered_branches
    
    with open(os.path.join(save_dir, 'importance_scores.json'), 'w') as f:
        json.dump(all_importance, f, indent=2)
    
    with open(os.path.join(save_dir, 'importance_rankings.json'), 'w') as f:
        json.dump(all_rankings, f, indent=2)
    
    with open(os.path.join(save_dir, 'filtered_branches.json'), 'w') as f:
        json.dump(all_filtered, f, indent=2)
    
    visualize_importance(all_importance, all_hyper_weights, branch_info, save_dir)
    
    print(f"\n✓ Importance scores saved: {save_dir}/importance_scores.json")
    print(f"✓ Rankings saved: {save_dir}/importance_rankings.json")
    print(f"✓ Filtered branches saved: {save_dir}/filtered_branches.json")
    
    return all_importance, all_rankings, all_filtered

def visualize_importance(all_importance, all_hyper_weights, branch_info, save_dir):

    weight_names = list(next(iter(all_importance.values())).keys())
    num_domains = len(all_importance)
    num_weights = len(weight_names)
    
    importance_matrix = np.zeros((num_domains, num_weights))
    domain_labels = []
    
    for i, domain_key in enumerate(sorted(all_importance.keys())):
        domain_labels.append(f"D{i}")
        
        for j, weight_name in enumerate(weight_names):
            importance_matrix[i, j] = all_importance[domain_key][weight_name]
    
    weight_labels = []
    for name in weight_names:
        parts = name.split('.')
        stage = parts[0].replace('stage', 'S')
        block = parts[1]
        
        if name in branch_info:
            btype = branch_info[name]
            if btype == 'dense':
                branch = 'D'
            elif btype == 'physical_identity':
                branch = 'I'
            elif btype == 'virtual_identity':
                branch = 'V'
            elif btype == '1x1':
                branch = '1'
            else:
                branch = '?'
        else:
            branch = '?'
        
        weight_labels.append(f"{stage}.{block}.{branch}")
    
    plt.figure(figsize=(32, 10))
    
    importance_matrix_log = np.log10(importance_matrix + 1)
    
    sns.heatmap(
        importance_matrix_log,
        xticklabels=weight_labels,
        yticklabels=domain_labels,
        annot=False,
        cmap='viridis',
        cbar_kws={'label': 'log10(Importance + 1)'}
    )
    
    plt.title('Importance Scores (log scale) - D=Dense, I=Identity, 1=1x1', 
              fontsize=16, pad=20)
    plt.xlabel('Hyper Weight (Stage.Block.Branch)', fontsize=12)
    plt.ylabel('Domain', fontsize=12)
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_dir, 'importance_heatmap.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Importance heatmap saved: {save_dir}/importance_heatmap.png")
    
    df = pd.DataFrame(importance_matrix, 
                      index=domain_labels, 
                      columns=weight_labels)
    df.to_csv(os.path.join(save_dir, 'importance_matrix.csv'))
    print(f"✓ Importance matrix saved: {save_dir}/importance_matrix.csv")

def print_ranking_summary(all_rankings, all_filtered, branch_info, save_dir):
    print(f"\n{'='*80}")
    print(f"IMPORTANCE RANKINGS (Bottom 10 - Least Important)")
    print(f"{'='*80}\n")
    
    summary_file = os.path.join(save_dir, 'ranking_summary.txt')
    with open(summary_file, 'w') as f:
        for domain_key in sorted(all_rankings.keys()):
            ranking = all_rankings[domain_key]
            filtered = all_filtered[domain_key]
            bottom_10 = ranking[:10]
            
            output = f"Domain {domain_key}:\n"
            
            if filtered:
                output += f"  Filtered (not participating in ranking):\n"
                for branch in filtered:
                    output += f"    - {branch['name']} (weight={branch['weight']:.4f}, reason={branch['reason']})\n"
                output += "\n"
            
            output += f"  Least important (candidates for pruning):\n"
            for i, weight_name in enumerate(bottom_10, 1):
                btype = branch_info.get(weight_name, 'unknown')
                output += f"    {i:2d}. {weight_name} [{btype}]\n"
            output += "\n"
            
            print(output)
            f.write(output)
    
    print(f"✓ Ranking summary saved: {summary_file}")

def print_statistics(flops_dict, branch_info, all_importance, save_dir):
    print(f"\n{'='*80}")
    print(f"BRANCH STATISTICS")
    print(f"{'='*80}\n")
    
    stats_file = os.path.join(save_dir, 'branch_statistics.txt')
    with open(stats_file, 'w') as f:

        type_counts = {}
        type_flops = {}
        
        for name, btype in branch_info.items():
            if btype not in type_counts:
                type_counts[btype] = 0
                type_flops[btype] = []
            
            type_counts[btype] += 1
            if name in flops_dict:
                type_flops[btype].append(flops_dict[name])
        
        output = "Branch type distribution:\n"
        for btype in sorted(type_counts.keys()):
            count = type_counts[btype]
            avg_flops = np.mean(type_flops[btype]) if type_flops[btype] else 0
            output += f"  {btype:20s}: {count:2d} branches, avg FLOPS: {avg_flops/1e9:.4f} G\n"
        
        output += "\n"
        
        domain_key = 'domain_0'
        if domain_key in all_importance:
            importance = all_importance[domain_key]
            
            output += f"Importance score ranges (domain_0):\n"
            
            for btype in sorted(set(branch_info.values())):
                scores = [importance[name] for name, bt in branch_info.items() 
                         if bt == btype and name in importance]
                
                if scores:

                    scores_filtered = [s for s in scores if s < 1e9]
                    
                    if scores_filtered:
                        output += f"  {btype:20s}: [{min(scores_filtered):.4f}, {max(scores_filtered):.4f}]\n"
                    else:
                        output += f"  {btype:20s}: all marked as 1e10 (FLOPs=0 sentinel)\n"
        
        print(output)
        f.write(output)
    
    print(f"✓ Statistics saved: {stats_file}")

if __name__ == "__main__":
    MODEL_NAME = 'RepVGG-B1'

    WEIGHT_PATH = "/root/autodl-tmp/JSAC/RepVGG/weights/RepVGG-B1-train.pth"
    
    TRAINED_WEIGHTS_FILE = f"task1_trained_weights_{MODEL_NAME}/trained_hyper_weights.json"
    SAVE_DIR = f"task2_importance_rankings_fix_1211_{MODEL_NAME}"

    model = create_hyper_repvgg(WEIGHT_PATH, model_name='RepVGG-B1')

    print("="*80)
    print("Task 2: Calculate FLOPs and Importance Rankings (FIXED VERSION)")
    print("="*80)
    print("Fixes:")
    print("  1. Identity FLOPS: BatchNorm calculation (not 1x1 conv)")
    print("  2. Added 1x1 branch FLOPS calculation")
    print("  3. Filter virtual identity (not participating in ranking)")
    print("  4. Unified handling of all 1x1 branches (Stage 1-4 as real conv)")
    print("  5. Correct formula: importance = weight / (flops / 1e9)")
    print("="*80)
    
    print("\nStep 1: Calculating FLOPs for each branch...")
    
    flops_dict, branch_info = calculate_branch_flops(model)
    
    print(f"\n{'='*80}")
    print(f"FLOPs Summary")
    print(f"{'='*80}")
    total_flops = sum(flops_dict.values())
    print(f"Total FLOPs (all branches): {total_flops/1e9:.2f} GFLOPs")
    print(f"Number of branches: {len(flops_dict)}")
    
    dense_flops = sum(v for k, v in flops_dict.items() if 'dense' in k)
    identity_flops = sum(v for k, v in flops_dict.items() if 'identity' in k)
    onexone_flops = sum(v for k, v in flops_dict.items() if '1x1' in k)
    
    print(f"\nFLOPs by branch type:")
    print(f"  Dense (3x3):   {dense_flops/1e9:.2f} G ({dense_flops/total_flops*100:.1f}%)")
    print(f"  Identity:      {identity_flops/1e9:.2f} G ({identity_flops/total_flops*100:.1f}%)")
    print(f"  1x1 conv:      {onexone_flops/1e9:.2f} G ({onexone_flops/total_flops*100:.1f}%)")
    
    print(f"\nExample FLOPs (first 6 branches):")
    for i, (name, flops) in enumerate(list(flops_dict.items())[:6]):
        btype = branch_info.get(name, 'unknown')
        print(f"  {name:30s} [{btype:18s}]: {flops/1e6:.2f} MFLOPs")
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    with open(os.path.join(SAVE_DIR, 'branch_flops.json'), 'w') as f:
        json.dump({k: float(v) for k, v in flops_dict.items()}, f, indent=2)
    
    with open(os.path.join(SAVE_DIR, 'branch_info.json'), 'w') as f:
        json.dump(branch_info, f, indent=2)
    
    print(f"\n✓ FLOPs saved: {SAVE_DIR}/branch_flops.json")
    print(f"✓ Branch info saved: {SAVE_DIR}/branch_info.json")
    
    print(f"\nStep 2: Calculating importance scores...")
    all_importance, all_rankings, all_filtered = calculate_importance(
        TRAINED_WEIGHTS_FILE, 
        flops_dict,
        branch_info,
        SAVE_DIR
    )
    
    print_ranking_summary(all_rankings, all_filtered, branch_info, SAVE_DIR)
    
    print_statistics(flops_dict, branch_info, all_importance, SAVE_DIR)
    
    print("\n✓ Task 2 completed successfully!")
    print(f"✓ All results saved to: {SAVE_DIR}/")
    print(f"\n📊 Key improvements:")
    print(f"  - Virtual identity branches filtered out: {len(all_filtered['domain_0'])} branches")
    print(f"  - Physical identity uses BatchNorm FLOPS (~0.01 G)")
    print(f"  - All 1x1 branches use true FLOPs (no ghost branches)")
    print(f"  - Pruning candidates ranked by weight/FLOPS ratio")