import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
def create_noniid_partition(data_path, num_groups=10, save_dir='noniid_partitions'):
    from domain_transforms import get_domain_transform
    
    classes = sorted([d for d in os.listdir(data_path) 
                     if os.path.isdir(os.path.join(data_path, d))])
    
    num_classes = len(classes)
    print(f"Total classes: {num_classes}")
    print(f"Number of groups (domains): {num_groups}")
    
    classes_per_group = num_classes // num_groups
    print(f"Classes per group: {classes_per_group}")
    
    group_assignments = {}
    class_to_group = {}
    group_to_classes = defaultdict(list)
    
    for class_idx, class_name in enumerate(classes):
        group_id = class_idx // classes_per_group
        
        if group_id >= num_groups:
            group_id = num_groups - 1
        
        group_assignments[class_idx] = group_id
        class_to_group[class_name] = group_id
        group_to_classes[group_id].append(class_name)
    
    print("\n" + "="*80)
    print("GROUP ASSIGNMENTS WITH DOMAIN TRANSFORMS")
    print("="*80)
    
    for group_id in range(num_groups):
        classes_in_group = group_to_classes[group_id]
        _, domain_name, domain_desc = get_domain_transform(group_id)
        
        print(f"\n【Group {group_id} - Domain: {domain_name}】")
        print(f"  Transform: {domain_desc}")
        print(f"  Classes: {len(classes_in_group)}")
        print(f"  Class range: {', '.join(classes_in_group[:5])}", end="")
        if len(classes_in_group) > 5:
            print(f" ... (and {len(classes_in_group) - 5} more)")
        else:
            print()
        
        total_samples = 0
        for class_name in classes_in_group:
            class_path = os.path.join(data_path, class_name)
            num_samples = len([f for f in os.listdir(class_path) 
                             if f.endswith('.JPEG') or f.endswith('.jpg')])
            total_samples += num_samples
        
        print(f"  Total samples: {total_samples}")
    
    os.makedirs(save_dir, exist_ok=True)
    
    partition_info = {
        'num_groups': num_groups,
        'num_classes': num_classes,
        'classes_per_group': classes_per_group,
        'group_assignments': group_assignments,
        'class_to_group': class_to_group,
        'group_to_classes': {k: v for k, v in group_to_classes.items()},
        'domain_info': {}
    }
    
    for group_id in range(num_groups):
        _, domain_name, domain_desc = get_domain_transform(group_id)
        partition_info['domain_info'][str(group_id)] = {
            'name': domain_name,
            'description': domain_desc
        }
    
    with open(os.path.join(save_dir, 'partition_info.json'), 'w') as f:
        json.dump(partition_info, f, indent=2)
    
    print(f"\n✓ Partition info saved to {save_dir}/partition_info.json")
    
    return partition_info

def visualize_partition(partition_info, save_path='noniid_partitions/partition_visualization.png'):
    num_classes = partition_info['num_classes']
    num_groups = partition_info['num_groups']
    group_assignments = partition_info['group_assignments']
    
    matrix = np.zeros((num_groups, num_classes))
    
    for class_idx_str, group_id in group_assignments.items():
        class_idx = int(class_idx_str)
        matrix[group_id, class_idx] = 1
    
    plt.figure(figsize=(20, 6))
    
    cmap = plt.cm.Blues
    plt.imshow(matrix, aspect='auto', cmap=cmap, interpolation='nearest')
    
    plt.xlabel('Class Index', fontsize=12)
    plt.ylabel('Group ID', fontsize=12)
    plt.title('Non-IID Data Partition: Class Assignment to Groups', fontsize=14, pad=20)
    
    plt.yticks(range(num_groups), [f'Group {i}' for i in range(num_groups)])
    
    xticks = list(range(0, num_classes, 100))
    plt.xticks(xticks, xticks)
    
    cbar = plt.colorbar(label='Assignment', ticks=[0, 1])
    cbar.ax.set_yticklabels(['Not Assigned', 'Assigned'])
    
    for i in range(num_groups + 1):
        plt.axhline(i - 0.5, color='gray', linewidth=0.5, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved to {save_path}")
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle('Classes Distribution per Group', fontsize=16, y=1.02)
    
    axes = axes.flatten()
    
    for group_id in range(num_groups):
        ax = axes[group_id]
        
        classes_in_group = [int(k) for k, v in group_assignments.items() if v == group_id]
        classes_in_group.sort()
        
        ax.bar(range(len(classes_in_group)), [1] * len(classes_in_group), 
               color=f'C{group_id}', alpha=0.7)
        
        ax.set_title(f'Group {group_id}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Class Index in Group', fontsize=10)
        ax.set_ylabel('Indicator', fontsize=10)
        ax.set_ylim([0, 1.5])
        
        ax.text(0.5, 1.3, f'{len(classes_in_group)} classes\nRange: {min(classes_in_group)}-{max(classes_in_group)}',
                ha='center', va='top', fontsize=9, transform=ax.transData,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    detail_save_path = save_path.replace('.png', '_detail.png')
    plt.savefig(detail_save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Detailed visualization saved to {detail_save_path}")
    
    plt.close('all')

def get_group_subset_indices(partition_info, group_id, data_path):
    from torchvision import datasets

    full_dataset = datasets.ImageFolder(data_path)
    
    group_classes = partition_info['group_to_classes'][str(group_id)]
    
    class_to_idx = full_dataset.class_to_idx
    group_class_indices = [class_to_idx[c] for c in group_classes if c in class_to_idx]
    
    indices = [i for i, (_, label) in enumerate(full_dataset.samples) 
               if label in group_class_indices]
    
    print(f"Group {group_id}: {len(group_classes)} classes, {len(indices)} samples")
    
    return indices

if __name__ == "__main__":

    DATA_PATH = "/root/autodl-tmp/JSAC/RepVGG/imagenet_val"
    NUM_GROUPS = 10
    SAVE_DIR = "noniid_partitions"
    
    print("="*80)
    print("Creating Non-IID Data Partition")
    print("="*80)
    
    partition_info = create_noniid_partition(DATA_PATH, NUM_GROUPS, SAVE_DIR)
    
    print("\n" + "="*80)
    print("Creating Visualizations")
    print("="*80)
    visualize_partition(partition_info, 
                       os.path.join(SAVE_DIR, 'partition_visualization.png'))
    
    print("\n" + "="*80)
    print("✓ Non-IID partition creation completed!")
    print("="*80)