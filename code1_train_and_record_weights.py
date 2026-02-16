import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from collections import OrderedDict
from hyper_repvgg import create_hyper_repvgg
from domain_dataset import create_domain_dataloaders

def train_hyper_weights_pure(model, train_loader, full_val_loader, group_id, 
                             domain_name, domain_desc,
                             epochs=10, lr=0.01, l1_lambda=0.001, device='cuda'):
    model = model.to(device)
    model.eval()
    
    print(f"\n{'='*80}")
    print(f"Domain {group_id}: {domain_name}")
    print(f"Description: {domain_desc}")
    print(f"{'='*80}")
    
    print(f"\nInitial Evaluation (all hyper-weights = 1.0)...")
    initial_acc = evaluate(model, full_val_loader, device)
    print(f"✓ Initial validation accuracy: {initial_acc:.2f}%")
    
    hyper_params = []
    for name, param in model.named_parameters():
        if 'hyper_' in name:
            param.requires_grad = True
            hyper_params.append(param)
        else:
            param.requires_grad = False
    
    print(f"\n{'Training Configuration':-^80}")
    print(f"Total hyper parameters: {len(hyper_params)}")
    print(f"Trainable: {len(hyper_params)}")
    print(f"Frozen: 0 (no pruning)")
    print(f"{'-'*80}\n")
    
    model.train()
    optimizer = optim.SGD(hyper_params, lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    history = {
        'group_id': group_id,
        'domain_name': domain_name,
        'domain_desc': domain_desc,
        'initial_acc': initial_acc,
        'train_loss': [],
        'val_acc': [],
        'l1_penalty': []
    }
    
    best_acc = initial_acc
    best_hyper_weights = get_hyper_weights(model)
    
    print(f"{'='*80}")
    print(f"Starting Training")
    print(f"{'='*80}\n")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_l1 = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            l1_penalty = sum(torch.abs(p).sum() for p in hyper_params)
            total_loss = loss + l1_lambda * l1_penalty
            
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_l1 += l1_penalty.item()
            
            pbar.set_postfix({
                'loss': loss.item(),
                'l1': l1_penalty.item()
            })
        
        val_acc = evaluate(model, full_val_loader, device)
        
        history['train_loss'].append(epoch_loss / len(train_loader))
        history['val_acc'].append(val_acc)
        history['l1_penalty'].append(epoch_l1 / len(train_loader))
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_hyper_weights = get_hyper_weights(model)
        
        print(f"Epoch {epoch+1}: Val Acc = {val_acc:.2f}% (Best = {best_acc:.2f}%)")
    
    history['best_acc'] = best_acc
    history['improvement'] = best_acc - initial_acc
    
    print(f"\n{'='*80}")
    print(f"Domain {group_id} Training Complete!")
    print(f"{'='*80}")
    print(f"Initial Acc (hyper=1.0): {initial_acc:.2f}%")
    print(f"Best Acc (after training): {best_acc:.2f}%")
    print(f"Improvement: {best_acc - initial_acc:+.2f}%")
    print(f"{'='*80}\n")
    
    return history, best_hyper_weights

def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return 100 * correct / total

def get_hyper_weights(model):
    weights = OrderedDict()
    for name, param in model.named_parameters():
        if 'hyper_' in name:
            weights[name] = param.data.item()
    return weights

def run_training_experiment(data_path, weight_path, partition_info_path,
                           num_groups=10, epochs=10, lr=0.01, l1_lambda=0.001,
                           batch_size=32, save_dir="task1_trained_weights",
                           device='cuda'):
    os.makedirs(save_dir, exist_ok=True)
    
    with open(partition_info_path, 'r') as f:
        partition_info = json.load(f)
    
    all_results = {}
    all_hyper_weights = {}
    
    for group_id in range(num_groups):
        domain_info = partition_info['domain_info'][str(group_id)]
        domain_name = domain_info['name']
        domain_desc = domain_info['description']
        
        print(f"\n{'='*80}")
        print(f"PROCESSING DOMAIN {group_id}/{num_groups-1}: {domain_name}")
        print(f"{'='*80}\n")
        
        model = create_hyper_repvgg(weight_path, model_name='RepVGG-A0')
        
        train_loader, full_val_loader = create_domain_dataloaders(
            data_path=data_path,
            partition_info=partition_info,
            domain_id=group_id,
            batch_size=batch_size
        )
        
        history, trained_weights = train_hyper_weights_pure(
            model=model,
            train_loader=train_loader,
            full_val_loader=full_val_loader,
            group_id=group_id,
            domain_name=domain_name,
            domain_desc=domain_desc,
            epochs=epochs,
            lr=lr,
            l1_lambda=l1_lambda,
            device=device
        )
        
        domain_key = f'domain_{group_id}'
        all_results[domain_key] = history
        all_hyper_weights[domain_key] = trained_weights
    
    with open(os.path.join(save_dir, "training_results.json"), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    with open(os.path.join(save_dir, "trained_hyper_weights.json"), 'w') as f:
        json.dump(all_hyper_weights, f, indent=2)
    
    visualize_weights(all_hyper_weights, all_results, save_dir)
    
    print_summary(all_results)
    
    return all_results, all_hyper_weights

def visualize_weights(all_hyper_weights, all_results, save_dir):

    weight_names = list(next(iter(all_hyper_weights.values())).keys())
    num_domains = len(all_hyper_weights)
    num_weights = len(weight_names)
    
    weight_matrix = np.zeros((num_domains, num_weights))
    domain_labels = []
    
    for i, (domain_key, weights) in enumerate(sorted(all_hyper_weights.items())):
        domain_name = all_results[domain_key]['domain_name']
        domain_labels.append(f"D{i}\n{domain_name[:10]}")
        
        for j, weight_name in enumerate(weight_names):
            weight_matrix[i, j] = weights[weight_name]
    
    weight_labels = []
    for name in weight_names:

        parts = name.split('.')
        stage = parts[0].replace('stage', 'S')
        block = parts[1]
        branch = 'D' if 'dense' in parts[2] else ('I' if 'identity' in parts[2] else '1')
        weight_labels.append(f"{stage}.{block}.{branch}")
    
    plt.figure(figsize=(24, 8))
    
    sns.heatmap(
        weight_matrix,
        xticklabels=weight_labels,
        yticklabels=domain_labels,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn',
        center=1.0,
        vmin=0,
        vmax=2.0,
        cbar_kws={'label': 'Hyper Weight Value'}
    )
    
    plt.title('Trained Hyper Weights Across Domains', fontsize=16, pad=20)
    plt.xlabel('Hyper Weight (Stage.Block.Branch)', fontsize=12)
    plt.ylabel('Domain', fontsize=12)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_dir, 'hyper_weights_heatmap.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Heatmap saved: {save_dir}/hyper_weights_heatmap.png")
    
    import pandas as pd
    df = pd.DataFrame(weight_matrix, 
                      index=domain_labels, 
                      columns=weight_labels)
    df.to_csv(os.path.join(save_dir, 'hyper_weights_matrix.csv'))
    print(f"✓ Weight matrix saved: {save_dir}/hyper_weights_matrix.csv")

def print_summary(all_results):
    print(f"\n{'='*80}")
    print(f"EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    print(f"{'Domain':<30} {'Initial':<10} {'Final':<10} {'Improvement':<12}")
    print(f"{'-'*80}")
    for domain_key in sorted(all_results.keys()):
        result = all_results[domain_key]
        print(f"{result['domain_name']:<30} "
              f"{result['initial_acc']:>8.2f}% "
              f"{result['best_acc']:>8.2f}% "
              f"{result['improvement']:>+10.2f}%")
    
    print(f"{'='*80}\n")

if __name__ == "__main__":
    DATA_PATH = "/root/autodl-tmp/JSAC/RepVGG/imagenet_val"
    WEIGHT_PATH = "/root/autodl-tmp/JSAC/RepVGG/weights/RepVGG-A0-train.pth"

    PARTITION_INFO_PATH = "/root/autodl-tmp/JSAC/RepVGG/noniid_partitions/partition_info.json"
    
    NUM_GROUPS = 10
    EPOCHS = 3
    LEARNING_RATE = 0.01
    L1_LAMBDA = 0.001
    BATCH_SIZE = 32

    SAVE_DIR = "task1_trained_weights_RepVGG-A0"
    
    print("="*80)
    print("Task 1: Train and Record Hyper Weights")
    print("="*80)
    print("Purpose: Train hyper weights for each domain and visualize")
    print(f"Configuration:")
    print(f"  Domains: {NUM_GROUPS}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  LR: {LEARNING_RATE}, L1: {L1_LAMBDA}")
    print("="*80)
    
    all_results, all_hyper_weights = run_training_experiment(
        DATA_PATH, WEIGHT_PATH, PARTITION_INFO_PATH,
        num_groups=NUM_GROUPS,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        l1_lambda=L1_LAMBDA,
        batch_size=BATCH_SIZE,
        save_dir=SAVE_DIR
    )
    
    print("\n✓ Task 1 completed successfully!")
    print(f"✓ Results saved to: {SAVE_DIR}/")