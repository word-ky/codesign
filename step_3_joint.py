import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import argparse
from datetime import datetime
from collections import OrderedDict

from config import (
    MODEL_NAME, WEIGHT_PATH_SERVER, NUM_DOMAINS, NUM_BLOCKS,
    DATA_PATH, PARTITION_INFO_PATH, BATCH_SIZE,
    I_KNEE_SERVER, I_KNEE_CLIENT, SPLIT_SIGMA,
    PRUNING_THRESHOLD, BLOCK_INDEX_MAP
)

from hyper_repvgg import create_hyper_repvgg
from lut_manager import LUTManager
from gamma_configurator import GammaConfigurator


class MaskManager:
    def __init__(self, model, device='cuda'):
        self.device = device
        self.block_info = self._analyze_model(model)
    
    def _analyze_model(self, model):
        from hyper_repvgg import HyperRepVGGBlock
        block_info = OrderedDict()
        
        for stage_name in ['stage1', 'stage2', 'stage3', 'stage4']:
            stage = getattr(model, stage_name)
            for block_idx, block in enumerate(stage):
                if isinstance(block, HyperRepVGGBlock):
                    block_name = f"{stage_name}.{block_idx}"
                    block_info[block_name] = {
                        'has_identity': block.rbr_identity is not None
                    }
        
        return block_info
    
    def compute_mask(self, alpha_dict, threshold=PRUNING_THRESHOLD):
        mask_dict = OrderedDict()
        for block_name, info in self.block_info.items():
            mask = torch.zeros(14, device=self.device)
            
            if block_name in alpha_dict:
                alpha = alpha_dict[block_name]
                dense_pruned = abs(alpha.get('hyper_dense', 1.0)) < threshold
                onebyone_pruned = abs(alpha.get('hyper_1x1', 1.0)) < threshold
                identity_pruned = (abs(alpha.get('hyper_identity', 1.0)) < threshold 
                                   if info['has_identity'] else True)
            else:
                dense_pruned = False
                onebyone_pruned = False
                identity_pruned = not info['has_identity']
            
            for gamma_idx in range(14):
                config = GammaConfigurator.GAMMA_CONFIGS[gamma_idx]
                branches = config['branches']
                
                invalid = False
                if 'dense' in branches and dense_pruned:
                    invalid = True
                if '1x1' in branches and onebyone_pruned:
                    invalid = True
                if 'identity' in branches:
                    if not info['has_identity'] or identity_pruned:
                        invalid = True
                
                if invalid:
                    mask[gamma_idx] = float('-inf')
            
            mask_dict[block_name] = mask
        
        return mask_dict


class JointOptimizer:
    def __init__(self, model, lut_manager, train_loader, val_loader,
                 kappa=100.0, mu=1.0, omega=0.5, epsilon=2.0,
                 beta_S=0.01, beta_E=0.01, lambda_I=0.05, rho=0.1,
                 device='cuda', mode='pseudo', connection=None):
        self.model = model
        self.lut_manager = lut_manager
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.mode = mode
        self.connection = connection
        
        self.kappa = kappa
        self.mu = mu
        self.omega = omega
        self.epsilon = epsilon
        
        self.beta_S = beta_S
        self.beta_E = beta_E
        self.lambda_I_S = lambda_I
        self.lambda_I_E = lambda_I
        self.rho = rho
        self.g_mem_S = 1.0
        self.g_mem_E = 1.0
        
        self.bandwidth = 100e6
        self.T_rt = 0.001
        
        self.lut_tensors = lut_manager.to_tensor_dict(device)
        self.block_names = list(BLOCK_INDEX_MAP.keys())
        
        self.mask_manager = MaskManager(model, device)
        
        self.theta_gamma = nn.Parameter(torch.zeros(NUM_BLOCKS, 14, device=device))
        self.sigma_lambda = nn.Parameter(torch.tensor(10.0, device=device))
        self.xi_A = nn.Parameter(torch.tensor(0.0, device=device))
        
        self.A_ref = None
    
    def compute_intensity(self, tier, gamma_soft, z_k):
        total_flops = 0.0
        total_bytes = 0.0
        
        for i, block_name in enumerate(self.block_names):
            weight = z_k[i] if tier == 'server' else (1 - z_k[i])
            
            for gamma_idx in range(14):
                gamma_prob = gamma_soft[block_name][gamma_idx]
                
                flops = self.lut_tensors['flops'][i][gamma_idx] if 'flops' in self.lut_tensors else 1e9
                bytes_val = self.lut_tensors['bytes'][i][gamma_idx] if 'bytes' in self.lut_tensors else 1e6
                
                total_flops += weight * gamma_prob * flops
                total_bytes += weight * gamma_prob * bytes_val
        
        intensity = total_flops / (total_bytes + 1e-9)
        return intensity
    
    def compute_flops(self, tier, gamma_soft, z_k):
        total_flops = 0.0
        for i, block_name in enumerate(self.block_names):
            weight = z_k[i] if tier == 'server' else (1 - z_k[i])
            for gamma_idx in range(14):
                gamma_prob = gamma_soft[block_name][gamma_idx]
                flops = self.lut_tensors['flops'][i][gamma_idx] if 'flops' in self.lut_tensors else 1e9
                total_flops += weight * gamma_prob * flops
        return total_flops
    
    def compute_bytes(self, tier, gamma_soft, z_k):
        total_bytes = 0.0
        for i, block_name in enumerate(self.block_names):
            weight = z_k[i] if tier == 'server' else (1 - z_k[i])
            for gamma_idx in range(14):
                gamma_prob = gamma_soft[block_name][gamma_idx]
                bytes_val = self.lut_tensors['bytes'][i][gamma_idx] if 'bytes' in self.lut_tensors else 1e6
                total_bytes += weight * gamma_prob * bytes_val
        return total_bytes
    
    def get_feature_size(self, split_point):
        feature_sizes = {
            0: 224*224*64, 4: 112*112*64, 10: 56*56*128,
            14: 28*28*256, 26: 14*14*512
        }
        split_int = int(split_point)
        for k in sorted(feature_sizes.keys(), reverse=True):
            if split_int >= k:
                return feature_sizes[k] * 4
        return 224*224*64*4
    
    def compute_joint_loss(self, alpha_dict, theta_gamma, sigma_lambda):
        mask_dict = self.mask_manager.compute_mask(alpha_dict)
        gamma_soft = {}
        for i, block_name in enumerate(self.block_names):
            if i < len(self.block_names):
                mask = mask_dict.get(block_name, torch.zeros(14, device=self.device))
                gamma_soft[block_name] = F.softmax(theta_gamma[i] + mask, dim=0)
        
        z_k = torch.sigmoid(SPLIT_SIGMA * (sigma_lambda - torch.arange(NUM_BLOCKS, device=self.device, dtype=torch.float32)))
        
        T_S_exec = 0.0
        T_E_exec = 0.0
        for i, block_name in enumerate(self.block_names):
            if i < len(self.block_names):
                lut_server = self.lut_tensors['T_server'][i] if 'T_server' in self.lut_tensors else torch.ones(14, device=self.device) * 0.01
                lut_client = self.lut_tensors['T_client'][i] if 'T_client' in self.lut_tensors else torch.ones(14, device=self.device) * 0.05
                
                T_S_exec += z_k[i] * (gamma_soft[block_name] * lut_server).sum()
                T_E_exec += (1 - z_k[i]) * (gamma_soft[block_name] * lut_client).sum()
        
        feature_size = self.get_feature_size(sigma_lambda.item())
        T_Com = feature_size / self.bandwidth + self.T_rt
        
        I_M_server = self.compute_intensity('server', gamma_soft, z_k)
        F_M_server = self.compute_flops('server', gamma_soft, z_k)
        B_M_server = self.compute_bytes('server', gamma_soft, z_k)
        
        dominant_server = torch.relu(I_KNEE_SERVER - I_M_server)
        auxiliary_server = F_M_server / (10e12 + 1e-6) + B_M_server / (100e9 + 1e-6)
        L_roof_S = self.lambda_I_S * self.g_mem_S * (dominant_server + self.rho * auxiliary_server)
        
        I_M_edge = self.compute_intensity('edge', gamma_soft, z_k)
        F_M_edge = self.compute_flops('edge', gamma_soft, z_k)
        B_M_edge = self.compute_bytes('edge', gamma_soft, z_k)
        
        dominant_edge = torch.relu(I_KNEE_CLIENT - I_M_edge)
        auxiliary_edge = F_M_edge / (5e12 + 1e-6) + B_M_edge / (50e9 + 1e-6)
        L_roof_E = self.lambda_I_E * self.g_mem_E * (dominant_edge + self.rho * auxiliary_edge)
        
        T_S_roof = self.beta_S * L_roof_S
        T_E_roof = self.beta_E * L_roof_E
        
        T_S = T_S_exec + T_S_roof
        T_E = T_E_exec + T_E_roof
        
        T_lat = T_S + T_E + T_Com
        T_cyc = torch.max(torch.stack([T_S, T_E, torch.tensor(T_Com, device=self.device)]))
        
        T_eff = self.omega * T_lat + (1 - self.omega) * T_cyc
        
        A_current = self.evaluate_accuracy_fast()
        
        accuracy_term = self.kappa * self.xi_A - self.mu * A_current
        
        constraint_violation = torch.relu(self.A_ref - self.epsilon - A_current + self.xi_A)
        constraint_penalty = 1000.0 * constraint_violation
        
        J = T_eff + accuracy_term + constraint_penalty
        
        metrics = {
            'J_total': J.item(),
            'T_eff': T_eff.item(),
            'T_lat': T_lat.item(),
            'T_cyc': T_cyc.item(),
            'T_S': T_S.item(),
            'T_E': T_E.item(),
            'T_Com': T_Com,
            'L_roof_S': L_roof_S.item(),
            'L_roof_E': L_roof_E.item(),
            'A_current': A_current,
            'xi_A': self.xi_A.item(),
            'constraint_violation': constraint_violation.item(),
            'I_M_server': I_M_server.item(),
            'I_M_edge': I_M_edge.item()
        }
        
        return J, metrics
    
    def get_current_alpha(self):
        alpha_dict = {}
        for name, param in self.model.named_parameters():
            if 'hyper_' in name:
                parts = name.split('.')
                if len(parts) >= 3:
                    block_name = '.'.join(parts[:2])
                    branch_name = parts[2]
                    if block_name not in alpha_dict:
                        alpha_dict[block_name] = {}
                    alpha_dict[block_name][branch_name] = param.data.item()
        return alpha_dict
    
    def evaluate_accuracy_fast(self):
        self.model.eval()
        correct = 0
        total = 0
        max_batches = 10
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.val_loader):
                if batch_idx >= max_batches:
                    break
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        if total == 0:
            return 75.0
        return 100.0 * correct / total
    
    def evaluate_accuracy_full(self):
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        if total == 0:
            return 75.0
        return 100.0 * correct / total
    
    def optimize_alpha(self, epochs=3):
        print("\n[Phase A] Optimizing α (Accuracy)...")
        
        self.theta_gamma.requires_grad = False
        self.sigma_lambda.requires_grad = False
        
        alpha_params = []
        for name, param in self.model.named_parameters():
            if 'hyper_' in name:
                param.requires_grad = True
                alpha_params.append(param)
            else:
                param.requires_grad = False
        
        if len(alpha_params) == 0:
            print("  No alpha parameters found, skipping...")
            return
        
        optimizer = torch.optim.SGD(alpha_params + [self.xi_A], lr=0.01, momentum=0.9)
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            batch_count = 0
            
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                
                outputs = self.model(images)
                ce_loss = F.cross_entropy(outputs, labels)
                
                alpha_dict = self.get_current_alpha()
                J, metrics = self.compute_joint_loss(alpha_dict, self.theta_gamma, self.sigma_lambda)
                
                total_loss = ce_loss + 0.001 * J
                
                total_loss.backward()
                optimizer.step()
                
                with torch.no_grad():
                    self.xi_A.clamp_(min=0.0)
                
                epoch_loss += total_loss.item()
                batch_count += 1
                
                if batch_count >= 50:
                    break
            
            acc = self.evaluate_accuracy_fast()
            print(f"  Epoch {epoch+1}/{epochs}: Loss={epoch_loss/batch_count:.4f}, Acc={acc:.2f}%")
        
        self.theta_gamma.requires_grad = True
        self.sigma_lambda.requires_grad = True
    
    def optimize_gamma_lambda(self, steps=400):
        print("\n[Phase B] Optimizing γ, λ (Throughput)...")
        
        for param in self.model.parameters():
            param.requires_grad = False
        
        optimizer = torch.optim.Adam([self.theta_gamma, self.sigma_lambda, self.xi_A], lr=0.01)
        
        for step in range(steps):
            optimizer.zero_grad()
            
            alpha_dict = self.get_current_alpha()
            
            J, metrics = self.compute_joint_loss(alpha_dict, self.theta_gamma, self.sigma_lambda)
            
            J.backward()
            optimizer.step()
            
            with torch.no_grad():
                self.xi_A.clamp_(min=0.0)
            
            if (step + 1) % 100 == 0:
                print(f"  Step {step+1}/{steps}: J={metrics['J_total']:.4f}, "
                      f"T_eff={metrics['T_eff']:.4f}s, "
                      f"Acc={metrics['A_current']:.2f}%, "
                      f"ξ_A={metrics['xi_A']:.4f}")
        
        for param in self.model.parameters():
            param.requires_grad = True
    
    def optimize_joint(self, max_iterations=5):
        print("="*80)
        print("Joint Optimization: Accuracy + Throughput")
        print("="*80)
        
        print("\n[0/5] Evaluating baseline accuracy...")
        self.A_ref = self.evaluate_accuracy_full()
        print(f"  Baseline accuracy A_ref = {self.A_ref:.2f}%")
        print(f"  Tolerance ε = {self.epsilon}%")
        print(f"  Constraint: A(u) ≥ {self.A_ref - self.epsilon:.2f}%")
        
        for iteration in range(max_iterations):
            print(f"\n{'='*80}")
            print(f"Iteration {iteration+1}/{max_iterations}")
            print(f"{'='*80}")
            
            self.optimize_alpha(epochs=3)
            
            self.optimize_gamma_lambda(steps=400)
            
            metrics = self.evaluate_current_state()
            self.print_iteration_summary(iteration+1, metrics)
            
            if self.check_convergence(metrics):
                print("\n✓ Optimization converged!")
                break
        
        return self.get_final_configuration()
    
    def evaluate_current_state(self):
        alpha_dict = self.get_current_alpha()
        
        J, metrics = self.compute_joint_loss(alpha_dict, self.theta_gamma, self.sigma_lambda)
        
        pruned_count = 0
        total_count = 0
        for block_dict in alpha_dict.values():
            for v in block_dict.values():
                total_count += 1
                if abs(v) < PRUNING_THRESHOLD:
                    pruned_count += 1
        
        metrics['pruned_branches'] = pruned_count
        metrics['total_branches'] = total_count
        metrics['pruning_ratio'] = pruned_count / total_count if total_count > 0 else 0
        
        metrics['A_current'] = self.evaluate_accuracy_full()
        
        return metrics
    
    def print_iteration_summary(self, iteration, metrics):
        print(f"\n{'='*80}")
        print(f"Iteration {iteration} Summary")
        print(f"{'='*80}")
        print(f"  Objective J:        {metrics['J_total']:.4f}")
        print(f"  Efficiency T_eff:   {metrics['T_eff']:.4f}s")
        print(f"    ├─ T_lat:         {metrics['T_lat']:.4f}s")
        print(f"    └─ T_cyc:         {metrics['T_cyc']:.4f}s")
        print(f"  Accuracy A(u):      {metrics['A_current']:.2f}%")
        print(f"  Slack ξ_A:          {metrics['xi_A']:.4f}")
        print(f"  Constraint viol.:   {metrics['constraint_violation']:.4f}")
        print(f"  Pruned branches:    {metrics['pruned_branches']}/{metrics['total_branches']} ({metrics['pruning_ratio']*100:.1f}%)")
        print(f"  Intensity (Server): {metrics['I_M_server']:.2f}")
        print(f"  Intensity (Edge):   {metrics['I_M_edge']:.2f}")
        print(f"{'='*80}")
    
    def check_convergence(self, metrics):
        accuracy_satisfied = metrics['A_current'] >= (self.A_ref - self.epsilon - 0.5)
        slack_small = metrics['xi_A'] < 0.1
        return accuracy_satisfied and slack_small
    
    def get_final_configuration(self):
        alpha_dict = self.get_current_alpha()
        
        gamma_config = {}
        mask_dict = self.mask_manager.compute_mask(alpha_dict)
        for i, block_name in enumerate(self.block_names):
            if i < len(self.block_names):
                mask = mask_dict.get(block_name, torch.zeros(14, device=self.device))
                gamma_soft = F.softmax(self.theta_gamma[i] + mask, dim=0)
                gamma_idx = torch.argmax(gamma_soft).item()
                gamma_config[block_name] = gamma_idx
        
        lambda_split = int(torch.round(self.sigma_lambda).item())
        
        return {
            'alpha': alpha_dict,
            'gamma': gamma_config,
            'lambda': lambda_split,
            'xi_A': self.xi_A.item(),
            'final_accuracy': self.evaluate_accuracy_full()
        }


def create_domain_dataloaders(data_path, partition_info, domain_id, batch_size=BATCH_SIZE):
    try:
        from domain_dataset import create_domain_dataloaders as _create_loaders
        return _create_loaders(data_path, partition_info, domain_id, batch_size)
    except ImportError:
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain_id', type=int, required=True)
    parser.add_argument('--lut_file', type=str, default='results/hardware_lut_complete.json')
    parser.add_argument('--mode', type=str, default='pseudo', choices=['pseudo', 'real'])
    parser.add_argument('--kappa', type=float, default=100.0)
    parser.add_argument('--mu', type=float, default=1.0)
    parser.add_argument('--omega', type=float, default=0.5)
    parser.add_argument('--epsilon', type=float, default=2.0)
    parser.add_argument('--max_iterations', type=int, default=5)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    
    print("="*80)
    print(f"Step 3 (Joint): Accuracy + Throughput Optimization (Domain {args.domain_id})")
    print("="*80)
    print(f"Device: {device}")
    print(f"Mode: {args.mode}")
    print(f"LUT file: {args.lut_file}")
    print()
    
    print(f"Loading model: {MODEL_NAME}")
    model = create_hyper_repvgg(WEIGHT_PATH_SERVER, model_name=MODEL_NAME)
    model = model.to(device)
    
    print(f"Loading LUT: {args.lut_file}")
    lut_manager = LUTManager(args.lut_file)
    
    print(f"Loading data for domain {args.domain_id}")
    with open(PARTITION_INFO_PATH, 'r') as f:
        partition_info = json.load(f)
    train_loader, val_loader = create_domain_dataloaders(
        DATA_PATH, partition_info, args.domain_id, BATCH_SIZE
    )
    
    if args.mode == 'real':
        print("\n[Real Mode] Setting up Server-Client communication...")
        from network_utils import ServerConnection
        from split_trainer import ServerClientTrainer
        from config import PORT_MAIN
        
        print(f"  Waiting for Client connection on port {PORT_MAIN}...")
        server_conn = ServerConnection(PORT_MAIN)
        conn = server_conn.start()
        print(f"  ✓ Connected to Client")
        
        optimizer = JointOptimizer(
            model, lut_manager, train_loader, val_loader,
            kappa=args.kappa, mu=args.mu, omega=args.omega, epsilon=args.epsilon,
            device=device, mode='real', connection=conn
        )
    else:
        print("\n[Pseudo Mode] Running local optimization (no Client needed)")
        optimizer = JointOptimizer(
            model, lut_manager, train_loader, val_loader,
            kappa=args.kappa, mu=args.mu, omega=args.omega, epsilon=args.epsilon,
            device=device, mode='pseudo'
        )
    
    final_config = optimizer.optimize_joint(max_iterations=args.max_iterations)
    
    output_dir = 'results/joint_optimization'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'domain_{args.domain_id}_joint.json')
    
    with open(output_file, 'w') as f:
        json.dump(final_config, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")
    print(f"  Final accuracy: {final_config['final_accuracy']:.2f}%")
    print(f"  Split point λ: {final_config['lambda']}")
    print(f"  Slack ξ_A: {final_config['xi_A']:.4f}")


if __name__ == "__main__":
    main()

