import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from config import (
    NUM_BLOCKS, BLOCK_INDEX_MAP, INDEX_BLOCK_MAP,
    SPLIT_SIGMA, FIXED_SPLIT_POINT,
    LR_GAMMA, LR_LAMBDA, THROUGHPUT_OPTIM_STEPS,
    PRUNING_THRESHOLD,

    OMEGA_ROOFLINE_CLIENT, OMEGA_ROOFLINE_SERVER,
    OMEGA_TIME_SERVER, OMEGA_TIME_CLIENT, OMEGA_TRANS,
    OMEGA_INTENSITY,

    I_KNEE_SERVER, I_KNEE_CLIENT,
    ALPHA_ROOFLINE_LOWER, BETA_ROOFLINE_UPPER,
    CLIENT_INTENSITY_UPPER_RATIO, SERVER_INTENSITY_UPPER_RATIO
)
from gamma_configurator import GammaConfigurator
from intensity_analyzer import compute_flops_bytes

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
    
    def get_valid_gamma_count(self, mask_dict):
        counts = {}
        for block_name, mask in mask_dict.items():
            counts[block_name] = (mask != float('-inf')).sum().item()
        return counts

class ThroughputOptimizer:
    def __init__(self, lut_manager, mask_manager, device='cuda', fixed_lambda=None):
        self.lut_manager = lut_manager
        self.mask_manager = mask_manager
        self.device = device
        self.fixed_lambda = fixed_lambda

        self.lut_tensors = lut_manager.to_tensor_dict(device)
        self.block_names = self.lut_tensors['block_names']
        self.num_blocks = len(self.block_names)

        self.theta_gamma = nn.Parameter(
            torch.zeros(self.num_blocks, 14, device=device)
        )

        if fixed_lambda is not None:

            self.lambda_param = torch.tensor(float(fixed_lambda), device=device)
            self.lambda_is_learnable = False
        else:

            self.lambda_param = nn.Parameter(
                torch.tensor(float(FIXED_SPLIT_POINT), device=device)
            )
            self.lambda_is_learnable = True

        if self.lambda_is_learnable:
            self.optimizer = torch.optim.Adam([
                {'params': [self.theta_gamma], 'lr': LR_GAMMA},
                {'params': [self.lambda_param], 'lr': LR_LAMBDA}
            ])
        else:
            self.optimizer = torch.optim.Adam([
                {'params': [self.theta_gamma], 'lr': LR_GAMMA}
            ])

        self.current_mask = None
    
    def update_mask(self, alpha_dict):
        mask_dict = self.mask_manager.compute_mask(alpha_dict)

        mask_tensor = torch.zeros(self.num_blocks, 14, device=self.device)
        for i, block_name in enumerate(self.block_names):
            if block_name in mask_dict:
                mask_tensor[i] = mask_dict[block_name]
        
        self.current_mask = mask_tensor
        return mask_dict
    
    def compute_soft_gamma(self):
        if self.current_mask is None:
            mask = torch.zeros(self.num_blocks, 14, device=self.device)
        else:
            mask = self.current_mask
        
        gamma_soft = F.softmax(self.theta_gamma + mask, dim=-1)
        return gamma_soft
    
    def compute_soft_split(self):

        k = torch.arange(self.num_blocks, dtype=torch.float32, device=self.device)
        z = torch.sigmoid(SPLIT_SIGMA * (k - self.lambda_param))
        return z
    
    def compute_loss_roofline_client(self, gamma_soft, z):
        FLOPs = self.lut_tensors['FLOPs']
        Bytes = self.lut_tensors['Bytes']

        FLOPs_expect = (gamma_soft * FLOPs).sum(dim=-1)
        Bytes_expect = (gamma_soft * Bytes).sum(dim=-1)
        
        FLOPs_client = (z * FLOPs_expect).sum()
        Bytes_client = (z * Bytes_expect).sum() + 1e-6
        I_client = FLOPs_client / Bytes_client
        
        UPPER_BOUND = I_KNEE_CLIENT * CLIENT_INTENSITY_UPPER_RATIO
        
        loss_lower = ALPHA_ROOFLINE_LOWER * F.relu(I_KNEE_CLIENT - I_client) ** 2

        loss_upper = BETA_ROOFLINE_UPPER * F.relu(I_client - UPPER_BOUND) ** 2
        
        return loss_lower + loss_upper
    
    def compute_loss_roofline_server(self, gamma_soft, z):
        FLOPs_expect = (gamma_soft * self.lut_tensors['FLOPs']).sum(dim=-1)
        Bytes_expect = (gamma_soft * self.lut_tensors['Bytes']).sum(dim=-1)

        FLOPs_server = ((1 - z) * FLOPs_expect).sum()
        Bytes_server = ((1 - z) * Bytes_expect).sum() + 1e-6
        I_server = FLOPs_server / Bytes_server

        UPPER_BOUND = I_KNEE_SERVER * SERVER_INTENSITY_UPPER_RATIO
        loss = F.relu(I_server - UPPER_BOUND) ** 2

        return loss

    def compute_loss_intensity(self, gamma_soft, z):
        lut = self.lut_manager.lut

        total_loss = 0.0

        for s in range(self.num_blocks):
            block_name = self.block_names[s]

            flops_list = []
            bytes_list = []

            for gamma_idx in range(14):
                flops, bytes_val = compute_flops_bytes(lut, block_name, gamma_idx)
                flops_list.append(flops)
                bytes_list.append(bytes_val)

            flops_tensor = torch.tensor(flops_list, dtype=torch.float32, device=self.device)
            bytes_tensor = torch.tensor(bytes_list, dtype=torch.float32, device=self.device)

            expected_flops = (gamma_soft[s] * flops_tensor).sum()
            expected_bytes = (gamma_soft[s] * bytes_tensor).sum()

            I_s = expected_flops / (expected_bytes + 1e-8)

            server_weight = 1.0 - z[s]

            penalty = F.relu(I_KNEE_SERVER - I_s)

            w_s = 1.0
            total_loss = total_loss + server_weight * w_s * penalty

        return total_loss

    def compute_loss_time_server(self, gamma_soft, z):
        T_server = self.lut_tensors['T_server']

        T_expect = (gamma_soft * T_server).sum(dim=-1)
        
        loss = ((1 - z) * T_expect).sum()
        
        return loss
    
    def compute_loss_time_client(self, gamma_soft, z):
        T_client = self.lut_tensors['T_client']
        T_expect = (gamma_soft * T_client).sum(dim=-1)
        
        loss = (z * T_expect).sum()
        
        return loss
    
    def compute_loss_trans(self, gamma_soft, z):
        Comm_size = self.lut_tensors['Comm_size']
        N_stream = self.lut_tensors['N_stream']
        bandwidth = self.lut_tensors['bandwidth']

        n_streams_expect = (gamma_soft * N_stream.unsqueeze(0)).sum(dim=-1)
        
        z_diff = z[1:] - z[:-1]
        z_diff = F.relu(z_diff)
        
        comm_cost = torch.zeros(self.num_blocks, device=self.device)
        for k in range(self.num_blocks):
            n = n_streams_expect[k]

            n_floor = torch.floor(n).long().clamp(1, 3) - 1
            n_ceil = torch.ceil(n).long().clamp(1, 3) - 1
            w = n - torch.floor(n)
            
            if n_floor == n_ceil:
                comm_cost[k] = Comm_size[k, n_floor]
            else:
                comm_cost[k] = (1 - w) * Comm_size[k, n_floor] + w * Comm_size[k, n_ceil]
        
        trans_time = comm_cost / (bandwidth + 1e-6)
        
        loss = (z_diff * trans_time[:-1]).sum()
        
        return loss
    
    def compute_total_loss(self):
        gamma_soft = self.compute_soft_gamma()
        z = self.compute_soft_split()

        loss_roofline_client = self.compute_loss_roofline_client(gamma_soft, z)
        loss_roofline_server = self.compute_loss_roofline_server(gamma_soft, z)
        loss_time_server = self.compute_loss_time_server(gamma_soft, z)
        loss_time_client = self.compute_loss_time_client(gamma_soft, z)
        loss_trans = self.compute_loss_trans(gamma_soft, z)
        loss_intensity = self.compute_loss_intensity(gamma_soft, z)

        total_loss = (
            OMEGA_ROOFLINE_CLIENT * loss_roofline_client +
            OMEGA_ROOFLINE_SERVER * loss_roofline_server +
            OMEGA_TIME_SERVER * loss_time_server +
            OMEGA_TIME_CLIENT * loss_time_client +
            OMEGA_TRANS * loss_trans +
            OMEGA_INTENSITY * loss_intensity
        )

        return {
            'total': total_loss,
            'roofline_client': loss_roofline_client.item(),
            'roofline_server': loss_roofline_server.item(),
            'time_server': loss_time_server.item(),
            'time_client': loss_time_client.item(),
            'trans': loss_trans.item(),
            'intensity': loss_intensity.item()
        }
    
    def optimize_step(self):
        self.optimizer.zero_grad()
        losses = self.compute_total_loss()
        losses['total'].backward()
        
        self.optimizer.step()
        
        return losses
    
    def optimize(self, n_steps=THROUGHPUT_OPTIM_STEPS, verbose=True):
        history = []
        for step in range(n_steps):
            losses = self.optimize_step()
            history.append({
                'step': step,
                'total': losses['total'].item(),
                'roofline_client': losses['roofline_client'],
                'roofline_server': losses['roofline_server'],
                'time_server': losses['time_server'],
                'time_client': losses['time_client'],
                'trans': losses['trans'],
                'intensity': losses['intensity']
            })

            if verbose and (step + 1) % 20 == 0:
                print(f"  Step {step+1}/{n_steps}: Loss={losses['total'].item():.4f} "
                      f"(T_s={losses['time_server']:.2f}, T_c={losses['time_client']:.2f}, "
                      f"L_I={losses['intensity']:.2f})")

        return history
    
    def get_optimal_gamma(self):
        gamma_soft = self.compute_soft_gamma()
        gamma_hard = gamma_soft.argmax(dim=-1)
        result = OrderedDict()
        for i, block_name in enumerate(self.block_names):
            result[block_name] = gamma_hard[i].item()
        
        return result
    
    def get_optimal_lambda(self):
        if self.lambda_is_learnable:
            return int(round(self.lambda_param.item()))
        else:
            return int(round(self.lambda_param.item())) if isinstance(self.lambda_param, torch.Tensor) else int(self.fixed_lambda)
    def get_gamma_distribution(self):
        gamma_soft = self.compute_soft_gamma()
        result = OrderedDict()
        for i, block_name in enumerate(self.block_names):
            result[block_name] = gamma_soft[i].detach().cpu().numpy().tolist()
        
        return result
    
    def get_throughput_constant(self):
        with torch.no_grad():
            losses = self.compute_total_loss()
        return losses['total'].item()