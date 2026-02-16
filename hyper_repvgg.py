import torch
import torch.nn as nn
import torch.nn.functional as F
from repvgg import RepVGGBlock, func_dict 
from collections import OrderedDict
class HyperRepVGGBlock(nn.Module):
    def __init__(self, original_block):
        super(HyperRepVGGBlock, self).__init__()

        self.rbr_dense = original_block.rbr_dense
        self.rbr_1x1 = original_block.rbr_1x1
        self.rbr_identity = original_block.rbr_identity
        self.nonlinearity = original_block.nonlinearity
        
        self.deploy = getattr(original_block, 'deploy', False)
        
        for param in self.rbr_dense.parameters():
            param.requires_grad = False
        for param in self.rbr_1x1.parameters():
            param.requires_grad = False
        if self.rbr_identity is not None:
            for param in self.rbr_identity.parameters():
                param.requires_grad = False
        
        self.hyper_dense = nn.Parameter(torch.ones(1))
        self.hyper_1x1 = nn.Parameter(torch.ones(1))
        self.hyper_identity = nn.Parameter(torch.ones(1))
        
        self.register_buffer('fused_weight', None)
        self.register_buffer('fused_bias', None)
        
        self.active_gamma = None
        self.is_fused = False
    
    def _pad_1x1_to_3x3(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        return F.pad(kernel1x1, (1, 1, 1, 1))
    def _get_identity_kernel(self):
        if self.rbr_identity is None:
            return 0

        input_dim = self.rbr_dense[0].weight.shape[1]
        output_dim = self.rbr_dense[0].weight.shape[0]
        
        if input_dim != output_dim:
            return 0
        
        kernel = torch.zeros((output_dim, input_dim, 3, 3), 
                            dtype=self.rbr_dense[0].weight.dtype,
                            device=self.rbr_dense[0].weight.device)
        
        for i in range(min(input_dim, output_dim)):
            kernel[i, i, 1, 1] = 1.0
        
        return kernel
    
    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):

            kernel = branch[0].weight.data
            running_mean = branch[1].running_mean
            running_var = branch[1].running_var
            gamma = branch[1].weight
            beta = branch[1].bias
            eps = branch[1].eps
        else:

            assert isinstance(branch, nn.BatchNorm2d)
            
            if not hasattr(self, 'id_tensor'):
                input_dim = self.rbr_dense[0].in_channels
                output_dim = self.rbr_dense[0].out_channels
                kernel_value = torch.zeros((output_dim, input_dim, 3, 3), 
                                          dtype=self.rbr_dense[0].weight.dtype,
                                          device=self.rbr_dense[0].weight.device)
                for i in range(min(input_dim, output_dim)):
                    kernel_value[i, i, 1, 1] = 1.0
                self.id_tensor = kernel_value
            
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        
        return kernel * t, beta - running_mean * gamma / std
    
    def fuse_branches(self, gamma_idx):
        self.active_gamma = gamma_idx

        dense_kernel, dense_bias = self._fuse_bn_tensor(self.rbr_dense)
        conv1x1_kernel, conv1x1_bias = self._fuse_bn_tensor(self.rbr_1x1)
        identity_kernel, identity_bias = self._fuse_bn_tensor(self.rbr_identity)
        
        if gamma_idx == 0:
            self.fused_weight = self.hyper_dense.item() * dense_kernel
            self.fused_bias = self.hyper_dense.item() * dense_bias
            self.is_fused = True
            
        elif gamma_idx == 1:

            self.fused_weight = self.hyper_1x1.item() * conv1x1_kernel
            self.fused_bias = self.hyper_1x1.item() * conv1x1_bias
            self.is_fused = True
            
        elif gamma_idx == 2:

            self.fused_weight = self.hyper_identity.item() * identity_kernel[:, :, 1:2, 1:2]
            self.fused_bias = self.hyper_identity.item() * identity_bias
            self.is_fused = True
            
        elif gamma_idx == 4:
            fused = (self.hyper_dense.item() * dense_kernel + 
                    self.hyper_1x1.item() * self._pad_1x1_to_3x3(conv1x1_kernel))
            self.fused_weight = fused
            self.fused_bias = self.hyper_dense.item() * dense_bias + self.hyper_1x1.item() * conv1x1_bias
            self.is_fused = True
            
        elif gamma_idx == 6:
            fused = (self.hyper_dense.item() * dense_kernel + 
                    self.hyper_identity.item() * identity_kernel)
            self.fused_weight = fused
            self.fused_bias = self.hyper_dense.item() * dense_bias + self.hyper_identity.item() * identity_bias
            self.is_fused = True
            
        elif gamma_idx == 8:

            fused_1x1 = (self.hyper_1x1.item() * conv1x1_kernel + 
                        self.hyper_identity.item() * identity_kernel[:, :, 1:2, 1:2])
            self.fused_weight = fused_1x1
            self.fused_bias = self.hyper_1x1.item() * conv1x1_bias + self.hyper_identity.item() * identity_bias
            self.is_fused = True
            
        elif gamma_idx == 10:

            fused_main = (self.hyper_dense.item() * dense_kernel + 
                         self.hyper_1x1.item() * self._pad_1x1_to_3x3(conv1x1_kernel))
            self.fused_weight = fused_main
            self.fused_bias = self.hyper_dense.item() * dense_bias + self.hyper_1x1.item() * conv1x1_bias

            self.is_fused = False
            
        elif gamma_idx == 11:

            fused_main = (self.hyper_dense.item() * dense_kernel + 
                         self.hyper_identity.item() * identity_kernel)
            self.fused_weight = fused_main
            self.fused_bias = self.hyper_dense.item() * dense_bias + self.hyper_identity.item() * identity_bias

            self.is_fused = False
            
        elif gamma_idx == 12:

            fused_1x1 = (self.hyper_1x1.item() * conv1x1_kernel + 
                        self.hyper_identity.item() * identity_kernel[:, :, 1:2, 1:2])
            self.fused_weight = fused_1x1
            self.fused_bias = self.hyper_1x1.item() * conv1x1_bias + self.hyper_identity.item() * identity_bias

            self.is_fused = False
            
        elif gamma_idx == 13:
            fused = (self.hyper_dense.item() * dense_kernel + 
                    self.hyper_1x1.item() * self._pad_1x1_to_3x3(conv1x1_kernel) +
                    self.hyper_identity.item() * identity_kernel)
            self.fused_weight = fused
            self.fused_bias = (self.hyper_dense.item() * dense_bias + 
                              self.hyper_1x1.item() * conv1x1_bias + 
                              self.hyper_identity.item() * identity_bias)
            self.is_fused = True
            
        else:
            self.is_fused = False
    
    def forward(self, inputs):
        if self.deploy:
            return self.nonlinearity(self.rbr_reparam(inputs))
        
        gamma_idx = self.active_gamma
        
        if gamma_idx is None:
            id_out = 0
            if self.rbr_identity is not None:
                id_out = self.hyper_identity * self.rbr_identity(inputs)
            
            dense_out = self.hyper_dense * self.rbr_dense(inputs)
            x1x1_out = self.hyper_1x1 * self.rbr_1x1(inputs)
            
            return self.nonlinearity(dense_out + x1x1_out + id_out)
        
        if self.is_fused and gamma_idx in [0, 4, 6, 13]:
            stride = self.rbr_dense[0].stride
            padding = self.rbr_dense[0].padding
            dilation = self.rbr_dense[0].dilation
            groups = self.rbr_dense[0].groups
            
            out = F.conv2d(inputs, self.fused_weight, self.fused_bias,
                          stride=stride, padding=padding, dilation=dilation, groups=groups)
            return self.nonlinearity(out)
        
        elif self.is_fused and gamma_idx in [1, 2, 8]:
            stride = self.rbr_dense[0].stride

            groups = self.rbr_1x1[0].groups if gamma_idx in [1, 8] else 1
            out = F.conv2d(inputs, self.fused_weight, self.fused_bias,
                          stride=stride, padding=0, dilation=1, groups=groups)
            return self.nonlinearity(out)
    
        elif gamma_idx == 10:
            stride = self.rbr_dense[0].stride
            padding = self.rbr_dense[0].padding
            dilation = self.rbr_dense[0].dilation
            groups = self.rbr_dense[0].groups
            
            fused_out = F.conv2d(inputs, self.fused_weight, self.fused_bias,
                                stride=stride, padding=padding, dilation=dilation, groups=groups)
            
            id_out = 0
            if self.rbr_identity is not None:
                id_out = self.hyper_identity * self.rbr_identity(inputs)
            
            return self.nonlinearity(fused_out + id_out)
        
        elif gamma_idx == 11:
            stride = self.rbr_dense[0].stride
            padding = self.rbr_dense[0].padding
            dilation = self.rbr_dense[0].dilation
            groups = self.rbr_dense[0].groups
            
            fused_out = F.conv2d(inputs, self.fused_weight, self.fused_bias,
                                stride=stride, padding=padding, dilation=dilation, groups=groups)
            
            x1x1_out = self.hyper_1x1 * self.rbr_1x1(inputs)
            
            return self.nonlinearity(fused_out + x1x1_out)
        
        elif gamma_idx == 12:
            stride = self.rbr_dense[0].stride

            groups_1x1 = self.rbr_1x1[0].groups
            fused_out = F.conv2d(inputs, self.fused_weight, self.fused_bias,
                                stride=stride, padding=0, dilation=1, groups=groups_1x1)

            dense_out = self.hyper_dense * self.rbr_dense(inputs)

            return self.nonlinearity(fused_out + dense_out)
        
        else:
            out = 0
            
            if gamma_idx in [3, 5, 9]:
                out = out + self.hyper_dense * self.rbr_dense(inputs)
            
            if gamma_idx in [3, 7, 9]:
                out = out + self.hyper_1x1 * self.rbr_1x1(inputs)
            
            if gamma_idx in [5, 7, 9] and self.rbr_identity is not None:
                out = out + self.hyper_identity * self.rbr_identity(inputs)
            
            return self.nonlinearity(out)
    
    def get_hyper_weights(self):
        return {
            'dense': self.hyper_dense.item(),
            '1x1': self.hyper_1x1.item(),
            'identity': self.hyper_identity.item() if self.rbr_identity is not None else None
        }
    def get_fusion_info(self):
        return {
            'active_gamma': self.active_gamma,
            'is_fused': self.is_fused,
            'has_fused_weight': self.fused_weight is not None
        }

class HyperRepVGG(nn.Module):
    def __init__(self, base_model):
        super(HyperRepVGG, self).__init__()

        self.in_planes = base_model.in_planes
        self.override_groups_map = base_model.override_groups_map
        self.deploy = base_model.deploy
        
        self.stage0 = base_model.stage0
        for param in self.stage0.parameters():
            param.requires_grad = False
        
        self.stage1 = self._replace_blocks(base_model.stage1)
        self.stage2 = self._replace_blocks(base_model.stage2)
        self.stage3 = self._replace_blocks(base_model.stage3)
        self.stage4 = self._replace_blocks(base_model.stage4)
        
        self.gap = base_model.gap
        self.linear = base_model.linear
        for param in self.linear.parameters():
            param.requires_grad = False
    
    def _replace_blocks(self, stage):
        new_stage = nn.Sequential()
        for name, module in stage.named_children():
            if isinstance(module, RepVGGBlock):
                new_stage.add_module(name, HyperRepVGGBlock(module))
            else:
                new_stage.add_module(name, module)
                for param in module.parameters():
                    param.requires_grad = False
        return new_stage
    
    def forward(self, x):
        out = self.stage0(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
    def get_all_hyper_weights(self):
        hyper_weights = OrderedDict()
        stage_names = ['stage1', 'stage2', 'stage3', 'stage4']
        for stage_name in stage_names:
            stage = getattr(self, stage_name)
            for block_idx, module in enumerate(stage):
                if isinstance(module, HyperRepVGGBlock):
                    key = f"{stage_name}.block{block_idx}"
                    hyper_weights[key] = module.get_hyper_weights()
        
        return hyper_weights
    
    def count_hyper_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def create_hyper_repvgg(weight_path, model_name='RepVGG-B3'):
    if model_name not in func_dict:
        raise ValueError(f"Unknown model name: {model_name}. Available: {list(func_dict.keys())}")
    create_func = func_dict[model_name]
    base_model = create_func(deploy=False)
    
    if weight_path is not None:
        print(f"Loading base weights from {weight_path}...")
        checkpoint = torch.load(weight_path, map_location='cpu')
        
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        
        if list(checkpoint.keys())[0].startswith('module.'):
            checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
        
        base_model.load_state_dict(checkpoint)
    else:
        print("Note: Creating HyperRepVGG structure with random initialization (No base weights loaded).")

    base_model.eval()
    
    hyper_model = HyperRepVGG(base_model)
    
    print(f"✓ HyperRepVGG created: {model_name}")
    if weight_path is None:
        print("  (Initialized in structure-only mode)")
        
    print(f"  Output classes: 1000 (original ImageNet)")
    print(f"  Total parameters: {sum(p.numel() for p in hyper_model.parameters()):,}")
    print(f"  Trainable hyper-parameters: {hyper_model.count_hyper_parameters()}")
    print(f"  Frozen base parameters: {sum(p.numel() for p in hyper_model.parameters() if not p.requires_grad):,}")
    
    return hyper_model

if __name__ == "__main__":
    print("="*80)
    print("Testing HyperRepVGG with TRUE FUSION")
    print("="*80)
    
    model = create_hyper_repvgg(None, model_name='RepVGG-B3')
    model.eval()
    
    print("\n" + "="*80)
    print("Testing Fusion Capability")
    print("="*80)
    
    test_block = model.stage1[0]
    
    print("\n[Test 1] γ=3 (并行): dense + 1×1")
    test_block.fuse_branches(3)
    print(f"  Fusion info: {test_block.get_fusion_info()}")
    
    print("\n[Test 2] γ=4 (融合): dense + 1×1 → 3×3")
    test_block.fuse_branches(4)
    print(f"  Fusion info: {test_block.get_fusion_info()}")
    print(f"  Fused weight shape: {test_block.fused_weight.shape if test_block.fused_weight is not None else None}")
    
    print("\n[Test 3] γ=13 (全融合): dense + 1×1 + identity → 3×3")
    test_block.fuse_branches(13)
    print(f"  Fusion info: {test_block.get_fusion_info()}")
    print(f"  Fused weight shape: {test_block.fused_weight.shape if test_block.fused_weight is not None else None}")
    
    print("\n" + "="*80)
    print("Testing Forward Pass")
    print("="*80)
    
    x = torch.randn(2, 3, 224, 224)
    
    with torch.no_grad():

        test_block.fuse_branches(3)
        y_parallel = model(x)
        print(f"✓ Parallel mode (γ=3): output shape {y_parallel.shape}")
        
        test_block.fuse_branches(4)
        y_fused = model(x)
        print(f"✓ Fused mode (γ=4): output shape {y_fused.shape}")
        
        test_block.fuse_branches(13)
        y_all_fused = model(x)
        print(f"✓ All-fused mode (γ=13): output shape {y_all_fused.shape}")
    
    print("\n✓ All tests passed!")