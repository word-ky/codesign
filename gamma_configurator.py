import torch
import torch.nn as nn
class GammaConfigurator:

    GAMMA_CONFIGS = {
        0: {'name': 'SOLO_K3',    'branches': ['dense'],                        'n_streams': 1, 'fuse': True},
        1: {'name': 'SOLO_K1',    'branches': ['1x1'],                          'n_streams': 1, 'fuse': True},
        2: {'name': 'SOLO_ID',    'branches': ['identity'],                     'n_streams': 1, 'fuse': True},
        3: {'name': 'PARA_3_1',   'branches': ['dense', '1x1'],                 'n_streams': 2, 'fuse': False},
        4: {'name': 'FUSED_3_1',  'branches': ['dense', '1x1'],                 'n_streams': 1, 'fuse': True},
        5: {'name': 'PARA_3_I',   'branches': ['dense', 'identity'],            'n_streams': 2, 'fuse': False},
        6: {'name': 'FUSED_3_I',  'branches': ['dense', 'identity'],            'n_streams': 1, 'fuse': True},
        7: {'name': 'PARA_1_I',   'branches': ['1x1', 'identity'],              'n_streams': 2, 'fuse': False},
        8: {'name': 'FUSED_1_I',  'branches': ['1x1', 'identity'],              'n_streams': 1, 'fuse': True},
        9: {'name': 'PARA_ALL',   'branches': ['dense', '1x1', 'identity'],     'n_streams': 3, 'fuse': False},
        10: {'name': 'PART_31_I', 'branches': ['dense', '1x1', 'identity'],     'n_streams': 2, 'fuse': 'partial'},
        11: {'name': 'PART_3I_1', 'branches': ['dense', '1x1', 'identity'],     'n_streams': 2, 'fuse': 'partial'},
        12: {'name': 'PART_1I_3', 'branches': ['dense', '1x1', 'identity'],     'n_streams': 2, 'fuse': 'partial'},
        13: {'name': 'FUSED_ALL', 'branches': ['dense', '1x1', 'identity'],     'n_streams': 1, 'fuse': True}
    }
    
    @staticmethod
    def apply_gamma_to_block(block, gamma_idx):
        config = GammaConfigurator.GAMMA_CONFIGS[gamma_idx]
        branches = config['branches']

        has_identity = (block.rbr_identity is not None)
        
        if 'identity' in branches and not has_identity:
            branches = [b for b in branches if b != 'identity']
            
            if not branches:
                branches = ['dense']
        
        block.hyper_dense.data.fill_(0.0)
        block.hyper_1x1.data.fill_(0.0)
        block.hyper_identity.data.fill_(0.0)
        
        fuse_mode = config['fuse']
        
        if fuse_mode == False:
            if 'dense' in branches:
                block.hyper_dense.data.fill_(1.0)
            if '1x1' in branches:
                block.hyper_1x1.data.fill_(1.0)
            if 'identity' in branches and has_identity:
                block.hyper_identity.data.fill_(1.0)
                
        elif fuse_mode == True:

            if 'dense' in branches:
                block.hyper_dense.data.fill_(1.0)
            if '1x1' in branches:
                block.hyper_1x1.data.fill_(1.0)
            if 'identity' in branches and has_identity:
                block.hyper_identity.data.fill_(1.0)
                
        elif fuse_mode == 'partial':

            if 'dense' in branches:
                block.hyper_dense.data.fill_(1.0)
            if '1x1' in branches:
                block.hyper_1x1.data.fill_(1.0)
            if 'identity' in branches and has_identity:
                block.hyper_identity.data.fill_(1.0)
        
        if hasattr(block, 'fuse_branches'):
            block.fuse_branches(gamma_idx)
        else:

            block.active_gamma = gamma_idx
    
    @staticmethod
    def is_gamma_valid_for_block(block, gamma_idx):
        config = GammaConfigurator.GAMMA_CONFIGS[gamma_idx]
        branches = config['branches']
        has_identity = (block.rbr_identity is not None)

        if 'identity' in branches and not has_identity:
            return False
        
        return True
    
    @staticmethod
    def get_gamma_name(gamma_idx):
        return GammaConfigurator.GAMMA_CONFIGS[gamma_idx]['name']
    @staticmethod
    def get_n_streams(gamma_idx):
        return GammaConfigurator.GAMMA_CONFIGS[gamma_idx]['n_streams']
    @staticmethod
    def is_fused(gamma_idx):
        return GammaConfigurator.GAMMA_CONFIGS[gamma_idx]['fuse'] != False
    @staticmethod
    def get_fusion_type(gamma_idx):
        return GammaConfigurator.GAMMA_CONFIGS[gamma_idx]['fuse']
    @staticmethod
    def get_valid_gammas_for_block(block):
        valid_gammas = []
        for gamma_idx in range(14):
            if GammaConfigurator.is_gamma_valid_for_block(block, gamma_idx):
                valid_gammas.append(gamma_idx)
        return valid_gammas
    
    @staticmethod
    def print_gamma_info(gamma_idx):
        config = GammaConfigurator.GAMMA_CONFIGS[gamma_idx]
        print(f"γ={gamma_idx} ({config['name']})")
        print(f"  Branches: {config['branches']}")
        print(f"  N_streams: {config['n_streams']}")
        print(f"  Fusion: {config['fuse']}")

if __name__ == "__main__":
    print("="*80)
    print("Gamma Configurator - Configuration Overview")
    print("="*80)
    
    print("\n【单分支配置】")
    for gamma_idx in [0, 1, 2]:
        GammaConfigurator.print_gamma_info(gamma_idx)
        print()
    
    print("\n【并行配置 (n_streams > 1, fuse=False)】")
    for gamma_idx in [3, 5, 7, 9]:
        GammaConfigurator.print_gamma_info(gamma_idx)
        print()
    
    print("\n【完全融合配置 (n_streams=1, fuse=True)】")
    for gamma_idx in [4, 6, 8, 13]:
        GammaConfigurator.print_gamma_info(gamma_idx)
        print()
    
    print("\n【部分融合配置 (n_streams=2, fuse='partial')】")
    for gamma_idx in [10, 11, 12]:
        GammaConfigurator.print_gamma_info(gamma_idx)
        print()
    
    print("="*80)
    print("✓ Gamma configurations loaded successfully")
    print("="*80)