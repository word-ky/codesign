import torch
from collections import OrderedDict
from hyper_repvgg import HyperRepVGGBlock
from gamma_configurator import GammaConfigurator

class GammaMaskBuilder:
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device

        self.branch_structure = self._inspect_branch_structure()
        self.block_names = list(self.branch_structure.keys())

    def _inspect_branch_structure(self):
        branch_structure = OrderedDict()
        stage_names = ['stage1', 'stage2', 'stage3', 'stage4']

        for stage_name in stage_names:
            stage = getattr(self.model, stage_name)

            for block_idx, module in enumerate(stage):
                if isinstance(module, HyperRepVGGBlock):
                    block_name = f"{stage_name}.{block_idx}"

                    has_dense = module.rbr_dense is not None
                    has_1x1 = module.rbr_1x1 is not None
                    has_identity = module.rbr_identity is not None

                    branch_structure[block_name] = {
                        'dense': has_dense,
                        '1x1': has_1x1,
                        'identity': has_identity
                    }

        return branch_structure

    def _convert_pruned_weights_to_branches(self, pruned_weights):
        pruned_branches = {}
        for weight_name in pruned_weights:

            parts = weight_name.split('.')
            if len(parts) < 3:
                continue

            block_name = '.'.join(parts[:2])
            branch_type = parts[2].replace('hyper_', '')

            if block_name not in pruned_branches:
                pruned_branches[block_name] = {
                    'dense': False,
                    '1x1': False,
                    'identity': False
                }

            pruned_branches[block_name][branch_type] = True

        return pruned_branches

    def _compute_effective_branches(self, pruned_branches):
        effective_branches = OrderedDict()
        for block_name in self.block_names:
            structure = self.branch_structure[block_name]
            pruned = pruned_branches.get(block_name, {
                'dense': False,
                '1x1': False,
                'identity': False
            })

            effective = {
                'dense': structure['dense'] and not pruned['dense'],
                '1x1': structure['1x1'] and not pruned['1x1'],
                'identity': structure['identity'] and not pruned['identity']
            }

            effective['count'] = sum([effective['dense'], effective['1x1'], effective['identity']])
            effective_branches[block_name] = effective

        return effective_branches

    def build_mask(self, pruned_weights):
        pruned_branches = self._convert_pruned_weights_to_branches(pruned_weights)

        effective_branches = self._compute_effective_branches(pruned_branches)

        num_blocks = len(self.block_names)
        mask_tensor = torch.zeros(num_blocks, 14, device=self.device)

        for block_idx, block_name in enumerate(self.block_names):
            effective = effective_branches[block_name]

            count = effective['count']
            has_dense = effective['dense']
            has_1x1 = effective['1x1']
            has_identity = effective['identity']

            for gamma_idx in range(14):
                config = GammaConfigurator.GAMMA_CONFIGS[gamma_idx]
                required_branches = config['branches']

                invalid = False

                if 'dense' in required_branches and not has_dense:
                    invalid = True
                if '1x1' in required_branches and not has_1x1:
                    invalid = True
                if 'identity' in required_branches and not has_identity:
                    invalid = True

                required_set = set(required_branches)
                effective_set = set()
                if has_dense:
                    effective_set.add('dense')
                if has_1x1:
                    effective_set.add('1x1')
                if has_identity:
                    effective_set.add('identity')

                if count == 1:

                    if gamma_idx >= 3:
                        invalid = True
                    elif required_set != effective_set:
                        invalid = True

                elif count == 2:

                    if gamma_idx < 3 or gamma_idx > 8:
                        invalid = True
                    elif required_set != effective_set:
                        invalid = True

                elif count == 3:

                    if gamma_idx < 9:
                        invalid = True
                    elif required_set != effective_set:
                        invalid = True

                else:

                    invalid = True

                if invalid:
                    mask_tensor[block_idx, gamma_idx] = float('-inf')

        return mask_tensor

    def get_branch_info(self):
        return {
            'branch_structure': self.branch_structure,
            'block_names': self.block_names,
            'num_blocks': len(self.block_names)
        }
    def print_mask_summary(self, mask_tensor, pruned_weights=None):
        print("\n" + "=" * 80)
        print("Gamma Mask Summary")
        print("=" * 80)
        total_valid = 0
        total_invalid = 0

        if pruned_weights:
            pruned_branches = self._convert_pruned_weights_to_branches(pruned_weights)
            effective_branches = self._compute_effective_branches(pruned_branches)
        else:
            effective_branches = self._compute_effective_branches({})

        count_distribution = {1: 0, 2: 0, 3: 0}
        for effective in effective_branches.values():
            if effective['count'] > 0:
                count_distribution[effective['count']] += 1

        print(f"\nEffective Branch Distribution:")
        print(f"  Single-branch blocks:  {count_distribution[1]} (should use gamma 0-2)")
        print(f"  Dual-branch blocks:    {count_distribution[2]} (should use gamma 3-8)")
        print(f"  Triple-branch blocks:  {count_distribution[3]} (should use gamma 9-13)")

        for i in range(len(self.block_names)):
            valid = (mask_tensor[i] != float('-inf')).sum().item()
            total_valid += valid
            total_invalid += (14 - valid)

        print(f"\nGamma Configuration Statistics:")
        print(f"  Total valid configs:   {total_valid}")
        print(f"  Total invalid configs: {total_invalid}")
        print(f"  Average per block:     {total_valid / len(self.block_names):.2f} / 14")

        print(f"\nDetailed Info (first 5 blocks):")
        print(f"{'Block':<12} {'Structure':<15} {'Effective':<15} {'Valid Gamma'}")
        print("-" * 80)

        for i, block_name in enumerate(self.block_names[:5]):
            structure = self.branch_structure[block_name]
            effective = effective_branches[block_name]

            def format_branches(branches):
                parts = []
                if branches.get('dense', False):
                    parts.append('D')
                if branches.get('1x1', False):
                    parts.append('1')
                if branches.get('identity', False):
                    parts.append('I')
                return '+'.join(parts) if parts else 'None'

            structure_str = format_branches(structure)
            effective_str = format_branches(effective)

            valid_gammas = (mask_tensor[i] != float('-inf')).nonzero(as_tuple=True)[0].tolist()
            valid_str = ','.join(map(str, valid_gammas))

            print(f"{block_name:<12} {structure_str:<15} {effective_str:<15} {valid_str}")

        print("=" * 80)

if __name__ == '__main__':
    import os
    import json
    from config import MODEL_NAME, WEIGHT_PATH_SERVER
    from hyper_repvgg import create_hyper_repvgg
    print("Testing Gamma Mask Builder")
    print("=" * 80)

    if os.path.exists(WEIGHT_PATH_SERVER):
        model = create_hyper_repvgg(WEIGHT_PATH_SERVER, model_name=MODEL_NAME)
    else:
        print("Warning: Using structure-only model")
        model = create_hyper_repvgg(None, model_name=MODEL_NAME)

    model.eval()

    builder = GammaMaskBuilder(model, device='cpu')

    print("\n[Test 1] No Pruning")
    mask_no_pruning = builder.build_mask([])
    builder.print_mask_summary(mask_no_pruning)

    accuracy_file = 'results/accuracy_results/domain_0_accuracy.json'
    if os.path.exists(accuracy_file):
        with open(accuracy_file, 'r') as f:
            accuracy_data = json.load(f)

        pruned_weights = accuracy_data['final_state']['pruned_weights']

        print(f"\n[Test 2] With Pruning ({len(pruned_weights)} weights pruned)")
        mask_with_pruning = builder.build_mask(pruned_weights)
        builder.print_mask_summary(mask_with_pruning, pruned_weights)

        print("\n[Key Test Cases]")
        if 'stage1.0' in builder.block_names:
            idx = builder.block_names.index('stage1.0')
            valid = (mask_with_pruning[idx] != float('-inf')).nonzero(as_tuple=True)[0].tolist()
            print(f"stage1.0 valid gamma: {valid} (expected: [3, 4])")

        if 'stage1.1' in builder.block_names:
            idx = builder.block_names.index('stage1.1')
            valid = (mask_with_pruning[idx] != float('-inf')).nonzero(as_tuple=True)[0].tolist()

            pruned_branches = builder._convert_pruned_weights_to_branches(pruned_weights)
            if 'stage1.1' in pruned_branches and pruned_branches['stage1.1']['dense']:
                print(f"stage1.1 valid gamma: {valid} (expected: [7, 8] after dense pruning)")
            else:
                print(f"stage1.1 valid gamma: {valid}")

    print("\n" + "=" * 80)
    print("Test completed!")