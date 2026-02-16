import torch
import copy
import json
import os
from collections import OrderedDict
from tqdm import tqdm
from config import (
    MAX_PRUNING_ROUNDS, CANDIDATE_POOL_SIZE, EPOCHS_PER_TEST,
    THRESHOLD_PERCENT, L1_LAMBDA, BLOCK_INDEX_MAP
)

def freeze_single_weight(model, weight_name):
    parts = weight_name.split('.')
    module = model
    for part in parts[:-1]:
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    param_name = parts[-1]
    param = getattr(module, param_name)
    
    original_value = param.data.item()
    param.data.fill_(0.0)
    param.requires_grad = False
    
    return param, original_value

def get_hyper_weights(model):
    weights = OrderedDict()
    for name, param in model.named_parameters():
        if 'hyper_' in name:
            weights[name] = param.data.item()
    return weights

def get_alpha_dict(model):
    alpha_dict = {}
    for stage_name in ['stage1', 'stage2', 'stage3', 'stage4']:
        stage = getattr(model, stage_name)
        for block_idx, block in enumerate(stage):
            if hasattr(block, 'hyper_dense'):
                block_name = f"{stage_name}.{block_idx}"
                alpha_dict[block_name] = {
                    'hyper_dense': block.hyper_dense.data.item(),
                    'hyper_1x1': block.hyper_1x1.data.item(),
                    'hyper_identity': block.hyper_identity.data.item() if block.rbr_identity is not None else 0.0
                }
    
    return alpha_dict

def compute_flops_params(pruned_weights, branch_flops):
    pruned_set = set(pruned_weights)
    total_flops = 0
    for weight_name, flops in branch_flops.items():
        if weight_name not in pruned_set:
            total_flops += flops
    
    total_params = sum(flops for name, flops in branch_flops.items() 
                       if name not in pruned_set) / 1e6
    
    return total_flops / 1e9, total_params

def progressive_pruning(
    model,
    trainer,
    train_loader,
    val_loader,
    importance_ranking,
    initial_acc,
    best_acc,
    branch_flops,
    domain_id,
    max_rounds=MAX_PRUNING_ROUNDS,
    candidate_pool_size=CANDIDATE_POOL_SIZE,
    epochs_per_test=EPOCHS_PER_TEST,
    threshold_percent=THRESHOLD_PERCENT,
    device='cuda'
):
    model = model.to(device)
    print(f"\n{'='*80}")
    print(f"Progressive Pruning - Domain {domain_id}")
    print(f"{'='*80}")
    print(f"Initial Acc (hyper=1.0): {initial_acc:.2f}%")
    print(f"Best Acc (Code 1):       {best_acc:.2f}%")
    print(f"Early stopping threshold: {initial_acc:.2f}% - {threshold_percent}% = {initial_acc - threshold_percent:.2f}%")
    
    print(f"\n{'='*80}")
    print(f"Debug 1: Verifying Initial Accuracy")
    print(f"{'='*80}")
    verified_acc = trainer.evaluate(val_loader)
    print(f"Verified initial_acc: {verified_acc:.2f}%")
    print(f"Passed initial_acc:   {initial_acc:.2f}%")
    print(f"Difference:           {abs(verified_acc - initial_acc):.2f}%")
    if abs(verified_acc - initial_acc) > 5.0:
        print(f"⚠️  WARNING: Large discrepancy! Model state may be corrupted.")
    else:
        print(f"✓ Initial accuracy verified.")
    print(f"{'='*80}\n")
    
    initial_flops, initial_params = compute_flops_params([], branch_flops)
    
    candidate_pool = importance_ranking[:candidate_pool_size].copy()
    print(f"Candidate pool initialized with {len(candidate_pool)} weights")
    
    pruned_weights = []
    pruning_history = []
    
    best_acc_so_far = initial_acc
    
    pruning_history.append({
        'round': 0,
        'num_pruned': 0,
        'pruned_weights': [],
        'acc': initial_acc,
        'flops': initial_flops,
        'params': initial_params,
        'acc_drop_from_initial': 0.0,
        'acc_vs_best': initial_acc - best_acc
    })
    
    current_acc = initial_acc
    stopped = False
    final_round = 0
    
    for round_num in range(1, max_rounds + 1):
        print(f"\n{'-'*80}")
        print(f"Round {round_num}/{max_rounds}")
        print(f"{'-'*80}")
        print(f"Candidate pool size: {len(candidate_pool)}")
        print(f"Already pruned: {len(pruned_weights)} weights")
        print(f"Current accuracy: {current_acc:.2f}%")
        
        if len(candidate_pool) == 0:
            print("⚠️ Candidate pool is empty, stopping.")
            break
        
        print(f"\nTesting {len(candidate_pool)} candidates...")
        test_results = []
        
        for i, candidate in enumerate(candidate_pool):
            print(f"\n  [{i+1}/{len(candidate_pool)}] Testing: {candidate}")
            
            temp_model = copy.deepcopy(model)
            temp_model = temp_model.to(device)
            
            temp_pruned = pruned_weights + [candidate]
            
            print(f"\n    {'Debug 3: Checking Frozen Weight':-^76}")
            frozen_param, original_value = freeze_single_weight(temp_model, candidate)
            print(f"    Weight name:      {candidate}")
            print(f"    Original value:   {original_value:.6f}")
            print(f"    After freezing:   {frozen_param.data.item():.6f}")
            print(f"    requires_grad:    {frozen_param.requires_grad}")
            
            print(f"\n    Sample hyper weights (first 5):")
            count = 0
            for name, param in temp_model.named_parameters():
                if 'hyper_' in name and count < 5:
                    print(f"      {name}: {param.data.item():.6f}, grad={param.requires_grad}")
                    count += 1
            print(f"    {'-'*76}\n")
            
            print(f"    {'Debug 2: Checking Model Output Before Training':-^76}")
            temp_model.eval()
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224).to(device)
                dummy_output = temp_model(dummy_input)
                print(f"    Sample output (first 5 logits): {dummy_output[0, :5].cpu().numpy()}")
                print(f"    Output sum:   {dummy_output.sum().item():.4f}")
                print(f"    Output max:   {dummy_output.max().item():.4f}")
                print(f"    Output min:   {dummy_output.min().item():.4f}")
                print(f"    Output mean:  {dummy_output.mean().item():.4f}")
                print(f"    Output std:   {dummy_output.std().item():.4f}")
                
                if torch.isnan(dummy_output).any():
                    print(f"    ⚠️  WARNING: Output contains NaN!")
                if torch.isinf(dummy_output).any():
                    print(f"    ⚠️  WARNING: Output contains Inf!")
                if dummy_output.abs().max() < 1e-6:
                    print(f"    ⚠️  WARNING: Output is nearly zero!")
            print(f"    {'-'*76}\n")
            temp_model.train()
            
            original_model = trainer.model
            
            trainer.model = temp_model
            
            if hasattr(trainer, 'split_model'):
                from split_trainer import SplitModelServer
                trainer.split_model = SplitModelServer(temp_model, trainer.split_point)
            
            print(f"    Training {epochs_per_test} epochs with per-epoch evaluation...")
            epoch_accs, best_acc, best_epoch = trainer.train_with_eval(
                train_loader, val_loader,
                epochs=epochs_per_test,
                pruned_weights=temp_pruned
            )
            
            test_acc = best_acc
            acc_drop_from_initial = initial_acc - test_acc
            acc_drop_from_best_so_far = best_acc_so_far - test_acc
            
            test_results.append({
                'weight': candidate,
                'acc': test_acc,
                'epoch_accs': epoch_accs,
                'best_epoch': best_epoch,
                'acc_drop_from_initial': acc_drop_from_initial,
                'acc_drop_from_best_so_far': acc_drop_from_best_so_far,
                'model_state': temp_model.state_dict()
            })
            
            print(f"    Result: Best Acc = {test_acc:.2f}% (at Epoch {best_epoch+1})")
            print(f"            Drop from initial = {acc_drop_from_initial:+.2f}%")
            print(f"            Drop from best so far = {acc_drop_from_best_so_far:+.2f}%")
            
            trainer.model = original_model
            if hasattr(trainer, 'split_model'):
                from split_trainer import SplitModelServer
                trainer.split_model = SplitModelServer(original_model, trainer.split_point)
            
            del temp_model
            if device == 'cuda':
                torch.cuda.empty_cache()
        
        best_result = max(test_results, key=lambda x: x['acc'])
        
        print(f"\n{'Best Candidate':-^80}")
        print(f"Weight: {best_result['weight']}")
        print(f"Accuracy: {best_result['acc']:.2f}%")
        print(f"Drop from initial: {best_result['acc_drop_from_initial']:+.2f}%")
        print(f"{'-'*80}")
        
        if round_num == 1:
            if best_result['acc_drop_from_initial'] > threshold_percent:
                print(f"\n⚠️ First round: Drop {best_result['acc_drop_from_initial']:.2f}% > threshold {threshold_percent}%")
                print(f"⛔ Stopping pruning without applying any candidate.")
                
                pruning_history.append({
                    'round': round_num,
                    'candidates_tested': len(candidate_pool),
                    'test_results': [{k: v for k, v in r.items() if k != 'model_state'} 
                                     for r in test_results],
                    'best_candidate': best_result['weight'],
                    'best_acc': best_result['acc'],
                    'best_epoch': best_result.get('best_epoch', 0),
                    'epoch_accs': best_result.get('epoch_accs', []),
                    'acc_drop_from_initial': best_result['acc_drop_from_initial'],
                    'acc_drop_from_best_so_far': best_result['acc_drop_from_initial'],
                    'threshold_check': 'fail',
                    'stopped': True,
                    'reason': f"First round drop {best_result['acc_drop_from_initial']:.2f}% exceeds threshold {threshold_percent}%"
                })
                
                stopped = True
                final_round = round_num - 1
                current_acc = initial_acc
                break
            threshold_check = "pass"
        else:

            acc_drop_from_best_so_far = best_acc_so_far - best_result['acc']
            if acc_drop_from_best_so_far > threshold_percent:
                print(f"\n❌ Drop from historical best: {acc_drop_from_best_so_far:.2f}% > {threshold_percent}%")
                print(f"❌ Best candidate acc: {best_result['acc']:.2f}%")
                print(f"❌ Historical best: {best_acc_so_far:.2f}%")
                print(f"❌ Stopping pruning at Round {round_num}")
                print(f"✓ Final pruning rounds: {round_num - 1}")
                
                pruning_history.append({
                    'round': round_num,
                    'candidates_tested': len(candidate_pool),
                    'test_results': [{k: v for k, v in r.items() if k != 'model_state'} 
                                     for r in test_results],
                    'best_candidate': best_result['weight'],
                    'best_acc': best_result['acc'],
                    'best_epoch': best_result.get('best_epoch', 0),
                    'epoch_accs': best_result.get('epoch_accs', []),
                    'acc_drop_from_initial': best_result['acc_drop_from_initial'],
                    'acc_drop_from_best_so_far': acc_drop_from_best_so_far,
                    'threshold_check': 'fail',
                    'stopped': True,
                    'reason': f"Best candidate drops {acc_drop_from_best_so_far:.2f}% from historical best (> {threshold_percent}%)"
                })
                
                stopped = True
                final_round = round_num - 1
                break
            
            threshold_check = "pass"
        
        print(f"\n✅ Applying pruning: {best_result['weight']}")
        print(f"✅ Loading model state from testing (no retraining)")
        
        model.load_state_dict(best_result['model_state'])
        pruned_weights.append(best_result['weight'])
        candidate_pool.remove(best_result['weight'])
        
        next_candidates = [w for w in importance_ranking 
                          if w not in pruned_weights and w not in candidate_pool]
        if next_candidates:
            refilled_candidate = next_candidates[0]
            candidate_pool.append(refilled_candidate)
            print(f"✓ Refilled candidate pool with: {refilled_candidate}")
        else:
            print(f"⚠️ No more candidates available for refill")
        
        current_acc = best_result['acc']
        current_flops, current_params = compute_flops_params(pruned_weights, branch_flops)
        
        best_acc_so_far = max(best_acc_so_far, current_acc)
        acc_drop_from_initial = initial_acc - current_acc
        acc_vs_best = current_acc - best_acc
        
        print(f"\n{'Round ' + str(round_num) + ' Results':-^80}")
        print(f"Pruned: {best_result['weight']}")
        print(f"Total pruned: {len(pruned_weights)} weights")
        print(f"Accuracy: {current_acc:.2f}%")
        print(f"  vs Initial: {-acc_drop_from_initial:+.2f}%")
        print(f"  vs Best:    {acc_vs_best:+.2f}%")
        print(f"FLOPs: {current_flops:.2f} GFLOPs ({(1-current_flops/initial_flops)*100:.1f}% reduction)")
        print(f"{'-'*80}")
        
        pruning_history.append({
            'round': round_num,
            'candidates_tested': len(test_results),
            'test_results': [{k: v for k, v in r.items() if k != 'model_state'} 
                             for r in test_results],
            'best_candidate': best_result['weight'],
            'best_acc': best_result['acc'],
            'best_epoch': best_result.get('best_epoch', 0),
            'epoch_accs': best_result.get('epoch_accs', []),
            'acc_drop_from_initial': best_result['acc_drop_from_initial'],
            'acc_drop_from_best_so_far': best_acc_so_far - best_result['acc'],
            'threshold_check': threshold_check,
            'num_pruned': len(pruned_weights),
            'pruned_weights': pruned_weights.copy(),
            'acc': current_acc,
            'flops': current_flops,
            'params': current_params,
            'acc_vs_best': acc_vs_best,
            'historical_best': best_acc_so_far
        })
        
        final_round = round_num
    
    final_flops, final_params = compute_flops_params(pruned_weights, branch_flops)
    final_acc_drop = initial_acc - current_acc
    final_acc_vs_best = current_acc - best_acc
    
    result = {
        'domain_id': domain_id,
        'initial_acc': initial_acc,
        'best_acc': best_acc,
        'initial_flops': initial_flops,
        'initial_params': initial_params,
        'pruning_history': pruning_history,
        'final_state': {
            'num_pruned': len(pruned_weights),
            'pruned_weights': pruned_weights,
            'final_acc': current_acc,
            'final_flops': final_flops,
            'final_params': final_params,
            'acc_drop_from_initial': final_acc_drop,
            'acc_vs_best': final_acc_vs_best,
            'flops_reduction_percent': (1 - final_flops / initial_flops) * 100,
            'stopped_early': stopped,
            'completed_rounds': final_round
        },
        'final_model_state': model.state_dict(),
        'final_alpha_dict': get_alpha_dict(model)
    }
    
    print(f"\n{'='*80}")
    print(f"Domain {domain_id} Progressive Pruning Complete!")
    print(f"{'='*80}")
    print(f"Initial Acc:        {initial_acc:.2f}%")
    print(f"Best Acc (Code 1):  {best_acc:.2f}%")
    print(f"Final Acc:          {current_acc:.2f}%")
    print(f"  vs Initial:       {-final_acc_drop:+.2f}%")
    print(f"  vs Best:          {final_acc_vs_best:+.2f}%")
    print(f"FLOPs reduction:    {(1 - final_flops / initial_flops) * 100:.1f}%")
    print(f"Completed rounds:   {result['final_state']['completed_rounds']}")
    print(f"{'='*80}\n")
    
    return result