import torch
import json
import os
import argparse
from datetime import datetime
from config import MODEL_NAME, WEIGHT_PATH_SERVER

from hyper_repvgg import create_hyper_repvgg
from lut_manager import LUTMeasurer

def measure_server_lut(args):
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    print("="*80)
    print("Step 1: LUT Measurement (Server Side)")
    print("="*80)
    print(f"Device: {device}")
    print(f"Model: {MODEL_NAME}")
    print()
    
    print("[1/3] Creating model...")
    if os.path.exists(WEIGHT_PATH_SERVER):
        model = create_hyper_repvgg(WEIGHT_PATH_SERVER, model_name=MODEL_NAME)
    else:
        print(f"  Warning: Weight file not found, creating structure only...")
        model = create_hyper_repvgg(None, model_name=MODEL_NAME)
    
    model = model.to(device)
    model.eval()
    print(f"  ✓ Model created")
    
    print("\n[2/3] Measuring server-side latency...")
    measurer = LUTMeasurer(model, device=device)
    server_results = measurer.measure_all_blocks()
    
    server_results_formatted = {}
    for block_name, gamma_dict in server_results.items():
        server_results_formatted[block_name] = {
            str(k): v for k, v in gamma_dict.items()
        }
    
    print("\n[3/3] Saving results...")
    output_data = {
        'metadata': {
            'device': 'server',
            'gpu_name': torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU',
            'model': MODEL_NAME,
            'timestamp': str(datetime.now())
        },
        'T_server': server_results_formatted,
        'model_info': {
            'block_input_shapes': measurer.block_input_shapes,
            'block_output_shapes': measurer.block_output_shapes
        }
    }
    
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✓ Server LUT saved to: {args.output}")
    print(f"  Total blocks measured: {len(server_results_formatted)}")
    print("="*80)

def measure_client_lut(args):
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    print("="*80)
    print("Step 1: LUT Measurement (Client Side)")
    print("="*80)
    print(f"Device: {device}")
    print(f"Model: {MODEL_NAME}")
    print()
    
    print("[1/3] Creating model...")

    model = create_hyper_repvgg(None, model_name=MODEL_NAME)
    model = model.to(device)
    model.eval()
    print(f"  ✓ Model created (structure only)")
    
    print("\n[2/3] Measuring client-side latency...")
    measurer = LUTMeasurer(model, device=device)
    client_results = measurer.measure_all_blocks()
    
    client_results_formatted = {}
    for block_name, gamma_dict in client_results.items():
        client_results_formatted[block_name] = {
            str(k): v for k, v in gamma_dict.items()
        }
    
    print("\n[3/3] Saving results...")
    output_data = {
        'metadata': {
            'device': 'client',
            'gpu_name': torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU',
            'model': MODEL_NAME,
            'timestamp': str(datetime.now())
        },
        'T_client': client_results_formatted
    }
    
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✓ Client LUT saved to: {args.output}")
    print(f"  Total blocks measured: {len(client_results_formatted)}")
    print("="*80)

def merge_luts(args):
    print("="*80)
    print("Step 1: LUT Measurement (Merge)")
    print("="*80)
    print(f"Server LUT: {args.server_lut}")
    print(f"Client LUT: {args.client_lut}")
    print()

    print("[1/2] Loading LUTs...")
    if not os.path.exists(args.server_lut):
        raise FileNotFoundError(f"Server LUT not found: {args.server_lut}")
    
    with open(args.server_lut, 'r') as f:
        server_data = json.load(f)
    
    print(f"  ✓ Loaded server LUT")
    
    if not os.path.exists(args.client_lut):
        raise FileNotFoundError(f"Client LUT not found: {args.client_lut}")
    
    with open(args.client_lut, 'r') as f:
        client_data = json.load(f)
    
    print(f"  ✓ Loaded client LUT")
    
    print("\n[2/2] Building complete LUT...")
    from lut_manager import build_complete_lut
    
    comm_model = {
        'bandwidth_mbps': 100.0,
        'latency_ms': 1.0
    }
    
    complete_lut = build_complete_lut(
        server_data['T_server'],
        client_data['T_client'],
        comm_model,
        model_info=server_data.get('model_info', {})
    )
    
    complete_lut['metadata']['server_device'] = server_data['metadata'].get('gpu_name', 'Unknown')
    complete_lut['metadata']['client_device'] = client_data['metadata'].get('gpu_name', 'Unknown')
    complete_lut['metadata']['merge_timestamp'] = str(datetime.now())
    
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    with open(args.output, 'w') as f:
        json.dump(complete_lut, f, indent=2)
    
    print(f"✓ Complete LUT saved to: {args.output}")
    print(f"  Server device: {complete_lut['metadata']['server_device']}")
    print(f"  Client device: {complete_lut['metadata']['client_device']}")
    print("="*80)

def main():
    parser = argparse.ArgumentParser(
        description='Step 1: LUT Measurement'
    )
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['server', 'client', 'merge'],
        help='Measurement mode'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path'
    )
    parser.add_argument(
        '--server_lut',
        type=str,
        default='results/lut_server.json',
        help='Server LUT file (for merge mode)'
    )
    parser.add_argument(
        '--client_lut',
        type=str,
        default='results/lut_client.json',
        help='Client LUT file (for merge mode)'
    )
    parser.add_argument(
        '--cpu',
        action='store_true',
        help='Use CPU instead of GPU'
    )
    
    args = parser.parse_args()
    
    if args.output is None:
        if args.mode == 'server':
            args.output = 'results/lut_server.json'
        elif args.mode == 'client':
            args.output = 'results/lut_client.json'
        else:
            args.output = 'results/hardware_lut_complete.json'
    
    if args.mode == 'server':
        measure_server_lut(args)
    elif args.mode == 'client':
        measure_client_lut(args)
    else:
        merge_luts(args)

if __name__ == '__main__':
    main()