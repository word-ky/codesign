
import torch
import torch.nn as nn
import json
import time
import numpy as np
from datetime import datetime
from collections import OrderedDict
from config import (
    NUM_BLOCKS, BLOCK_INDEX_MAP, INDEX_BLOCK_MAP,
    I_KNEE_SERVER, I_KNEE_CLIENT, CMD
)

class LUTMeasurer:
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.block_input_shapes = {}
        self.block_output_shapes = {}
        
        self._compute_all_input_shapes()
    
    def _compute_all_input_shapes(self):
        from hyper_repvgg import HyperRepVGGBlock
        print("[LUT] Computing input shapes for all blocks...")
        
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        hooks = []
        
        def make_hook(block_name):
            def hook_fn(module, input, output):
                self.block_input_shapes[block_name] = tuple(input[0].shape)
                self.block_output_shapes[block_name] = tuple(output.shape)
            return hook_fn
        
        for stage_name in ['stage1', 'stage2', 'stage3', 'stage4']:
            stage = getattr(self.model, stage_name)
            for block_idx, block in enumerate(stage):
                if isinstance(block, HyperRepVGGBlock):
                    block_name = f"{stage_name}.{block_idx}"
                    handle = block.register_forward_hook(make_hook(block_name))
                    hooks.append(handle)
        
        self.model.eval()
        with torch.no_grad():
            _ = self.model(dummy_input)
        
        for handle in hooks:
            handle.remove()
        
        print(f"[LUT] Computed input shapes for {len(self.block_input_shapes)} blocks")
    
    def benchmark_block(self, block, block_name, gamma_idx, n_warmup=10, n_repeat=100):
        from gamma_configurator import GammaConfigurator

        if not GammaConfigurator.is_gamma_valid_for_block(block, gamma_idx):
            return None, None
        
        GammaConfigurator.apply_gamma_to_block(block, gamma_idx)
        
        input_shape = self.block_input_shapes.get(block_name, None)
        if input_shape is None:
            raise ValueError(f"Missing input shape for block {block_name}, cannot measure latency.")
        dummy_input = torch.randn(*input_shape).to(self.device)
        
        block.eval()
        
        with torch.no_grad():
            for _ in range(n_warmup):
                _ = block(dummy_input)
        
        times = []
        with torch.no_grad():
            for _ in range(n_repeat):
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                
                start = time.perf_counter()
                _ = block(dummy_input)
                
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                
                end = time.perf_counter()
                times.append((end - start) * 1000)
        
        return np.mean(times), np.std(times)
    
    def measure_all_blocks(self):
        from hyper_repvgg import HyperRepVGGBlock
        from gamma_configurator import GammaConfigurator
        results = {}
        
        print(f"\n[LUT] Measuring all blocks on {self.device}...")
        
        for stage_name in ['stage1', 'stage2', 'stage3', 'stage4']:
            stage = getattr(self.model, stage_name)
            
            for block_idx, block in enumerate(stage):
                if not isinstance(block, HyperRepVGGBlock):
                    continue
                
                block_name = f"{stage_name}.{block_idx}"
                results[block_name] = {}
                
                has_identity = (block.rbr_identity is not None)
                print(f"\n  [{block_name}] has_identity: {has_identity}")
                
                for gamma_idx in range(14):
                    gamma_name = GammaConfigurator.get_gamma_name(gamma_idx)
                    
                    avg_time, std_time = self.benchmark_block(
                        block, block_name, gamma_idx
                    )
                    
                    if avg_time is not None:
                        results[block_name][gamma_idx] = {
                            'gamma_name': gamma_name,
                            'time_ms': float(avg_time),
                            'std_ms': float(std_time),
                            'status': 'ok'
                        }
                        print(f"    γ={gamma_idx} ({gamma_name}): {avg_time:.3f} ms")
                    else:
                        results[block_name][gamma_idx] = {
                            'gamma_name': gamma_name,
                            'time_ms': None,
                            'std_ms': None,
                            'status': 'invalid'
                        }
        
        print(f"\n[LUT] Measurement complete: {len(results)} blocks")
        return results

class LUTManager:
    def __init__(self, lut_file=None):
        self.lut = None
        if lut_file:
            self.load(lut_file)
    
    def load(self, lut_file):
        with open(lut_file, 'r') as f:
            self.lut = json.load(f)
        print(f"[LUT] Loaded: {lut_file}")
        return self.lut
    def save(self, lut_file):
        with open(lut_file, 'w') as f:
            json.dump(self.lut, f, indent=2)
        print(f"[LUT] Saved: {lut_file}")
    def get_T_server(self, block_name, gamma_idx):
        if block_name in self.lut['T_server']:
            val = self.lut['T_server'][block_name][gamma_idx]
            return val if val != -1.0 else None
        return None
    def get_T_client(self, block_name, gamma_idx):
        if block_name in self.lut['T_client']:
            val = self.lut['T_client'][block_name][gamma_idx]
            return val if val != -1.0 else None
        return None
    def get_comm_size(self, block_name, n_streams):
        if block_name in self.lut['Comm_size']:
            return self.lut['Comm_size'][block_name][n_streams - 1]
        return 0.0
    def get_intensity(self, block_name, gamma_idx):
        if block_name in self.lut['Intensity']:
            return self.lut['Intensity'][block_name][gamma_idx]
        return 0.0
    def get_n_streams(self, gamma_idx):
        return self.lut['N_stream'][gamma_idx]
    def get_bandwidth(self):
        return self.lut['bandwidth']
    def get_all_block_names(self):
        server_blocks = set(self.lut.get('T_server', {}).keys())
        client_blocks = set(self.lut.get('T_client', {}).keys())
        return sorted(server_blocks | client_blocks, key=lambda x: BLOCK_INDEX_MAP.get(x, 999))
    def _compute_flops_bytes(self, block_name, gamma_idx):
        shapes = self.lut.get('block_shapes', {})
        inp = shapes.get('input', {}).get(block_name)
        out = shapes.get('output', {}).get(block_name)
        
        if inp is None or out is None:
            return 0.0, 1.0
            
        B, Cin, Hin, Win = inp
        _, Cout, Hout, Wout = out
        
        gamma_kernels = {
            0: (['k3'], 1), 1: (['k1'], 1), 2: (['id'], 1),
            3: (['k3', 'k1'], 2), 4: (['k3'], 1), 5: (['k3', 'id'], 2),
            6: (['k3'], 1), 7: (['k1', 'id'], 2), 8: (['k1'], 1),
            9: (['k3', 'k1', 'id'], 3), 10: (['k3', 'id'], 2),
            11: (['k3', 'k1'], 2), 12: (['k1', 'k3'], 2), 13: (['k3'], 1),
        }
        
        kernels, n_streams = gamma_kernels.get(gamma_idx, ([], 1))
        
        flops = 0.0
        for k in kernels:
            if k == 'k3':
                flops += 2.0 * B * Cout * Hout * Wout * Cin * 9
            elif k == 'k1':
                flops += 2.0 * B * Cout * Hout * Wout * Cin
            elif k == 'id':
                flops += 0.0
                
        bytes_rw = n_streams * B * Cin * Hin * Win * 4.0 + B * Cout * Hout * Wout * 4.0
        if bytes_rw == 0: bytes_rw = 1.0
        
        return flops, bytes_rw
    
    def update_for_pruned_alpha(self, alpha_status):

        pass
    
    def to_tensor_dict(self, device='cuda'):
        block_names = self.get_all_block_names()
        num_blocks = len(block_names)

        T_server = torch.zeros(num_blocks, 14, device=device)
        for i, block_name in enumerate(block_names):
            if block_name in self.lut.get('T_server', {}):
                for j in range(14):
                    val = self.lut['T_server'][block_name][j]
                    T_server[i, j] = val if val != -1.0 else 1000.0
        
        T_client = torch.zeros(num_blocks, 14, device=device)
        for i, block_name in enumerate(block_names):
            if block_name in self.lut.get('T_client', {}):
                for j in range(14):
                    val = self.lut['T_client'][block_name][j]
                    T_client[i, j] = val if val != -1.0 else 1000.0
        
        Intensity = torch.zeros(num_blocks, 14, device=device)
        for i, block_name in enumerate(block_names):
            if block_name in self.lut.get('Intensity', {}):
                for j in range(14):
                    Intensity[i, j] = self.lut['Intensity'][block_name][j]
        
        Comm_size = torch.zeros(num_blocks, 3, device=device)
        for i, block_name in enumerate(block_names):
            if block_name in self.lut.get('Comm_size', {}):
                for j in range(3):
                    Comm_size[i, j] = self.lut['Comm_size'][block_name][j]
        
        N_stream = torch.tensor(self.lut['N_stream'], dtype=torch.float32, device=device)
        
        bandwidth = self.lut['bandwidth']['bandwidth_mbps']
        
        FLOPs = torch.zeros(num_blocks, 14, device=device)
        Bytes = torch.zeros(num_blocks, 14, device=device)
        
        for i, block_name in enumerate(block_names):
            for j in range(14):
                f, b = self._compute_flops_bytes(block_name, j)
                FLOPs[i, j] = f
                Bytes[i, j] = b
        
        return {
            'T_server': T_server,
            'T_client': T_client,
            'Intensity': Intensity,
            'Comm_size': Comm_size,
            'N_stream': N_stream,
            'bandwidth': bandwidth,
            'FLOPs': FLOPs,
            'Bytes': Bytes,
            'block_names': block_names
        }

def build_complete_lut(server_results, client_results, comm_model, model_info=None):
    from gamma_configurator import GammaConfigurator
    ERR_MODEL_INFO = "尚未计算实测 Intensity/Comm_size：缺少 model_info（输入/输出 shape）。"
    ERR_BLOCK_SHAPE = "尚未计算实测 Intensity/Comm_size：缺少 {block} 的输入/输出 shape。"

    if model_info is None:
        raise ValueError(ERR_MODEL_INFO)
    block_input_shapes = model_info.get('block_input_shapes', {})
    block_output_shapes = model_info.get('block_output_shapes', {})
    if not block_input_shapes or not block_output_shapes:
        raise ValueError(ERR_MODEL_INFO)

    def _get_shapes(block_name):
        inp = block_input_shapes.get(block_name)
        out = block_output_shapes.get(block_name)
        if inp is None or out is None:
            raise ValueError(ERR_BLOCK_SHAPE.format(block=block_name))
        return inp, out

    def _compute_intensity(block_name, gamma_idx):
        inp, out = _get_shapes(block_name)
        B, Cin, Hin, Win = inp
        _, Cout, Hout, Wout = out

        gamma_kernels = {
            0: (['k3'], 1),
            1: (['k1'], 1),
            2: (['id'], 1),
            3: (['k3', 'k1'], 2),
            4: (['k3'], 1),
            5: (['k3', 'id'], 2),
            6: (['k3'], 1),
            7: (['k1', 'id'], 2),
            8: (['k1'], 1),
            9: (['k3', 'k1', 'id'], 3),
            10: (['k3', 'id'], 2),
            11: (['k3', 'k1'], 2),
            12: (['k1', 'k3'], 2),
            13: (['k3'], 1),
        }
        if gamma_idx not in gamma_kernels:
            raise ValueError(f"Unsupported gamma_idx {gamma_idx}")

        kernels, n_streams = gamma_kernels[gamma_idx]

        flops = 0.0
        for k in kernels:
            if k == 'k3':
                flops += 2.0 * B * Cout * Hout * Wout * Cin * 9
            elif k == 'k1':
                flops += 2.0 * B * Cout * Hout * Wout * Cin
            elif k == 'id':
                flops += 0.0

        bytes_rw = n_streams * B * Cin * Hin * Win * 4.0 + B * Cout * Hout * Wout * 4.0
        if bytes_rw == 0:
            return 0.0
        return float(flops / bytes_rw)

    def _compute_comm_sizes(block_name):
        _, out = _get_shapes(block_name)
        B, C, H, W = out
        base_mb = B * C * H * W * 4.0 / 1e6
        return [base_mb * n for n in (1, 2, 3)]
    lut = {
        'metadata': {
            'build_time': str(datetime.now()),
            'model': 'RepVGG-B3'
        },
        'T_server': {},
        'T_client': {},
        'Comm_size': {},
        'Intensity': {},
        'N_stream': [GammaConfigurator.get_n_streams(i) for i in range(14)],
        'bandwidth': comm_model,
        'roofline': {
            'I_knee_server': I_KNEE_SERVER,
            'I_knee_client': I_KNEE_CLIENT
        },
        'block_shapes': {
            'input': block_input_shapes,
            'output': block_output_shapes
        }
    }

    for block_name, gamma_dict in server_results.items():
        lut['T_server'][block_name] = [
            gamma_dict[str(i)]['time_ms'] if gamma_dict[str(i)].get('status') == 'ok' else -1.0
            for i in range(14)
        ]

    for block_name, gamma_dict in client_results.items():
        lut['T_client'][block_name] = [
            gamma_dict[str(i)]['time_ms'] if gamma_dict[str(i)].get('status') == 'ok' else -1.0
            for i in range(14)
        ]

    all_blocks = sorted(
        set(lut['T_server'].keys()) | set(lut['T_client'].keys()),
        key=lambda x: BLOCK_INDEX_MAP.get(x, 999)
    )

    for block_name in all_blocks:
        lut['Comm_size'][block_name] = _compute_comm_sizes(block_name)
        lut['Intensity'][block_name] = [
            _compute_intensity(block_name, gamma_idx) for gamma_idx in range(14)
        ]

    return lut

def measure_communication_bandwidth(conn, test_sizes_mb=[0.1, 0.5, 1.0, 2.0, 5.0], n_repeat=5):
    from network_utils import send_message, recv_message
    results = []
    
    print("\n[Comm] Measuring communication bandwidth...")
    
    for size_mb in test_sizes_mb:
        times = []
        num_elements = int(size_mb * 262144)
        
        for _ in range(n_repeat):
            tensor = torch.randn(num_elements)
            
            start = time.perf_counter()
            send_message(conn, CMD.FORWARD_DATA, tensor)
            cmd, ack = recv_message(conn)
            end = time.perf_counter()
            
            times.append((end - start) * 1000)
        
        avg_time = np.mean(times)
        results.append({'size_mb': size_mb, 'time_ms': avg_time})
        print(f"  {size_mb:.1f} MB: {avg_time:.2f} ms")
    
    sizes = np.array([r['size_mb'] for r in results])
    times = np.array([r['time_ms'] for r in results])
    k, b = np.polyfit(sizes, times, 1)
    
    bandwidth = 1000 / k if k > 0 else 0.0
    
    print(f"\n[Comm] Model: T = {k:.4f} × size + {b:.2f}")
    print(f"[Comm] Bandwidth: {bandwidth:.2f} MB/s")
    
    return {
        'k': float(k),
        'b': float(b),
        'bandwidth_mbps': float(bandwidth)
    }