
import time
import sys
import json
from datetime import datetime
import torch

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        props = torch.cuda.get_device_properties(0)
        info = {
            "name": torch.cuda.get_device_name(0),
            "major": props.major,
            "minor": props.minor,
            "total_memory_gb": round(props.total_memory / 1024**3, 2),
            "multi_processor_count": props.multi_processor_count,
            "clock_rate_mhz": round(getattr(props, "clockRate", 0) / 1000, 2),
            "memory_clock_rate_mhz": round(getattr(props, "memoryClockRate", 0) / 1000, 2),
            "memory_bus_width_bits": getattr(props, "memoryBusWidth", 0),
        }
    else:
        device = torch.device("cpu")
        info = {"name": "CPU", "note": "CUDA not available"}
    return device, info

def measure_peak_performance(device, iterations=100, matrix_size=4096):
    if device.type == "cpu":
        matrix_size = min(matrix_size, 2048)
    A = torch.randn(matrix_size, matrix_size, dtype=torch.float32, device=device)
    B = torch.randn(matrix_size, matrix_size, dtype=torch.float32, device=device)

    for _ in range(5):
        _ = torch.matmul(A, B)
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.time()
    for _ in range(iterations):
        C = torch.matmul(A, B)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start

    flops_per_matmul = 2 * matrix_size ** 3
    total_flops = flops_per_matmul * iterations
    peak_gflops = total_flops / elapsed / 1e9

    del A, B, C
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return peak_gflops, elapsed

def measure_memory_bandwidth(device, iterations=100, array_size=100_000_000):
    if device.type == "cpu":
        array_size = min(array_size, 10_000_000)
    A = torch.randn(array_size, dtype=torch.float32, device=device)
    B = torch.empty(array_size, dtype=torch.float32, device=device)
    for _ in range(5):
        B.copy_(A)
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.time()
    for _ in range(iterations):
        B.copy_(A)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start

    bytes_per_copy = array_size * 4 * 2
    total_bytes = bytes_per_copy * iterations
    bandwidth_gbs = total_bytes / elapsed / 1e9

    del A, B
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return bandwidth_gbs, elapsed

def calculate_i_knee(peak_gflops, bandwidth_gbs):
    return peak_gflops / bandwidth_gbs if bandwidth_gbs > 0 else 0.0

def save_results(filename, device_info, peak, bw, i_knee, extras=None):
    data = {
        "timestamp": datetime.now().isoformat(),
        "device_info": device_info,
        "peak_performance_gflops": round(peak, 2),
        "memory_bandwidth_gbs": round(bw, 2),
        "i_knee_flops_per_byte": round(i_knee, 2),
    }
    if extras:
        data.update(extras)
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    print(f"结果已保存: {filename}")

def main():
    print("\nGPU/CPU Roofline 参数测量")
    device, info = get_device()
    print(f"使用设备: {info}")

    print("\n测 Peak Performance ...")
    peak, t_peak = measure_peak_performance(device)
    print(f"Peak: {peak:.2f} GFLOPS (耗时 {t_peak:.2f}s)")

    print("\n测 Memory Bandwidth ...")
    bw, t_bw = measure_memory_bandwidth(device)
    print(f"Bandwidth: {bw:.2f} GB/s (耗时 {t_bw:.2f}s)")

    i_knee = calculate_i_knee(peak, bw)
    print(f"\nI_knee: {i_knee:.2f} FLOPs/Byte")

    save_results("roofline_measurements.json", info, peak, bw, i_knee)

if __name__ == "__main__":
    main()