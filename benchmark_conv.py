import time
import torch
from src.models.video_vae_v3.modules.causal_inflation_lib import InflatedCausalConv3d
from src.models.video_vae_v3.modules.types import MemoryState

def run_benchmark():
    conv = InflatedCausalConv3d(in_channels=16, out_channels=16, kernel_size=3, padding=1, inflation_mode='3d')
    conv.memory_limit = 0.00005  # Force splitting in memory_limit_conv

    x = torch.randn(1, 16, 4, 32, 32)

    # Warmup
    for _ in range(10):
        conv(x, memory_state=MemoryState.DISABLED)

    iterations = 2000
    t0 = time.perf_counter()
    for _ in range(iterations):
        conv(x, memory_state=MemoryState.DISABLED)
    t1 = time.perf_counter()

    elapsed = t1 - t0
    avg_ms = (elapsed / iterations) * 1000
    print(f"Total time for {iterations} iterations: {elapsed:.4f}s")
    print(f"Average time per iteration: {avg_ms:.4f}ms")

if __name__ == "__main__":
    run_benchmark()
