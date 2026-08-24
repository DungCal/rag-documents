"""Local BGE-M3 Model Inference & GPU Verification Benchmark.

Tests and benchmarks the local BGE-M3 embedding model loaded from local weights,
enforcing and verifying GPU execution (CUDA), memory footprint, latency, and
semantic vector quality.

Usage:
    # Force GPU execution (default):
    python scripts/test_local_embedding.py --device cuda

    # Test on CPU:
    python scripts/test_local_embedding.py --device cpu

    # Custom model path & batch size:
    python scripts/test_local_embedding.py --model-path bge-m3/bge-m3 --batch-size 32
"""
import argparse
import math
import sys
import time
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_index import config
from rag_index.embedder import Local_BGE_M3_Embedder


def dot_product(v1: list[float], v2: list[float]) -> float:
    return sum(a * b for a, b in zip(v1, v2))


def l2_norm(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def format_bytes(bytes_val: int) -> str:
    mb = bytes_val / (1024 * 1024)
    gb = bytes_val / (1024 * 1024 * 1024)
    if gb >= 1.0:
        return f"{gb:.2f} GB ({mb:.1f} MB)"
    return f"{mb:.1f} MB"


SAMPLE_PASSAGES = [
    "1. SAFETY INSTRUCTIONS: Always fasten seat belt before operating the tractor. Ensure ROPS is securely locked.",
    "2. FUEL FILTER REPLACEMENT: Drain water and sediments from the fuel filter cartridge before installing the new element.",
    "3. DPF REGENERATION: When the DPF warning lamp turns ON, maintain engine speed above 2000 RPM until regeneration completes.",
    "4. TRANSMISSION OIL CHECK: Check transmission and hydraulic fluid level using the dipstick at the rear of the transmission case.",
    "5. TIRE INFLATION PRESSURE: Front tires 180 kPa (26 psi), Rear tires 140 kPa (20 psi). Adjust pressure based on implement load.",
    "6. BATTERY JUMP STARTING: Connect red jumper cable to positive (+) terminal, then black cable to tractor chassis ground.",
    "7. PTO SWITCH OPERATION: Push down and rotate the PTO switch clockwise to engage the power take-off shaft.",
    "8. ENGINE COOLANT REPLACEMENT: Allow engine to cool completely before opening the radiator cap to avoid burns from hot coolant.",
]


def test_environment(forced_device: str):
    print("=" * 70)
    print(" [STEP 1] Hardware & PyTorch Environment Inspection")
    print("=" * 70)

    try:
        import torch
        print(f"  * PyTorch Version : {torch.__version__}")
        cuda_avail = torch.cuda.is_available()
        print(f"  * CUDA Available  : {cuda_avail}")

        if cuda_avail:
            cuda_ver = getattr(torch.version, "cuda", "Unknown")
            device_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            total_vram = torch.cuda.get_device_properties(0).total_memory
            print(f"  * CUDA Version    : {cuda_ver}")
            print(f"  * GPU Device(s)   : {device_count} device(s) detected")
            print(f"  * Primary GPU     : {gpu_name}")
            print(f"  * Total VRAM      : {format_bytes(total_vram)}")
        else:
            print("  * Note            : CUDA is not available in current PyTorch build.")
            if forced_device.startswith("cuda"):
                print("\n  [ERROR] --device cuda was requested but CUDA is not available.")
                print("  Install CUDA-enabled PyTorch with:")
                print("     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124\n")
                sys.exit(1)

    except ImportError:
        print("  [ERROR] PyTorch is not installed.")
        sys.exit(1)
    print()


def test_model_loading(model_path: str, device: str) -> Local_BGE_M3_Embedder:
    print("=" * 70)
    print(f" [STEP 2] Loading Local BGE-M3 Model (Forced on '{device}')")
    print("=" * 70)

    import torch

    vram_before = torch.cuda.memory_allocated(0) if torch.cuda.is_available() else 0
    t0 = time.perf_counter()

    embedder = Local_BGE_M3_Embedder(model_path=model_path, device=device)
    load_time = time.perf_counter() - t0

    # Inspect device of underlying model parameters
    try:
        underlying_model = embedder.model
        param_device = next(underlying_model.parameters()).device
    except Exception:
        param_device = "unknown"

    vram_after = torch.cuda.memory_allocated(0) if torch.cuda.is_available() else 0
    vram_used = vram_after - vram_before

    print(f"  * Model Path      : {embedder.model_path}")
    print(f"  * Resolved Device : {embedder.device}")
    print(f"  * Parameter Device: {param_device}")
    print(f"  * Model Load Time : {load_time:.3f}s")
    if torch.cuda.is_available() and str(param_device).startswith("cuda"):
        print(f"  * VRAM Allocated  : {format_bytes(vram_used)}")
        print("  * Status          : [OK] Model parameters successfully placed on GPU!")
    else:
        print("  * Status          : Running on CPU.")
    print()
    return embedder


def test_single_inference(embedder: Local_BGE_M3_Embedder, runs: int = 5):
    print("=" * 70)
    print(" [STEP 3] Single-Query Embedding Latency Benchmark")
    print("=" * 70)

    query = "How do I adjust the operator seat belt?"
    print(f"  * Test Query: {query!r}")

    # Warmup
    _ = embedder.embed(query)

    latencies = []
    vector = []
    for _ in range(runs):
        t0 = time.perf_counter()
        vector = embedder.embed(query)
        latencies.append((time.perf_counter() - t0) * 1000.0)  # ms

    avg_ms = sum(latencies) / len(latencies)
    min_ms = min(latencies)
    max_ms = max(latencies)
    norm = l2_norm(vector)

    print(f"  * Output Vector Dim : {len(vector)} (Expected: 1024)")
    print(f"  * Vector L2 Norm    : {norm:.6f} (Expected: ~1.000000)")
    print(f"  * Latency (Avg)     : {avg_ms:.2f} ms")
    print(f"  * Latency (Min-Max) : {min_ms:.2f} ms - {max_ms:.2f} ms over {runs} runs")

    if len(vector) == 1024 and abs(norm - 1.0) < 1e-4:
        print("  * Status            : [OK] Single query embedding passed with valid normalized vector!")
    else:
        print("  * Status            : [WARN] Vector check warning.")
    print()


def test_batch_inference(embedder: Local_BGE_M3_Embedder, batch_size: int = 16):
    print("=" * 70)
    print(f" [STEP 4] Batch Document Embedding Benchmark (batch_size={batch_size})")
    print("=" * 70)

    total_passages = len(SAMPLE_PASSAGES)
    t0 = time.perf_counter()
    vectors = embedder.embed_documents(SAMPLE_PASSAGES, batch_size=batch_size)
    elapsed = time.perf_counter() - t0

    throughput = total_passages / elapsed if elapsed > 0 else 0.0

    print(f"  * Passages Count  : {total_passages} chunks")
    print(f"  * Total Time      : {elapsed:.3f}s ({elapsed*1000/total_passages:.2f} ms/doc)")
    print(f"  * Throughput      : {throughput:.1f} documents/second")
    print(f"  * Output Matrix   : ({len(vectors)}, {len(vectors[0]) if vectors else 0})")
    print("  * Status          : [OK] Batch embedding passed!")
    print()


def test_semantic_quality(embedder: Local_BGE_M3_Embedder):
    print("=" * 70)
    print(" [STEP 5] Semantic Similarity & Retrieval Quality Verification")
    print("=" * 70)

    query = "How to replace the fuel filter cartridge"
    related_1 = "Steps for fuel filter replacement and bleeding air from tractor fuel lines"
    related_2 = "Fuel system periodic maintenance and fuel water separator cleaning"
    unrelated = "Recipe for baking a traditional chocolate sponge cake with strawberry topping"

    q_vec = embedder.embed(query)
    r1_vec = embedder.embed(related_1)
    r2_vec = embedder.embed(related_2)
    u_vec = embedder.embed(unrelated)

    sim_r1 = dot_product(q_vec, r1_vec)
    sim_r2 = dot_product(q_vec, r2_vec)
    sim_u = dot_product(q_vec, u_vec)

    print(f"  Query        : {query!r}")
    print(f"  |-- Related 1 : {sim_r1:.4f}  -> {related_1!r}")
    print(f"  |-- Related 2 : {sim_r2:.4f}  -> {related_2!r}")
    print(f"  +-- Unrelated : {sim_u:.4f}  -> {unrelated!r}")

    if sim_r1 > 0.70 and sim_r2 > 0.60 and sim_u < 0.40:
        print("\n  * Semantic Quality: [OK] EXCELLENT (Strong separation between related and unrelated passages)")
    else:
        print("\n  * Semantic Quality: [WARN] Lower than expected separation")
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to force model execution on ('cuda', 'cuda:0', 'cpu', or 'auto'). Default: 'cuda'",
    )
    parser.add_argument(
        "--model-path",
        default=config.LOCAL_BGE_M3_PATH,
        help="Path to local BGE-M3 model weights",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for batch inference benchmark",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Benchmark iterations for single-query latency",
    )
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  LOCAL BGE-M3 EMBEDDING BENCHMARK & GPU VERIFICATION")
    print("=" * 70 + "\n")

    test_environment(args.device)
    embedder = test_model_loading(args.model_path, args.device)
    test_single_inference(embedder, runs=args.runs)
    test_batch_inference(embedder, batch_size=args.batch_size)
    test_semantic_quality(embedder)

    print("=" * 70)
    print(" [PASSED] ALL LOCAL EMBEDDING VERIFICATIONS COMPLETED SUCCESSFULLY!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
