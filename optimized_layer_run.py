import time
from benchmark import Benchmark

def run_optimized_benchmark():
    """
    Demonstrates using the optimization engine layer.
    The optimization engine dynamically adjusts memory context, limit limits,
    and LLM selection based on real-time hardware telemetry.
    """
    print("=== Initiating Optimized Layer Inference ===")
    layer_benchmark = Benchmark()
    
    test_prompts = [
        "Explain the concept of quantum entanglement and its implications.",
        "Write a Python script to reverse a linked list.",
        "Summarize the plot of the movie Inception.",
        "Translate 'Hello, how are you?' into French, Spanish, and German.",
        "Write a 500-word essay on the impact of artificial intelligence."
    ]
    
    # Run UNOPTIMIZED baseline first
    print("\n[Stage 1] Executing Unoptimized Baseline (Direct to LLM)...")
    for prompt in test_prompts:
        print(f"-> Unoptimized: {prompt[:40]}...")
        layer_benchmark.run_inference(prompt, use_optimizer=False, static_model="llama3.2:latest")
        time.sleep(1) # simulate brief pause
        
    print("\n============================================\n")
    
    # Run OPTIMIZED inference
    print("[Stage 2] Executing Inference via Optimization Engine...")
    for prompt in test_prompts:
        print(f"-> Optimized: {prompt[:40]}...")
        layer_benchmark.run_inference(prompt, use_optimizer=True)
        time.sleep(1)
        
    print("\n=== Benchmarks Completed ===")
    print("Optimization telemetry has been written to logs/results.csv")
    print("You can now view the visualizations in layer_metrics_analysis.ipynb")

if __name__ == "__main__":
    run_optimized_benchmark()
