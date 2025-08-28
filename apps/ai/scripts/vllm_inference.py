import os
import sys
import argparse
from vllm import LLM, SamplingParams

DEFAULT_LLM_DIR = "/capstor/scratch/cscs/sshen/workspace/spcl-atlahs/apps/ai/llm-models"

def parse_args():
    parser = argparse.ArgumentParser(description="VLLM Multi-Node Inference Script")
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-70B-Instruct",
        help="Model name or path"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Tokenizer name or path (defaults to model if not specified)"
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=4,
        help="Number of GPUs per node for tensor parallelism"
    )
    parser.add_argument(
        "--pipeline-parallel-size",
        type=int,
        default=1,
        help="Number of nodes for pipeline parallelism"
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.95,
        help="GPU memory utilization ratio"
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=8192,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--download-dir",
        type=str,
        default=None,
        help="Directory to download models (defaults to LLM_MODEL_DIR env var)"
    )
    parser.add_argument(
        "--distributed-executor-backend",
        type=str,
        default="mp",
        choices=["mp", "ray", "external_launcher"],
        help="Distributed executor backend"
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code when loading models"
    )
    parser.add_argument(
        "--dtype", 
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Model data type"
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        choices=["awq", "gptq", "squeezellm", "fp8"],
        help="Quantization method"
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=8192,
        help="Maximum number of batched tokens"
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=256,
        help="Maximum number of sequences"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p sampling parameter"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default=None,
        help="File containing prompts (one per line)"
    )
    
    return parser.parse_args()

def load_prompts(prompt_file=None):
    """Load prompts from file or use defaults"""
    if prompt_file and os.path.exists(prompt_file):
        with open(prompt_file, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]
        print(f"[INFO] Loaded {len(prompts)} prompts from {prompt_file}")
    else:
        prompts = [
            "Hello, world!",
            "What is the capital of France?",
            "What is the capital of Germany?",
            "What is the capital of Italy?",
            "What is the capital of Spain?",
            "Explain quantum computing in simple terms.",
            "What are the main challenges in climate change?",
            "How does machine learning work?",
        ]
        print(f"[INFO] Using {len(prompts)} default prompts")
    
    return prompts

def run_vllm() -> None:
    args = parse_args()
    
    # Set tokenizer to model if not specified
    tokenizer_name = args.tokenizer or args.model
    
    # Get download directory
    download_dir = args.download_dir or os.environ.get("LLM_MODEL_DIR", DEFAULT_LLM_DIR)
    
    # Calculate total tensor parallel size across all nodes
    total_tensor_parallel_size = args.tensor_parallel_size * args.pipeline_parallel_size
    
    # Load prompts
    prompts = load_prompts(args.prompt_file)
    
    print(f"[INFO] Multi-Node VLLM Inference Configuration:")
    print(f"[INFO] Model: {args.model}")
    print(f"[INFO] Tokenizer: {tokenizer_name}")
    print(f"[INFO] Tensor Parallel Size (per node): {args.tensor_parallel_size}")
    print(f"[INFO] Pipeline Parallel Size (nodes): {args.pipeline_parallel_size}")
    print(f"[INFO] Total Tensor Parallel Size: {total_tensor_parallel_size}")
    print(f"[INFO] GPU Memory Utilization: {args.gpu_memory_utilization}")
    print(f"[INFO] Max Model Length: {args.max_model_len}")
    print(f"[INFO] Distributed Backend: {args.distributed_executor_backend}")
    print(f"[INFO] Data Type: {args.dtype}")
    if args.quantization:
        print(f"[INFO] Quantization: {args.quantization}")
    print(f"[INFO] Download Directory: {download_dir}")
    
    # Initialize LLM with multi-node configuration

    llm = LLM(
        model=args.model,
        tokenizer=tokenizer_name,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        download_dir=download_dir,
        distributed_executor_backend=args.distributed_executor_backend,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
        quantization=args.quantization,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        seed=42,
    )

    print(f"[INFO] Model loaded successfully!")
    print(f"[INFO] Starting inference with {len(prompts)} prompts...")

    sampling_params = SamplingParams(
        temperature=args.temperature, 
        top_p=args.top_p, 
        max_tokens=args.max_tokens
    )

    # Generate responses
    for i, out in enumerate(llm.generate(prompts, sampling_params)):
        print(f"\n[RESPONSE {i+1}]")
        print(f"Prompt: {prompts[i]}")
        print(f"Response: {out.outputs[0].text}")
        print("-" * 80)

    print(f"[INFO] Inference completed successfully!")


if __name__ == "__main__":
    run_vllm()