import os
import sys
import argparse
from vllm import LLM, SamplingParams

DEFAULT_LLM_DIR = "/capstor/scratch/cscs/sshen/workspace/spcl-atlahs/apps/ai/llm-models"

def run_vllm() -> None:
    # print(f"[DEBUG] HF_TOKEN: {os.environ['HF_TOKEN']}")
    prompts = [
        "Hello, world!",
        "What is the capital of France?",
        "What is the capital of Germany?",
        "What is the capital of Italy?",
        "What is the capital of Spain?",
    ]

    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    tokenizer_name = "meta-llama/Llama-3.1-8B-Instruct"
    gpu_memory_utilization = 0.95
    tensor_parallel_size = 4
    download_dir = os.environ.get("LLM_MODEL_DIR", DEFAULT_LLM_DIR)

    llm = LLM(
        model=model_name,
        tokenizer=tokenizer_name,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        download_dir=download_dir,
        distributed_executor_backend="mp"
    )

    print(f"[INFO] Running VLLM Inference...")
    print(f"[INFO] Model: {model_name}")
    print(f"[INFO] Tokenizer: {tokenizer_name}")
    print(f"[INFO] GPU Memory Utilization: {gpu_memory_utilization}")
    print(f"[INFO] Tensor Parallel Size: {tensor_parallel_size}")

    sampling_params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=1024)

    for out in llm.generate(prompts, sampling_params):
        print(f"[INFO] Response: {out.outputs[0].text}")


if __name__ == "__main__":
    run_vllm()