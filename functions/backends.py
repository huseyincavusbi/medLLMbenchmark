"""
Inference Backends for Medical LLM Benchmarking
================================================
Provides a unified interface for different LLM inference backends.

Backends:
    - HuggingFaceBackend: Sequential inference, good for debugging
    - VLLMBackend: Batched inference, optimized for speed

Usage:
    from functions.backends import create_backend
    
    # Create backend
    backend = create_backend("vllm", "./models/medgemma-27b-it")
    
    # Generate responses
    prompts = ["What is diabetes?", "What causes fever?"]
    responses = backend.generate_batch(prompts, max_tokens=512)
"""

import os
import time
from abc import ABC, abstractmethod
from typing import List, Optional

# Get HuggingFace token from environment
HF_TOKEN = os.environ.get("HF_TOKEN", None)


class InferenceBackend(ABC):
    """Abstract base class for inference backends"""
    
    @abstractmethod
    def generate_batch(self, prompts: List[str], max_tokens: int = 512) -> List[str]:
        """
        Generate responses for a batch of prompts.
        
        Args:
            prompts: List of input prompts
            max_tokens: Maximum tokens to generate per response
            
        Returns:
            List of generated responses
        """
        pass
    
    @abstractmethod
    def get_info(self) -> dict:
        """Return backend information for logging"""
        pass


class HuggingFaceBackend(InferenceBackend):
    """
    HuggingFace Transformers backend.
    
    Processes prompts sequentially. Good for debugging and development.
    """
    
    def __init__(self, model_path: str, device: str = "auto"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print("\n" + "=" * 70)
        print("Loading HuggingFace Backend")
        print("=" * 70)
        print(f"Model: {model_path}")
        print(f"Device: {device}")
        
        start_time = time.time()
        
        # Load tokenizer
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True,
            token=HF_TOKEN
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        print("Loading model (this may take 1-3 minutes)...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            token=HF_TOKEN
        )
        self.model.eval()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        
        load_time = time.time() - start_time
        
        # Report memory
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1e9
            print(f"[OK] Model loaded in {load_time:.1f}s")
            print(f"GPU Memory: {allocated:.2f}GB allocated")
        
        print("=" * 70 + "\n")
    
    def generate_batch(self, prompts: List[str], max_tokens: int = 512) -> List[str]:
        """Generate responses sequentially (one at a time)"""
        import torch
        from tqdm import tqdm
        
        results = []
        
        for prompt in tqdm(prompts, desc="Generating (HF)"):
            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=4096
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=1.0,  # Required when do_sample=False
                    do_sample=False,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode (remove input prompt)
            generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            results.append(response.strip())
        
        return results
    
    def get_info(self) -> dict:
        return {
            "backend": "huggingface",
            "model_path": self.model_path,
            "device": str(self.device)
        }


class VLLMBackend(InferenceBackend):
    """
    vLLM backend for high-throughput batched inference.
    
    Uses PagedAttention and continuous batching for optimal GPU utilization.
    """
    
    def __init__(self, model_path: str, tensor_parallel_size: int = 1):
        from vllm import LLM, SamplingParams
        
        print("\n" + "=" * 70)
        print("Loading vLLM Backend")
        print("=" * 70)
        print(f"Model: {model_path}")
        print(f"Tensor Parallel Size: {tensor_parallel_size}")
        
        start_time = time.time()
        
        # Initialize vLLM
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
            max_model_len=4096,
            gpu_memory_utilization=0.90
        )
        
        self.model_path = model_path
        self.SamplingParams = SamplingParams
        
        load_time = time.time() - start_time
        print(f"[OK] Model loaded in {load_time:.1f}s")
        print("=" * 70 + "\n")
    
    def generate_batch(self, prompts: List[str], max_tokens: int = 512) -> List[str]:
        """Generate responses for all prompts in optimized batches"""
        
        sampling_params = self.SamplingParams(
            max_tokens=max_tokens,
            temperature=0.0,
            repetition_penalty=1.1
        )
        
        print(f"Generating {len(prompts)} responses with vLLM...")
        start_time = time.time()
        
        outputs = self.llm.generate(prompts, sampling_params)
        
        elapsed = time.time() - start_time
        total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        print(f"Generated {total_tokens} tokens in {elapsed:.1f}s ({total_tokens/elapsed:.1f} tok/s)")
        
        return [output.outputs[0].text.strip() for output in outputs]
    
    def get_info(self) -> dict:
        return {
            "backend": "vllm",
            "model_path": self.model_path
        }


def create_backend(backend_type: str, model_path: str, **kwargs) -> InferenceBackend:
    """
    Factory function to create an inference backend.
    
    Args:
        backend_type: "hf" for HuggingFace, "vllm" for vLLM
        model_path: Path to the model directory
        **kwargs: Additional backend-specific arguments
        
    Returns:
        InferenceBackend instance
    """
    if backend_type.lower() == "vllm":
        tensor_parallel = kwargs.get("tensor_parallel_size", 1)
        return VLLMBackend(model_path, tensor_parallel_size=tensor_parallel)
    elif backend_type.lower() in ("hf", "huggingface"):
        device = kwargs.get("device", "auto")
        return HuggingFaceBackend(model_path, device=device)
    else:
        raise ValueError(f"Unknown backend type: {backend_type}. Use 'hf' or 'vllm'.")


def test_backend(backend_type: str, model_path: str):
    """Quick test to verify backend works"""
    print(f"\nTesting {backend_type} backend with model: {model_path}")
    
    backend = create_backend(backend_type, model_path)
    
    test_prompts = [
        "What is the most common cause of chest pain?",
        "List 3 symptoms of diabetes."
    ]
    
    responses = backend.generate_batch(test_prompts, max_tokens=100)
    
    print("\nTest Results:")
    for i, (prompt, response) in enumerate(zip(test_prompts, responses)):
        print(f"\n[{i+1}] Prompt: {prompt[:50]}...")
        print(f"    Response: {response[:100]}...")
    
    print("\n[OK] Backend test passed!")
    return True


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python backends.py <backend_type> <model_path>")
        print("  backend_type: 'hf' or 'vllm'")
        print("  model_path: path to model directory")
        sys.exit(1)
    
    test_backend(sys.argv[1], sys.argv[2])
