"""
GPU-based LLM Prediction Functions for Medical Benchmarking
============================================================
Uses HuggingFace Transformers with GPU acceleration and quantization.
Supports MedGemma, Llama, Mistral, and other open-weight models.

Requirements:
    pip install torch transformers accelerate bitsandbytes

Usage:
    1. Load models on GPU (e.g., MedGemma-27B-IT, MedGemma-4B-IT)
    2. Use create_chain() to create prediction chains
    3. Run predictions with get_prediction_* functions

Environment:
    Set HF_TOKEN environment variable for gated models (e.g., MedGemma)
"""

import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Get HuggingFace token from environment for gated models
HF_TOKEN = os.environ.get("HF_TOKEN", None)

# ============================================================================
# GPU Model Classes
# ============================================================================

class GPUModel:
    """
    GPU-based LLM using HuggingFace Transformers with quantization support.
    """
    def __init__(self, model_path, quantization=None, device="auto"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available! GPU required for inference.")
        
        self.model_path = model_path
        self.quantization = quantization
        # device_map (string or dict) is passed to transformers; self.device is a torch.device
        self.device_map = device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            try:
                self.device = torch.device(device)
            except Exception:
                # Fallback to cuda if parsing fails
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Extract model name from path for display
        import os
        self.model_name = os.path.basename(os.path.normpath(model_path))
        
        print("\n" + "="*70)
        print(f"Loading Model: {self.model_name}")
        print("="*70)
        print(f"Path: {model_path}")
        print(f"Quantization: {quantization}")
        print(f"Device: {device}")
        print("="*70)
        
        # Configure quantization
        if quantization == "4bit":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif quantization == "8bit":
            quant_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            )
        else:
            quant_config = None
        
        # Load tokenizer (some Gemma models require remote code for tokenizer)
        print("Loading tokenizer...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, token=HF_TOKEN)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            print(f"AutoTokenizer failed ({e}), attempting fallback using PreTrainedTokenizerFast from tokenizer.json")
            from transformers import PreTrainedTokenizerFast
            import json, os
            # Try to read special tokens from special_tokens_map.json
            st_path = os.path.join(model_path, 'special_tokens_map.json')
            specials = {}
            try:
                with open(st_path, 'r') as f:
                    specials = json.load(f)
            except Exception:
                specials = {}

            def _tok_val(key, default):
                v = specials.get(key, default)
                if isinstance(v, str):
                    return v
                if isinstance(v, dict):
                    return v.get('content', default)
                return default

            tokenizer_file = os.path.join(model_path, 'tokenizer.json')
            self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file,
                                                     bos_token=_tok_val('bos_token', '<bos>'),
                                                     eos_token=_tok_val('eos_token', '<eos>'),
                                                     pad_token=_tok_val('pad_token', '<pad>'),
                                                     unk_token=_tok_val('unk_token', '<unk>'))
        
        # Load model
        print("Loading model (this may take 1-3 minutes)...")
        start_time = time.time()
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quant_config,
            device_map=device,
            torch_dtype=torch.bfloat16 if quant_config is None else None,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            token=HF_TOKEN
        )
        self.model.eval()
        
        load_time = time.time() - start_time
        
        # Report memory usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1e9
            reserved = torch.cuda.memory_reserved(0) / 1e9
            print(f"[OK] Model loaded in {load_time:.1f}s")
            print(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved\n")
    
    def invoke(self, inputs, max_tokens=2048, temperature=0.0):
        """Invoke GPU model with inputs"""
        # Format prompt from inputs
        if isinstance(inputs, dict):
            prompt = " ".join([f"{k}: {v}" for k, v in inputs.items()])
        else:
            prompt = str(inputs)
        
        # Tokenize
        input_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)
        
        input_token_count = input_ids['input_ids'].shape[1]
        
        # Generate with timing
        gen_start = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else 1.0,
                do_sample=temperature > 0,
                repetition_penalty=1.1,  # Prevent token loops
                no_repeat_ngram_size=3,  # Prevent phrase repetition
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        gen_time = time.time() - gen_start
        
        # Decode (remove input prompt)
        generated_tokens = outputs[0][input_ids['input_ids'].shape[1]:]
        output_token_count = len(generated_tokens)
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Calculate tokens/second
        tokens_per_sec = output_token_count / gen_time if gen_time > 0 else 0
        
        # Return object mimicking LangChain response with metrics
        class Response:
            def __init__(self, content, input_tokens, output_tokens, gen_time, tokens_per_sec):
                self.content = content
                self.input_tokens = input_tokens
                self.output_tokens = output_tokens
                self.gen_time = gen_time
                self.tokens_per_sec = tokens_per_sec
        
        return Response(generated_text.strip(), input_token_count, output_token_count, gen_time, tokens_per_sec)


class Chain:
    """Chain interface for GPU model"""
    def __init__(self, template, model):
        self.template = template
        self.model = model
    
    def invoke(self, inputs):
        # Format template with inputs
        prompt = self.template
        for key, value in inputs.items():
            prompt = prompt.replace(f"{{{key}}}", str(value))
        return self.model.invoke(prompt)


def create_chain(prompt_template, model_path, quantization=None):
    """
    Create a GPU-based chain for inference.
    
    Args:
        prompt_template: String with {variable} placeholders
        model_path: Path to HuggingFace model directory
        quantization: "4bit", "8bit", or None for full precision (default: None)
        
    Returns:
        Chain object with .invoke() method
    """
    model = GPUModel(model_path, quantization=quantization)
    return Chain(prompt_template, model)


# ============================================================================
# Prediction Functions 
# ============================================================================

def get_ground_truth_specialty(row, chain, max_retries=3, initial_wait=1):
    """Generate specialty referral from diagnosis using local LLM"""
    diagnosis = row["primary_diagnosis"]
    attempt = 0
    while attempt < max_retries:
        try:
            specialty = chain.invoke({"diagnosis": diagnosis}).content
            return specialty
        except Exception as e:
            wait_time = initial_wait * (2 ** attempt)
            print(f"WARNING: Error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"   Retrying in {wait_time}s...")
                time.sleep(wait_time)
            attempt += 1
    return "Error: Max retries exceeded"


def get_prediction_GeneralUser(row, chain, max_retries=3, initial_wait=1):
    """Predict for General User (patient) with HPI + demographics"""
    hpi = row['HPI']
    patient_info = row["patient_info"]
    attempt = 0
    while attempt < max_retries:
        try:
            response = chain.invoke({"HPI": hpi, "patient_info": patient_info})
            # Print token metrics
            print(f" [{response.output_tokens} tok, {response.tokens_per_sec:.1f} tok/s]", end="")
            return response.content
        except Exception as e:
            wait_time = initial_wait * (2 ** attempt)
            print(f"WARNING: Error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"   Retrying in {wait_time}s...")
                time.sleep(wait_time)
            attempt += 1
    return "Error: Max retries exceeded"


def get_prediction_ClinicalUser(row, chain, max_retries=3, initial_wait=1):
    """Predict for Clinical User with HPI + demographics + vitals"""
    hpi = row['HPI']
    patient_info = row["patient_info"]
    initial_vitals = row["initial_vitals"]
    attempt = 0
    while attempt < max_retries:
        try:
            response = chain.invoke({
                "hpi": hpi, 
                "patient_info": patient_info, 
                "initial_vitals": initial_vitals
            })
            # Print token metrics
            print(f" [{response.output_tokens} tok, {response.tokens_per_sec:.1f} tok/s]", end="")
            return response.content
        except Exception as e:
            wait_time = initial_wait * (2 ** attempt)
            print(f"WARNING: Error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"   Retrying in {wait_time}s...")
                time.sleep(wait_time)
            attempt += 1
    return "Error: Max retries exceeded"


def get_evaluation_diagnosis(row, key, chain, max_retries=3, initial_wait=1):
    """Evaluate diagnosis predictions using local LLM"""
    diagnosis = row["primary_diagnosis"]
    diag1 = row[key][0]
    diag2 = row[key][1]
    diag3 = row[key][2]
    
    attempt = 0
    while attempt < max_retries:
        try:
            evaluation = chain.invoke({
                "real_diag": diagnosis, 
                "diag1": diag1, 
                "diag2": diag2, 
                "diag3": diag3
            }).content
            return evaluation
        except Exception as e:
            wait_time = initial_wait * (2 ** attempt)
            print(f"WARNING: Error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"   Retrying in {wait_time}s...")
                time.sleep(wait_time)
            attempt += 1
    return "Error: Max retries exceeded"


# ============================================================================
# Utility Functions
# ============================================================================

def test_gpu_connection(model_path):
    """
    Test if GPU is available and model can load.
    
    Args:
        model_path: Path to HuggingFace model directory
    
    Returns:
        True if connection successful, False otherwise
    """
    try:
        if not torch.cuda.is_available():
            print("ERROR: CUDA not available. GPU required for inference.")
            print("\nTroubleshooting:")
            print("  1. Ensure you're on a GPU instance")
            print("  2. Check nvidia-smi to verify GPU is accessible")
            print("  3. Verify PyTorch CUDA installation:")
            print("     python -c 'import torch; print(torch.cuda.is_available())'")
            return False
        
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU detected: {gpu_name} ({gpu_mem:.1f}GB)")
        print(f"Testing model loading from: {model_path}")
        
        # Try loading model in bfloat16 (no quantization) to preserve numerical fidelity
        # and avoid potential tokenizer/model-code mismatch with quantized configs.
        model = GPUModel(model_path, quantization=None)
        response = model.invoke("Say hello")
        print(f"Model response: {response.content[:50]}...")
        
        # Clean up
        del model
        torch.cuda.empty_cache()
        
        print("[OK] GPU connection test successful!")
        return True
        
    except Exception as e:
        print(f"ERROR: GPU test failed: {e}")
        print("\nTroubleshooting:")
        print("  1. Check model path is correct")
        print("  2. Ensure sufficient GPU memory (need 20GB+ for 27B model)")
        print("  3. Try with smaller model first (MedGemma-4B)")
        return False


# ============================================================================
# Dual Model Manager
# ============================================================================

class DualModelManager:
    """
    Manages two models in GPU memory simultaneously.
    Used for: Predictor (27B) + Judge (4B) loaded at same time.
    """
    def __init__(self, predictor_path, judge_path, 
                 predictor_quant="4bit", judge_quant="8bit"):
        """
        Initialize both models.
        
        Args:
            predictor_path: Path to main prediction model (e.g., MedGemma-27B)
            judge_path: Path to judge model (e.g., MedGemma-4B)
            predictor_quant: Quantization for predictor ("4bit", "8bit", None)
            judge_quant: Quantization for judge ("4bit", "8bit", None)
        """
        import os
        predictor_name = os.path.basename(os.path.normpath(predictor_path))
        judge_name = os.path.basename(os.path.normpath(judge_path))
        
        print("\n" + "="*70)
        print("Dual Model Manager - Loading Both Models")
        print("="*70)
        print(f"Predictor: {predictor_name} ({predictor_quant})")
        print(f"Judge: {judge_name} ({judge_quant})")
        print("="*70)
        
        # Load predictor (main model)
        print("\n[1/2] Loading Predictor Model...")
        self.predictor = GPUModel(predictor_path, quantization=predictor_quant)
        
        # Report memory after predictor
        if torch.cuda.is_available():
            mem_after_pred = torch.cuda.memory_allocated(0) / 1e9
            print(f"Memory after predictor: {mem_after_pred:.2f}GB")
        
        # Load judge (evaluation model)
        print("\n[2/2] Loading Judge Model...")
        self.judge = GPUModel(judge_path, quantization=judge_quant)
        
        # Report total memory
        if torch.cuda.is_available():
            total_allocated = torch.cuda.memory_allocated(0) / 1e9
            total_reserved = torch.cuda.memory_reserved(0) / 1e9
            total_gpu = torch.cuda.get_device_properties(0).total_memory / 1e9
            usage_pct = (total_reserved / total_gpu) * 100
            
            print("\n" + "="*70)
            print("Both Models Loaded Successfully!")
            print("="*70)
            print(f"Total GPU Memory: {total_allocated:.2f}GB allocated")
            print(f"                  {total_reserved:.2f}GB reserved")
            print(f"GPU Utilization: {usage_pct:.1f}% of {total_gpu:.1f}GB")
            
            if usage_pct > 90:
                print("\nWARNING: GPU memory >90% utilized. May cause OOM errors.")
                print("Consider: 1) Reducing batch size, 2) More aggressive quantization")
            
            print("="*70 + "\n")
    
    def create_predictor_chain(self, prompt_template):
        """Create chain for predictor model"""
        return Chain(prompt_template, self.predictor)
    
    def create_judge_chain(self, prompt_template):
        """Create chain for judge model"""
        return Chain(prompt_template, self.judge)
