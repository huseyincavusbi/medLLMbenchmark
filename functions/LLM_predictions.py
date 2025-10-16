"""
Local LLM Prediction Functions for Medical Benchmarking
=========================================================
Uses local models via LM Studio (OpenAI-compatible API).
Supports MedGemma, Llama, Mistral, and other open-weight models.

Requirements:
    pip install openai

Usage:
    1. Open LM Studio and load a model (e.g., MedGemma-4B-IT)
    2. Start the local server (http://localhost:1234/v1)
    3. Use create_local_chain() to create prediction chains
"""

import time

# ============================================================================
# Local LLM Core Classes
# ============================================================================

class LocalLLM:
    """
    Local LLM wrapper using LM Studio's OpenAI-compatible API
    """
    def __init__(self, base_url="http://localhost:1234/v1", model_name="local-model"):
        try:
            from openai import OpenAI
            self.client = OpenAI(base_url=base_url, api_key="not-needed")
            self.model_name = model_name
            self.base_url = base_url
        except ImportError:
            raise ImportError("OpenAI package required. Run: pip install openai")
    
    def invoke(self, inputs, max_tokens=500, temperature=0.0):
        """Invoke local LLM with inputs dict"""
        # Format prompt from inputs
        if isinstance(inputs, dict):
            prompt = " ".join([f"{k}: {v}" for k, v in inputs.items()])
        else:
            prompt = str(inputs)
        
        messages = [{"role": "user", "content": prompt}]
        
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Return object mimicking LangChain response
        class Response:
            def __init__(self, content):
                self.content = content
        
        return Response(completion.choices[0].message.content)


class LocalChain:
    """Chain-like interface for local LLM"""
    def __init__(self, template, llm):
        self.template = template
        self.llm = llm
    
    def invoke(self, inputs):
        # Format template with inputs
        prompt = self.template
        for key, value in inputs.items():
            prompt = prompt.replace(f"{{{key}}}", str(value))
        return self.llm.invoke(prompt)


def create_local_chain(prompt_template, base_url="http://localhost:1234/v1"):
    """
    Create a local LLM chain compatible with existing code
    
    Args:
        prompt_template: String with {variable} placeholders
        base_url: LM Studio server URL
        
    Returns:
        LocalChain object with .invoke() method
    """
    llm = LocalLLM(base_url=base_url)
    return LocalChain(prompt_template, llm)


# ============================================================================
# Prediction Functions (Local LLM)
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
            print(f"⚠️  Error (attempt {attempt + 1}/{max_retries}): {e}")
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
            response = chain.invoke({"HPI": hpi, "patient_info": patient_info}).content
            return response
        except Exception as e:
            wait_time = initial_wait * (2 ** attempt)
            print(f"⚠️  Error (attempt {attempt + 1}/{max_retries}): {e}")
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
            }).content
            return response
        except Exception as e:
            wait_time = initial_wait * (2 ** attempt)
            print(f"⚠️  Error (attempt {attempt + 1}/{max_retries}): {e}")
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
            print(f"⚠️  Error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"   Retrying in {wait_time}s...")
                time.sleep(wait_time)
            attempt += 1
    return "Error: Max retries exceeded"


# ============================================================================
# Utility Functions
# ============================================================================

def test_local_llm_connection(base_url="http://localhost:1234/v1"):
    """
    Test if LM Studio is running and responding.
    
    Usage:
        from functions.LLM_predictions import test_local_llm_connection
        if test_local_llm_connection():
            print("LM Studio ready!")
    
    Returns:
        True if connection successful, False otherwise
    """
    try:
        llm = LocalLLM(base_url=base_url)
        response = llm.invoke("Say hello")
        print(f"✅ Local LLM connected: {response.content[:50]}...")
        return True
    except Exception as e:
        print(f"❌ Local LLM connection failed: {e}")
        print(f"\n💡 Troubleshooting:")
        print(f"   1. Open LM Studio")
        print(f"   2. Load a model (e.g., MedGemma-4B-IT, Llama-3.1-8B)")
        print(f"   3. Go to 'Local Server' tab")
        print(f"   4. Click 'Start Server'")
        print(f"   5. Verify server is running at: {base_url}")
        return False
