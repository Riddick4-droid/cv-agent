"""
llm_client.py - Unified LLM Client with OpenAI primary and local DeepSeek fallback.
Supports quantization (4-bit/8-bit) via bitsandbytes for local models.
Implements retries, token counting, and graceful fallback.
"""

import time
from typing import Optional, List, Dict, Any
from pathlib import Path
import torch
import httpx

#openai
from openai import OpenAI, APIError, RateLimitError, APITimeoutError

#local imports
from .exceptions import LLMServiceError, ModelLoadError, ConfigurationError
from .logger import get_logger
from .config import get_config

logger = get_logger(__name__)

#opensource imports
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import transformers
    TRANSFORMER_AVAILABLE = True
except ImportError:
    TRANSFORMER_AVAILABLE=False
    logger.warning("transformers not installed. Local Model fallback disabled")


def estimate_tokens(text:str)->int:
    """Rough token count (4 chars ~ 1 token for English). For local model use."""
    return len(text) // 4

class LLMClient:
    """
    Unified LLM Client.
    Primary: OpenAI API.
    Fallback: Local quantized DeepSeek model (if enabled and available).
    """
    def __init__(self, trace_id: Optional[str]=None):
        """
        Initialize the LLM client.
        Loads config and sets up OpenAI client. Local model is lazy-loaded only on fallback.
        """
        self.config = get_config()
        self.trace_id = trace_id or "llm-client"
        self.openai_client = None
        self.local_model = None
        self.tokenizer = None
        self.local_device=None

        #setup openai client if API key exists
        if self.config.openai_api_key:
            self.openai_client = OpenAI(
                    api_key=self.config.openai_api_key,
                    timeout=self.config.llm_timeout,
                    max_retries=0, 
                        )
            logger.info("Openai client initialized")
        else:
            logger.warning("No Openai API key found. Primary LLM disabled")

        
        #setup fallback llm
        self.fallback_enabled = self.config.use_local_fallback and TRANSFORMER_AVAILABLE

        if self.fallback_enabled:
            logger.info("Local fallback model enabled (deepseek quantized)")
        elif self.config.use_local_fallback and not TRANSFORMER_AVAILABLE:
            logger.warning("Local fallback enabled but transformers not installed. Run: pip install transformers")
        
        #retry settings
        self.max_retries = self.config.llm_max_retries
        self.base_delay = 1 #this is is seconds

    def _call_openai_with_retry(self, messages: List[Dict[str, str]])->str:
        """
        Call OpenAI API with retries on transient errors.
        Returns response text.
        """
        for attempt in range(self.max_retries + 1):
            try:
                response = self.openai_client.chat.completions.create(
                    model=self.config.llm_model,
                    messages=messages,
                    temperature=self.config.llm_temperature,
                    #max_tokens=self.config.llm_max_tokens
                )
                content = response.choices[0].message.content
                if content is None:
                    raise LLMServiceError("Openai returned empty response")
                return content.strip()
            except (RateLimitError, APITimeoutError) as e:
                wait = self.base_delay * (2**attempt)
                logger.warning(f"Openai transient error (attempt: {attempt +1}/{self.max_retries+1}):{e}. Retrying in {wait}s")
                time.sleep(wait)
                continue
            except APIError as e:
                logger.error(f"Openai API error (non-retryable): {e}")
                raise LLMServiceError(f"Openai API error: {e}") from e
            except Exception as e:
                # Unexpected errors
                logger.error(f"Unexpected error calling OpenAI: {e}")
                raise LLMServiceError(f"OpenAI call failed: {e}") from e
        raise LLMServiceError("OpenAI: Max retries exceeded")   
    
    def _load_local_model(self):
        """
        Lazy-load the local DeepSeek model with quantization.
        Only called when fallback is needed.
        """
        if self.local_model is not None:
            return
        if not self.fallback_enabled:
            raise ModelLoadError("Local fallback not enabled or transformers not installed")
        
        logger.info(f"Loading local model: {self.config.local_model_path}")
        logger.info(f"Quantization bits: {self.config.quantization_bits}")

        #determine device
        if self.config.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self.config.device

        self.local_device = device
        logger.info(f"Using device: {device}")

        #configguring bitandbytes config
        bits = self.config.quantization_bits
        if bits == 4:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif bits == 8:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            raise ModelLoadError(f"Unsupported quantization bits: {bits}")
        
        try:
            self.local_tokenizer = AutoTokenizer.from_pretrained(
                self.config.local_model_path,
                trust_remote_code=True
            )
            self.local_model = AutoModelForCausalLM.from_pretrained(
                self.config.local_model_path,
                quanitization_config = bnb_config,
                device_map="auto" if device =="cuda" else None,
                trust_remote_code=True,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            ) 

            #if cpu
            if device == "cpu":
                self.local_model = self.local_model.to("cpu")
            
            logger.info(f"Local model loaded successfully on {device}")
        except Exception as e:
            logger.error(f"Failed to load local model: {e}")
            raise ModelLoadError(f"Local model loading failed: {e}") from e
    
    def _call_local_model(self, messages: List[Dict[str,str]])->str:
        """
        Generate response using local quantized DeepSeek model.
        Assumes model is already loaded.
        """
        if self.local_model is None:
            self._load_local_model()
        
        #build prompt from messages (simple chat format)
        prompt_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role =="user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        prompt = "\n".join(prompt_parts) + "\nAssitant:"

        #tokenize
        inputs = self.local_tokenizer(prompt, return_tensors = "pt").to(self.local_device)

        #generate
        with torch.no_grad():
            outputs = self.local_model.generate(**inputs, 
                                                max_new_tokens=self.config.llm_max_tokens, 
                                                temperature=self.config.llm_temperature,
                                                do_sample=True, 
                                                pad_token_id=self.local_tokenizer.eos_token_id)
            # Decode
            full_output = self.local_tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the input prompt from the output
            response = full_output[len(prompt):].strip()
        return response
    
    def generate(self,prompt: str, system_prompt: Optional[str] = None, use_fallback: bool = False) -> str:
        """
        Generates a response from the LLM.
        
        Args:
            prompt: User prompt string.
            system_prompt: Optional system instruction.
            use_fallback: If True, bypass OpenAI and use local model directly (for testing).
        
        Returns:
            Generated text response.
        
        Raises:
            LLMServiceError: If both primary and fallback fail.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # If forced fallback, skip OpenAI
        if use_fallback:
            logger.info("Using forced local fallback mode")
            if not self.fallback_enabled:
                raise LLMServiceError("Local fallback not available but forced.")
            try:
                self._load_local_model()
                return self._call_local_model(messages=messages)
            except Exception as e:
                raise LLMServiceError(f"Local model generation failed: {e}") from e
        
        # Primary llm: OpenAI
        if self.openai_client:
            try:
                logger.debug("Calling OpenAI...")
                response = self._call_openai_with_retry(messages)
                logger.debug("OpenAI call successful")
                return response
            except Exception as e:
                logger.error(f"OpenAI failed: {e}")
                if self.fallback_enabled:
                    logger.info("Attempting fallback to local model...")
                    try:
                        self._load_local_model()
                        return self._call_local_model(messages)
                    except Exception as fallback_e:
                        raise LLMServiceError(
                            f"Both OpenAI and local fallback failed. "
                            f"OpenAI error: {e}, Fallback error: {fallback_e}"
                        ) from fallback_e
                else:
                    raise LLMServiceError(f"OpenAI failed and no fallback: {e}") from e
        else:
            # No OpenAI client (missing key) – use local if available
            if self.fallback_enabled:
                logger.info("No OpenAI client, using local model")
                try:
                    self._load_local_model()
                    return self._call_local_model(messages)
                except Exception as e:
                    raise LLMServiceError(f"Local model generation failed: {e}") from e
            else:
                raise LLMServiceError("No LLM provider available (OpenAI key missing and local fallback disabled)")
            
if __name__ == "__main__":
    # Load environment and config
    from .config import get_config
    config = get_config()
    
    # Test with a simple prompt
    client = LLMClient(trace_id="test")
    
    # Test OpenAI (if key available)
    if config.openai_api_key:
        print("Testing OpenAI...")
        try:
            response = client.generate("What is the capital of Ghana?'")
            print(f"OpenAI response: {response}")
        except Exception as e:
            print(f"OpenAI test failed: {e}")
    
    # Test local fallback (if enabled)
    if config.use_local_fallback:
        print("Testing local model (fallback)...")
        try:
            response = client.generate("What is the capital of Ghana'", use_fallback=False)
            print(f"Local response: {response}")
        except Exception as e:
            print(f"Local model test failed: {e}")