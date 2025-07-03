import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from django.conf import settings
from typing import Optional, Dict, Any
import logging
import gc
import os

# Use the updated import to fix deprecation warning
try:
    from langchain_huggingface import HuggingFacePipeline
except ImportError:
    # Fallback to the old import if the new package isn't installed
    from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

logger = logging.getLogger(__name__)


class Phi3ModelService:
    """Service for managing Phi-3-mini-128k-instruct local model optimized for RAG"""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.llm = None
        self.device = self._get_optimal_device()
        self.model_loaded = False
        self.max_retries = 3
        self.model_name = "microsoft/Phi-3-mini-128k-instruct"

    def _get_optimal_device(self) -> str:
        """Determine the best available device with M1 optimization"""
        try:
            # Check for force CPU environment variable
            if os.getenv("FORCE_CPU_ONLY", "false").lower() == "true":
                logger.info("Forced CPU usage via FORCE_CPU_ONLY environment variable")
                return "cpu"

            # Check CUDA first (most reliable)
            if torch.cuda.is_available():
                logger.info("Using CUDA device")
                return "cuda"

            # For MPS (M1/M2 Macs), enable with optimizations for Phi-3
            if torch.backends.mps.is_available():
                logger.info("Using MPS device (Apple Silicon) - optimized for Phi-3")
                return "mps"

            # Fallback to CPU
            logger.info("Using CPU device")
            return "cpu"

        except Exception as e:
            logger.warning(f"Error detecting device, using CPU: {e}")
            return "cpu"

    def initialize_model(self) -> bool:
        """Initialize Phi-3 model with device fallback"""
        if self.model_loaded:
            return True

        # Try with current device, fallback if it fails
        for attempt in range(2):  # Try current device, then CPU fallback
            try:
                device_to_use = self.device if attempt == 0 else "cpu"
                if attempt == 1:
                    logger.warning(
                        f"Falling back from {self.device} to CPU due to errors"
                    )
                    self.device = "cpu"

                return self._load_model_on_device(device_to_use)

            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed on {device_to_use}: {e}")
                if attempt == 0 and self.device != "cpu":
                    # Clean up before trying CPU
                    self.cleanup_model()
                    continue
                else:
                    # Final attempt failed
                    logger.error("❌ Failed to load Phi-3 on any device")
                    return False

        return False

    def _load_model_on_device(self, device: str) -> bool:
        """Load Phi-3 model on specific device with optimizations"""
        try:
            logger.info(f"Loading Phi-3-mini-128k-instruct on {device}...")

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Device-specific model loading optimized for Phi-3
            if device == "cuda":
                model_kwargs = {
                    "torch_dtype": torch.float16,
                    "device_map": "auto",
                    "trust_remote_code": True,
                    "low_cpu_mem_usage": True,
                }
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name, **model_kwargs
                )

                self.pipeline = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                )

            elif device == "mps":
                # Optimized MPS settings for Phi-3 on Apple Silicon
                model_kwargs = {
                    "torch_dtype": torch.float16,  # Use float16 for better performance on M1
                    "trust_remote_code": True,
                    "low_cpu_mem_usage": True,
                }
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name, **model_kwargs
                )

                # Move to MPS with optimizations
                try:
                    self.model = self.model.to("mps")

                    self.pipeline = pipeline(
                        "text-generation",
                        model=self.model,
                        tokenizer=self.tokenizer,
                        device=0,  # Use device 0 for MPS
                        torch_dtype=torch.float16,
                        trust_remote_code=True,
                    )
                except Exception as mps_error:
                    logger.error(f"MPS model placement failed: {mps_error}")
                    raise mps_error

            else:  # CPU
                model_kwargs = {
                    "torch_dtype": torch.float32,
                    "trust_remote_code": True,
                    "low_cpu_mem_usage": True,
                }
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name, **model_kwargs
                )

                self.pipeline = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=-1,  # -1 for CPU
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                )

            # Create LangChain wrapper
            self.llm = HuggingFacePipeline(pipeline=self.pipeline)
            self.model_loaded = True
            logger.info(f"✅ Phi-3-mini-128k-instruct loaded successfully on {device}!")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to load Phi-3 on {device}: {e}")
            self.cleanup_model()
            raise e

    def generate_response(
        self,
        prompt: str,
        max_tokens: int = 500,  # Increased default for Phi-3's better capabilities
        temperature: float = 0.7,
        system_prompt: str = "You are a helpful assistant.",
    ) -> Optional[str]:
        """Generate response using Phi-3 with enhanced RAG capabilities"""

        if not self.model_loaded and not self.initialize_model():
            return None

        try:
            # Format prompt for Phi-3 chat format
            formatted_prompt = self._format_chat_prompt(prompt, system_prompt)

            # Generate response with device-specific error handling
            try:
                # Use more compatible generation parameters for Phi-3
                outputs = self.pipeline(
                    formatted_prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True if temperature > 0 else False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    return_full_text=False,
                    # Phi-3 specific optimizations - simplified for compatibility
                    top_p=0.9 if temperature > 0 else None,
                    top_k=50 if temperature > 0 else None,
                    # Add these parameters to avoid cache issues
                    use_cache=False,  # Disable caching to avoid DynamicCache issues
                    clean_up_tokenization_spaces=True,
                )
            except RuntimeError as e:
                if "MPS" in str(e) or "Metal" in str(e):
                    logger.error(
                        f"MPS error during generation, attempting CPU fallback: {e}"
                    )
                    # Try to reinitialize on CPU
                    self.cleanup_model()
                    self.device = "cpu"
                    if self.initialize_model():
                        return self.generate_response(
                            prompt, max_tokens, temperature, system_prompt
                        )
                    else:
                        return None
                else:
                    raise e
            except Exception as e:
                # If pipeline fails, try direct model generation
                logger.warning(f"Pipeline generation failed: {e}")
                logger.info("Attempting direct model generation...")
                return self._direct_generate(formatted_prompt, max_tokens, temperature)

            if outputs and len(outputs) > 0:
                response = outputs[0]["generated_text"].strip()
                # Clean up the response
                response = self._clean_response(response)
                return response

            return None

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return None

    def _direct_generate(
        self, formatted_prompt: str, max_tokens: int, temperature: float
    ) -> Optional[str]:
        """Direct generation bypass for compatibility issues"""
        try:
            # Tokenize input
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt")

            # Move to device
            if self.device != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate with model directly
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature if temperature > 0 else 1.0,
                    do_sample=True if temperature > 0 else False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    # Disable problematic features
                    use_cache=False,
                    attention_mask=inputs.get("attention_mask"),
                )

            # Decode response
            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )

            return self._clean_response(response)

        except Exception as e:
            logger.error(f"Direct generation also failed: {e}")
            return None

    def _format_chat_prompt(self, user_message: str, system_prompt: str) -> str:
        """Format prompt for Phi-3 chat format"""
        # Phi-3 uses a different chat template format
        return f"<|system|>\n{system_prompt}<|end|>\n<|user|>\n{user_message}<|end|>\n<|assistant|>\n"

    def _clean_response(self, response: str) -> str:
        """Clean up Phi-3 model response"""
        # Remove special tokens and extra whitespace
        response = response.replace("<|", "").replace("|>", "").replace("<|end|>", "")
        response = (
            response.replace("<|system|>", "")
            .replace("<|user|>", "")
            .replace("<|assistant|>", "")
        )
        response = response.strip()

        # Remove repetitive patterns
        lines = response.split("\n")
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line and line not in cleaned_lines[-3:]:  # Avoid recent repetitions
                cleaned_lines.append(line)

        return "\n".join(cleaned_lines)

    def cleanup_model(self):
        """Clean up model from memory"""
        try:
            if self.model:
                del self.model
            if self.tokenizer:
                del self.tokenizer
            if self.pipeline:
                del self.pipeline
            if self.llm:
                del self.llm

            self.model = None
            self.tokenizer = None
            self.pipeline = None
            self.llm = None
            self.model_loaded = False

            # Force garbage collection
            gc.collect()

            # Clear device-specific cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif self.device == "mps" and torch.backends.mps.is_available():
                try:
                    torch.mps.empty_cache()
                except Exception as e:
                    logger.warning(f"Error clearing MPS cache: {e}")

            logger.info("🧹 Phi-3 model cleaned from memory")

        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")

    def is_available(self) -> bool:
        """Check if model is available"""
        return self.model_loaded or self.initialize_model()

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "name": "Phi-3-mini-128k-instruct",
            "context_length": "128K tokens",
            "device": self.device,
            "loaded": self.model_loaded,
            "memory_usage": self._get_memory_usage() if self.model_loaded else 0,
            "optimized_for": ["RAG", "Apple Silicon", "Long Context"],
        }

    def _get_memory_usage(self) -> float:
        """Get current memory usage in GB"""
        try:
            if self.device == "cuda" and torch.cuda.is_available():
                return torch.cuda.memory_allocated() / 1e9
            elif self.device == "mps" and torch.backends.mps.is_available():
                # Estimated memory usage for Phi-3-mini on MPS
                return 3.8  # Rough estimate for Phi-3-mini in float16
            return 0.0
        except Exception:
            return 0.0

    def get_context_stats(self) -> Dict[str, Any]:
        """Get context window statistics - key advantage of Phi-3"""
        return {
            "max_context_tokens": 128000,  # 128K context window
            "effective_context": "120K tokens",  # Leave room for generation
            "context_advantage": "16x larger than original TinyLlama",
            "rag_benefit": "Can process much longer retrieved documents",
            "multi_turn_benefit": "Maintains longer conversation history",
        }
