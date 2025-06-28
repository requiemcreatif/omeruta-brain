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


class TinyLlamaService:
    """Service for managing TinyLlama local model"""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.llm = None
        self.device = self._get_optimal_device()
        self.model_loaded = False
        self.max_retries = 3

    def _get_optimal_device(self) -> str:
        """Determine the best available device with proper fallback"""
        try:
            # Check for force CPU environment variable
            if os.getenv("FORCE_CPU_ONLY", "false").lower() == "true":
                logger.info("Forced CPU usage via FORCE_CPU_ONLY environment variable")
                return "cpu"

            # Check CUDA first (most reliable)
            if torch.cuda.is_available():
                logger.info("Using CUDA device")
                return "cuda"

            # For MPS, be very conservative due to Metal shader issues
            if torch.backends.mps.is_available():
                # Disable MPS entirely for now due to Metal shader compilation issues
                logger.warning(
                    "MPS available but disabled due to Metal shader compilation issues"
                )
                return "cpu"

            # Fallback to CPU
            logger.info("Using CPU device")
            return "cpu"

        except Exception as e:
            logger.warning(f"Error detecting device, using CPU: {e}")
            return "cpu"

    def initialize_model(self) -> bool:
        """Initialize TinyLlama model with device fallback"""
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
                    logger.error("❌ Failed to load TinyLlama on any device")
                    return False

        return False

    def _load_model_on_device(self, device: str) -> bool:
        """Load model on specific device"""
        try:
            model_name = settings.AI_MODELS["LOCAL_MODELS"]["tinyllama"]["model_name"]
            logger.info(f"Loading TinyLlama on {device}...")

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Device-specific model loading
            if device == "cuda":
                model_kwargs = {
                    "torch_dtype": torch.float16,
                    "low_cpu_mem_usage": True,
                    "device_map": "auto",
                }
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name, **model_kwargs
                )

                self.pipeline = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device_map="auto",
                    torch_dtype=torch.float16,
                )

            elif device == "mps":
                # More conservative MPS settings
                model_kwargs = {
                    "torch_dtype": torch.float32,
                    "low_cpu_mem_usage": True,
                }
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name, **model_kwargs
                )

                # Move to MPS with error handling
                try:
                    self.model = self.model.to("mps")

                    self.pipeline = pipeline(
                        "text-generation",
                        model=self.model,
                        tokenizer=self.tokenizer,
                        device=0,  # Use device 0 for MPS
                        torch_dtype=torch.float32,
                    )
                except Exception as mps_error:
                    logger.error(f"MPS model placement failed: {mps_error}")
                    raise mps_error

            else:  # CPU
                model_kwargs = {
                    "torch_dtype": torch.float32,
                    "low_cpu_mem_usage": True,
                }
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name, **model_kwargs
                )

                self.pipeline = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=-1,  # -1 for CPU
                    torch_dtype=torch.float32,
                )

            # Create LangChain wrapper
            self.llm = HuggingFacePipeline(pipeline=self.pipeline)
            self.model_loaded = True
            logger.info(f"✅ TinyLlama loaded successfully on {device}!")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to load TinyLlama on {device}: {e}")
            self.cleanup_model()
            raise e

    def generate_response(
        self,
        prompt: str,
        max_tokens: int = 200,
        temperature: float = 0.7,
        system_prompt: str = "You are a helpful assistant.",
    ) -> Optional[str]:
        """Generate response using TinyLlama with error handling"""

        if not self.model_loaded and not self.initialize_model():
            return None

        try:
            # Format prompt for TinyLlama chat format
            formatted_prompt = self._format_chat_prompt(prompt, system_prompt)

            # Generate response with device-specific error handling
            try:
                outputs = self.pipeline(
                    formatted_prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    return_full_text=False,
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

            if outputs and len(outputs) > 0:
                response = outputs[0]["generated_text"].strip()
                # Clean up the response
                response = self._clean_response(response)
                return response

            return None

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return None

    def _format_chat_prompt(self, user_message: str, system_prompt: str) -> str:
        """Format prompt for TinyLlama chat format"""
        return f"<|system|>\n{system_prompt}</s>\n<|user|>\n{user_message}</s>\n<|assistant|>\n"

    def _clean_response(self, response: str) -> str:
        """Clean up model response"""
        # Remove special tokens and extra whitespace
        response = response.replace("<|", "").replace("|>", "").replace("</s>", "")
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

            logger.info("🧹 Model cleaned from memory")

        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")

    def is_available(self) -> bool:
        """Check if model is available"""
        return self.model_loaded or self.initialize_model()

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "name": "TinyLlama-1.1B-Chat",
            "device": self.device,
            "loaded": self.model_loaded,
            "memory_usage": self._get_memory_usage() if self.model_loaded else 0,
        }

    def _get_memory_usage(self) -> float:
        """Get current memory usage in GB"""
        try:
            if self.device == "cuda" and torch.cuda.is_available():
                return torch.cuda.memory_allocated() / 1e9
            elif self.device == "mps" and torch.backends.mps.is_available():
                # MPS doesn't have direct memory monitoring, return estimated
                return 0.5  # Rough estimate for TinyLlama
            return 0.0
        except Exception:
            return 0.0
