import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from django.conf import settings
from typing import Optional, Dict, Any
import logging
import gc

logger = logging.getLogger(__name__)


class TinyLlamaService:
    """Service for managing TinyLlama local model"""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        # Properly detect device with MPS support
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        self.model_loaded = False
        self.max_retries = 3

    def initialize_model(self) -> bool:
        """Initialize TinyLlama model"""
        if self.model_loaded:
            return True

        try:
            model_name = settings.AI_MODELS["LOCAL_MODELS"]["tinyllama"]["model_name"]

            logger.info(f"Loading TinyLlama on {self.device}...")

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model with proper device handling
            if self.device == "cuda":
                model_kwargs = {
                    "torch_dtype": torch.float16,
                    "low_cpu_mem_usage": True,
                    "device_map": "auto",
                }
            elif self.device == "mps":
                model_kwargs = {
                    "torch_dtype": torch.float32,  # MPS works better with float32
                    "low_cpu_mem_usage": True,
                }
            else:
                model_kwargs = {
                    "torch_dtype": torch.float32,
                    "low_cpu_mem_usage": True,
                }

            # Load model without automatic device mapping for MPS
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, **model_kwargs
            )

            # Move model to device after loading (except for CUDA with device_map)
            if self.device != "cuda":
                self.model = self.model.to(self.device)

            # Create pipeline with proper device configuration
            if self.device == "cuda":
                self.pipeline = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device_map="auto",
                    torch_dtype=torch.float16,
                )
            else:
                self.pipeline = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=0 if self.device == "mps" else -1,  # 0 for MPS, -1 for CPU
                    torch_dtype=torch.float32,
                )

            self.model_loaded = True
            logger.info("✅ TinyLlama loaded successfully!")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to load TinyLlama: {e}")
            self.cleanup_model()
            return False

    def generate_response(
        self,
        prompt: str,
        max_tokens: int = 200,
        temperature: float = 0.7,
        system_prompt: str = "You are a helpful assistant.",
    ) -> Optional[str]:
        """Generate response using TinyLlama"""

        if not self.model_loaded and not self.initialize_model():
            return None

        try:
            # Format prompt for TinyLlama chat format
            formatted_prompt = self._format_chat_prompt(prompt, system_prompt)

            # Generate response
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
        if self.model:
            del self.model
        if self.tokenizer:
            del self.tokenizer
        if self.pipeline:
            del self.pipeline

        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.model_loaded = False

        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

        logger.info("🧹 Model cleaned from memory")

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
        if self.device == "cuda" and torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1e9
        return 0.0
