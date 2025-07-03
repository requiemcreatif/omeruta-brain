import os
import time
import logging
from typing import Dict, Any, Optional, List
import threading
import multiprocessing

logger = logging.getLogger(__name__)

try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler

    MLX_AVAILABLE = True
    logger.info("✅ MLX framework available")
except ImportError as e:
    MLX_AVAILABLE = False
    logger.warning(f"❌ MLX not available: {e}")
    logger.warning("Install with: pip install mlx-lm")


class MLXPhi3Service:
    """
    MLX-optimized Phi3 service for Apple Silicon
    Provides significant performance improvements over CPU-only inference
    """

    def __init__(
        self,
        model_name: str = "microsoft/Phi-3-mini-128k-instruct",
        auto_load: bool = True,
    ):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        self.load_time = 0
        self.device_info = self._get_device_info()
        self._lock = threading.Lock()
        self._process_id = os.getpid()

        # MLX-specific settings
        self.max_tokens = 512
        self.temperature = 0.7
        self.top_p = 0.9

        logger.info(f"🚀 Initializing MLX Phi3 Service (PID: {self._process_id})")
        logger.info(f"   Model: {model_name}")
        logger.info(f"   Device: {self.device_info}")

        # Check for multiprocessing safety
        if self._is_multiprocessing_safe():
            # Auto-load model if requested
            if auto_load and MLX_AVAILABLE:
                logger.info("🔄 Auto-loading model...")
                self.load_model()
        else:
            logger.warning(
                "⚠️ Multiprocessing detected - delaying model load for safety"
            )

    def _is_multiprocessing_safe(self) -> bool:
        """Check if we're in a safe environment for MLX loading"""
        try:
            # Check if we're in a forked process
            if hasattr(multiprocessing, "current_process"):
                current_process = multiprocessing.current_process()
                if current_process.name != "MainProcess":
                    logger.warning(f"Running in subprocess: {current_process.name}")
                    return False

            # Check if we're in Celery worker
            if "celery" in str(multiprocessing.current_process().name).lower():
                logger.warning("Running in Celery worker - using lazy loading")
                return False

            return True
        except Exception as e:
            logger.warning(f"Error checking multiprocessing safety: {e}")
            return False

    def _get_device_info(self) -> Dict[str, Any]:
        """Get Apple Silicon device information"""
        try:
            if MLX_AVAILABLE:
                # MLX automatically uses the best available device
                return {
                    "framework": "MLX",
                    "device": "Apple Silicon (GPU + CPU unified)",
                    "memory_type": "unified",
                    "optimization": "Native Apple Silicon",
                    "process_id": os.getpid(),
                }
            else:
                return {
                    "framework": "None",
                    "device": "MLX not available",
                    "memory_type": "system",
                    "optimization": "None",
                    "process_id": os.getpid(),
                }
        except Exception as e:
            logger.error(f"Error getting device info: {e}")
            return {"error": str(e), "process_id": os.getpid()}

    def load_model(self) -> bool:
        """Load the Phi3 model using MLX with multiprocessing safety"""
        if not MLX_AVAILABLE:
            logger.error("MLX not available - cannot load model")
            return False

        if self.is_loaded:
            # Check if we're in the same process
            current_pid = os.getpid()
            if current_pid != self._process_id:
                logger.warning(
                    f"Process changed from {self._process_id} to {current_pid} - reloading model"
                )
                self.is_loaded = False
                self.model = None
                self.tokenizer = None
                self._process_id = current_pid
            else:
                logger.info("Model already loaded in current process")
                return True

        try:
            with self._lock:
                if (
                    self.is_loaded and os.getpid() == self._process_id
                ):  # Double-check after acquiring lock
                    return True

                logger.info(
                    f"🔄 Loading {self.model_name} with MLX (PID: {os.getpid()})..."
                )
                start_time = time.time()

                # Load model and tokenizer using MLX with error handling
                try:
                    self.model, self.tokenizer = load(self.model_name)
                except Exception as load_error:
                    logger.error(f"MLX load failed: {load_error}")
                    # Try to clean up any partial state
                    self.model = None
                    self.tokenizer = None
                    return False

                self.load_time = time.time() - start_time
                self.is_loaded = True
                self._process_id = os.getpid()

                logger.info(
                    f"✅ Model loaded successfully in {self.load_time:.2f}s (PID: {self._process_id})"
                )
                logger.info(f"   Memory: Unified Apple Silicon memory")
                logger.info(f"   Optimization: Native MLX acceleration")

                return True

        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            self.is_loaded = False
            self.model = None
            self.tokenizer = None
            return False

    def generate_response(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> Optional[str]:
        """Generate response using MLX-optimized inference with safety checks"""

        if not self.is_available():
            logger.error("Model not available for generation")
            # Try to load model if not loaded
            if not self.is_loaded:
                logger.info("Attempting to load model on-demand...")
                if not self.load_model():
                    return None
            else:
                return None

        try:
            # Use provided parameters or defaults
            max_tokens = max_tokens or self.max_tokens
            temperature = temperature or self.temperature
            top_p = kwargs.get("top_p", self.top_p)

            # Prepare the full prompt using Phi-3 chat format
            if system_prompt:
                full_prompt = f"<|system|>\n{system_prompt}<|end|>\n<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
            else:
                full_prompt = f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"

            logger.debug(
                f"Generating response with MLX (max_tokens={max_tokens}, temp={temperature}, top_p={top_p})"
            )

            start_time = time.time()

            # Create sampler with MLX-LM API
            sampler = make_sampler(
                temp=temperature,
                top_p=top_p,
                top_k=kwargs.get("top_k", 0),  # 0 means no top_k filtering
            )

            # Generate using MLX-LM with proper API and error handling
            try:
                response = generate(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    prompt=full_prompt,
                    max_tokens=max_tokens,
                    sampler=sampler,
                    verbose=False,
                )
            except Exception as gen_error:
                logger.error(f"MLX generation failed: {gen_error}")
                # Try to reload model if generation fails
                logger.info("Attempting to reload model after generation failure...")
                self.unload_model()
                if self.load_model():
                    logger.info("Model reloaded, retrying generation...")
                    response = generate(
                        model=self.model,
                        tokenizer=self.tokenizer,
                        prompt=full_prompt,
                        max_tokens=max_tokens,
                        sampler=sampler,
                        verbose=False,
                    )
                else:
                    return None

            generation_time = time.time() - start_time

            # Clean up the response
            if response:
                # Remove any remaining special tokens and the original prompt
                response = response.replace("<|end|>", "").strip()

                # Remove the prompt from the response if it's included
                if full_prompt in response:
                    response = response.replace(full_prompt, "").strip()

                logger.debug(f"✅ Response generated in {generation_time:.2f}s")
                logger.debug(f"   Response length: {len(response)} chars")

                return response
            else:
                logger.warning("Empty response generated")
                return None

        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            return None

    def is_available(self) -> bool:
        """Check if the model is loaded and ready"""
        # Check if we're in the same process
        if self.is_loaded and os.getpid() != self._process_id:
            logger.warning(f"Process changed - marking model as unavailable")
            self.is_loaded = False
            self.model = None
            self.tokenizer = None

        return MLX_AVAILABLE and self.is_loaded and self.model is not None

    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information"""
        info = {
            "name": self.model_name,
            "loaded": self.is_loaded,
            "mlx_available": MLX_AVAILABLE,
            "load_time_seconds": float(self.load_time),
            "device_info": self.device_info,
            "framework": "MLX" if MLX_AVAILABLE else "None",
            "optimization": "Apple Silicon Native" if MLX_AVAILABLE else "None",
            "device": str(self.device_info.get("device", "unknown")),
            "process_id": os.getpid(),
            "original_process_id": self._process_id,
        }

        if self.is_loaded:
            info.update(
                {
                    "max_tokens": int(self.max_tokens),
                    "temperature": float(self.temperature),
                    "top_p": float(self.top_p),
                    "memory_usage": "Unified Apple Silicon Memory",
                    "performance": "GPU + CPU Unified Processing",
                }
            )

        return info

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            "framework": "MLX",
            "device": "Apple Silicon",
            "memory_model": "Unified",
            "load_time_seconds": self.load_time,
            "available": self.is_available(),
            "optimization_level": "Native" if MLX_AVAILABLE else "None",
            "expected_speedup": "5-10x vs CPU" if MLX_AVAILABLE else "None",
            "process_id": os.getpid(),
            "multiprocessing_safe": self._is_multiprocessing_safe(),
        }

    def unload_model(self):
        """Unload the model to free memory"""
        with self._lock:
            if self.is_loaded:
                self.model = None
                self.tokenizer = None
                self.is_loaded = False
                logger.info("🗑️ Model unloaded")

    def __del__(self):
        """Cleanup on destruction"""
        try:
            self.unload_model()
        except:
            pass


# Global instance with lazy loading
_mlx_phi3_instance = None
_instance_lock = threading.Lock()


def get_mlx_phi3_service() -> MLXPhi3Service:
    """Get shared MLX Phi3 service instance with lazy loading"""
    global _mlx_phi3_instance

    if _mlx_phi3_instance is None:
        with _instance_lock:
            if _mlx_phi3_instance is None:
                # Use lazy loading for multiprocessing safety
                _mlx_phi3_instance = MLXPhi3Service(auto_load=False)

    return _mlx_phi3_instance
