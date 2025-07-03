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

            # Create sampler with MLX-LM API (only supported parameters)
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
                # Log raw response for debugging
                logger.debug(f"Raw response: {repr(response[:200])}")

                # Remove any remaining special tokens and the original prompt
                response = response.replace("<|end|>", "").strip()

                # Remove the prompt from the response if it's included
                if full_prompt in response:
                    response = response.replace(full_prompt, "").strip()

                # Clean up repetitive assistant tokens and patterns
                cleaned_response = self._clean_response(response)

                logger.debug(f"✅ Response generated in {generation_time:.2f}s")
                logger.debug(f"   Original length: {len(response)} chars")
                logger.debug(f"   Cleaned length: {len(cleaned_response)} chars")
                logger.debug(f"   Final response: {repr(cleaned_response[:100])}")

                # Ensure we have a meaningful response
                if len(cleaned_response.strip()) < 5:
                    logger.warning(
                        f"Response too short after cleaning: {repr(cleaned_response)}"
                    )
                    # Return original if cleaning made it too short
                    return response if len(response.strip()) >= 5 else None

                return cleaned_response
            else:
                logger.warning("Empty response generated")
                return None

        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            return None

    def _clean_response(self, response: str) -> str:
        """Clean up MLX response to remove repetitive patterns and tokens"""
        if not response:
            return response

        # Remove special tokens
        response = (
            response.replace("<|assistant|>", "")
            .replace("<|user|>", "")
            .replace("<|system|>", "")
        )
        response = response.replace("<|end|>", "").strip()

        # Split by assistant tokens and take only the first complete response
        parts = response.split("<|assistant|>")
        if len(parts) > 1:
            # Take the first non-empty part
            for part in parts:
                cleaned_part = part.strip()
                if (
                    cleaned_part and len(cleaned_part) > 3
                ):  # Minimum meaningful length (reduced)
                    response = cleaned_part
                    break

        # Advanced repetition removal - handle semantic duplicates
        response = self._remove_semantic_repetitions(response)

        # Ensure proper ending
        if response and not response.endswith((".", "!", "?")):
            response += "."

        return response.strip()

    def _remove_semantic_repetitions(self, text: str) -> str:
        """Remove semantically similar repetitive sentences"""
        if not text:
            return text

        # For very short responses, be less aggressive
        if len(text) < 50:
            # Just remove exact duplicates for short responses
            return self._remove_exact_duplicates(text)

        # Split into sentences more carefully
        import re

        # Split by sentence endings but preserve them
        sentences = re.split(r"([.!?]+)", text)

        # Reconstruct sentences with their punctuation
        full_sentences = []
        for i in range(0, len(sentences) - 1, 2):
            sentence = sentences[i].strip()
            punctuation = sentences[i + 1] if i + 1 < len(sentences) else ""
            if sentence:
                full_sentences.append(sentence + punctuation)

        # If we have a final sentence without punctuation
        if len(sentences) % 2 == 1 and sentences[-1].strip():
            full_sentences.append(sentences[-1].strip())

        # Remove exact duplicates and very similar sentences
        unique_sentences = []
        seen_content = set()

        for sentence in full_sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Create a normalized version for comparison
            normalized = self._normalize_sentence(sentence)

            # Check if this is a meaningful sentence (not just punctuation)
            if len(normalized) < 3:  # Reduced threshold
                continue

            # Check for exact matches and very similar content
            is_duplicate = False
            for seen in seen_content:
                if self._sentences_too_similar(normalized, seen):
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_sentences.append(sentence)
                seen_content.add(normalized)

        # Join sentences back together
        result = " ".join(unique_sentences)

        # Clean up extra spaces
        result = re.sub(r"\s+", " ", result).strip()

        return result

    def _remove_exact_duplicates(self, text: str) -> str:
        """Remove only exact duplicate sentences for short responses"""
        import re

        # Split by sentence endings but preserve them
        sentences = re.split(r"([.!?]+)", text)

        # Reconstruct sentences with their punctuation
        full_sentences = []
        for i in range(0, len(sentences) - 1, 2):
            sentence = sentences[i].strip()
            punctuation = sentences[i + 1] if i + 1 < len(sentences) else ""
            if sentence:
                full_sentences.append(sentence + punctuation)

        # If we have a final sentence without punctuation
        if len(sentences) % 2 == 1 and sentences[-1].strip():
            full_sentences.append(sentences[-1].strip())

        # Remove only exact duplicates
        unique_sentences = []
        seen_exact = set()

        for sentence in full_sentences:
            sentence_clean = sentence.strip()
            if sentence_clean and sentence_clean.lower() not in seen_exact:
                unique_sentences.append(sentence)
                seen_exact.add(sentence_clean.lower())

        result = " ".join(unique_sentences)
        return re.sub(r"\s+", " ", result).strip()

    def _normalize_sentence(self, sentence: str) -> str:
        """Normalize sentence for comparison"""
        import re

        # Remove punctuation and convert to lowercase
        normalized = re.sub(r"[^\w\s]", "", sentence.lower())

        # Remove extra whitespace
        normalized = re.sub(r"\s+", " ", normalized).strip()

        return normalized

    def _sentences_too_similar(
        self, sent1: str, sent2: str, threshold: float = 0.6
    ) -> bool:
        """Check if two sentences are too similar using simple word overlap"""
        if not sent1 or not sent2:
            return False

        # Exact match
        if sent1 == sent2:
            return True

        # Check for key phrase repetition (common in factual statements)
        # Extract key phrases (3+ consecutive words)
        import re

        words1 = sent1.split()
        words2 = sent2.split()

        # Check for overlapping 3-word phrases
        if len(words1) >= 3 and len(words2) >= 3:
            phrases1 = set()
            phrases2 = set()

            for i in range(len(words1) - 2):
                phrase = " ".join(words1[i : i + 3]).lower()
                phrases1.add(phrase)

            for i in range(len(words2) - 2):
                phrase = " ".join(words2[i : i + 3]).lower()
                phrases2.add(phrase)

            # If they share significant phrase overlap, they're likely repetitive
            if phrases1 and phrases2:
                phrase_overlap = len(phrases1.intersection(phrases2))
                phrase_union = len(phrases1.union(phrases2))
                phrase_similarity = (
                    phrase_overlap / phrase_union if phrase_union > 0 else 0
                )

                if phrase_similarity >= 0.5:  # 50% phrase overlap indicates repetition
                    return True

        # Word-based similarity
        words1_set = set(word.lower() for word in words1)
        words2_set = set(word.lower() for word in words2)

        if not words1_set or not words2_set:
            return False

        # Calculate Jaccard similarity
        intersection = len(words1_set.intersection(words2_set))
        union = len(words1_set.union(words2_set))

        similarity = intersection / union if union > 0 else 0

        return similarity >= threshold

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
