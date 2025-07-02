import os
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import numpy as np
import torch
from django.conf import settings
from crawler.models import CrawledPage
import logging

logger = logging.getLogger(__name__)


class EnhancedVectorSearchService:
    """Enhanced search service with local embeddings"""

    def __init__(self):
        self.embedding_model = None
        # Detect device properly
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "cpu"  # Use CPU for embeddings on MPS to avoid issues
        else:
            self.device = "cpu"
        self._load_embedding_model()

    def _load_embedding_model(self):
        """Load local embedding model"""
        try:
            # Set Hugging Face token if available
            hf_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
            if hf_token:
                os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token

            model_name = settings.VECTOR_SETTINGS["EMBEDDING_MODEL"]

            # Load with specific device configuration
            self.embedding_model = SentenceTransformer(
                model_name,
                device=self.device,
                use_auth_token=hf_token if hf_token else None,
            )

            logger.info(f"✅ Embedding model {model_name} loaded on {self.device}")
        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {e}")
            # If rate limited, provide helpful message
            if "429" in str(e) or "rate limit" in str(e).lower():
                logger.warning(
                    "🚨 Hugging Face rate limit detected. Consider adding HUGGINGFACE_TOKEN to environment variables."
                )
                logger.warning(
                    "   Get a free token at: https://huggingface.co/settings/tokens"
                )

            # Fallback: try loading on CPU
            try:
                model_name = settings.VECTOR_SETTINGS["EMBEDDING_MODEL"]
                self.embedding_model = SentenceTransformer(
                    model_name,
                    device="cpu",
                    use_auth_token=hf_token if hf_token else None,
                )
                self.device = "cpu"
                logger.info(f"✅ Embedding model {model_name} loaded on CPU (fallback)")
            except Exception as fallback_error:
                logger.error(
                    f"❌ Failed to load embedding model on CPU fallback: {fallback_error}"
                )
                self.embedding_model = None

    def search_crawled_content(
        self, query: str, limit: int = 5, threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Search through crawled content using your existing data"""

        if not self.embedding_model:
            return []

        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])[0]

            # Get relevant crawled pages
            pages = CrawledPage.objects.filter(  # pylint: disable=no-member
                success=True, clean_markdown__isnull=False
            ).exclude(clean_markdown="")[
                :100
            ]  # Limit for performance

            results = []
            for page in pages:
                content = page.clean_markdown or ""
                if len(content) < 50:  # Skip very short content
                    continue

                # Chunk the content
                chunks = self._chunk_content(content)

                for i, chunk in enumerate(chunks):
                    # Generate chunk embedding
                    chunk_embedding = self.embedding_model.encode([chunk])[0]

                    # Calculate similarity
                    similarity = np.dot(query_embedding, chunk_embedding) / (
                        np.linalg.norm(query_embedding)
                        * np.linalg.norm(chunk_embedding)
                    )

                    if similarity > threshold:
                        results.append(
                            {
                                "content": chunk,
                                "similarity": float(similarity),
                                "page_title": page.title or "Untitled",
                                "page_url": page.url,
                                "page_id": str(page.id),
                                "chunk_index": i,
                            }
                        )

            # Sort by similarity and return top results
            results.sort(key=lambda x: x["similarity"], reverse=True)
            return results[:limit]

        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

    def _chunk_content(
        self, content: str, chunk_size: int = 1000, overlap: int = 100
    ) -> List[str]:
        """Split content into overlapping chunks"""
        words = content.split()
        chunks = []

        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i : i + chunk_size]
            chunk = " ".join(chunk_words)
            chunks.append(chunk)

            if i + chunk_size >= len(words):
                break

        return chunks

    def get_context_for_query(self, query: str, max_length: int = 2000) -> str:
        """Get relevant context for a query from your crawled data"""
        search_results = self.search_crawled_content(query, limit=10)

        context_pieces = []
        total_length = 0

        for result in search_results:
            content = result["content"]
            source = f"Source: {result['page_title']}"

            # Truncate content if too long for a single piece
            max_content_length = (
                max_length - len(source) - 50
            )  # Leave room for formatting
            if len(content) > max_content_length:
                content = content[:max_content_length] + "..."

            piece = f"{source}\n{content}"

            if total_length + len(piece) > max_length:
                # Try to fit a truncated version
                remaining_space = max_length - total_length - len(source) - 50
                if remaining_space > 100:  # Only add if meaningful content can fit
                    truncated_content = content[:remaining_space] + "..."
                    piece = f"{source}\n{truncated_content}"
                    context_pieces.append(piece)
                break

            context_pieces.append(piece)
            total_length += len(piece)

        return "\n\n---\n\n".join(context_pieces)
