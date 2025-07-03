import time
import hashlib
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

# Force CPU-only processing BEFORE importing PyTorch-based libraries
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# Force PyTorch to use CPU
import torch

torch.set_default_device("cpu")
if hasattr(torch.backends, "mps"):
    torch.backends.mps.is_available = lambda: False
    torch.backends.mps.is_built = lambda: False

from sentence_transformers import SentenceTransformer, CrossEncoder
from django.db import connection
from django.conf import settings
from django.core.cache import cache
from ..models import KnowledgeEmbedding, QueryCache
from crawler.models import CrawledPage
import logging

# Suppress linter warnings for Django model managers
# pylint: disable=no-member

logger = logging.getLogger(__name__)


class PgVectorSearchService:
    """Production-ready vector search with pgvector and advanced RAG"""

    def __init__(self):
        self.embedding_model = None
        self.cross_encoder = None
        self.config = settings.VECTOR_SETTINGS
        self._initialize_models()

    def _initialize_models(self):
        """Initialize embedding and cross-encoder models"""
        try:
            # Set Hugging Face token if available
            hf_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
            if hf_token:
                os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
                logger.info("🔑 Using Hugging Face authentication token")

            # Initialize embedding model with explicit CPU device
            self.embedding_model = SentenceTransformer(
                self.config["EMBEDDING_MODEL"],
                device="cpu",  # Use CPU to avoid GPU conflicts with main model
                use_auth_token=hf_token if hf_token else None,
            )
            logger.info(f"✅ Embedding model loaded: {self.config['EMBEDDING_MODEL']}")

            # Initialize cross-encoder for re-ranking if enabled
            if self.config.get("USE_CROSS_ENCODER", False):
                self.cross_encoder = CrossEncoder(
                    self.config["CROSS_ENCODER_MODEL"],
                    device="cpu",
                )
                logger.info(
                    f"✅ Cross-encoder loaded: {self.config['CROSS_ENCODER_MODEL']}"
                )

        except Exception as e:
            logger.error(f"❌ Failed to initialize search models: {e}")
            # If rate limited, provide helpful message
            if "429" in str(e) or "rate limit" in str(e).lower():
                logger.warning(
                    "🚨 Hugging Face rate limit detected. Consider adding HUGGINGFACE_TOKEN to environment variables."
                )
                logger.warning(
                    "   Get a free token at: https://huggingface.co/settings/tokens"
                )
            self.embedding_model = None
            self.cross_encoder = None

    def enhanced_search(
        self, query: str, filters: Dict = None, use_cache: bool = True
    ) -> Dict[str, Any]:
        """Enhanced search with query expansion, re-ranking, and caching"""

        start_time = time.time()

        # Check cache first
        if use_cache:
            cached_result = self._get_cached_result(query)
            if cached_result:
                logger.info(f"🎯 Cache hit for query: {query[:50]}...")
                return cached_result

        # Step 1: Query expansion
        expanded_queries = self._expand_query(query)

        # Step 2: Multi-query vector search
        all_candidates = []
        for expanded_query in expanded_queries:
            candidates = self._vector_search(
                expanded_query, filters, top_k=self.config["RERANK_TOP_K"]
            )
            all_candidates.extend(candidates)

        # Step 3: Remove duplicates and combine scores
        unique_candidates = self._deduplicate_results(all_candidates)

        # Step 4: Re-rank with cross-encoder if available
        if self.cross_encoder and len(unique_candidates) > self.config["FINAL_TOP_K"]:
            reranked_results = self._rerank_with_cross_encoder(query, unique_candidates)
        else:
            reranked_results = unique_candidates

        # Step 5: Ensure diversity
        final_results = self._ensure_diversity(
            reranked_results[: self.config["FINAL_TOP_K"]]
        )

        # Step 6: Prepare response
        retrieval_time = (time.time() - start_time) * 1000  # Convert to ms

        response = {
            "results": final_results,
            "retrieval_time_ms": int(retrieval_time),
            "total_candidates": len(all_candidates),
            "unique_candidates": len(unique_candidates),
            "final_count": len(final_results),
            "query_expansions": expanded_queries,
            "used_cross_encoder": bool(self.cross_encoder),
        }

        # Cache the result
        if use_cache:
            self._cache_result(query, response)

        return response

    def _expand_query(self, query: str) -> List[str]:
        """Expand query with variations for better recall"""

        if not self.config.get("QUERY_EXPANSION", True):
            return [query]

        expansions = [query]  # Original query first

        # Add question variations
        query_lower = query.lower().strip()

        if not query_lower.startswith(("what", "how", "why", "when", "where", "who")):
            expansions.extend(
                [
                    f"What is {query}?",
                    f"How does {query} work?",
                    f"Explain {query}",
                ]
            )

        # Add related terms
        if len(query.split()) <= 3:  # Only for short queries
            expansions.extend(
                [
                    f"{query} explanation",
                    f"{query} definition",
                    f"{query} overview",
                ]
            )

        return expansions[:5]  # Limit to 5 expansions

    def _vector_search(
        self, query: str, filters: Dict = None, top_k: int = 10
    ) -> List[Dict]:
        """Perform vector similarity search using pgvector"""

        if not self.embedding_model:
            logger.error("Embedding model not available")
            return []

        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query)

            # Build SQL query with filters
            sql_query = """
                SELECT
                    ke.id,
                    ke.chunk_text,
                    ke.chunk_index,
                    ke.metadata,
                    ke.content_quality_score,
                    ke.semantic_density,
                    cp.title as page_title,
                    cp.url as page_url,
                    cp.meta_description,
                    ke.embedding <=> %s::vector as distance
                FROM knowledge_base_knowledgeembedding ke
                JOIN crawler_crawledpage cp ON ke.page_id = cp.id
                WHERE cp.success = true
                AND ke.content_quality_score > 0.3
            """

            params = [query_embedding.tolist()]

            # Add filters
            if filters:
                if filters.get("min_quality"):
                    sql_query += " AND ke.content_quality_score >= %s"
                    params.append(filters["min_quality"])

                if filters.get("page_urls"):
                    placeholders = ",".join(["%s"] * len(filters["page_urls"]))
                    sql_query += f" AND cp.url IN ({placeholders})"
                    params.extend(filters["page_urls"])

                if filters.get("content_types"):
                    sql_query += " AND ke.metadata->>'has_code' = %s"
                    params.append(
                        str(filters["content_types"].get("code", False)).lower()
                    )

            # Add similarity threshold and ordering
            sql_query += """
                AND ke.embedding <=> %s::vector < %s
                ORDER BY distance ASC
                LIMIT %s
            """
            params.extend([query_embedding.tolist(), 0.8, top_k])

            # Execute query
            with connection.cursor() as cursor:
                cursor.execute(sql_query, params)
                rows = cursor.fetchall()

            # Convert to result format
            results = []
            for row in rows:
                results.append(
                    {
                        "id": str(row[0]),
                        "text": row[1],
                        "chunk_index": row[2],
                        "metadata": row[3],
                        "quality_score": row[4],
                        "semantic_density": row[5],
                        "page_title": row[6],
                        "page_url": row[7],
                        "page_description": row[8],
                        "distance": row[9],
                        "similarity": 1 - row[9],  # Convert distance to similarity
                    }
                )

            return results

        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return []

    def _deduplicate_results(self, candidates: List[Dict]) -> List[Dict]:
        """Remove duplicate results and combine scores"""

        seen_chunks = {}

        for candidate in candidates:
            chunk_id = candidate["id"]

            if chunk_id in seen_chunks:
                # Combine scores (take the best)
                existing = seen_chunks[chunk_id]
                if candidate["similarity"] > existing["similarity"]:
                    seen_chunks[chunk_id] = candidate
            else:
                seen_chunks[chunk_id] = candidate

        # Sort by similarity
        unique_results = list(seen_chunks.values())
        unique_results.sort(key=lambda x: x["similarity"], reverse=True)

        return unique_results

    def _rerank_with_cross_encoder(
        self, query: str, candidates: List[Dict]
    ) -> List[Dict]:
        """Re-rank results using cross-encoder for better relevance"""

        if not self.cross_encoder:
            return candidates

        try:
            # Prepare query-document pairs
            pairs = [(query, candidate["text"]) for candidate in candidates]

            # Get cross-encoder scores
            cross_scores = self.cross_encoder.predict(pairs)

            # Combine vector similarity and cross-encoder scores
            for i, candidate in enumerate(candidates):
                vector_score = candidate["similarity"]
                cross_score = cross_scores[i]

                # Weighted combination: 60% vector + 40% cross-encoder
                combined_score = 0.6 * vector_score + 0.4 * cross_score
                candidate["cross_encoder_score"] = float(cross_score)
                candidate["combined_score"] = combined_score

            # Sort by combined score
            candidates.sort(key=lambda x: x["combined_score"], reverse=True)

            return candidates

        except Exception as e:
            logger.error(f"Cross-encoder re-ranking error: {e}")
            return candidates

    def _ensure_diversity(self, results: List[Dict]) -> List[Dict]:
        """Ensure diversity in final results to avoid repetitive content"""

        if not results or len(results) <= 2:
            return results

        diverse_results = [results[0]]  # Always include the top result

        for candidate in results[1:]:
            # Check similarity with already selected results
            too_similar = False

            for selected in diverse_results:
                # Simple text overlap check
                candidate_words = set(candidate["text"].lower().split())
                selected_words = set(selected["text"].lower().split())

                overlap = len(candidate_words & selected_words)
                union = len(candidate_words | selected_words)

                if union > 0:
                    jaccard_similarity = overlap / union
                    if jaccard_similarity > self.config.get("DIVERSITY_THRESHOLD", 0.8):
                        too_similar = True
                        break

            if not too_similar:
                diverse_results.append(candidate)

            # Stop when we have enough diverse results
            if len(diverse_results) >= self.config["FINAL_TOP_K"]:
                break

        return diverse_results

    def _get_cached_result(self, query: str) -> Optional[Dict]:
        """Get cached result for query"""

        query_hash = hashlib.sha256(query.encode()).hexdigest()
        cache_key = f"search_result:{query_hash}"

        try:
            cached = cache.get(cache_key)
            if cached:
                # Update access tracking
                from django.utils import timezone
                from django.db import models

                QueryCache.objects.filter(query_hash=query_hash).update(
                    access_count=models.F("access_count") + 1,
                    last_accessed=timezone.now(),
                )
                return cached
        except Exception as e:
            logger.error(f"Cache retrieval error: {e}")

        return None

    def _cache_result(self, query: str, result: Dict):
        """Cache search result"""

        query_hash = hashlib.sha256(query.encode()).hexdigest()
        cache_key = f"search_result:{query_hash}"

        try:
            # Cache in Redis for fast access
            cache.set(cache_key, result, timeout=3600)  # 1 hour

            # Store in database for persistence and analytics
            QueryCache.objects.update_or_create(
                query_hash=query_hash,
                defaults={
                    "original_query": query,
                    "expanded_queries": result.get("query_expansions", []),
                    "relevant_chunks": [r["id"] for r in result["results"]],
                    "retrieval_time_ms": result["retrieval_time_ms"],
                },
            )
        except Exception as e:
            logger.error(f"Cache storage error: {e}")

    def get_search_stats(self) -> Dict[str, Any]:
        """Get search service statistics"""

        stats = {
            "embedding_model_available": bool(self.embedding_model),
            "cross_encoder_available": bool(self.cross_encoder),
            "embedding_model_name": self.config["EMBEDDING_MODEL"],
            "cross_encoder_name": self.config.get("CROSS_ENCODER_MODEL", "None"),
            "config": self.config,
        }

        if self.embedding_model:
            stats["embedding_dimensions"] = self.config["EMBEDDING_DIMENSIONS"]

        return stats
