import time
import logging
from typing import List, Dict, Any, Optional
from django.conf import settings
from .pgvector_search import PgVectorSearchService

logger = logging.getLogger(__name__)


class EnhancedRAGService:
    """Production-ready RAG service with advanced retrieval and generation"""

    def __init__(self):
        self.search_service = PgVectorSearchService()
        self.config = settings.VECTOR_SETTINGS

    def generate_response(
        self, query: str, context_filters: Dict = None, response_config: Dict = None
    ) -> Dict[str, Any]:
        """Generate enhanced response using RAG pipeline"""

        start_time = time.time()

        # Step 1: Retrieve relevant context
        search_result = self.search_service.enhanced_search(
            query=query, filters=context_filters or {}
        )

        # Step 2: Build context for LLM
        context = self._build_context(search_result["results"])

        # Step 3: Prepare enhanced prompt
        enhanced_prompt = self._create_enhanced_prompt(
            query, context, response_config or {}
        )

        total_time = (time.time() - start_time) * 1000

        return {
            "enhanced_prompt": enhanced_prompt,
            "context": context,
            "search_metadata": {
                "retrieval_time_ms": search_result["retrieval_time_ms"],
                "total_time_ms": int(total_time),
                "sources_used": len(search_result["results"]),
                "query_expansions": search_result["query_expansions"],
                "used_cross_encoder": search_result["used_cross_encoder"],
            },
            "sources": [
                {
                    "title": result["page_title"],
                    "url": result["page_url"],
                    "relevance_score": result.get(
                        "combined_score", result["similarity"]
                    ),
                    "quality_score": result["quality_score"],
                    "chunk_preview": (
                        result["text"][:150] + "..."
                        if len(result["text"]) > 150
                        else result["text"]
                    ),
                }
                for result in search_result["results"]
            ],
        }

    def _build_context(self, search_results: List[Dict]) -> str:
        """Build coherent context from search results"""

        if not search_results:
            return "No relevant information found in the knowledge base."

        # Group results by page to maintain coherence
        pages = {}
        for result in search_results:
            page_url = result["page_url"]
            if page_url not in pages:
                pages[page_url] = {
                    "title": result["page_title"],
                    "url": page_url,
                    "chunks": [],
                }
            pages[page_url]["chunks"].append(result)

        # Sort chunks within each page by index
        for page_data in pages.values():
            page_data["chunks"].sort(key=lambda x: x["chunk_index"])

        # Build context string
        context_parts = []

        for page_data in pages.values():
            # Add page header
            context_parts.append(f"\n--- Source: {page_data['title']} ---")
            context_parts.append(f"URL: {page_data['url']}")

            # Add chunks
            for chunk in page_data["chunks"]:
                context_parts.append(f"\n{chunk['text']}")

        return "\n".join(context_parts)

    def _create_enhanced_prompt(self, query: str, context: str, config: Dict) -> str:
        """Create enhanced prompt with context and instructions"""

        response_style = config.get("style", "informative")
        max_length = config.get("max_length", "medium")
        include_sources = config.get("include_sources", True)

        # Style-specific instructions
        style_instructions = {
            "concise": "Provide a brief, direct answer focusing on the most important points.",
            "detailed": "Provide a comprehensive, detailed explanation with examples where relevant.",
            "analytical": "Analyze the information critically and provide insights or conclusions.",
            "informative": "Provide a clear, well-structured explanation that's easy to understand.",
        }

        # Length instructions
        length_instructions = {
            "short": "Keep the response to 1-2 paragraphs.",
            "medium": "Aim for 2-4 paragraphs with good detail.",
            "long": "Provide a thorough response with multiple sections if needed.",
        }

        prompt = f"""You are an AI assistant with access to a knowledge base. Answer the user's question based on the provided context.

CONTEXT:
{context}

QUESTION: {query}

INSTRUCTIONS:
- {style_instructions.get(response_style, style_instructions['informative'])}
- {length_instructions.get(max_length, length_instructions['medium'])}
- Base your answer primarily on the provided context
- If the context doesn't contain enough information, acknowledge this clearly
- Be accurate and avoid making assumptions beyond what's in the context"""

        if include_sources:
            prompt += "\n- Reference specific sources when possible"

        prompt += "\n\nRESPONSE:"

        return prompt

    def evaluate_response_quality(
        self, query: str, response: str, context: str
    ) -> Dict[str, float]:
        """Evaluate response quality metrics"""

        # Simple heuristic-based evaluation
        # In production, you might use more sophisticated methods

        scores = {}

        # Relevance: Check if response addresses the query
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        query_coverage = (
            len(query_words & response_words) / len(query_words) if query_words else 0
        )
        scores["relevance"] = min(1.0, query_coverage * 2)  # Scale to 0-1

        # Context usage: Check if response uses provided context
        context_words = set(context.lower().split())
        context_usage = (
            len(context_words & response_words) / len(context_words)
            if context_words
            else 0
        )
        scores["context_usage"] = min(1.0, context_usage * 10)  # Scale to 0-1

        # Completeness: Check response length relative to query complexity
        query_complexity = len(query.split())
        response_length = len(response.split())

        if query_complexity <= 5:  # Simple query
            ideal_length = 30
        elif query_complexity <= 10:  # Medium query
            ideal_length = 60
        else:  # Complex query
            ideal_length = 100

        length_ratio = min(response_length / ideal_length, 1.0)
        scores["completeness"] = length_ratio

        # Overall score
        scores["overall"] = (
            scores["relevance"] + scores["context_usage"] + scores["completeness"]
        ) / 3

        return scores

    def get_rag_stats(self) -> Dict[str, Any]:
        """Get RAG service statistics"""

        search_stats = self.search_service.get_search_stats()

        return {
            "search_service_status": search_stats,
            "config": self.config,
            "rag_ready": bool(search_stats.get("embedding_model_available", False)),
        }
