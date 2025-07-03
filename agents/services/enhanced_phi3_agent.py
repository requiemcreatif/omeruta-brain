import time
import logging
from typing import Dict, Any, List
from django.db import connection
from knowledge_base.models import KnowledgeEmbedding
from knowledge_base.services.embedding_generator import EmbeddingGenerationService

# Try to import MLX service first, fallback to CPU
try:
    from .mlx_phi3_service import MLXPhi3Service, MLX_AVAILABLE

    if MLX_AVAILABLE:
        logger = logging.getLogger(__name__)
        logger.info("🚀 MLX available - using Apple Silicon optimization")
except ImportError:
    MLX_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("MLX not available - falling back to CPU implementation")

# Fallback to CPU implementation
if not MLX_AVAILABLE:
    from .phi3_model_service import Phi3ModelService

# pylint: disable=no-member
logger = logging.getLogger(__name__)


class EnhancedPhi3Agent:
    """Enhanced Phi-3 agent with MLX optimization for Apple Silicon and CPU fallback"""

    def __init__(self, agent_type: str = "general"):
        self.agent_type = agent_type
        self.embedding_service = EmbeddingGenerationService()
        self.use_mlx = MLX_AVAILABLE

        # Initialize the appropriate service
        if self.use_mlx:
            logger.info("🚀 Using MLX-optimized Phi3 service for Apple Silicon")
            self.llm_service = MLXPhi3Service()
        else:
            logger.info("🔄 Using CPU-based Phi3 service")
            self.llm_service = Phi3ModelService()

        # Enhanced system prompts optimized for Phi-3's capabilities
        self.system_prompts = {
            "general": """You are Omeruta Brain, an intelligent AI assistant with access to a knowledge base and enhanced by Phi-3's 128K context window.
            
CRITICAL INSTRUCTIONS:
- Answer ONLY based on the provided context below
- If the context is empty or insufficient, clearly state that you don't have enough information
- Never speculate about future events or make predictions
- If asked about future dates (like 2026), explain that you can only provide information from your knowledge base
- Be honest about the limitations of your knowledge
- With your 128K context window, you can process much longer retrieved documents for comprehensive answers""",
            "research": """You are a research assistant powered by Phi-3 with 128K context capability. Analyze the provided context thoroughly and provide comprehensive explanations based only on the available information. Your extended context window allows for deeper analysis of longer documents.""",
            "qa": """You are a Q&A specialist powered by Phi-3. Provide direct, accurate answers based strictly on the provided context. If the context doesn't contain the answer, say so clearly. Your 128K context window enables processing of extensive source material.""",
            "content_analyzer": """You are a content analysis expert powered by Phi-3's extended context capabilities. Analyze the provided content for key themes and concepts, but only based on what is actually present in the text. Your 128K context window allows for comprehensive analysis of longer documents.""",
        }

    def process_message(
        self,
        message: str,
        use_context: bool = True,
        conversation_history: List[Dict] = None,
        response_config: Dict = None,
        context_filters: Dict = None,
    ) -> Dict[str, Any]:
        """Process message with MLX optimization when available"""

        start_time = time.time()
        response_config = response_config or {}

        try:
            # Step 1: Check if model is available
            if not self.llm_service.is_available():
                return self._create_error_response(
                    "Local model is not available. Please check the setup.",
                    "model_unavailable",
                )

            # Step 2: Direct context retrieval if enabled
            context_info = None
            sources = []

            if use_context:
                try:
                    context_info = self._direct_search(message, context_filters or {})
                    sources = context_info.get("sources", [])
                except Exception as e:
                    logger.warning("Context retrieval failed: %s", e)
                    context_info = None

            # Step 3: Prepare prompt
            if context_info and context_info.get("context"):
                prompt = self._create_prompt_with_context(
                    message, context_info["context"]
                )
                used_context = True
            else:
                # No context available - be honest about it
                system_prompt = self.system_prompts.get(
                    self.agent_type, self.system_prompts["general"]
                )
                prompt = f"""{system_prompt}

CONTEXT: No relevant information found in the knowledge base.

USER QUESTION: {message}

RESPONSE: I don't have any relevant information in my knowledge base to answer your question about "{message}". My knowledge base appears to be limited and doesn't contain information on this topic."""
                used_context = False

            # Step 4: Generate response with optimization
            if self.use_mlx:
                # Use MLX for Apple Silicon optimization
                response = self.llm_service.generate_response(
                    prompt=message,  # MLX service handles system prompts differently
                    max_tokens=response_config.get("max_tokens", 500),
                    temperature=response_config.get("temperature", 0.7),
                    system_prompt=(
                        self.system_prompts.get(
                            self.agent_type, self.system_prompts["general"]
                        )
                        if not used_context
                        else None
                    ),
                )
            else:
                # Use CPU implementation
                response = self.llm_service.generate_response(
                    prompt=prompt,
                    max_tokens=response_config.get("max_tokens", 500),
                    temperature=response_config.get("temperature", 0.7),
                )

            processing_time = (time.time() - start_time) * 1000

            # Step 5: Evaluate response quality
            quality_scores = self._evaluate_response(
                message,
                response,
                context_info.get("context", "") if context_info else "",
            )

            # Determine model info
            model_info = self.llm_service.get_model_info()
            model_used = model_info.get("name", "Phi-3-mini-128k-instruct")
            optimization = "MLX Apple Silicon" if self.use_mlx else "CPU"

            return {
                "status": "success",
                "response": response,
                "used_context": used_context,
                "sources": sources,
                "quality_scores": quality_scores,
                "processing_time_ms": int(processing_time),
                "model_used": model_used,
                "agent_type": self.agent_type,
                "optimization": optimization,
                "framework": "MLX" if self.use_mlx else "Transformers",
                "device": model_info.get("device", "unknown"),
            }

        except Exception as e:
            logger.error("Error processing message: %s", e)
            return self._create_error_response(str(e), "processing_error")

    def _direct_search(self, query: str, filters: Dict) -> Dict[str, Any]:
        """Direct database search for context retrieval"""
        try:
            # Generate query embedding using the search service
            if not self.embedding_service.search_service.embedding_model:
                logger.error("Embedding model not available")
                return {"context": "", "sources": []}

            query_embedding = (
                self.embedding_service.search_service.embedding_model.encode(query)
            )
            if query_embedding is None:
                return {"context": "", "sources": []}

            # Convert to list for PostgreSQL vector format
            query_embedding_list = query_embedding.tolist()

            # Direct SQL query with pgvector
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT 
                        ke.chunk_text,
                        ke.content_quality_score,
                        cp.title as page_title,
                        cp.url as page_url,
                        (ke.embedding <=> %s::vector) as distance
                    FROM knowledge_base_knowledgeembedding ke
                    JOIN crawler_crawledpage cp ON ke.page_id = cp.id
                    WHERE cp.success = true
                    ORDER BY ke.embedding <=> %s::vector
                    LIMIT 10
                    """,
                    [query_embedding_list, query_embedding_list],
                )

                rows = cursor.fetchall()

            context_parts = []
            sources = []

            for row in rows:
                chunk_text, quality_score, page_title, page_url, distance = row

                context_parts.append(f"Source: {page_title}\nContent: {chunk_text}")
                sources.append(
                    {
                        "title": page_title,
                        "url": page_url,
                        "quality_score": quality_score,
                        "relevance_score": 1 - distance,
                        "content_preview": (
                            chunk_text[:150] + "..."
                            if len(chunk_text) > 150
                            else chunk_text
                        ),
                    }
                )

            context = "\n\n---\n\n".join(context_parts)

            return {"context": context, "sources": sources}

        except Exception as e:
            logger.error(f"Direct search error: {e}")
            return {"context": "", "sources": []}

    def _create_prompt_with_context(self, message: str, context: str) -> str:
        """Create a prompt with context"""
        system_prompt = self.system_prompts.get(
            self.agent_type, self.system_prompts["general"]
        )

        return f"""{system_prompt}

CONTEXT FROM KNOWLEDGE BASE:
{context}

USER QUESTION: {message}

RESPONSE:"""

    def _evaluate_response(
        self, question: str, response: str, context: str
    ) -> Dict[str, float]:
        """Evaluate response quality"""
        try:
            scores = {}

            # Basic quality metrics
            if response:
                scores["response_length"] = min(len(response) / 500, 1.0)
                scores["context_relevance"] = 0.8 if context else 0.3
                scores["completeness"] = 0.9 if len(response) > 50 else 0.5
            else:
                scores = {
                    "response_length": 0.0,
                    "context_relevance": 0.0,
                    "completeness": 0.0,
                }

            return scores

        except Exception as e:
            logger.error(f"Error evaluating response: {e}")
            return {"error": 0.0}

    def _create_error_response(
        self, error_message: str, error_type: str
    ) -> Dict[str, Any]:
        """Create standardized error response"""
        return {
            "status": "error",
            "error_message": error_message,
            "error_type": error_type,
            "response": f"I encountered an error: {error_message}",
            "used_context": False,
            "sources": [],
            "processing_time_ms": 0,
            "model_used": "none",
            "agent_type": self.agent_type,
            "optimization": "none",
        }

    def get_knowledge_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics with optimization info"""
        try:
            from crawler.models import CrawledPage

            total_pages = CrawledPage.objects.filter(success=True).count()
            total_embeddings = KnowledgeEmbedding.objects.count()

            model_info = self.llm_service.get_model_info()

            return {
                "total_crawled_pages": total_pages,
                "pages_with_content": total_pages,
                "total_embeddings": total_embeddings,
                "model_available": self.llm_service.is_available(),
                "search_available": bool(
                    self.embedding_service.search_service.embedding_model
                ),
                "optimization": "MLX Apple Silicon" if self.use_mlx else "CPU",
                "framework": "MLX" if self.use_mlx else "Transformers",
                "device": model_info.get("device", "unknown"),
                "load_time_seconds": model_info.get("load_time_seconds", 0),
                "expected_speedup": "5-10x" if self.use_mlx else "1x",
            }
        except Exception as e:
            logger.error("Error getting knowledge stats: %s", e)
            return {
                "total_crawled_pages": 0,
                "pages_with_content": 0,
                "total_embeddings": 0,
                "model_available": False,
                "search_available": False,
                "error": str(e),
                "optimization": "none",
            }


# Backward compatibility alias
EnhancedTinyLlamaAgent = EnhancedPhi3Agent
