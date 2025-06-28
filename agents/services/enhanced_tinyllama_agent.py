import time
import logging
from typing import Dict, Any, List
from .local_model_service import TinyLlamaService
from knowledge_base.services.enhanced_rag import EnhancedRAGService
from knowledge_base.services.pgvector_search import PgVectorSearchService
from knowledge_base.services.embedding_generator import EmbeddingGenerationService

logger = logging.getLogger(__name__)


class EnhancedTinyLlamaAgent:
    """Enhanced TinyLlama agent with advanced RAG capabilities"""

    def __init__(self, agent_type: str = "general"):
        self.agent_type = agent_type
        self.llm_service = TinyLlamaService()
        self.rag_service = EnhancedRAGService()
        self.search_service = PgVectorSearchService()
        self.embedding_service = EmbeddingGenerationService()

        # Enhanced system prompts
        self.system_prompts = {
            "general": """You are Omeruta Brain, an intelligent AI assistant with access to a comprehensive knowledge base.
            You provide accurate, helpful, and well-structured responses based on the provided context.
            Always cite your sources when referencing specific information.""",
            "research": """You are a research assistant specializing in analysis and synthesis.
            Analyze the provided context thoroughly, identify key insights, and provide comprehensive explanations.
            Compare different perspectives when available and highlight important findings.""",
            "qa": """You are a Q&A specialist focused on providing direct, accurate answers.
            Give concise but complete answers based on the context. If information is incomplete,
            clearly state what is known and what might need additional research.""",
            "content_analyzer": """You are a content analysis expert. Analyze the provided content for
            key themes, important concepts, and actionable insights. Structure your analysis clearly
            with main points and supporting details.""",
        }

    def process_message(
        self,
        message: str,
        use_context: bool = True,
        conversation_history: List[Dict] = None,
        response_config: Dict = None,
        context_filters: Dict = None,
    ) -> Dict[str, Any]:
        """Process message with enhanced RAG pipeline"""

        start_time = time.time()
        response_config = response_config or {}

        try:
            # Step 1: Check if model is available
            if not self.llm_service.is_available():
                return self._create_error_response(
                    "Local model is not available. Please check the setup.",
                    "model_unavailable",
                )

            # Step 2: Enhanced context retrieval if enabled
            context_info = None
            if use_context:
                try:
                    context_info = self.rag_service.generate_response(
                        query=message,
                        context_filters=context_filters,
                        response_config=response_config,
                    )
                except Exception as e:
                    logger.warning("Context retrieval failed: %s", e)
                    context_info = None

            # Step 3: Prepare prompt
            if context_info and context_info["context"]:
                prompt = context_info["enhanced_prompt"]
                used_context = True
            else:
                # Fallback to basic prompt without context
                system_prompt = self.system_prompts.get(
                    self.agent_type, self.system_prompts["general"]
                )
                prompt = f"{system_prompt}\n\nUser: {message}\nAssistant:"
                used_context = False

            # Step 4: Generate response
            response = self.llm_service.generate_response(
                prompt=prompt,
                max_tokens=response_config.get("max_tokens", 300),
                temperature=response_config.get("temperature", 0.7),
            )

            # Step 5: Post-process response
            if context_info and used_context:
                # Evaluate response quality
                quality_scores = self.rag_service.evaluate_response_quality(
                    query=message, response=response, context=context_info["context"]
                )
            else:
                quality_scores = {}

            processing_time = (time.time() - start_time) * 1000

            # Step 6: Build comprehensive response
            result = {
                "response": response,
                "model_used": "tinyllama-local",
                "agent_type": self.agent_type,
                "used_context": used_context,
                "processing_time_ms": int(processing_time),
                "quality_scores": quality_scores,
                "sources": context_info.get("sources", []) if context_info else [],
                "search_metadata": (
                    context_info.get("search_metadata", {}) if context_info else {}
                ),
                "status": "success",
            }

            return result

        except Exception as e:
            logger.error("Error processing message: %s", e)
            return self._create_error_response(str(e), "processing_error")

    def _create_error_response(
        self, error_message: str, error_type: str
    ) -> Dict[str, Any]:
        """Create standardized error response"""
        return {
            "response": f"I apologize, but I encountered an error: {error_message}",
            "model_used": "error",
            "agent_type": self.agent_type,
            "used_context": False,
            "processing_time_ms": 0,
            "quality_scores": {},
            "sources": [],
            "search_metadata": {},
            "status": "error",
            "error_type": error_type,
            "error_message": error_message,
        }

    def get_knowledge_stats(self) -> Dict[str, Any]:
        """Get comprehensive knowledge base statistics"""
        try:
            embedding_stats = self.embedding_service.get_embedding_stats()

            # Add search service stats
            search_stats = {
                "embedding_model_available": bool(self.search_service.embedding_model),
                "cross_encoder_available": bool(self.search_service.cross_encoder),
                "search_config": self.search_service.config,
            }

            # Add RAG service stats
            rag_stats = self.rag_service.get_rag_stats()

            return {
                **embedding_stats,
                **search_stats,
                "rag_service_ready": rag_stats["rag_ready"],
                "agent_type": self.agent_type,
                "llm_available": self.llm_service.is_available(),
                "llm_info": (
                    self.llm_service.get_model_info()
                    if self.llm_service.is_available()
                    else {}
                ),
            }
        except Exception as e:
            logger.error("Error getting knowledge stats: %s", e)
            return {"error": str(e)}

    def search_knowledge_base(self, query: str, filters: Dict = None) -> Dict[str, Any]:
        """Direct search of the knowledge base"""
        try:
            return self.search_service.enhanced_search(query, filters)
        except Exception as e:
            logger.error("Knowledge base search error: %s", e)
            return {"error": str(e), "results": []}

    def get_conversation_context(
        self, query: str, max_context_length: int = 2000
    ) -> Dict[str, Any]:
        """Get context for a query without generating a full response"""
        try:
            context_info = self.rag_service.generate_response(
                query=query, response_config={"include_sources": True}
            )

            # Truncate context if too long
            context = context_info["context"]
            if len(context) > max_context_length:
                context = context[:max_context_length] + "...\n[Context truncated]"
                context_info["context"] = context
                context_info["truncated"] = True
            else:
                context_info["truncated"] = False

            return context_info
        except Exception as e:
            logger.error("Context retrieval error: %s", e)
            return {"error": str(e), "context": "", "sources": []}

    def process_batch_messages(
        self, messages: List[Dict], use_context: bool = True
    ) -> List[Dict[str, Any]]:
        """Process multiple messages in batch"""
        results = []

        for msg_data in messages:
            message = msg_data.get("message", "")
            config = msg_data.get("config", {})
            filters = msg_data.get("filters", {})

            if message.strip():
                result = self.process_message(
                    message=message,
                    use_context=use_context,
                    response_config=config,
                    context_filters=filters,
                )
                result["original_message"] = message
                results.append(result)
            else:
                results.append(
                    {
                        "error": "Empty message",
                        "original_message": message,
                        "status": "error",
                    }
                )

        return results
