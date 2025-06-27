from .local_model_service import TinyLlamaService
from .enhanced_search_service import EnhancedVectorSearchService
from crawler.models import CrawledPage
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class TinyLlamaAgent:
    """Agent powered by TinyLlama with access to your crawled knowledge base"""

    def __init__(self, agent_type: str = "general"):
        self.agent_type = agent_type
        self.llm_service = TinyLlamaService()
        self.search_service = EnhancedVectorSearchService()
        self.system_prompts = {
            "general": "You are a helpful AI assistant. Answer questions accurately and concisely based on the provided context.",
            "research": "You are a research assistant. Help users find and analyze information from the knowledge base.",
            "qa": "You are a Q&A assistant. Provide direct, accurate answers to user questions using the available context.",
        }

    def process_message(
        self, message: str, use_context: bool = True, max_tokens: int = 300
    ) -> Dict[str, Any]:
        """Process user message and generate response"""

        # Check if model is available
        if not self.llm_service.is_available():
            return {
                "response": "Local model is not available. Please check the setup.",
                "model_used": "none",
                "context_used": False,
                "error": "Model initialization failed",
            }

        try:
            # Get relevant context from your crawled data
            context = ""
            context_used = False
            context_sources = 0

            if use_context and self._needs_context(message):
                search_results = self.search_service.search_crawled_content(message)
                if search_results:
                    context = self.search_service.get_context_for_query(message)
                    context_used = bool(context)
                    context_sources = len(search_results)

            # Prepare the full prompt
            system_prompt = self.system_prompts.get(
                self.agent_type, self.system_prompts["general"]
            )

            if context:
                enhanced_prompt = f"""{system_prompt}

Context from knowledge base:
{context}

Please answer the user's question based on the context above. If the context doesn't contain relevant information, say so clearly."""
            else:
                enhanced_prompt = system_prompt

            # Generate response
            response = self.llm_service.generate_response(
                prompt=message, max_tokens=max_tokens, system_prompt=enhanced_prompt
            )

            if response is None:
                return {
                    "response": "Sorry, I encountered an error generating a response.",
                    "model_used": "tinyllama",
                    "context_used": context_used,
                    "context_sources": context_sources,
                    "error": "Generation failed",
                }

            return {
                "response": response,
                "model_used": "tinyllama",
                "context_used": context_used,
                "context_sources": context_sources,
                "model_info": self.llm_service.get_model_info(),
            }

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return {
                "response": f"An error occurred: {str(e)}",
                "model_used": "tinyllama",
                "context_used": False,
                "context_sources": 0,
                "error": str(e),
            }

    def _needs_context(self, message: str) -> bool:
        """Determine if message needs knowledge base context"""
        # Simple heuristic - can be enhanced
        context_keywords = [
            "what",
            "how",
            "explain",
            "tell me about",
            "information about",
            "details",
            "describe",
            "summary",
            "definition",
        ]
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in context_keywords)

    def get_available_knowledge_stats(self) -> Dict[str, Any]:
        """Get stats about available knowledge base"""
        try:
            total_pages = CrawledPage.objects.filter(success=True).count()
            pages_with_content = (
                CrawledPage.objects.filter(success=True, clean_markdown__isnull=False)
                .exclude(clean_markdown="")
                .count()
            )

            return {
                "total_crawled_pages": total_pages,
                "pages_with_content": pages_with_content,
                "model_available": self.llm_service.is_available(),
                "search_available": self.search_service.embedding_model is not None,
            }
        except Exception as e:
            logger.error(f"Error getting knowledge stats: {e}")
            return {"error": str(e)}
