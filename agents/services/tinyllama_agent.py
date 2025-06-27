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

        # Enhanced system prompts based on your domain
        self.system_prompts = {
            "general": """You are Omeruta Brain, an intelligent AI assistant with access to a curated knowledge base.
            Answer questions accurately and cite your sources when using provided context.
            If you don't know something, say so clearly.""",
            "research": """You are a research specialist within Omeruta Brain. You excel at:
            - Analyzing and synthesizing information from multiple sources
            - Identifying key insights and patterns
            - Providing comprehensive yet concise summaries
            - Suggesting related topics for further exploration""",
            "qa": """You are a Q&A specialist within Omeruta Brain. You provide:
            - Direct, accurate answers to specific questions
            - Clear explanations with examples when helpful
            - Citations to sources when using provided context
            - Honest acknowledgment when information is not available""",
            "content_analyzer": """You are a content analysis specialist within Omeruta Brain. You:
            - Analyze the quality and credibility of information
            - Compare different perspectives on topics
            - Identify biases or gaps in content
            - Summarize key themes and insights""",
        }

    def _classify_question_type(self, message: str) -> str:
        """Classify the type of question for better handling"""
        message_lower = message.lower()

        # Factual questions - need context
        factual_keywords = [
            "what is",
            "what are",
            "how does",
            "explain",
            "define",
            "tell me about",
        ]
        if any(keyword in message_lower for keyword in factual_keywords):
            return "factual"

        # Analytical questions - may need context
        analytical_keywords = [
            "compare",
            "analyze",
            "evaluate",
            "pros and cons",
            "advantages",
            "disadvantages",
        ]
        if any(keyword in message_lower for keyword in analytical_keywords):
            return "analytical"

        # Procedural questions - may need context
        procedural_keywords = ["how to", "steps", "process", "procedure", "guide"]
        if any(keyword in message_lower for keyword in procedural_keywords):
            return "procedural"

        # Opinion/creative questions - usually don't need context
        opinion_keywords = ["think", "opinion", "believe", "feel", "create", "generate"]
        if any(keyword in message_lower for keyword in opinion_keywords):
            return "opinion"

        return "general"

    def _needs_context(self, message: str) -> bool:
        """Enhanced context detection"""
        question_type = self._classify_question_type(message)

        # These question types typically benefit from context
        context_types = ["factual", "analytical", "procedural"]

        # Also check for specific domain keywords from your crawled content
        domain_keywords = [
            "cryptocurrency",
            "bitcoin",
            "blockchain",
            "digital currency",
            "tech",
            "technology",
        ]
        has_domain_keywords = any(
            keyword in message.lower() for keyword in domain_keywords
        )

        return question_type in context_types or has_domain_keywords

    def process_message(
        self, message: str, use_context: bool = True, max_tokens: int = 300
    ) -> Dict[str, Any]:
        """Enhanced message processing with better context handling"""

        if not self.llm_service.is_available():
            return {
                "response": "Local model is not available. Please check the setup.",
                "model_used": "none",
                "context_used": False,
                "error": "Model initialization failed",
            }

        try:
            # Enhanced context retrieval
            context = ""
            context_used = False
            context_sources = 0

            if use_context and self._needs_context(message):
                search_results = self.search_service.search_crawled_content(message)
                if search_results:
                    context = self.search_service.get_context_for_query(message)
                    context_used = bool(context)
                    context_sources = len(search_results)

            # Get question type for appropriate system prompt
            question_type = self._classify_question_type(message)
            base_prompt = self.system_prompts.get(
                self.agent_type, self.system_prompts["general"]
            )

            if context:
                enhanced_prompt = f"""{base_prompt}

Context from knowledge base:
{context}

Please answer the user's question based on the context above. If the context doesn't contain relevant information, say so clearly and provide what you know from your general knowledge."""
            else:
                enhanced_prompt = base_prompt

            # Generate response with appropriate max_tokens based on question complexity
            if question_type in ["analytical", "procedural"]:
                max_tokens = min(
                    int(max_tokens * 1.5), 500
                )  # Longer responses for complex questions

            response = self.llm_service.generate_response(
                prompt=message,
                max_tokens=int(max_tokens),
                system_prompt=enhanced_prompt,
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
                "question_type": question_type,
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
