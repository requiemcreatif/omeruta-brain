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
- If the context contains relevant information, use it to answer the question directly
- If the context is empty or insufficient, clearly state that you don't have enough information
- When the context contains factual information (including dates, names, events), provide that information as your answer
- Only refuse to answer if the context is truly empty or doesn't contain relevant information
- Be honest about the limitations of your knowledge when context is insufficient
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

            # Step 2: Analyze question type and determine response characteristics
            question_analysis = self._classify_question_type(message)

            # Override max_tokens from question analysis if not explicitly set
            if "max_tokens" not in response_config:
                response_config["max_tokens"] = question_analysis["max_tokens"]

            # Step 3: Direct context retrieval if enabled
            context_info = None
            sources = []

            if use_context:
                try:
                    context_info = self._direct_search(message, context_filters or {})
                    sources = context_info.get("sources", [])
                except Exception as e:
                    logger.warning("Context retrieval failed: %s", e)
                    context_info = None

            # Step 4: Prepare prompt with intelligent response instructions
            if context_info and context_info.get("context"):
                prompt = self._create_adaptive_prompt_with_context(
                    message, context_info["context"], question_analysis
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

            # Step 5: Generate response with optimization
            if self.use_mlx:
                # Use MLX for Apple Silicon optimization
                if used_context:
                    # When context is available, use the full prompt with context
                    response = self.llm_service.generate_response(
                        prompt=prompt,  # Full prompt with context
                        max_tokens=response_config.get("max_tokens", 500),
                        temperature=response_config.get("temperature", 0.7),
                    )
                else:
                    # When no context, use system prompt + message
                    response = self.llm_service.generate_response(
                        prompt=message,
                        max_tokens=response_config.get("max_tokens", 500),
                        temperature=response_config.get("temperature", 0.7),
                        system_prompt=self.system_prompts.get(
                            self.agent_type, self.system_prompts["general"]
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
                "question_analysis": question_analysis,
                "adaptive_response": True,
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

RESPONSE: Based on the information provided in the knowledge base above, """

    def _create_adaptive_prompt_with_context(
        self, message: str, context: str, question_analysis: Dict[str, Any]
    ) -> str:
        """Create an adaptive prompt that adjusts response style based on question type"""
        system_prompt = self.system_prompts.get(
            self.agent_type, self.system_prompts["general"]
        )

        response_instructions = self._get_response_instructions(question_analysis)

        return f"""{system_prompt}

CONTEXT FROM KNOWLEDGE BASE:
{context}

USER QUESTION: {message}

RESPONSE INSTRUCTIONS: {response_instructions}

IMPORTANT: Provide ONE complete answer only. Do not repeat yourself or generate multiple responses.

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

    def _classify_question_type(self, message: str) -> Dict[str, Any]:
        """Classify question type and determine appropriate response characteristics"""
        message_lower = message.lower().strip()

        # Question type classification
        question_type = "general"
        response_style = "balanced"
        max_tokens = 300

        # Simple factual questions - short, direct answers
        simple_factual = [
            "who is",
            "who was",
            "what is",
            "what was",
            "when is",
            "when was",
            "where is",
            "where was",
            "which is",
            "which was",
        ]
        if any(pattern in message_lower for pattern in simple_factual):
            question_type = "simple_factual"
            response_style = "concise"
            max_tokens = 250  # Increased to ensure complete responses

        # Complex analytical questions - longer, detailed answers
        analytical_patterns = [
            "analyze",
            "compare",
            "evaluate",
            "explain how",
            "explain why",
            "pros and cons",
            "advantages and disadvantages",
            "impact of",
            "relationship between",
            "differences between",
        ]
        if any(pattern in message_lower for pattern in analytical_patterns):
            question_type = "analytical"
            response_style = "comprehensive"
            max_tokens = 500

        # Procedural questions - step-by-step answers
        procedural_patterns = [
            "how to",
            "steps to",
            "process of",
            "procedure for",
            "guide to",
            "tutorial",
            "instructions",
        ]
        if any(pattern in message_lower for pattern in procedural_patterns):
            question_type = "procedural"
            response_style = "structured"
            max_tokens = 400

        # List-based questions - organized answers
        list_patterns = [
            "list of",
            "examples of",
            "types of",
            "kinds of",
            "what are the",
            "name some",
            "give me",
        ]
        if any(pattern in message_lower for pattern in list_patterns):
            question_type = "list_based"
            response_style = "organized"
            max_tokens = 350

        # Definition questions - brief but complete
        definition_patterns = [
            "define",
            "definition of",
            "meaning of",
            "what does",
            "what means",
        ]
        if any(pattern in message_lower for pattern in definition_patterns):
            question_type = "definition"
            response_style = "precise"
            max_tokens = 200

        return {
            "type": question_type,
            "style": response_style,
            "max_tokens": max_tokens,
            "complexity": self._assess_complexity(message),
        }

    def _assess_complexity(self, message: str) -> str:
        """Assess question complexity based on length and keywords"""
        word_count = len(message.split())

        complex_indicators = [
            "comprehensive",
            "detailed",
            "thorough",
            "in-depth",
            "elaborate",
            "extensive",
            "complete analysis",
        ]

        simple_indicators = ["briefly", "quick", "short", "simple", "just", "only"]

        if any(indicator in message.lower() for indicator in complex_indicators):
            return "high"
        elif any(indicator in message.lower() for indicator in simple_indicators):
            return "low"
        elif word_count > 15:
            return "high"
        elif word_count < 5:
            return "low"
        else:
            return "medium"

    def _get_response_instructions(self, question_analysis: Dict[str, Any]) -> str:
        """Generate response instructions based on question analysis"""
        style = question_analysis["style"]
        question_type = question_analysis["type"]
        complexity = question_analysis["complexity"]

        instructions = {
            "concise": "Provide a complete, informative answer using clear sentences. Be direct but ensure you fully answer the question.",
            "comprehensive": "Provide a detailed, thorough analysis. Include multiple perspectives and implications.",
            "structured": "Organize your response in clear steps or sections. Use numbering or bullets if helpful.",
            "organized": "Present information in a well-organized format. Group related items together.",
            "precise": "Give a clear, accurate definition or explanation. Be specific and avoid ambiguity.",
            "balanced": "Provide a complete but appropriately sized response based on the question's needs.",
        }

        base_instruction = instructions.get(style, instructions["balanced"])

        # Add complexity-based modifiers
        if complexity == "low":
            base_instruction += " Keep it simple and to the point."
        elif complexity == "high":
            base_instruction += " Provide comprehensive detail and context."

        # Add question-type specific guidance
        if question_type == "simple_factual":
            base_instruction += " Provide the key facts in complete, well-formed sentences. Ensure your answer is informative and complete."
        elif question_type == "definition":
            base_instruction += (
                " Start with a clear definition, then add brief context if relevant."
            )

        return base_instruction
