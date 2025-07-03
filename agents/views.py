from rest_framework import viewsets, permissions, status
from rest_framework.decorators import action
from rest_framework.response import Response
from .services.enhanced_phi3_agent import EnhancedPhi3Agent
from .services.conversation_memory import ConversationMemory
from knowledge_base.services.enhanced_rag import EnhancedRAGService
from knowledge_base.services.pgvector_search import PgVectorSearchService
from knowledge_base.tasks import batch_generate_embeddings
from .tasks import (
    process_user_message_async,
    process_multiagent_query,
    auto_expand_knowledge_async,
    analyze_content_quality_batch,
    health_check,
    process_live_research_async,
)
from celery.result import AsyncResult
from django.core.cache import cache
import time
import logging
from django.views import View
from django.shortcuts import render
from .services.live_research_agent import LiveResearchAgent
from django.utils import timezone

logger = logging.getLogger(__name__)


class AIAssistantView(View):
    template_name = "agents/ai_assistant.html"

    def get(self, request, *args, **kwargs):
        # Use enhanced agent for better stats
        agent = EnhancedPhi3Agent()
        status = agent.get_knowledge_stats()
        model_info = (
            agent.llm_service.get_model_info()
            if agent.llm_service.is_available()
            else {}
        )

        available_agent_types = list(agent.system_prompts.keys())

        def format_agent_type(type_name):
            if type_name == "qa":
                return "Q&A"
            return type_name.replace("_", " ").title()

        formatted_agent_types = [
            {"value": type_name, "name": format_agent_type(type_name)}
            for type_name in available_agent_types
        ]

        initial_status = {
            "knowledge_stats": status,
            "model_info": model_info,
            "agent_type": agent.agent_type,
            "available_agent_types": formatted_agent_types,
        }

        context = {
            "initial_status": initial_status,
            "conversation_id": request.session.get("conversation_id"),
        }
        return render(request, self.template_name, context)


class Phi3AgentViewSet(viewsets.GenericViewSet):
    """Enhanced ViewSet with advanced RAG and conversation memory"""

    permission_classes = [permissions.IsAuthenticated]

    # Class-level shared instances (singleton pattern)
    _shared_agent = None
    _shared_search_service = None
    _shared_rag_service = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Use shared instances to avoid reloading models
        if Phi3AgentViewSet._shared_agent is None:
            logger.info("🚀 Initializing shared Phi-3 models (first time)...")
            Phi3AgentViewSet._shared_agent = EnhancedPhi3Agent()
            Phi3AgentViewSet._shared_search_service = PgVectorSearchService()
            Phi3AgentViewSet._shared_rag_service = EnhancedRAGService()
            logger.info("✅ Shared Phi-3 models initialized successfully")
        else:
            logger.info("♻️ Reusing existing Phi-3 models (no reload needed)")

        self.agent = Phi3AgentViewSet._shared_agent
        self.enhanced_agent = Phi3AgentViewSet._shared_agent  # Same instance
        self.search_service = Phi3AgentViewSet._shared_search_service
        self.rag_service = Phi3AgentViewSet._shared_rag_service

    @action(detail=False, methods=["post"])
    def chat(self, request):
        """Enhanced chat with conversation memory"""
        message = request.data.get("message", "")
        use_context = request.data.get("use_context", True)
        max_tokens = request.data.get("max_tokens", 300)
        conversation_id = request.data.get("conversation_id")

        if not message:
            return Response(
                {"error": "Message is required"}, status=status.HTTP_400_BAD_REQUEST
            )

        # Start timing the response
        start_time = time.time()

        try:
            # Initialize conversation memory
            memory = ConversationMemory(conversation_id)

            # Get conversation context
            conversation_context = memory.get_conversation_context()

            # Enhanced message with conversation context
            if conversation_context:
                enhanced_message = f"Recent conversation context:\n{conversation_context}\n\nCurrent question: {message}"
            else:
                enhanced_message = message

            # Process with enhanced agent
            result = self.agent.process_message(
                message=enhanced_message,
                use_context=use_context,
                response_config={"max_tokens": max_tokens},
            )

            # Calculate response time
            response_time_ms = int((time.time() - start_time) * 1000)

            # Store in conversation memory
            memory.add_exchange(
                user_message=message,
                agent_response=result["response"],
                metadata={
                    "context_used": result.get("context_used", False),
                    "context_sources": result.get("context_sources", 0),
                    "question_type": result.get("question_type", "general"),
                    "response_time_ms": response_time_ms,
                },
            )

            # Add conversation_id and response time to response
            result["conversation_id"] = memory.conversation_id
            result["response_time_ms"] = response_time_ms

            # Log usage for analytics (you can enhance this later)
            logger.info(
                "Chat request processed: user=%s, question_type=%s, context_used=%s, response_time=%sms",
                request.user.id,
                result.get("question_type"),
                result.get("context_used"),
                response_time_ms,
            )

            return Response(result)

        except Exception as e:
            logger.error("Error in chat endpoint: %s", e)
            return Response(
                {"error": "An unexpected error occurred", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    @action(detail=False, methods=["post"])
    def clear_conversation(self, request):
        """Clear conversation memory"""
        conversation_id = request.data.get("conversation_id")
        if conversation_id:
            memory = ConversationMemory(conversation_id)
            memory.clear_conversation()
            return Response(
                {"message": "Conversation cleared", "conversation_id": conversation_id}
            )
        return Response({"error": "conversation_id required"}, status=400)

    @action(detail=False, methods=["get"])
    def conversation_history(self, request):
        """Get conversation history"""
        conversation_id = request.query_params.get("conversation_id")
        if not conversation_id:
            return Response({"error": "conversation_id required"}, status=400)

        memory = ConversationMemory(conversation_id)
        history = memory.get_full_conversation()
        summary = memory.get_conversation_summary()

        return Response(
            {"conversation_id": conversation_id, "history": history, "summary": summary}
        )

    @action(detail=False, methods=["get"])
    def conversation_summary(self, request):
        """Get conversation summary"""
        conversation_id = request.query_params.get("conversation_id")
        if not conversation_id:
            return Response({"error": "conversation_id required"}, status=400)

        memory = ConversationMemory(conversation_id)
        summary = memory.get_conversation_summary()

        return Response({"conversation_id": conversation_id, "summary": summary})

    @action(detail=False, methods=["get"], url_path="status")
    def status(self, request):
        """Get agent and knowledge base status"""
        try:
            # Use cached model info if available to avoid triggering loads
            if Phi3AgentViewSet._shared_agent is not None:
                stats = self.agent.get_knowledge_stats()
                model_info = self.agent.llm_service.get_model_info()
            else:
                # Return basic info without initializing models
                stats = {
                    "models_loaded": False,
                    "message": "Models not yet initialized",
                }
                model_info = {
                    "loaded": False,
                    "name": "Phi-3-mini-128k-instruct",
                    "device": "cpu",
                }

            return Response(
                {
                    "knowledge_stats": stats,
                    "model_info": model_info,
                    "agent_type": (
                        self.agent.agent_type
                        if Phi3AgentViewSet._shared_agent
                        else "general"
                    ),
                    "available_agent_types": [
                        "general",
                        "research",
                        "qa",
                        "content_analyzer",
                        "live_research",
                    ],
                    "models_shared": Phi3AgentViewSet._shared_agent is not None,
                }
            )
        except Exception as e:
            logger.error("Error getting status: %s", e)
            return Response(
                {
                    "error": "Failed to get status",
                    "knowledge_stats": {"error": str(e)},
                    "model_info": {"loaded": False, "error": str(e)},
                    "agent_type": "general",
                    "available_agent_types": [
                        "general",
                        "research",
                        "qa",
                        "content_analyzer",
                        "live_research",
                    ],
                    "models_shared": False,
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    @action(detail=False, methods=["post"], url_path="change_agent_type")
    def change_agent_type(self, request):
        agent_type = request.data.get("agent_type")
        if not agent_type:
            return Response(
                {"error": "agent_type is required"}, status=status.HTTP_400_BAD_REQUEST
            )

        # Validate agent type
        valid_agent_types = [
            "general",
            "research",
            "qa",
            "content_analyzer",
            "live_research",
        ]
        if agent_type not in valid_agent_types:
            return Response(
                {"error": f"Invalid agent type. Must be one of: {valid_agent_types}"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            # Simply change the agent type without reloading models
            old_agent_type = self.agent.agent_type
            self.agent.agent_type = agent_type

            # Update the session
            request.session["agent_type"] = agent_type

            logger.info(
                "Agent type changed from % s to %s for user %s",
                old_agent_type,
                agent_type,
                request.user.id,
            )

            return Response(
                {
                    "message": f"Agent type changed to {agent_type}",
                    "agent_type": agent_type,
                    "system_prompt": self.agent.system_prompts.get(agent_type, ""),
                },
                status=status.HTTP_200_OK,
            )
        except ValueError as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            logger.error("Error changing agent type: %s", e)
            return Response(
                {"error": "An unexpected error occurred"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    # Async Endpoints with Celery Integration

    @action(detail=False, methods=["post"])
    def chat_async(self, request):
        """Start async chat processing with real-time status updates"""
        message = request.data.get("message", "")
        use_context = request.data.get("use_context", True)
        max_tokens = request.data.get("max_tokens", 300)
        conversation_id = request.data.get("conversation_id")
        agent_type = request.data.get("agent_type", "general")

        if not message:
            return Response(
                {"error": "Message is required"}, status=status.HTTP_400_BAD_REQUEST
            )

        # Validate agent type
        if agent_type not in ["general", "research", "qa", "content_analyzer"]:
            agent_type = "general"

        # Start async task
        task = process_user_message_async.delay(
            message=message,
            user_id=request.user.id,
            conversation_id=conversation_id,
            use_context=use_context,
            agent_type=agent_type,
            max_tokens=max_tokens,
        )

        return Response(
            {
                "task_id": task.id,
                "status": "processing",
                "message": "Your request is being processed. Use the task_id to check status.",
                "check_status_url": f"/api/agents/async/status/{task.id}/",
                "conversation_id": conversation_id,
                "agent_type": agent_type,
            }
        )

    @action(detail=False, methods=["get"], url_path="async/status/(?P<task_id>[^/.]+)")
    def check_task_status(self, request, task_id):
        """Check status of async task with detailed progress"""
        try:
            # Get task result
            task_result = AsyncResult(task_id)

            # Get cached status for more detailed progress
            cached_status = cache.get(f"task_status:{task_id}")

            if cached_status:
                response_data = cached_status.copy()
                response_data["task_id"] = task_id
                response_data["celery_status"] = task_result.status

                # If completed, include result
                if (
                    cached_status["status"] == "completed"
                    and "result_key" in cached_status
                ):
                    result = cache.get(cached_status["result_key"])
                    if result:
                        response_data["result"] = result

                return Response(response_data)

            # Fallback to Celery status only
            return Response(
                {
                    "task_id": task_id,
                    "status": task_result.status.lower(),
                    "ready": task_result.ready(),
                    "progress": 0,
                    "message": "Processing...",
                    "result": task_result.result if task_result.ready() else None,
                }
            )

        except Exception as e:
            logger.error("Error checking task status: %s", e)
            return Response(
                {"error": "Failed to get task status", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    @action(detail=False, methods=["post"])
    def multiagent_async(self, request):
        """Start async multi-agent processing with intelligent agent selection"""
        query = request.data.get("query", "")
        conversation_id = request.data.get("conversation_id")
        agent_preference = request.data.get("agent_preference")

        if not query:
            return Response(
                {"error": "Query is required"}, status=status.HTTP_400_BAD_REQUEST
            )

        # Start async task
        task = process_multiagent_query.delay(
            query=query,
            user_id=request.user.id,
            conversation_id=conversation_id,
            agent_preference=agent_preference,
        )

        return Response(
            {
                "task_id": task.id,
                "status": "processing",
                "message": "Multi-agent query is being processed with intelligent routing.",
                "check_status_url": f"/api/agents/async/status/{task.id}/",
                "conversation_id": conversation_id,
            }
        )

    @action(detail=False, methods=["post"])
    def expand_knowledge(self, request):
        """Trigger async knowledge expansion for a topic"""
        topic = request.data.get("topic", "")
        max_urls = request.data.get("max_urls", 5)

        if not topic:
            return Response(
                {"error": "Topic is required"}, status=status.HTTP_400_BAD_REQUEST
            )

        # Start async knowledge expansion
        task = auto_expand_knowledge_async.delay(topic, max_urls)

        return Response(
            {
                "task_id": task.id,
                "status": "processing",
                "message": f'Knowledge expansion for "{topic}" started.',
                "check_status_url": f"/api/agents/async/status/{task.id}/",
                "topic": topic,
                "max_urls": max_urls,
            }
        )

    @action(detail=False, methods=["post"])
    def analyze_quality(self, request):
        """Trigger async content quality analysis"""
        task = analyze_content_quality_batch.delay()

        return Response(
            {
                "task_id": task.id,
                "status": "processing",
                "message": "Content quality analysis started.",
                "check_status_url": f"/api/agents/async/status/{task.id}/",
            }
        )

    @action(detail=False, methods=["get"])
    def health_async(self, request):
        """Get async system health check"""
        task = health_check.delay()

        return Response(
            {
                "task_id": task.id,
                "status": "processing",
                "message": "Health check started.",
                "check_status_url": f"/api/agents/async/status/{task.id}/",
            }
        )

    @action(detail=False, methods=["get"])
    def async_info(self, request):
        """Get information about async capabilities"""
        return Response(
            {
                "async_enabled": True,
                "available_async_endpoints": [
                    "chat_async",
                    "multiagent_async",
                    "expand_knowledge",
                    "analyze_quality",
                    "health_async",
                ],
                "status_check_endpoint": "/api/agents/async/status/{task_id}/",
                "average_response_times": {
                    "chat_async": "3-8 seconds",
                    "multiagent_async": "5-15 seconds",
                    "expand_knowledge": "30-120 seconds",
                    "analyze_quality": "60-300 seconds",
                },
                "benefits": [
                    "Non-blocking API responses",
                    "Real-time progress updates",
                    "Better user experience",
                    "Handles concurrent requests",
                    "Automatic retries on failure",
                ],
            }
        )

    @action(detail=False, methods=["post"])
    def enhanced_chat(self, request):
        """Enhanced chat with advanced RAG pipeline"""
        try:
            data = request.data
            message = data.get("message", "").strip()

            if not message:
                return Response(
                    {"error": "Message is required"}, status=status.HTTP_400_BAD_REQUEST
                )

            # Extract configuration
            config = {
                "agent_type": data.get("agent_type", "general"),
                "use_context": data.get("use_context", True),
                "conversation_id": data.get("conversation_id"),
                "response_config": {
                    "style": data.get("style", "informative"),
                    "max_length": data.get("max_length", "medium"),
                    "max_tokens": data.get("max_tokens", 400),
                    "temperature": data.get("temperature", 0.7),
                    "include_sources": data.get("include_sources", True),
                },
                "context_filters": {
                    "min_quality": data.get("min_quality", 0.3),
                    "page_urls": data.get("page_urls"),
                    "content_types": data.get("content_types", {}),
                },
            }

            # Update agent type if different
            if config["agent_type"] != self.enhanced_agent.agent_type:
                self.enhanced_agent.agent_type = config["agent_type"]

            # Handle conversation memory if conversation_id provided
            # conversation_history = None
            if config["conversation_id"]:
                memory = ConversationMemory(config["conversation_id"])
                conversation_context = memory.get_conversation_context()
                if conversation_context:
                    # Enhance message with conversation context
                    enhanced_message = f"Recent conversation context:\n{conversation_context}\n\nCurrent question: {message}"
                else:
                    enhanced_message = message
            else:
                enhanced_message = message

            # Process message with enhanced agent
            result = self.enhanced_agent.process_message(
                message=enhanced_message,
                use_context=config["use_context"],
                response_config=config["response_config"],
                context_filters=config["context_filters"],
            )

            # Store in conversation memory if conversation_id provided
            if config["conversation_id"]:
                memory = ConversationMemory(config["conversation_id"])
                memory.add_exchange(
                    user_message=message,
                    agent_response=result["response"],
                    metadata={
                        "context_used": result.get("used_context", False),
                        "context_sources": len(result.get("sources", [])),
                        "agent_type": result.get("agent_type", "general"),
                        "response_time_ms": result.get("processing_time_ms", 0),
                        "quality_scores": result.get("quality_scores", {}),
                    },
                )
                result["conversation_id"] = memory.conversation_id

            return Response(result, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error("Enhanced chat error: %s", e)
            return Response(
                {"error": "Failed to process message: %s" % str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    @action(detail=False, methods=["post"])
    def search_knowledge(self, request):
        """Direct knowledge base search with enhanced results"""
        try:
            query = request.data.get("query", "").strip()
            if not query:
                return Response(
                    {"error": "Query is required"}, status=status.HTTP_400_BAD_REQUEST
                )

            filters = request.data.get("filters", {})
            result = self.search_service.enhanced_search(query, filters)

            return Response(result, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error("Knowledge search error: %s", e)
            return Response(
                {"error": "Search failed: %s" % str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    @action(detail=False, methods=["get"])
    def enhanced_stats(self, request):
        """Get comprehensive enhanced knowledge base statistics"""
        try:
            stats = self.enhanced_agent.get_knowledge_stats()
            return Response(stats, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error("Enhanced stats error: %s", e)
            return Response(
                {"error": "Failed to get stats: %s" % str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    @action(detail=False, methods=["post"])
    def generate_embeddings(self, request):
        """Trigger embedding generation for pages"""
        try:
            page_ids = request.data.get("page_ids", [])
            force_regenerate = request.data.get("force_regenerate", False)
            use_async = request.data.get("async", True)

            if use_async:
                if page_ids:
                    task = batch_generate_embeddings.delay(page_ids, force_regenerate)
                else:
                    task = batch_generate_embeddings.delay(None, force_regenerate)

                return Response(
                    {
                        "task_id": task.id,
                        "status": "started",
                        "message": "Embedding generation started in background",
                    },
                    status=status.HTTP_202_ACCEPTED,
                )
            else:
                # Synchronous processing (for small batches only)
                if len(page_ids) > 10:
                    return Response(
                        {"error": "Use async=true for more than 10 pages"},
                        status=status.HTTP_400_BAD_REQUEST,
                    )

                from knowledge_base.services.embedding_generator import (
                    EmbeddingGenerationService,
                )

                embedding_service = EmbeddingGenerationService()
                result = embedding_service.batch_process_pages(
                    page_ids, force_regenerate
                )

                return Response(result, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error("Embedding generation error: %s", e)
            return Response(
                {"error": "Failed to generate embeddings: %s" % str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    @action(detail=False, methods=["get"])
    def vectorization_stats(self, request):
        """Get vectorization statistics"""
        try:
            from knowledge_base.services.embedding_generator import (
                EmbeddingGenerationService,
            )
            from crawler.models import CrawledPage

            embedding_service = EmbeddingGenerationService()
            stats = embedding_service.get_embedding_stats()

            # Get unprocessed pages count
            unprocessed_pages = (
                CrawledPage.objects.filter(
                    success=True,
                    is_processed_for_embeddings=False,
                    clean_markdown__isnull=False,
                )
                .exclude(clean_markdown="")
                .count()
            )

            stats.update(
                {
                    "unprocessed_pages": unprocessed_pages,
                    "can_vectorize": unprocessed_pages > 0,
                }
            )

            return Response(stats, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error("Vectorization stats error: %s", e)
            return Response(
                {"error": "Failed to get vectorization stats: %s" % str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    @action(detail=False, methods=["post"])
    def get_context(self, request):
        """Get context for a query without generating full response"""
        try:
            query = request.data.get("query", "").strip()
            if not query:
                return Response(
                    {"error": "Query is required"}, status=status.HTTP_400_BAD_REQUEST
                )

            max_length = request.data.get("max_context_length", 2000)
            context_info = self.enhanced_agent.get_conversation_context(
                query, max_length
            )

            return Response(context_info, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error("Get context error: %s", e)
            return Response(
                {"error": "Failed to get context: %s" % str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    @action(detail=False, methods=["post"])
    def live_research(self, request):
        """Conduct live internet research on a topic"""
        try:
            data = request.data
            research_topic = data.get("topic", "").strip()

            if not research_topic:
                return Response(
                    {"error": "Research topic is required"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            # Configuration options
            config = {
                "max_sources": data.get("max_sources", 8),
                "include_local_kb": data.get("include_local_kb", True),
                "research_depth": data.get("research_depth", "comprehensive"),
                "use_live_research": data.get("use_live_research", True),
            }

            # Initialize live research agent
            live_research_agent = LiveResearchAgent()

            # Conduct research asynchronously
            async def conduct_research():
                return await live_research_agent.enhanced_research_chat(
                    message=research_topic,
                    use_live_research=config["use_live_research"],
                    max_sources=config["max_sources"],
                    research_depth=config["research_depth"],
                )

            # Run async research
            import asyncio

            result = asyncio.run(conduct_research())

            # Add processing metadata
            result.update(
                {
                    "processing_time_ms": int(
                        result.get("research_methodology", {}).get(
                            "research_time_seconds", 0
                        )
                        * 1000
                    ),
                    "conversation_id": data.get("conversation_id"),
                    "timestamp": timezone.now().isoformat(),
                    "research_topic": research_topic,
                    "config_used": config,
                }
            )

            return Response(result, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error("Live research error: %s", e, exc_info=True)
            return Response(
                {
                    "error": "Live research failed",
                    "details": str(e),
                    "response": "I encountered an error while conducting live research. Please try again or use a different research topic.",
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    @action(detail=False, methods=["get"])
    def research_capabilities(self, request):
        """Get information about research capabilities"""
        try:
            live_research_agent = LiveResearchAgent()
            capabilities = live_research_agent.get_research_capabilities()

            return Response(capabilities, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error("Error getting research capabilities: %s", e, exc_info=True)
            return Response(
                {"error": "Failed to get research capabilities"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class AsyncAgentViewSet(viewsets.GenericViewSet):
    """Dedicated viewset for async operations with clean URLs"""

    permission_classes = [permissions.IsAuthenticated]

    @action(detail=False, methods=["post"], url_path="chat_async")
    def chat_async(self, request):
        """Start async chat processing with real-time status updates"""
        message = request.data.get("message", "")
        use_context = request.data.get("use_context", True)
        max_tokens = request.data.get("max_tokens", 300)
        conversation_id = request.data.get("conversation_id")
        agent_type = request.data.get("agent_type", "general")

        if not message:
            return Response(
                {"error": "Message is required"}, status=status.HTTP_400_BAD_REQUEST
            )

        # Validate agent type
        if agent_type not in ["general", "research", "qa", "content_analyzer"]:
            agent_type = "general"

        # Start async task
        task = process_user_message_async.delay(
            message=message,
            user_id=request.user.id,
            conversation_id=conversation_id,
            use_context=use_context,
            agent_type=agent_type,
            max_tokens=max_tokens,
        )

        return Response(
            {
                "task_id": task.id,
                "status": "processing",
                "message": "Your request is being processed. Use the task_id to check status.",
                "check_status_url": f"/api/agents/async/status/{task.id}/",
                "conversation_id": conversation_id,
                "agent_type": agent_type,
            }
        )

    @action(detail=False, methods=["get"], url_path="status/(?P<task_id>[^/.]+)")
    def check_task_status(self, request, task_id):
        """Check status of async task with detailed progress"""
        try:
            # Get task result
            task_result = AsyncResult(task_id)

            # Get cached status for more detailed progress
            cached_status = cache.get(f"task_status:{task_id}")

            if cached_status:
                response_data = cached_status.copy()
                response_data["task_id"] = task_id
                response_data["celery_status"] = task_result.status

                # If completed, include result
                if (
                    cached_status["status"] == "completed"
                    and "result_key" in cached_status
                ):
                    result = cache.get(cached_status["result_key"])
                    if result:
                        response_data["result"] = result

                return Response(response_data)

            # Fallback to Celery status only
            return Response(
                {
                    "task_id": task_id,
                    "status": task_result.status.lower(),
                    "ready": task_result.ready(),
                    "progress": 0,
                    "message": "Processing...",
                    "result": task_result.result if task_result.ready() else None,
                }
            )

        except Exception as e:
            logger.error("Error checking task status: %s", e)
            return Response(
                {"error": "Failed to get task status", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    @action(detail=False, methods=["post"], url_path="multiagent_async")
    def multiagent_async(self, request):
        """Start async multi-agent processing"""
        query = request.data.get("query", "")
        conversation_id = request.data.get("conversation_id")
        agent_preference = request.data.get("agent_preference")

        if not query:
            return Response(
                {"error": "Query is required"}, status=status.HTTP_400_BAD_REQUEST
            )

        # Start async task
        task = process_multiagent_query.delay(
            query=query,
            user_id=request.user.id,
            conversation_id=conversation_id,
            agent_preference=agent_preference,
        )

        return Response(
            {
                "task_id": task.id,
                "status": "processing",
                "message": "Multi-agent query is being processed.",
                "check_status_url": f"/api/agents/async/status/{task.id}/",
                "conversation_id": conversation_id,
            }
        )

    @action(detail=False, methods=["post"], url_path="expand_knowledge")
    def expand_knowledge(self, request):
        """Trigger async knowledge expansion"""
        topic = request.data.get("topic", "")
        max_urls = request.data.get("max_urls", 5)

        if not topic:
            return Response(
                {"error": "Topic is required"}, status=status.HTTP_400_BAD_REQUEST
            )

        task = auto_expand_knowledge_async.delay(topic, max_urls)

        return Response(
            {
                "task_id": task.id,
                "status": "processing",
                "message": f'Knowledge expansion for "{topic}" started.',
                "check_status_url": f"/api/agents/async/status/{task.id}/",
                "topic": topic,
                "max_urls": max_urls,
            }
        )

    @action(detail=False, methods=["post"], url_path="live_research_async")
    def live_research_async(self, request):
        """Start async live research processing with real-time status updates"""
        research_topic = request.data.get("topic", "").strip()
        max_sources = request.data.get("max_sources", 5)
        include_local_kb = request.data.get("include_local_kb", True)
        research_depth = request.data.get("research_depth", "comprehensive")
        use_live_research = request.data.get("use_live_research", True)
        conversation_id = request.data.get("conversation_id")

        if not research_topic:
            return Response(
                {"error": "Research topic is required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Validate research depth
        if research_depth not in ["surface", "comprehensive", "deep"]:
            research_depth = "comprehensive"

        # Limit max sources to prevent excessive processing time
        max_sources = min(max_sources, 10)

        # Start async live research task
        task = process_live_research_async.delay(
            research_topic=research_topic,
            user_id=request.user.id,
            max_sources=max_sources,
            include_local_kb=include_local_kb,
            research_depth=research_depth,
            use_live_research=use_live_research,
            conversation_id=conversation_id,
        )

        return Response(
            {
                "task_id": task.id,
                "status": "processing",
                "message": "Live research started. Use the task_id to check status.",
                "check_status_url": f"/api/agents/async/status/{task.id}/",
                "research_topic": research_topic,
                "conversation_id": conversation_id,
                "config": {
                    "max_sources": max_sources,
                    "research_depth": research_depth,
                    "use_live_research": use_live_research,
                    "include_local_kb": include_local_kb,
                },
            }
        )
