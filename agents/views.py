from rest_framework import viewsets, permissions, status
from rest_framework.decorators import action
from rest_framework.response import Response
from .services.tinyllama_agent import TinyLlamaAgent
from .services.conversation_memory import ConversationMemory
from .tasks import (
    process_user_message_async,
    process_multiagent_query,
    auto_expand_knowledge_async,
    analyze_content_quality_batch,
    health_check,
)
from celery.result import AsyncResult
from django.core.cache import cache
import time
import logging
from django.views.generic import TemplateView

logger = logging.getLogger(__name__)


class AIAssistantView(TemplateView):
    template_name = "agents/ai_assistant.html"


class TinyLlamaViewSet(viewsets.GenericViewSet):
    """Enhanced ViewSet with conversation memory"""

    permission_classes = [permissions.IsAuthenticated]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent = TinyLlamaAgent()

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

            # Process with agent
            result = self.agent.process_message(
                message=enhanced_message, use_context=use_context, max_tokens=max_tokens
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
                f"Chat request processed: user={request.user.id}, "
                f"question_type={result.get('question_type')}, "
                f"context_used={result.get('context_used')}, "
                f"response_time={response_time_ms}ms"
            )

            return Response(result)

        except Exception as e:
            logger.error(f"Error in chat endpoint: {e}")
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

    @action(detail=False, methods=["get"])
    def status(self, request):
        """Get agent and knowledge base status"""
        stats = self.agent.get_available_knowledge_stats()
        model_info = self.agent.llm_service.get_model_info()

        return Response(
            {
                "knowledge_stats": stats,
                "model_info": model_info,
                "agent_type": self.agent.agent_type,
                "available_agent_types": list(self.agent.system_prompts.keys()),
            }
        )

    @action(detail=False, methods=["post"])
    def change_agent_type(self, request):
        """Change the agent type (general, research, qa, content_analyzer)"""
        agent_type = request.data.get("agent_type", "general")

        if agent_type not in self.agent.system_prompts:
            return Response(
                {
                    "error": f"Invalid agent type. Available types: {list(self.agent.system_prompts.keys())}"
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Create new agent with specified type
        self.agent = TinyLlamaAgent(agent_type=agent_type)

        return Response(
            {
                "message": f"Agent type changed to {agent_type}",
                "agent_type": agent_type,
                "system_prompt": self.agent.system_prompts[agent_type],
            }
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
            logger.error(f"Error checking task status: {e}")
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
            logger.error(f"Error checking task status: {e}")
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
