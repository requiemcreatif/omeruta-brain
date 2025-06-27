from rest_framework import viewsets, permissions, status
from rest_framework.decorators import action
from rest_framework.response import Response
from .services.tinyllama_agent import TinyLlamaAgent


class TinyLlamaViewSet(viewsets.GenericViewSet):
    """ViewSet for TinyLlama agent interactions"""

    permission_classes = [permissions.IsAuthenticated]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent = TinyLlamaAgent()

    @action(detail=False, methods=["post"])
    def chat(self, request):
        """Chat with TinyLlama agent"""
        message = request.data.get("message", "")
        use_context = request.data.get("use_context", True)
        max_tokens = request.data.get("max_tokens", 300)

        if not message:
            return Response(
                {"error": "Message is required"}, status=status.HTTP_400_BAD_REQUEST
            )

        result = self.agent.process_message(
            message=message, use_context=use_context, max_tokens=max_tokens
        )

        return Response(result)

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
            }
        )
