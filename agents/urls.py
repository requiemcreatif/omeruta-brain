from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import Phi3AgentViewSet, AsyncAgentViewSet, AIAssistantView

app_name = "agents"

router = DefaultRouter()
router.register(r"phi3", Phi3AgentViewSet, basename="phi3")
router.register(r"async", AsyncAgentViewSet, basename="async")

# Legacy support - keep tinyllama routes for backward compatibility during transition
router.register(r"tinyllama", Phi3AgentViewSet, basename="tinyllama-legacy")

urlpatterns = [
    path("", AIAssistantView.as_view(), name="ai_assistant"),
    path("api/agents/", include(router.urls)),
]
