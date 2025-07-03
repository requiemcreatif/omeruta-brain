from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import Phi3AgentViewSet, AsyncAgentViewSet, AIAssistantView

app_name = "agents"

router = DefaultRouter()
router.register(r"phi3", Phi3AgentViewSet, basename="phi3")
router.register(r"async", AsyncAgentViewSet, basename="async")

# Phi3 is now the primary and only model

urlpatterns = [
    path("", AIAssistantView.as_view(), name="ai_assistant"),
    path("api/agents/", include(router.urls)),
]
