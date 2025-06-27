from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import TinyLlamaViewSet, AsyncAgentViewSet

router = DefaultRouter()
router.register(r"tinyllama", TinyLlamaViewSet, basename="tinyllama")
router.register(r"async", AsyncAgentViewSet, basename="async")

urlpatterns = [
    path("", include(router.urls)),
]
