from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import TinyLlamaViewSet

router = DefaultRouter()
router.register(r"tinyllama", TinyLlamaViewSet, basename="tinyllama")

urlpatterns = [
    path("", include(router.urls)),
]
