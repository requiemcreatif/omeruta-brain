from django.shortcuts import render
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework.permissions import AllowAny
from rest_framework import status
from django.views.generic import TemplateView
from django.contrib.auth.mixins import LoginRequiredMixin

# Create your views here.


@api_view(["GET"])
@permission_classes([AllowAny])
def health_check(request):
    """
    Simple health check endpoint
    """
    return Response(
        {"status": "healthy", "message": "Omeruta Brain API is running successfully"},
        status=status.HTTP_200_OK,
    )


@api_view(["GET"])
def api_info(request):
    """
    Provides basic information about the API.
    """
    return Response(
        {
            "name": "Omeruta Brain API",
            "version": "1.0.0",
            "documentation": "/api/docs/",
            "endpoints": {
                "health": "/core/health/",
                "info": "/core/info/",
                "auth": "/auth/",
                "crawler": "/crawler/",
                "agents": "/ai/api/agents/",
            },
        }
    )


class DashboardView(LoginRequiredMixin, TemplateView):
    template_name = "core/index.html"
