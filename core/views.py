from django.shortcuts import render
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework.permissions import AllowAny
from rest_framework import status
from django.views.generic import TemplateView

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
@permission_classes([AllowAny])
def api_info(request):
    """
    API information endpoint
    """
    return Response(
        {
            "name": "Omeruta Brain API",
            "version": "1.0.0",
            "description": "AI-powered brain and knowledge management system",
            "endpoints": {
                "health": "/api/health/",
                "info": "/api/info/",
                "admin": "/admin/",
            },
        },
        status=status.HTTP_200_OK,
    )


class DashboardView(TemplateView):
    template_name = "core/index.html"
