from django.urls import path
from . import views

app_name = "core"

urlpatterns = [
    path("health/", views.health_check, name="health_check"),
    path("info/", views.api_info, name="api_info"),
    path("dashboard/", views.DashboardView.as_view(), name="dashboard"),
]
