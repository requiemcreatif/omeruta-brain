from django.urls import path
from rest_framework_simplejwt.views import TokenRefreshView
from . import views

app_name = "authentication"

urlpatterns = [
    # User authentication
    path("register/", views.UserRegistrationView.as_view(), name="user_register"),
    path("login/", views.UserLoginView.as_view(), name="user_login"),
    path("logout/", views.user_logout, name="user_logout"),
    path("api-logout/", views.user_logout, name="api_logout"),  # Alternative endpoint
    path("profile/", views.UserProfileView.as_view(), name="user_profile"),
    path(
        "change-password/", views.ChangePasswordView.as_view(), name="change_password"
    ),
    path("check-auth/", views.CheckAuthStatusView.as_view(), name="check_auth"),
    path("get-tokens/", views.GetJWTTokensView.as_view(), name="get_tokens"),
    # JWT token management
    path("token/refresh/", TokenRefreshView.as_view(), name="token_refresh"),
    # Admin authentication
    path("admin/login/", views.AdminLoginView.as_view(), name="admin_login"),
    path("admin/dashboard/", views.admin_dashboard, name="admin_dashboard"),
]
