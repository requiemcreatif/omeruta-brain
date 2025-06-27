from django.urls import path
from rest_framework_simplejwt.views import TokenRefreshView
from . import views

app_name = "authentication"

urlpatterns = [
    # User authentication
    path("register/", views.UserRegistrationView.as_view(), name="user_register"),
    path("login/", views.UserLoginView.as_view(), name="user_login"),
    path("logout/", views.UserLogoutView.as_view(), name="user_logout"),
    path("profile/", views.UserProfileView.as_view(), name="user_profile"),
    path(
        "change-password/", views.ChangePasswordView.as_view(), name="change_password"
    ),
    path("check-auth/", views.check_auth_status, name="check_auth"),
    # JWT token management
    path("token/refresh/", TokenRefreshView.as_view(), name="token_refresh"),
    # Admin authentication
    path("admin/login/", views.AdminLoginView.as_view(), name="admin_login"),
    path("admin/dashboard/", views.admin_dashboard, name="admin_dashboard"),
]
