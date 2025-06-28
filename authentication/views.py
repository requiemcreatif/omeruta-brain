from django.shortcuts import render, redirect
from rest_framework import status, generics, permissions
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework_simplejwt.views import TokenObtainPairView
from django.contrib.auth import get_user_model, login, logout, authenticate

from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from django.views import View
from django.contrib import messages
from django.http import JsonResponse

from .serializers import (
    UserRegistrationSerializer,
    UserLoginSerializer,
    UserProfileSerializer,
    ChangePasswordSerializer,
)

User = get_user_model()


class UserRegistrationView(generics.CreateAPIView):
    queryset = User.objects.all()
    serializer_class = UserRegistrationSerializer
    permission_classes = [permissions.AllowAny]

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.save()

        # Generate JWT tokens
        refresh = RefreshToken.for_user(user)

        return Response(
            {
                "user": UserProfileSerializer(user).data,
                "tokens": {
                    "refresh": str(refresh),
                    "access": str(refresh.access_token),
                },
                "message": "User registered successfully",
            },
            status=status.HTTP_201_CREATED,
        )


@method_decorator(csrf_exempt, name="dispatch")
class UserLoginView(APIView):
    """
    Unified login view that handles both web forms and API calls properly
    """

    permission_classes = [permissions.AllowAny]

    def get(self, request):
        """Render the login page"""
        # If user is already authenticated, redirect to dashboard
        if request.user.is_authenticated:
            return redirect("core:dashboard")
        return render(request, "authentication/login.html")

    def post(self, request):
        """Handle login for both API and form submissions"""
        # Determine if this is an API call or form submission
        is_api_call = (
            request.content_type == "application/json"
            or "application/json" in request.META.get("HTTP_ACCEPT", "")
            or request.META.get("HTTP_X_REQUESTED_WITH") == "XMLHttpRequest"
        )

        if is_api_call:
            return self._handle_api_login(request)
        else:
            return self._handle_form_login(request)

    def _handle_api_login(self, request):
        """Handle API login with JWT tokens"""
        try:
            serializer = UserLoginSerializer(data=request.data)
            serializer.is_valid(raise_exception=True)
            user = serializer.validated_data["user"]

            # Generate JWT tokens
            refresh = RefreshToken.for_user(user)

            # Also log them into the session for web interface
            login(request, user)

            return Response(
                {
                    "user": UserProfileSerializer(user).data,
                    "tokens": {
                        "refresh": str(refresh),
                        "access": str(refresh.access_token),
                    },
                    "message": "Login successful",
                },
                status=status.HTTP_200_OK,
            )
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

    def _handle_form_login(self, request):
        """Handle form login with session authentication"""
        email = request.POST.get("email", "").strip()
        password = request.POST.get("password", "").strip()

        if not email or not password:
            messages.error(request, "Email and password are required.")
            return render(request, "authentication/login.html")

        # Authenticate user
        user = authenticate(request, username=email, password=password)

        if user is not None:
            if user.is_active:
                # Log the user in (creates session)
                login(request, user)

                # Generate JWT tokens and store in session for API access
                refresh = RefreshToken.for_user(user)
                request.session["jwt_access_token"] = str(refresh.access_token)
                request.session["jwt_refresh_token"] = str(refresh)

                messages.success(request, "Login successful!")

                # Redirect to intended page or dashboard
                next_url = request.GET.get("next")
                if next_url:
                    return redirect(next_url)
                else:
                    return redirect("core:dashboard")
            else:
                messages.error(request, "Account is disabled.")
        else:
            messages.error(request, "Invalid email or password.")

        return render(request, "authentication/login.html")


@csrf_exempt
def user_logout(request):
    """Handle logout for both GET and POST requests"""
    if request.method == "GET":
        # Handle web logout
        logout(request)
        messages.success(request, "You have been logged out successfully.")
        return redirect("home")

    elif request.method == "POST":
        # Handle API logout
        try:
            # Parse JSON data if present
            data = {}
            if request.content_type == "application/json":
                import json

                data = json.loads(request.body.decode("utf-8"))

            # Try to blacklist the refresh token if provided
            refresh_token = data.get("refresh_token")
            if refresh_token:
                token = RefreshToken(refresh_token)
                token.blacklist()

            # Also logout from session
            logout(request)

            return JsonResponse(
                {"message": "Successfully logged out"},
                status=200,
            )
        except Exception as e:
            # Even if token blacklisting fails, logout from session
            logout(request)
            return JsonResponse(
                {"message": "Logged out (token may have been invalid)"},
                status=200,
            )

    return JsonResponse({"error": "Method not allowed"}, status=405)


class CheckAuthStatusView(View):
    """Check if user is authenticated and return user data - Plain Django view"""

    @method_decorator(csrf_exempt)
    def dispatch(self, request, *args, **kwargs):
        return super().dispatch(request, *args, **kwargs)

    def get(self, request):
        if not request.user.is_authenticated:
            return JsonResponse({"error": "Not authenticated"}, status=401)

        return JsonResponse(
            {
                "authenticated": True,
                "user": UserProfileSerializer(request.user).data,
                "is_admin": request.user.is_staff or request.user.is_superuser,
            },
            status=200,
        )


class GetJWTTokensView(View):
    """Get JWT tokens for authenticated session user - Plain Django view"""

    @method_decorator(csrf_exempt)
    def dispatch(self, request, *args, **kwargs):
        return super().dispatch(request, *args, **kwargs)

    def post(self, request):
        # Check if user is authenticated via session
        if not request.user.is_authenticated:
            return JsonResponse({"error": "Not authenticated"}, status=401)

        try:
            # Generate new tokens
            refresh = RefreshToken.for_user(request.user)

            # Store in session for future reference
            request.session["jwt_access_token"] = str(refresh.access_token)
            request.session["jwt_refresh_token"] = str(refresh)

            return JsonResponse(
                {
                    "tokens": {
                        "refresh": str(refresh),
                        "access": str(refresh.access_token),
                    },
                    "user": UserProfileSerializer(request.user).data,
                },
                status=200,
            )
        except Exception as e:
            return JsonResponse(
                {"error": f"Token generation failed: {str(e)}"},
                status=500,
            )


class UserProfileView(generics.RetrieveUpdateAPIView):
    serializer_class = UserProfileSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_object(self):
        return self.request.user


class ChangePasswordView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        serializer = ChangePasswordSerializer(
            data=request.data, context={"request": request}
        )
        serializer.is_valid(raise_exception=True)

        user = request.user
        user.set_password(serializer.validated_data["new_password"])
        user.save()

        return Response(
            {"message": "Password changed successfully"}, status=status.HTTP_200_OK
        )


# Admin-specific views
class AdminLoginView(TokenObtainPairView):
    """Enhanced login view for admin users"""

    permission_classes = [permissions.AllowAny]

    def post(self, request, *args, **kwargs):
        response = super().post(request, *args, **kwargs)

        if response.status_code == 200:
            # Verify if user is admin
            serializer = UserLoginSerializer(data=request.data)
            if serializer.is_valid():
                user = serializer.validated_data.get("user")
                if user and (user.is_staff or user.is_superuser):
                    user_data = UserProfileSerializer(user).data
                    response.data.update(
                        {
                            "user": user_data,
                            "message": "Admin login successful",
                            "is_admin": True,
                        }
                    )
                else:
                    return Response(
                        {"error": "Access denied. Admin privileges required."},
                        status=status.HTTP_403_FORBIDDEN,
                    )
        return response


@api_view(["GET"])
@permission_classes([permissions.IsAuthenticated])
def admin_dashboard(request):
    """Admin dashboard data endpoint"""
    if not (request.user.is_staff or request.user.is_superuser):
        return Response(
            {"error": "Admin access required"}, status=status.HTTP_403_FORBIDDEN
        )

    # Get some basic stats
    total_users = User.objects.count()
    active_users = User.objects.filter(is_active=True).count()
    staff_users = User.objects.filter(is_staff=True).count()

    return Response(
        {
            "stats": {
                "total_users": total_users,
                "active_users": active_users,
                "staff_users": staff_users,
            },
            "user": UserProfileSerializer(request.user).data,
        },
        status=status.HTTP_200_OK,
    )
