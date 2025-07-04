"""
URL configuration for omeruta_brain_project project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from authentication import views as auth_views
from django.views.generic import RedirectView

urlpatterns = [
    path("", auth_views.UserLoginView.as_view(), name="home"),
    path("admin/", admin.site.urls),
    path("core/", include("core.urls")),
    path("ai/", include("agents.urls")),
    path("auth/", include("authentication.urls")),
    path("crawler/", include("crawler.urls")),
    path("accounts/", include("allauth.urls")),
    path("__reload__/", include("django_browser_reload.urls")),
    # Redirect /login to the home page
    path("login/", RedirectView.as_view(pattern_name="home", permanent=True)),
    # Dedicated logout endpoint to bypass DRF interference
    path("simple-logout/", auth_views.user_logout, name="simple_logout"),
]

# Serve static and media files during development
if settings.DEBUG:
    from django.contrib.staticfiles.urls import staticfiles_urlpatterns

    urlpatterns += staticfiles_urlpatterns()
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
