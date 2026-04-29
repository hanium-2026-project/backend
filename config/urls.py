"""Root URL configuration for REST and admin endpoints."""

from django.contrib import admin
from django.urls import include, path


urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/", include("parking.urls")),
]
