"""WebSocket URL routing for the parking app."""

from django.urls import path

from .consumers import DashboardConsumer


websocket_urlpatterns = [
    path("ws/dashboard/", DashboardConsumer.as_asgi()),
]
