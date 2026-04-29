"""Application configuration for the parking domain."""

from django.apps import AppConfig


class ParkingConfig(AppConfig):
    """Register the parking control app and its domain models."""

    default_auto_field = "django.db.models.BigAutoField"
    name = "parking"
