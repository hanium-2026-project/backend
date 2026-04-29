"""Small CORS middleware used to avoid a hard dependency for local MVP runs."""

from django.conf import settings
from django.http import HttpRequest, HttpResponse


class SimpleCorsMiddleware:
    """Allow the React development server to call the REST API.

    A dedicated package such as django-cors-headers is preferable in production,
    but the MVP keeps this narrow middleware so tests and local execution do not
    depend on extra configuration.
    """

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request: HttpRequest) -> HttpResponse:
        if request.method == "OPTIONS":
            response = HttpResponse(status=204)
        else:
            response = self.get_response(request)

        origin = request.headers.get("Origin")
        if origin in getattr(settings, "CORS_ALLOWED_ORIGINS", []):
            response["Access-Control-Allow-Origin"] = origin
            response["Vary"] = "Origin"
            response["Access-Control-Allow-Methods"] = "GET,POST,PUT,PATCH,DELETE,OPTIONS"
            response["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
        return response
