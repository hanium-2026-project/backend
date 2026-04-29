"""Seed deterministic demo data for local development."""

from django.core.management.base import BaseCommand

from parking.services import seed_demo_data


class Command(BaseCommand):
    """Create a small parking lot, spots, and cameras for MVP demos."""

    help = "Seed demo parking lot, spots, and camera records."

    def handle(self, *args, **options) -> None:
        seed_demo_data()
        self.stdout.write(self.style.SUCCESS("Seeded demo parking data."))
