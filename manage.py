#!/usr/bin/env python3
"""Django command-line entrypoint for the parking control backend."""

import os
import sys


def main() -> None:
    """Run administrative Django commands with the project settings loaded."""
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
    from django.core.management import execute_from_command_line

    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
