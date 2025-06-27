from django.core.management.base import BaseCommand
import subprocess
import sys
import os


class Command(BaseCommand):
    help = "Start ASGI server with WebSocket support using Daphne"

    def add_arguments(self, parser):
        parser.add_argument(
            "--port",
            type=int,
            default=8000,
            help="Port to run the server on (default: 8000)",
        )
        parser.add_argument(
            "--bind",
            type=str,
            default="127.0.0.1",
            help="IP address to bind to (default: 127.0.0.1)",
        )

    def handle(self, *args, **options):
        port = options["port"]
        bind = options["bind"]
        verbosity = options.get("verbosity", 1)

        # Set up environment
        os.environ.setdefault(
            "DJANGO_SETTINGS_MODULE", "omeruta_brain_project.settings"
        )

        self.stdout.write(
            self.style.SUCCESS(
                "🚀 Starting ASGI server with WebSocket support...\n"
                f"Server will be available at: http://{bind}:{port}\n"
                f"WebSocket endpoint: ws://{bind}:{port}/ws/agent/\n"
            )
        )

        # Check if daphne is installed
        try:
            import daphne
        except ImportError:
            self.stdout.write(
                self.style.ERROR(
                    "❌ Daphne is not installed. Install it with:\n"
                    "pip install daphne\n"
                )
            )
            return

        # Build the command
        cmd = [
            sys.executable,
            "-m",
            "daphne",
            "-b",
            bind,
            "-p",
            str(port),
            "omeruta_brain_project.asgi:application",
        ]

        if verbosity >= 2:
            cmd.append("-v2")
        elif verbosity == 0:
            cmd.append("-v0")

        try:
            self.stdout.write(f"Command: {' '.join(cmd)}")
            self.stdout.write(
                self.style.WARNING(
                    "Features available:\n"
                    "✅ HTTP API endpoints\n"
                    "✅ WebSocket real-time updates\n"
                    "✅ Async task monitoring\n"
                    "✅ Live progress tracking\n"
                    "\nPress Ctrl+C to stop the server\n"
                )
            )

            subprocess.run(cmd, check=True)

        except subprocess.CalledProcessError as e:
            self.stdout.write(self.style.ERROR(f"Failed to start ASGI server: {e}"))
        except KeyboardInterrupt:
            self.stdout.write(self.style.WARNING("\n⏹️  ASGI server stopped by user"))

    def handle_help(self):
        self.stdout.write(
            """
ASGI WebSocket Server Management Command

This starts a production-ready ASGI server with WebSocket support using Daphne.

Usage Examples:
  python manage.py start_websocket_server
  python manage.py start_websocket_server --port 8080
  python manage.py start_websocket_server --bind 0.0.0.0 --port 8000

Features:
  - Full HTTP API support (all existing endpoints)
  - Real-time WebSocket communication
  - Live task progress updates
  - User authentication via WebSocket
  - Auto-reconnection support

WebSocket Endpoints:
  - ws://localhost:8000/ws/agent/ (main agent communication)

Prerequisites:
  1. Redis server must be running
  2. At least one Celery worker should be running
  3. Install Daphne: pip install daphne

Options:
  --port: Port to run the server on (default: 8000)
  --bind: IP address to bind to (default: 127.0.0.1, use 0.0.0.0 for external access)
  --verbosity: Log verbosity level (0=minimal, 1=normal, 2=verbose)
        """
        )
