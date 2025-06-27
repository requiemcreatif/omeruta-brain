from django.core.management.base import BaseCommand
import subprocess
import sys
import os


class Command(BaseCommand):
    help = "Start Celery Beat scheduler for periodic tasks"

    def add_arguments(self, parser):
        parser.add_argument(
            "--loglevel",
            type=str,
            default="info",
            help="Log level (debug, info, warning, error)",
        )
        parser.add_argument(
            "--pidfile", type=str, default="celerybeat.pid", help="Path to PID file"
        )
        parser.add_argument(
            "--schedule",
            type=str,
            default="celerybeat-schedule",
            help="Path to schedule database file",
        )

    def handle(self, *args, **options):
        loglevel = options["loglevel"]
        pidfile = options["pidfile"]
        schedule = options["schedule"]

        # Set up environment
        os.environ.setdefault(
            "DJANGO_SETTINGS_MODULE", "omeruta_brain_project.settings"
        )

        self.stdout.write(
            self.style.SUCCESS(
                "🕐 Starting Celery Beat scheduler for periodic tasks...\n"
                "This will handle scheduled background operations:"
            )
        )

        # List of periodic tasks that will be scheduled
        periodic_tasks = [
            "⏰ Cleanup expired tasks (every hour)",
            "🔄 Process embeddings (every 30 minutes)",
            "📊 Update content freshness (daily)",
            "🧠 Analyze content quality (weekly)",
        ]

        for task in periodic_tasks:
            self.stdout.write(f"  {task}")

        self.stdout.write("\n")

        cmd = [
            sys.executable,
            "-m",
            "celery",
            "-A",
            "omeruta_brain_project",
            "beat",
            "--loglevel",
            loglevel,
            "--pidfile",
            pidfile,
            "--schedule",
            schedule,
        ]

        try:
            self.stdout.write(f"Command: {' '.join(cmd)}")
            self.stdout.write(
                self.style.WARNING(
                    "Note: Make sure Redis is running and Celery workers are started!\n"
                    "Press Ctrl+C to stop the scheduler\n"
                )
            )

            subprocess.run(cmd, check=True)

        except subprocess.CalledProcessError as e:
            self.stdout.write(self.style.ERROR(f"Failed to start Beat scheduler: {e}"))
        except KeyboardInterrupt:
            self.stdout.write(self.style.WARNING("\n⏹️  Beat scheduler stopped by user"))

        # Cleanup
        try:
            if os.path.exists(pidfile):
                os.remove(pidfile)
                self.stdout.write(f"Cleaned up PID file: {pidfile}")
        except Exception as e:
            self.stdout.write(self.style.WARNING(f"Could not clean up PID file: {e}"))

    def handle_help(self):
        self.stdout.write(
            """
Celery Beat Scheduler Management Command

The Beat scheduler handles periodic background tasks:

🔄 Automated Tasks:
  - Cleanup expired tasks and conversations (every hour)
  - Process unprocessed pages for embeddings (every 30 minutes)
  - Update content freshness scores (daily at 2 AM)
  - Analyze content quality in batches (weekly on Monday)

Usage Examples:
  python manage.py start_celery_beat
  python manage.py start_celery_beat --loglevel debug
  python manage.py start_celery_beat --pidfile /tmp/celerybeat.pid

Prerequisites:
  1. Redis server must be running
  2. At least one Celery worker should be running
  3. Run: python manage.py start_celery_worker --queue all

Options:
  --loglevel: Log level - debug, info, warning, error (default: info)
  --pidfile: Path to PID file (default: celerybeat.pid)
  --schedule: Path to schedule database file (default: celerybeat-schedule)
        """
        )
