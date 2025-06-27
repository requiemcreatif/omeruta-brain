from django.core.management.base import BaseCommand
import subprocess
import sys
import os


class Command(BaseCommand):
    help = "Start Celery workers for different queues"

    def add_arguments(self, parser):
        parser.add_argument(
            "--queue",
            type=str,
            default="all",
            help="Queue to process (ai_high_priority, ai_processing, crawling, embeddings, or all)",
        )
        parser.add_argument(
            "--concurrency", type=int, default=2, help="Number of concurrent workers"
        )
        parser.add_argument(
            "--loglevel",
            type=str,
            default="info",
            help="Log level (debug, info, warning, error)",
        )

    def handle(self, *args, **options):
        queue = options["queue"]
        concurrency = options["concurrency"]
        loglevel = options["loglevel"]

        # Set up environment
        os.environ.setdefault(
            "DJANGO_SETTINGS_MODULE", "omeruta_brain_project.settings"
        )

        if queue == "all":
            self.stdout.write(
                self.style.SUCCESS(
                    "Starting Celery workers for all queues...\n"
                    "This will start multiple worker processes:"
                )
            )

            # Start different workers for different queues
            workers = [
                ("ai_high_priority", 1, "High priority AI tasks"),
                ("ai_processing", 2, "General AI processing"),
                ("crawling", 2, "Web crawling tasks"),
                ("embeddings", 1, "Embedding generation"),
            ]

            processes = []

            for queue_name, worker_count, description in workers:
                self.stdout.write(
                    f"Starting {worker_count} workers for {queue_name} ({description})"
                )

                cmd = [
                    sys.executable,
                    "-m",
                    "celery",
                    "-A",
                    "omeruta_brain_project",
                    "worker",
                    "--loglevel",
                    loglevel,
                    "--queues",
                    queue_name,
                    "--concurrency",
                    str(worker_count),
                    "--hostname",
                    f"worker-{queue_name}@%h",
                ]

                try:
                    process = subprocess.Popen(cmd)
                    processes.append((process, queue_name))
                    self.stdout.write(
                        self.style.SUCCESS(
                            f"✓ Started worker for {queue_name} (PID: {process.pid})"
                        )
                    )
                except Exception as e:
                    self.stdout.write(
                        self.style.ERROR(
                            f"✗ Failed to start worker for {queue_name}: {e}"
                        )
                    )

            if processes:
                self.stdout.write(
                    self.style.SUCCESS(
                        f"\n🚀 Started {len(processes)} worker processes!"
                        f"\nPress Ctrl+C to stop all workers"
                    )
                )

                try:
                    # Wait for all processes
                    for process, queue_name in processes:
                        process.wait()
                except KeyboardInterrupt:
                    self.stdout.write(
                        self.style.WARNING("\n⏹️  Stopping all workers...")
                    )
                    for process, queue_name in processes:
                        process.terminate()
                    self.stdout.write(self.style.SUCCESS("✓ All workers stopped"))

        else:
            # Start single queue worker
            self.stdout.write(f"Starting Celery worker for queue: {queue}")

            cmd = [
                sys.executable,
                "-m",
                "celery",
                "-A",
                "omeruta_brain_project",
                "worker",
                "--loglevel",
                loglevel,
                "--queues",
                queue,
                "--concurrency",
                str(concurrency),
                "--hostname",
                f"worker-{queue}@%h",
            ]

            try:
                self.stdout.write(f"Command: {' '.join(cmd)}")
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                self.stdout.write(self.style.ERROR(f"Failed to start worker: {e}"))
            except KeyboardInterrupt:
                self.stdout.write(self.style.WARNING("Worker stopped by user"))

    def handle_help(self):
        self.stdout.write(
            """
Celery Worker Management Command

Usage Examples:
  python manage.py start_celery_worker --queue all
  python manage.py start_celery_worker --queue ai_high_priority --concurrency 1
  python manage.py start_celery_worker --queue ai_processing --concurrency 2
  python manage.py start_celery_worker --queue crawling --concurrency 3

Available Queues:
  - ai_high_priority: User-facing AI responses (highest priority)
  - ai_processing: General AI tasks and analysis
  - crawling: Web crawling and data collection
  - embeddings: Vector embedding generation
  - all: Start workers for all queues (recommended for development)

Options:
  --queue: Which queue to process (default: all)
  --concurrency: Number of concurrent workers (default: 2)
  --loglevel: Log level - debug, info, warning, error (default: info)
        """
        )
