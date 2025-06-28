import os
from celery import Celery
from django.conf import settings

# Set the default Django settings module for the 'celery' program.
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "omeruta_brain_project.settings")

# Force CPU-only processing to avoid MPS crashes
os.environ.setdefault("FORCE_CPU_ONLY", "true")

# Additional environment variables to disable MPS/CUDA if needed
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

app = Celery("omeruta_brain_project")

# Using a string here means the worker doesn't have to serialize
# the configuration object to child processes.
app.config_from_object("django.conf:settings", namespace="CELERY")

# Enhanced configuration for AI workloads
app.conf.update(
    # Task routing for different types of work
    task_routes={
        "agents.tasks.*": {"queue": "ai_processing"},
        "agents.tasks.process_user_message_async": {"queue": "ai_high_priority"},
        "crawler.tasks.*": {"queue": "crawling"},
    },
    # Task result settings
    result_expires=3600,  # Results expire after 1 hour
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    # Concurrency settings
    worker_concurrency=4,  # Adjust based on CPU/GPU cores
    worker_prefetch_multiplier=1,  # Important for AI tasks
    # Task time limits
    task_soft_time_limit=100,  # 100 seconds soft limit
    task_time_limit=120,  # 120 seconds hard limit
    # Queue priorities
    task_default_priority=5,
    worker_hijack_root_logger=False,
    worker_log_color=False,
    worker_max_tasks_per_child=50,  # Restart workers after 50 tasks to prevent memory leaks
    worker_disable_rate_limits=True,
    # Beat schedule for periodic tasks
    beat_schedule={
        # Clean up expired tasks every hour
        "cleanup-expired-tasks": {
            "task": "agents.tasks.cleanup_expired_tasks",
            "schedule": 60.0 * 60.0,  # Every hour
        },
        # Process unprocessed embeddings every 30 minutes
        "process-embeddings": {
            "task": "agents.tasks.batch_process_unprocessed_pages",
            "schedule": 60.0 * 30.0,  # Every 30 minutes
        },
        # Check content freshness daily
        "update-freshness": {
            "task": "agents.tasks.update_content_freshness",
            "schedule": 60.0 * 60.0 * 24.0,  # Daily
        },
    },
    # Error handling
    task_annotations={
        "*": {
            "rate_limit": "10/s",
            "time_limit": 120,
            "soft_time_limit": 100,
        }
    },
    # Broker configuration
    broker_connection_retry_on_startup=True,
    broker_connection_retry=True,
    broker_connection_max_retries=10,
)

# Load task modules from all registered Django apps.
app.autodiscover_tasks()


@app.task(bind=True)
def debug_task(self):
    print(f"Request: {self.request!r}")
