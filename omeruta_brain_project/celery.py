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

# MLX-compatible worker configuration
app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    # Use thread-based workers for MLX compatibility
    worker_pool="threads",
    worker_concurrency=2,  # Adjust based on your CPU cores
    # Task routing for different queues
    task_routes={
        "agents.tasks.process_user_message_async": {"queue": "ai_high_priority"},
        "agents.tasks.process_multiagent_query": {"queue": "ai_processing"},
        "agents.tasks.process_live_research_async": {"queue": "ai_processing"},
        "knowledge_base.tasks.generate_embeddings_for_page": {"queue": "embeddings"},
        "knowledge_base.tasks.batch_generate_embeddings": {"queue": "embeddings"},
        "crawler.tasks.*": {"queue": "crawling"},
    },
    # MLX-specific settings
    worker_prefetch_multiplier=1,  # Prevent memory issues
    task_acks_late=True,
    worker_max_tasks_per_child=50,  # Restart workers periodically
)

# Load task modules from all registered Django apps.
app.autodiscover_tasks()


@app.task(bind=True)
def debug_task(self):
    print(f"Request: {self.request!r}")
