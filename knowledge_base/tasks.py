from celery import shared_task
from celery.utils.log import get_task_logger
from django.core.cache import cache
from .services.embedding_generator import EmbeddingGenerationService
from crawler.models import CrawledPage

logger = get_task_logger(__name__)


@shared_task(bind=True, max_retries=3)
def generate_embeddings_for_page(self, page_id: str, force_regenerate: bool = False):
    """Generate embeddings for a single page"""

    try:
        page = CrawledPage.objects.get(id=page_id)

        # Update task status
        cache.set(
            f"embedding_task:{self.request.id}",
            {
                "status": "processing",
                "progress": 0,
                "message": f"Processing page: {page.title}",
                "page_id": page_id,
            },
            timeout=300,
        )

        # Generate embeddings
        embedding_service = EmbeddingGenerationService()
        result = embedding_service.process_page(page, force_regenerate)

        # Update final status
        cache.set(
            f"embedding_task:{self.request.id}",
            {
                "status": "completed",
                "progress": 100,
                "message": "Embeddings generated successfully",
                "result": result,
            },
            timeout=300,
        )

        logger.info(f"Embeddings generated for page {page_id}: {result}")
        return result

    except CrawledPage.DoesNotExist:
        error_msg = f"Page {page_id} not found"
        logger.error(error_msg)
        return {"status": "error", "error": error_msg}
    except Exception as exc:
        logger.error(f"Error generating embeddings for page {page_id}: {exc}")
        raise self.retry(countdown=60, exc=exc)


@shared_task(bind=True)
def batch_generate_embeddings(
    self, page_ids: list = None, force_regenerate: bool = False
):
    """Generate embeddings for multiple pages"""

    try:
        embedding_service = EmbeddingGenerationService()

        # Update initial status
        cache.set(
            f"task_status:{self.request.id}",
            {
                "status": "processing",
                "progress": 0,
                "message": "Starting batch embedding generation",
            },
            timeout=3600,
        )

        # Process pages with progress updates
        result = embedding_service.batch_process_pages(
            page_ids, force_regenerate, task_id=self.request.id
        )

        # Update final status
        cache.set(
            f"task_status:{self.request.id}",
            {
                "status": "completed",
                "progress": 100,
                "message": "Batch processing completed",
                "result": result,
            },
            timeout=3600,
        )

        logger.info(f"Batch embedding generation completed: {result}")
        return result

    except Exception as exc:
        logger.error(f"Batch embedding generation failed: {exc}")
        cache.set(
            f"task_status:{self.request.id}",
            {"status": "failed", "progress": 0, "message": f"Error: {str(exc)}"},
            timeout=3600,
        )
        raise exc
