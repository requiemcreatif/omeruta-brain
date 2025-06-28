import logging
from typing import List, Dict, Optional, Any
from django.db import transaction
from django.conf import settings
from crawler.models import CrawledPage
from ..models import KnowledgeEmbedding
from .smart_chunker import SmartSemanticChunker
from .pgvector_search import PgVectorSearchService

logger = logging.getLogger(__name__)


class EmbeddingGenerationService:
    """Service for generating and managing embeddings"""

    def __init__(self):
        self.chunker = SmartSemanticChunker()
        self.search_service = PgVectorSearchService()

    def process_page(
        self, page: CrawledPage, force_regenerate: bool = False
    ) -> Dict[str, Any]:
        """Process a single page and generate embeddings"""

        if not page.success or not page.clean_markdown:
            return {"status": "skipped", "reason": "no_content"}

        # Check if already processed
        if not force_regenerate and page.is_processed_for_embeddings:
            existing_count = KnowledgeEmbedding.objects.filter(page=page).count()
            if existing_count > 0:
                return {"status": "already_processed", "chunk_count": existing_count}

        try:
            with transaction.atomic():
                # Remove existing embeddings if regenerating
                if force_regenerate:
                    KnowledgeEmbedding.objects.filter(page=page).delete()

                # Generate chunks
                page_metadata = {
                    "title": page.title,
                    "url": page.url,
                    "meta_description": page.meta_description,
                    "author": page.author,
                    "language": page.language,
                }

                chunks = self.chunker.chunk_content(page.clean_markdown, page_metadata)

                if not chunks:
                    return {"status": "no_chunks", "reason": "content_too_short"}

                # Generate embeddings for chunks
                embeddings_created = 0
                for chunk in chunks:
                    if self._create_embedding(page, chunk):
                        embeddings_created += 1

                # Update page status
                page.is_processed_for_embeddings = True
                page.save(update_fields=["is_processed_for_embeddings"])

                logger.info(
                    f"✅ Processed page {page.id}: {embeddings_created} embeddings created"
                )

                return {
                    "status": "success",
                    "chunk_count": embeddings_created,
                    "total_chunks": len(chunks),
                    "page_id": str(page.id),
                }

        except Exception as e:
            logger.error(f"❌ Error processing page {page.id}: {e}")
            return {"status": "error", "error": str(e)}

    def _create_embedding(self, page: CrawledPage, chunk: Dict) -> bool:
        """Create embedding for a single chunk"""

        try:
            # Generate embedding vector
            if not self.search_service.embedding_model:
                logger.error("Embedding model not available")
                return False

            embedding_vector = self.search_service.embedding_model.encode(chunk["text"])

            # Create KnowledgeEmbedding record
            KnowledgeEmbedding.objects.create(
                page=page,
                chunk_text=chunk["text"],
                chunk_index=chunk["index"],
                chunk_tokens=chunk["tokens"],
                embedding=embedding_vector.tolist(),
                metadata=chunk["metadata"],
                content_quality_score=chunk["quality_score"],
                semantic_density=chunk["semantic_density"],
            )

            return True

        except Exception as e:
            logger.error(f"Error creating embedding for chunk {chunk['index']}: {e}")
            return False

    def batch_process_pages(
        self, page_ids: List[str] = None, force_regenerate: bool = False
    ) -> Dict[str, Any]:
        """Process multiple pages in batch"""

        # Get pages to process
        if page_ids:
            pages = CrawledPage.objects.filter(id__in=page_ids, success=True)
        else:
            # Process unprocessed pages
            pages = CrawledPage.objects.filter(
                success=True,
                is_processed_for_embeddings=False,
                clean_markdown__isnull=False,
            ).exclude(clean_markdown="")

        results = {
            "total_pages": pages.count(),
            "processed": 0,
            "skipped": 0,
            "errors": 0,
            "total_embeddings": 0,
            "details": [],
        }

        for page in pages:
            result = self.process_page(page, force_regenerate)

            if result["status"] == "success":
                results["processed"] += 1
                results["total_embeddings"] += result["chunk_count"]
            elif result["status"] in ["skipped", "already_processed", "no_chunks"]:
                results["skipped"] += 1
            else:
                results["errors"] += 1

            results["details"].append(
                {
                    "page_id": str(page.id),
                    "url": page.url,
                    "title": page.title,
                    "result": result,
                }
            )

        logger.info(
            f"Batch processing complete: {results['processed']} processed, {results['skipped']} skipped, {results['errors']} errors"
        )
        return results

    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get statistics about embeddings"""

        from django.db.models import Count, Avg, Sum

        stats = KnowledgeEmbedding.objects.aggregate(
            total_embeddings=Count("id"),
            avg_quality=Avg("content_quality_score"),
            avg_density=Avg("semantic_density"),
            total_tokens=Sum("chunk_tokens"),
        )

        # Get pages with embeddings
        pages_with_embeddings = CrawledPage.objects.filter(
            is_processed_for_embeddings=True
        ).count()

        total_pages = CrawledPage.objects.filter(success=True).count()

        stats.update(
            {
                "pages_with_embeddings": pages_with_embeddings,
                "total_crawled_pages": total_pages,
                "processing_percentage": (
                    (pages_with_embeddings / total_pages * 100)
                    if total_pages > 0
                    else 0
                ),
            }
        )

        return stats

    def cleanup_orphaned_embeddings(self) -> Dict[str, int]:
        """Clean up orphaned embeddings"""

        try:
            # Remove embeddings for deleted pages
            orphaned_count = KnowledgeEmbedding.objects.filter(
                page__isnull=True
            ).delete()[0]

            # Remove embeddings for failed pages
            failed_count = KnowledgeEmbedding.objects.filter(
                page__success=False
            ).delete()[0]

            logger.info(
                f"Cleanup completed: {orphaned_count} orphaned, {failed_count} failed embeddings removed"
            )

            return {"orphaned_removed": orphaned_count, "failed_removed": failed_count}

        except Exception as e:
            logger.error(f"Embedding cleanup failed: {e}")
            return {"orphaned_removed": 0, "failed_removed": 0, "error": str(e)}

    def update_quality_scores(self, batch_size: int = 100) -> Dict[str, int]:
        """Recalculate quality scores for existing embeddings"""

        try:
            updated_count = 0

            # Process in batches
            embeddings = KnowledgeEmbedding.objects.filter(content_quality_score=0.0)[
                :batch_size
            ]

            for embedding in embeddings:
                new_quality = self.chunker._calculate_quality_score(
                    embedding.chunk_text
                )
                new_density = self.chunker._calculate_semantic_density(
                    embedding.chunk_text
                )

                embedding.content_quality_score = new_quality
                embedding.semantic_density = new_density
                embedding.save(
                    update_fields=["content_quality_score", "semantic_density"]
                )

                updated_count += 1

            logger.info(f"Updated quality scores for {updated_count} embeddings")
            return {"updated_count": updated_count}

        except Exception as e:
            logger.error(f"Quality score update failed: {e}")
            return {"updated_count": 0, "error": str(e)}
