import uuid
from django.db import models
from pgvector.django import VectorField
from crawler.models import CrawledPage


class KnowledgeEmbedding(models.Model):
    """Stores vector embeddings for content chunks with pgvector"""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    page = models.ForeignKey(
        CrawledPage, on_delete=models.CASCADE, related_name="embeddings"
    )

    # Content information
    chunk_text = models.TextField(help_text="The actual text chunk")
    chunk_index = models.IntegerField(help_text="Index of chunk within the page")
    chunk_tokens = models.IntegerField(help_text="Number of tokens in chunk")

    # Vector embedding (384 dimensions for all-MiniLM-L6-v2)
    embedding = VectorField(dimensions=384)

    # Metadata for enhanced retrieval
    metadata = models.JSONField(
        default=dict, help_text="Additional metadata for filtering"
    )

    # Quality metrics
    content_quality_score = models.FloatField(
        default=0.0, help_text="Quality score 0-1"
    )
    semantic_density = models.FloatField(
        default=0.0, help_text="Information density score"
    )

    # Processing information
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    processing_version = models.CharField(max_length=10, default="1.0")

    class Meta:
        indexes = [
            models.Index(fields=["page"]),
            models.Index(fields=["chunk_index"]),
            models.Index(fields=["content_quality_score"]),
            models.Index(fields=["created_at"]),
        ]
        unique_together = ["page", "chunk_index"]

    def __str__(self):
        return (
            f"Embedding {self.id} - Page: {self.page.title} - Chunk {self.chunk_index}"
        )


class QueryCache(models.Model):
    """Cache for frequently asked queries and their results"""

    query_hash = models.CharField(
        max_length=64, unique=True, help_text="SHA256 hash of query"
    )
    original_query = models.TextField()
    expanded_queries = models.JSONField(default=list)

    # Results
    relevant_chunks = models.JSONField(default=list)
    generated_response = models.TextField(null=True, blank=True)

    # Metrics
    retrieval_time_ms = models.IntegerField(default=0)
    generation_time_ms = models.IntegerField(default=0)
    user_rating = models.IntegerField(null=True, blank=True, help_text="1-5 rating")

    # Cache management
    created_at = models.DateTimeField(auto_now_add=True)
    access_count = models.IntegerField(default=1)
    last_accessed = models.DateTimeField(auto_now=True)

    class Meta:
        indexes = [
            models.Index(fields=["query_hash"]),
            models.Index(fields=["created_at"]),
            models.Index(fields=["access_count"]),
        ]
