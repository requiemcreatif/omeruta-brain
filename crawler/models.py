from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
import json
import uuid


class CrawlJob(models.Model):
    """Model to track crawl jobs and their status"""

    STATUS_CHOICES = [
        ("pending", "Pending"),
        ("running", "Running"),
        ("completed", "Completed"),
        ("failed", "Failed"),
        ("cancelled", "Cancelled"),
    ]

    STRATEGY_CHOICES = [
        ("single", "Single URL"),
        ("multi", "Multiple URLs"),
        ("deep_bfs", "Deep Crawl - Breadth First"),
        ("deep_dfs", "Deep Crawl - Depth First"),
        ("deep_best", "Deep Crawl - Best First"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    created_by = models.ForeignKey(User, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    # Job configuration
    strategy = models.CharField(
        max_length=20, choices=STRATEGY_CHOICES, default="single"
    )
    start_urls = models.JSONField(help_text="List of starting URLs")
    max_pages = models.PositiveIntegerField(
        default=10, help_text="Maximum pages to crawl"
    )
    max_depth = models.PositiveIntegerField(
        default=3, help_text="Maximum crawl depth for deep crawling"
    )

    # Filter patterns
    include_patterns = models.JSONField(
        null=True, blank=True, help_text="Regex patterns to include"
    )
    exclude_patterns = models.JSONField(
        null=True, blank=True, help_text="Regex patterns to exclude"
    )

    # Crawl settings
    delay_between_requests = models.FloatField(
        default=0.5, help_text="Delay between requests in seconds"
    )
    respect_robots_txt = models.BooleanField(default=True)
    use_cache = models.BooleanField(default=True)

    # Status tracking
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="pending")
    progress = models.PositiveIntegerField(
        default=0, help_text="Number of pages processed"
    )
    total_found = models.PositiveIntegerField(
        default=0, help_text="Total URLs discovered"
    )
    error_message = models.TextField(null=True, blank=True)

    # Results summary
    successful_crawls = models.PositiveIntegerField(default=0)
    failed_crawls = models.PositiveIntegerField(default=0)
    total_content_size = models.PositiveIntegerField(
        default=0, help_text="Total size of extracted content in bytes"
    )

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["status", "created_at"]),
            models.Index(fields=["created_by", "status"]),
        ]

    def __str__(self):
        return f"CrawlJob {self.id} - {self.strategy} - {self.status}"


class CrawledPage(models.Model):
    """Model to store individual crawled pages with clean markdown content"""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    crawl_job = models.ForeignKey(
        CrawlJob, on_delete=models.CASCADE, related_name="pages"
    )

    # Page identification
    url = models.URLField(max_length=2000)
    final_url = models.URLField(max_length=2000, help_text="Final URL after redirects")
    title = models.CharField(max_length=500, null=True, blank=True)

    # Crawl metadata
    crawled_at = models.DateTimeField(auto_now_add=True)
    depth = models.PositiveIntegerField(default=0, help_text="Depth from starting URL")
    status_code = models.PositiveIntegerField(null=True, blank=True)
    success = models.BooleanField(default=False)
    error_message = models.TextField(null=True, blank=True)

    # Content - optimized for vector database
    clean_markdown = models.TextField(
        null=True, blank=True, help_text="Cleaned markdown for vector DB"
    )
    word_count = models.PositiveIntegerField(default=0)
    content_hash = models.CharField(
        max_length=64, null=True, blank=True, help_text="SHA256 hash of clean content"
    )

    # Extracted metadata
    meta_description = models.TextField(null=True, blank=True)
    meta_keywords = models.TextField(null=True, blank=True)
    author = models.CharField(max_length=200, null=True, blank=True)
    publish_date = models.DateTimeField(null=True, blank=True)
    language = models.CharField(max_length=10, null=True, blank=True)

    # Links and references
    internal_links = models.JSONField(
        default=list, help_text="List of internal links found"
    )
    external_links = models.JSONField(
        default=list, help_text="List of external links found"
    )

    # Technical metadata
    content_type = models.CharField(max_length=100, null=True, blank=True)
    content_length = models.PositiveIntegerField(null=True, blank=True)
    response_time = models.FloatField(
        null=True, blank=True, help_text="Response time in seconds"
    )

    # Vector database preparation
    is_processed_for_embeddings = models.BooleanField(default=False)
    embedding_chunks = models.JSONField(
        default=list, help_text="Text chunks for embedding"
    )

    class Meta:
        ordering = ["-crawled_at"]
        unique_together = ["crawl_job", "url"]
        indexes = [
            models.Index(fields=["crawl_job", "success"]),
            models.Index(fields=["url"]),
            models.Index(fields=["content_hash"]),
            models.Index(fields=["is_processed_for_embeddings"]),
            models.Index(fields=["crawled_at"]),
        ]

    def __str__(self):
        return f"Page: {self.url} - {self.title or 'No Title'}"

    def get_content_for_vector_db(self):
        """Return the best content for vector database indexing"""
        return self.clean_markdown or ""

    def update_content_hash(self):
        """Update the content hash based on clean markdown"""
        import hashlib

        content = self.get_content_for_vector_db()
        if content:
            self.content_hash = hashlib.sha256(content.encode()).hexdigest()

    def save(self, *args, **kwargs):
        # Update content hash before saving
        self.update_content_hash()

        # Update word count
        content = self.get_content_for_vector_db()
        if content:
            self.word_count = len(content.split())

        super().save(*args, **kwargs)


class CrawlStatistics(models.Model):
    """Model to store crawl statistics and analytics"""

    crawl_job = models.OneToOneField(
        CrawlJob, on_delete=models.CASCADE, related_name="statistics"
    )

    # Content statistics
    total_words = models.PositiveIntegerField(default=0)
    total_characters = models.PositiveIntegerField(default=0)
    average_words_per_page = models.FloatField(default=0.0)

    # Link statistics
    total_internal_links = models.PositiveIntegerField(default=0)
    total_external_links = models.PositiveIntegerField(default=0)
    unique_domains_found = models.PositiveIntegerField(default=0)

    # Performance statistics
    total_crawl_time = models.FloatField(
        default=0.0, help_text="Total crawl time in seconds"
    )
    average_response_time = models.FloatField(
        default=0.0, help_text="Average response time in seconds"
    )
    pages_per_minute = models.FloatField(default=0.0)

    # Quality metrics
    success_rate = models.FloatField(
        default=0.0, help_text="Percentage of successful crawls"
    )
    content_quality_score = models.FloatField(
        default=0.0, help_text="Quality score based on content analysis"
    )

    # Error analysis
    common_errors = models.JSONField(
        default=dict, help_text="Common error types and counts"
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def calculate_statistics(self):
        """Calculate and update statistics based on crawled pages"""
        pages = self.crawl_job.pages.filter(success=True)

        if pages.exists():
            # Content statistics
            self.total_words = sum(page.word_count for page in pages)
            self.total_characters = sum(
                len(page.get_content_for_vector_db()) for page in pages
            )
            self.average_words_per_page = self.total_words / pages.count()

            # Link statistics
            self.total_internal_links = sum(len(page.internal_links) for page in pages)
            self.total_external_links = sum(len(page.external_links) for page in pages)

            # Performance statistics
            response_times = [
                page.response_time for page in pages if page.response_time
            ]
            if response_times:
                self.average_response_time = sum(response_times) / len(response_times)

            # Success rate
            total_pages = self.crawl_job.pages.count()
            if total_pages > 0:
                self.success_rate = (pages.count() / total_pages) * 100

        self.save()

    def __str__(self):
        return f"Statistics for {self.crawl_job}"
