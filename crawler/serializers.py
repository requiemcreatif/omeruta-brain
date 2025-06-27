from rest_framework import serializers
from django.contrib.auth.models import User
from .models import CrawlJob, CrawledPage, CrawlStatistics


class CrawlJobCreateSerializer(serializers.ModelSerializer):
    """Serializer for creating new crawl jobs"""

    class Meta:
        model = CrawlJob
        fields = [
            "strategy",
            "start_urls",
            "max_pages",
            "max_depth",
            "include_patterns",
            "exclude_patterns",
            "delay_between_requests",
            "respect_robots_txt",
            "use_cache",
        ]

    def validate_start_urls(self, value):
        """Validate that start_urls is a non-empty list"""
        if not value or not isinstance(value, list):
            raise serializers.ValidationError("start_urls must be a non-empty list")

        if len(value) == 0:
            raise serializers.ValidationError("At least one URL is required")

        # Basic URL validation
        for url in value:
            if not isinstance(url, str) or not url.strip():
                raise serializers.ValidationError("All URLs must be non-empty strings")

            if not (url.startswith("http://") or url.startswith("https://")):
                raise serializers.ValidationError(f"Invalid URL format: {url}")

        return value

    def validate_max_pages(self, value):
        """Validate max_pages is within reasonable limits"""
        if value < 1:
            raise serializers.ValidationError("max_pages must be at least 1")

        if value > 1000:  # Reasonable limit to prevent abuse
            raise serializers.ValidationError("max_pages cannot exceed 1000")

        return value

    def validate_max_depth(self, value):
        """Validate max_depth is within reasonable limits"""
        if value < 1:
            raise serializers.ValidationError("max_depth must be at least 1")

        if value > 10:  # Reasonable limit to prevent infinite crawling
            raise serializers.ValidationError("max_depth cannot exceed 10")

        return value

    def validate_delay_between_requests(self, value):
        """Validate delay is reasonable"""
        if value < 0:
            raise serializers.ValidationError(
                "delay_between_requests cannot be negative"
            )

        if value > 60:  # Max 1 minute delay
            raise serializers.ValidationError(
                "delay_between_requests cannot exceed 60 seconds"
            )

        return value


class CrawlJobSerializer(serializers.ModelSerializer):
    """Serializer for crawl job responses"""

    created_by = serializers.CharField(source="created_by.username", read_only=True)

    class Meta:
        model = CrawlJob
        fields = [
            "id",
            "created_by",
            "created_at",
            "updated_at",
            "strategy",
            "start_urls",
            "max_pages",
            "max_depth",
            "include_patterns",
            "exclude_patterns",
            "delay_between_requests",
            "respect_robots_txt",
            "use_cache",
            "status",
            "progress",
            "total_found",
            "error_message",
            "successful_crawls",
            "failed_crawls",
            "total_content_size",
        ]
        read_only_fields = [
            "id",
            "created_by",
            "created_at",
            "updated_at",
            "status",
            "progress",
            "total_found",
            "error_message",
            "successful_crawls",
            "failed_crawls",
            "total_content_size",
        ]


class CrawledPageSerializer(serializers.ModelSerializer):
    """Serializer for crawled pages"""

    crawl_job_id = serializers.UUIDField(source="crawl_job.id", read_only=True)
    content_preview = serializers.SerializerMethodField()

    class Meta:
        model = CrawledPage
        fields = [
            "id",
            "crawl_job_id",
            "url",
            "final_url",
            "title",
            "crawled_at",
            "depth",
            "status_code",
            "success",
            "error_message",
            "word_count",
            "content_hash",
            "meta_description",
            "meta_keywords",
            "author",
            "publish_date",
            "language",
            "content_type",
            "content_length",
            "response_time",
            "content_preview",
            "is_processed_for_embeddings",
        ]
        read_only_fields = ["id", "crawl_job_id", "crawled_at"]

    def get_content_preview(self, obj):
        """Return a preview of the content (first 200 characters)"""
        content = obj.get_content_for_vector_db()
        if content:
            return content[:200] + "..." if len(content) > 200 else content
        return None


class CrawledPageDetailSerializer(serializers.ModelSerializer):
    """Detailed serializer for individual crawled pages including full content"""

    crawl_job_id = serializers.UUIDField(source="crawl_job.id", read_only=True)

    class Meta:
        model = CrawledPage
        fields = [
            "id",
            "crawl_job_id",
            "url",
            "final_url",
            "title",
            "crawled_at",
            "depth",
            "status_code",
            "success",
            "error_message",
            "clean_markdown",
            "word_count",
            "content_hash",
            "meta_description",
            "meta_keywords",
            "author",
            "publish_date",
            "language",
            "content_type",
            "content_length",
            "response_time",
            "internal_links",
            "external_links",
            "embedding_chunks",
            "is_processed_for_embeddings",
        ]
        read_only_fields = ["id", "crawl_job_id", "crawled_at"]


class CrawlStatisticsSerializer(serializers.ModelSerializer):
    """Serializer for crawl statistics"""

    crawl_job_id = serializers.UUIDField(source="crawl_job.id", read_only=True)

    class Meta:
        model = CrawlStatistics
        fields = [
            "crawl_job_id",
            "total_words",
            "total_characters",
            "average_words_per_page",
            "total_internal_links",
            "total_external_links",
            "unique_domains_found",
            "total_crawl_time",
            "average_response_time",
            "pages_per_minute",
            "success_rate",
            "content_quality_score",
            "common_errors",
            "created_at",
            "updated_at",
        ]
        read_only_fields = ["crawl_job_id", "created_at", "updated_at"]


class CrawlJobSummarySerializer(serializers.ModelSerializer):
    """Minimal serializer for crawl job listings"""

    created_by = serializers.CharField(source="created_by.username", read_only=True)

    class Meta:
        model = CrawlJob
        fields = [
            "id",
            "created_by",
            "created_at",
            "strategy",
            "status",
            "progress",
            "successful_crawls",
            "failed_crawls",
            "total_content_size",
        ]
        read_only_fields = ["id", "created_by", "created_at"]


class ContentExtractionSerializer(serializers.Serializer):
    """Serializer for content extraction requests"""

    url = serializers.URLField(required=True)
    use_content_filter = serializers.BooleanField(default=True)
    filter_type = serializers.ChoiceField(
        choices=["pruning", "bm25"], default="pruning"
    )
    user_query = serializers.CharField(
        required=False, allow_blank=True, help_text="Required when using BM25 filter"
    )
    cache_mode = serializers.ChoiceField(
        choices=["enabled", "disabled", "bypass"], default="enabled"
    )

    def validate(self, data):
        """Validate that user_query is provided when using BM25 filter"""
        if data.get("filter_type") == "bm25" and not data.get("user_query"):
            raise serializers.ValidationError(
                "user_query is required when using BM25 filter"
            )
        return data


class ContentExtractionResponseSerializer(serializers.Serializer):
    """Serializer for content extraction responses"""

    url = serializers.URLField()
    final_url = serializers.URLField()
    success = serializers.BooleanField()
    error_message = serializers.CharField(required=False, allow_null=True)
    title = serializers.CharField(required=False, allow_null=True)
    clean_markdown = serializers.CharField(required=False, allow_null=True)
    word_count = serializers.IntegerField()
    content_length = serializers.IntegerField(required=False, allow_null=True)
    response_time = serializers.FloatField(required=False, allow_null=True)
    metadata = serializers.DictField(required=False)
    links = serializers.DictField(required=False)


class BulkContentExtractionSerializer(serializers.Serializer):
    """Serializer for bulk content extraction requests"""

    urls = serializers.ListField(
        child=serializers.URLField(),
        min_length=1,
        max_length=50,  # Reasonable limit for bulk operations
        help_text="List of URLs to extract content from",
    )
    use_content_filter = serializers.BooleanField(default=True)
    filter_type = serializers.ChoiceField(
        choices=["pruning", "bm25"], default="pruning"
    )
    user_query = serializers.CharField(
        required=False, allow_blank=True, help_text="Required when using BM25 filter"
    )
    cache_mode = serializers.ChoiceField(
        choices=["enabled", "disabled", "bypass"], default="enabled"
    )
    max_concurrent = serializers.IntegerField(
        default=5,
        min_value=1,
        max_value=10,
        help_text="Maximum number of concurrent requests",
    )

    def validate(self, data):
        """Validate that user_query is provided when using BM25 filter"""
        if data.get("filter_type") == "bm25" and not data.get("user_query"):
            raise serializers.ValidationError(
                "user_query is required when using BM25 filter"
            )
        return data


class BulkContentExtractionResponseSerializer(serializers.Serializer):
    """Serializer for bulk content extraction responses"""

    results = ContentExtractionResponseSerializer(many=True)
    summary = serializers.DictField()
