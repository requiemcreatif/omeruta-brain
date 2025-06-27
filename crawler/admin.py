from django.contrib import admin
from django.utils.html import format_html
from django.urls import reverse
from django.utils.safestring import mark_safe
from .models import CrawlJob, CrawledPage, CrawlStatistics


@admin.register(CrawlJob)
class CrawlJobAdmin(admin.ModelAdmin):
    """Admin interface for CrawlJob model"""

    list_display = [
        "id",
        "created_by",
        "strategy",
        "status",
        "progress_display",
        "successful_crawls",
        "failed_crawls",
        "created_at",
    ]
    list_filter = [
        "status",
        "strategy",
        "created_at",
        "respect_robots_txt",
        "use_cache",
    ]
    search_fields = ["id", "created_by__username", "start_urls"]
    readonly_fields = [
        "id",
        "created_at",
        "updated_at",
        "progress",
        "total_found",
        "successful_crawls",
        "failed_crawls",
        "total_content_size",
    ]

    fieldsets = (
        (
            "Basic Information",
            {"fields": ("id", "created_by", "created_at", "updated_at", "status")},
        ),
        (
            "Crawl Configuration",
            {
                "fields": (
                    "strategy",
                    "start_urls",
                    "max_pages",
                    "max_depth",
                    "include_patterns",
                    "exclude_patterns",
                )
            },
        ),
        (
            "Settings",
            {"fields": ("delay_between_requests", "respect_robots_txt", "use_cache")},
        ),
        (
            "Progress & Results",
            {
                "fields": (
                    "progress",
                    "total_found",
                    "successful_crawls",
                    "failed_crawls",
                    "total_content_size",
                    "error_message",
                )
            },
        ),
    )

    def progress_display(self, obj):
        """Display progress as a percentage"""
        if obj.total_found > 0:
            percentage = (obj.progress / obj.total_found) * 100
            return format_html(
                '<div style="width: 100px; background-color: #f0f0f0;">'
                '<div style="width: {}%; background-color: #4CAF50; height: 20px; text-align: center;">'
                "{}%</div></div>",
                percentage,
                round(percentage, 1),
            )
        return f"{obj.progress} pages"

    progress_display.short_description = "Progress"

    def get_queryset(self, request):
        """Optimize queryset with select_related"""
        return super().get_queryset(request).select_related("created_by")


@admin.register(CrawledPage)
class CrawledPageAdmin(admin.ModelAdmin):
    """Admin interface for CrawledPage model"""

    list_display = [
        "title_display",
        "url_display",
        "crawl_job_id",
        "success",
        "word_count",
        "depth",
        "crawled_at",
    ]
    list_filter = [
        "success",
        "crawled_at",
        "depth",
        "is_processed_for_embeddings",
        "crawl_job__strategy",
        "language",
    ]
    search_fields = ["title", "url", "final_url", "content_hash"]
    readonly_fields = [
        "id",
        "crawled_at",
        "content_hash",
        "word_count",
        "content_length",
    ]

    fieldsets = (
        (
            "Basic Information",
            {"fields": ("id", "crawl_job", "url", "final_url", "title")},
        ),
        (
            "Crawl Results",
            {
                "fields": (
                    "crawled_at",
                    "depth",
                    "status_code",
                    "success",
                    "error_message",
                    "response_time",
                )
            },
        ),
        (
            "Content",
            {
                "fields": (
                    "word_count",
                    "content_length",
                    "content_hash",
                    "clean_markdown",
                )
            },
        ),
        (
            "Metadata",
            {
                "fields": (
                    "meta_description",
                    "meta_keywords",
                    "author",
                    "publish_date",
                    "language",
                    "content_type",
                )
            },
        ),
        ("Links", {"fields": ("internal_links", "external_links")}),
        (
            "Vector Processing",
            {"fields": ("is_processed_for_embeddings", "embedding_chunks")},
        ),
    )

    def title_display(self, obj):
        """Display truncated title"""
        return (
            obj.title[:50] + "..."
            if obj.title and len(obj.title) > 50
            else obj.title or "No Title"
        )

    title_display.short_description = "Title"

    def url_display(self, obj):
        """Display clickable URL"""
        if obj.url:
            return format_html(
                '<a href="{}" target="_blank">{}</a>',
                obj.url,
                obj.url[:50] + "..." if len(obj.url) > 50 else obj.url,
            )
        return "No URL"

    url_display.short_description = "URL"

    def crawl_job_id(self, obj):
        """Display link to crawl job"""
        if obj.crawl_job:
            url = reverse("admin:crawler_crawljob_change", args=[obj.crawl_job.id])
            return format_html('<a href="{}">{}</a>', url, str(obj.crawl_job.id)[:8])
        return "No Job"

    crawl_job_id.short_description = "Job ID"

    def get_queryset(self, request):
        """Optimize queryset with select_related"""
        return super().get_queryset(request).select_related("crawl_job")


@admin.register(CrawlStatistics)
class CrawlStatisticsAdmin(admin.ModelAdmin):
    """Admin interface for CrawlStatistics model"""

    list_display = [
        "crawl_job_id",
        "total_words",
        "success_rate",
        "pages_per_minute",
        "total_crawl_time",
        "created_at",
    ]
    list_filter = ["created_at", "updated_at"]
    search_fields = ["crawl_job__id"]
    readonly_fields = ["crawl_job", "created_at", "updated_at"]

    fieldsets = (
        ("Job Information", {"fields": ("crawl_job", "created_at", "updated_at")}),
        (
            "Content Statistics",
            {"fields": ("total_words", "total_characters", "average_words_per_page")},
        ),
        (
            "Link Statistics",
            {
                "fields": (
                    "total_internal_links",
                    "total_external_links",
                    "unique_domains_found",
                )
            },
        ),
        (
            "Performance Statistics",
            {
                "fields": (
                    "total_crawl_time",
                    "average_response_time",
                    "pages_per_minute",
                )
            },
        ),
        (
            "Quality Metrics",
            {"fields": ("success_rate", "content_quality_score", "common_errors")},
        ),
    )

    def crawl_job_id(self, obj):
        """Display link to crawl job"""
        if obj.crawl_job:
            url = reverse("admin:crawler_crawljob_change", args=[obj.crawl_job.id])
            return format_html('<a href="{}">{}</a>', url, str(obj.crawl_job.id)[:8])
        return "No Job"

    crawl_job_id.short_description = "Job ID"

    def get_queryset(self, request):
        """Optimize queryset with select_related"""
        return super().get_queryset(request).select_related("crawl_job")


# Custom admin site configuration
admin.site.site_header = "Omeruta Brain Crawler Administration"
admin.site.site_title = "Crawler Admin"
admin.site.index_title = "Welcome to Omeruta Brain Crawler Administration"
