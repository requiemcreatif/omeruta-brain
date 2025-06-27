from django.urls import path, include
from rest_framework.routers import DefaultRouter

from .views import (
    CrawlJobListView,
    CrawlJobDetailView,
    CrawlJobCreateView,
    CrawlJobStopView,
    CrawledPageListView,
    CrawledPageDetailView,
    CrawlStatisticsView,
    ContentExtractionView,
    BulkContentExtractionView,
    crawler_health_check,
    crawler_stats_summary,
    pages_stats_summary,
)

app_name = "crawler"

urlpatterns = [
    # Health check and stats
    path("health/", crawler_health_check, name="health_check"),
    path("stats/", crawler_stats_summary, name="stats_summary"),
    path("pages/stats/", pages_stats_summary, name="pages_stats_summary"),
    # Crawl job management
    path("jobs/", CrawlJobListView.as_view(), name="job_list"),
    path("jobs/create/", CrawlJobCreateView.as_view(), name="job_create"),
    path("jobs/<uuid:pk>/", CrawlJobDetailView.as_view(), name="job_detail"),
    path("jobs/<uuid:job_id>/stop/", CrawlJobStopView.as_view(), name="job_stop"),
    path(
        "jobs/<uuid:job_id>/statistics/",
        CrawlStatisticsView.as_view(),
        name="job_statistics",
    ),
    # Crawled pages
    path(
        "pages/", CrawledPageListView.as_view(), name="all_pages"
    ),  # All pages for user
    path("jobs/<uuid:job_id>/pages/", CrawledPageListView.as_view(), name="job_pages"),
    path("pages/<uuid:pk>/", CrawledPageDetailView.as_view(), name="page_detail"),
    # Content extraction (without persistent jobs)
    path("extract/", ContentExtractionView.as_view(), name="extract_content"),
    path(
        "extract/bulk/",
        BulkContentExtractionView.as_view(),
        name="extract_bulk_content",
    ),
]
