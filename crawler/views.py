from django.shortcuts import render
import asyncio
import logging
from typing import Dict, Any
from django.http import JsonResponse
from django.shortcuts import get_object_or_404
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from django.db import transaction
from django.contrib.auth.decorators import login_required

from rest_framework import status, permissions
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.generics import ListAPIView, RetrieveAPIView
from rest_framework.pagination import PageNumberPagination

from .models import CrawlJob, CrawledPage, CrawlStatistics
from .serializers import (
    CrawlJobCreateSerializer,
    CrawlJobSerializer,
    CrawlJobSummarySerializer,
    CrawledPageSerializer,
    CrawledPageDetailSerializer,
    CrawlStatisticsSerializer,
    ContentExtractionSerializer,
    ContentExtractionResponseSerializer,
    BulkContentExtractionSerializer,
    BulkContentExtractionResponseSerializer,
)
from .services import CrawlerService

logger = logging.getLogger(__name__)


class StandardResultsSetPagination(PageNumberPagination):
    """Standard pagination for list views"""

    page_size = 20
    page_size_query_param = "page_size"
    max_page_size = 100


class CrawlJobListView(ListAPIView):
    """List all crawl jobs for the authenticated user"""

    serializer_class = CrawlJobSummarySerializer
    pagination_class = StandardResultsSetPagination
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        """Filter jobs by authenticated user"""
        return CrawlJob.objects.filter(created_by=self.request.user)


class CrawlJobDetailView(RetrieveAPIView):
    """Get detailed information about a specific crawl job"""

    serializer_class = CrawlJobSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        """Filter jobs by authenticated user"""
        return CrawlJob.objects.filter(created_by=self.request.user)


class CrawlJobCreateView(APIView):
    """Create and start a new crawl job"""

    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        """Create a new crawl job"""
        serializer = CrawlJobCreateSerializer(data=request.data)

        if serializer.is_valid():
            try:
                # Create the crawl job
                crawl_job = serializer.save(created_by=request.user)

                # Start the crawl job asynchronously
                # Note: In production, this should use a task queue like Celery
                import threading

                def run_crawl_job():
                    asyncio.run(self._execute_crawl_job_async(crawl_job))

                # Run in a separate thread to avoid blocking the response
                thread = threading.Thread(target=run_crawl_job)
                thread.daemon = True
                thread.start()

                # Return the created job
                response_serializer = CrawlJobSerializer(crawl_job)
                return Response(
                    response_serializer.data, status=status.HTTP_201_CREATED
                )

            except Exception as e:
                logger.error(f"Error creating crawl job: {e}", exc_info=True)
                return Response(
                    {"error": "Failed to create crawl job"},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                )

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    async def _execute_crawl_job_async(self, crawl_job: CrawlJob):
        """Execute the crawl job asynchronously"""
        try:
            crawler_service = CrawlerService()
            await crawler_service.execute_crawl_job(crawl_job)
        except Exception as e:
            logger.error(
                f"Error executing crawl job {crawl_job.id}: {e}", exc_info=True
            )
            # Update job status to failed
            crawl_job.status = "failed"
            crawl_job.error_message = str(e)
            crawl_job.save()


class CrawlJobStopView(APIView):
    """Stop a running crawl job"""

    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, job_id):
        """Stop a running crawl job"""
        crawl_job = get_object_or_404(CrawlJob, id=job_id, created_by=request.user)

        if crawl_job.status == "running":
            crawl_job.status = "cancelled"
            crawl_job.save()

            return Response(
                {
                    "message": "Crawl job stopped successfully",
                    "job_id": str(crawl_job.id),
                    "status": crawl_job.status,
                }
            )
        else:
            return Response(
                {"error": f"Cannot stop job with status: {crawl_job.status}"},
                status=status.HTTP_400_BAD_REQUEST,
            )


class CrawledPageListView(ListAPIView):
    """List crawled pages - either for a specific job or all pages for the user"""

    serializer_class = CrawledPageSerializer
    pagination_class = StandardResultsSetPagination
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        """Get pages - filtered by job if job_id provided, otherwise all user's pages"""
        job_id = self.kwargs.get("job_id")

        if job_id:
            # Get pages for specific job
            crawl_job = get_object_or_404(
                CrawlJob, id=job_id, created_by=self.request.user
            )
            queryset = CrawledPage.objects.filter(crawl_job=crawl_job)
        else:
            # Get all pages for user's jobs
            queryset = CrawledPage.objects.filter(
                crawl_job__created_by=self.request.user
            )

        # Optional filtering
        success_filter = self.request.query_params.get("success")
        if success_filter is not None:
            success_bool = success_filter.lower() in ["true", "1", "yes"]
            queryset = queryset.filter(success=success_bool)

        return queryset.order_by("-crawled_at")


class CrawledPageDetailView(RetrieveAPIView):
    """Get detailed information about a specific crawled page"""

    serializer_class = CrawledPageDetailSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        """Filter pages by job owner"""
        return CrawledPage.objects.filter(crawl_job__created_by=self.request.user)


class CrawlStatisticsView(RetrieveAPIView):
    """Get statistics for a specific crawl job"""

    serializer_class = CrawlStatisticsSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_object(self):
        """Get statistics for the specified job"""
        job_id = self.kwargs["job_id"]
        crawl_job = get_object_or_404(CrawlJob, id=job_id, created_by=self.request.user)

        # Get or create statistics
        stats, created = CrawlStatistics.objects.get_or_create(crawl_job=crawl_job)

        # Recalculate if needed
        if created or stats.updated_at < crawl_job.updated_at:
            stats.calculate_statistics()

        return stats


class ContentExtractionView(APIView):
    """Extract content from a single URL without creating a crawl job"""

    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        """Extract content from a single URL"""
        serializer = ContentExtractionSerializer(data=request.data)

        if serializer.is_valid():
            try:
                # Extract the content
                result = asyncio.run(
                    self._extract_content_async(serializer.validated_data)
                )

                response_serializer = ContentExtractionResponseSerializer(result)
                return Response(response_serializer.data)

            except Exception as e:
                logger.error(f"Error extracting content: {e}", exc_info=True)
                return Response(
                    {"error": "Failed to extract content"},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                )

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    async def _extract_content_async(
        self, validated_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract content asynchronously"""
        crawler_service = CrawlerService()

        try:
            await crawler_service.start_crawler()

            # Create a temporary crawl job for extraction
            temp_job = CrawlJob(
                created_by_id=1,  # Temporary
                start_urls=[validated_data["url"]],
                strategy="single",
            )

            # Configure extraction options
            config_options = {
                "use_content_filter": validated_data.get("use_content_filter", True),
                "filter_type": validated_data.get("filter_type", "pruning"),
                "user_query": validated_data.get("user_query"),
                "cache_mode": validated_data.get("cache_mode", "enabled"),
            }

            # Extract content
            page = await crawler_service.crawl_single_url(
                validated_data["url"], temp_job, config_options
            )

            # Format response
            result = {
                "url": page.url,
                "final_url": page.final_url,
                "success": page.success,
                "error_message": page.error_message,
                "title": page.title,
                "clean_markdown": page.clean_markdown,
                "word_count": page.word_count,
                "content_length": page.content_length,
                "response_time": page.response_time,
                "metadata": {
                    "meta_description": page.meta_description,
                    "meta_keywords": page.meta_keywords,
                    "author": page.author,
                    "language": page.language,
                    "content_type": page.content_type,
                },
                "links": {
                    "internal": page.internal_links,
                    "external": page.external_links,
                },
            }

            return result

        finally:
            await crawler_service.stop_crawler()


class BulkContentExtractionView(APIView):
    """Extract content from multiple URLs without creating a crawl job"""

    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        """Extract content from multiple URLs"""
        serializer = BulkContentExtractionSerializer(data=request.data)

        if serializer.is_valid():
            try:
                # Extract the content
                results = asyncio.run(
                    self._extract_bulk_content_async(serializer.validated_data)
                )

                response_serializer = BulkContentExtractionResponseSerializer(results)
                return Response(response_serializer.data)

            except Exception as e:
                logger.error(f"Error extracting bulk content: {e}", exc_info=True)
                return Response(
                    {"error": "Failed to extract content"},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                )

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    async def _extract_bulk_content_async(
        self, validated_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract content from multiple URLs asynchronously"""
        crawler_service = CrawlerService()

        try:
            await crawler_service.start_crawler()

            # Create a temporary crawl job for extraction
            temp_job = CrawlJob(
                created_by_id=1,  # Temporary
                start_urls=validated_data["urls"],
                strategy="multi",
            )

            # Configure extraction options
            config_options = {
                "use_content_filter": validated_data.get("use_content_filter", True),
                "filter_type": validated_data.get("filter_type", "pruning"),
                "user_query": validated_data.get("user_query"),
                "cache_mode": validated_data.get("cache_mode", "enabled"),
            }

            # Extract content from all URLs
            pages = await crawler_service.crawl_multiple_urls(
                validated_data["urls"],
                temp_job,
                config_options,
                max_concurrent=validated_data.get("max_concurrent", 5),
            )

            # Format responses
            results = []
            for page in pages:
                result = {
                    "url": page.url,
                    "final_url": page.final_url,
                    "success": page.success,
                    "error_message": page.error_message,
                    "title": page.title,
                    "clean_markdown": page.clean_markdown,
                    "word_count": page.word_count,
                    "content_length": page.content_length,
                    "response_time": page.response_time,
                    "metadata": {
                        "meta_description": page.meta_description,
                        "meta_keywords": page.meta_keywords,
                        "author": page.author,
                        "language": page.language,
                        "content_type": page.content_type,
                    },
                    "links": {
                        "internal": page.internal_links,
                        "external": page.external_links,
                    },
                }
                results.append(result)

            # Calculate summary
            successful = [r for r in results if r["success"]]
            failed = [r for r in results if not r["success"]]

            summary = {
                "total_urls": len(validated_data["urls"]),
                "successful": len(successful),
                "failed": len(failed),
                "total_words": sum(r["word_count"] for r in successful),
                "average_response_time": (
                    sum(r["response_time"] or 0 for r in successful) / len(successful)
                    if successful
                    else 0
                ),
            }

            return {"results": results, "summary": summary}

        finally:
            await crawler_service.stop_crawler()


@api_view(["GET"])
@permission_classes([permissions.IsAuthenticated])
def crawler_health_check(request):
    """Health check endpoint for the crawler service"""
    try:
        # Try to import Crawl4AI
        from crawl4ai import AsyncWebCrawler

        health_data = {
            "status": "healthy",
            "crawl4ai_available": True,
            "message": "Crawler service is operational",
        }

        return Response(health_data)

    except ImportError:
        health_data = {
            "status": "unhealthy",
            "crawl4ai_available": False,
            "message": "Crawl4AI is not installed",
        }

        return Response(health_data, status=status.HTTP_503_SERVICE_UNAVAILABLE)


@api_view(["GET"])
@permission_classes([permissions.IsAuthenticated])
def crawler_stats_summary(request):
    """Get overall crawler statistics for the user"""
    user = request.user

    # Get user's crawl jobs
    jobs = CrawlJob.objects.filter(created_by=user)

    # Calculate summary statistics
    total_jobs = jobs.count()
    completed_jobs = jobs.filter(status="completed").count()
    running_jobs = jobs.filter(status="running").count()
    failed_jobs = jobs.filter(status="failed").count()

    # Get page statistics
    pages = CrawledPage.objects.filter(crawl_job__created_by=user)
    total_pages = pages.count()
    successful_pages = pages.filter(success=True).count()
    total_words = sum(page.word_count for page in pages.filter(success=True))

    stats = {
        "jobs": {
            "total": total_jobs,
            "completed": completed_jobs,
            "running": running_jobs,
            "failed": failed_jobs,
            "success_rate": (
                (completed_jobs / total_jobs * 100) if total_jobs > 0 else 0
            ),
        },
        "pages": {
            "total": total_pages,
            "successful": successful_pages,
            "failed": total_pages - successful_pages,
            "success_rate": (
                (successful_pages / total_pages * 100) if total_pages > 0 else 0
            ),
        },
        "content": {
            "total_words": total_words,
            "average_words_per_page": (
                total_words / successful_pages if successful_pages > 0 else 0
            ),
        },
    }

    return Response(stats)


@api_view(["GET"])
@permission_classes([permissions.IsAuthenticated])
def pages_stats_summary(request):
    """Get detailed statistics about crawled pages for the user"""
    user = request.user

    # Get user's pages
    pages = CrawledPage.objects.filter(crawl_job__created_by=user)
    successful_pages = pages.filter(success=True)
    failed_pages = pages.filter(success=False)

    # Content statistics
    total_words = sum(page.word_count for page in successful_pages)
    total_chars = sum(page.content_length or 0 for page in successful_pages)

    # Response time statistics
    response_times = [
        page.response_time for page in successful_pages if page.response_time
    ]
    avg_response_time = (
        sum(response_times) / len(response_times) if response_times else 0
    )

    # Domain analysis
    from urllib.parse import urlparse

    domains = {}
    for page in successful_pages:
        try:
            domain = urlparse(page.url).netloc
            domains[domain] = domains.get(domain, 0) + 1
        except:
            continue

    # Content type analysis
    content_types = {}
    for page in successful_pages:
        if page.content_type:
            content_types[page.content_type] = (
                content_types.get(page.content_type, 0) + 1
            )

    # Language analysis
    languages = {}
    for page in successful_pages:
        if page.language:
            languages[page.language] = languages.get(page.language, 0) + 1

    # Recent pages
    recent_pages = pages.order_by("-crawled_at")[:10]

    stats = {
        "overview": {
            "total_pages": pages.count(),
            "successful_pages": successful_pages.count(),
            "failed_pages": failed_pages.count(),
            "success_rate": (
                round((successful_pages.count() / pages.count() * 100), 2)
                if pages.count() > 0
                else 0
            ),
        },
        "content_stats": {
            "total_words": total_words,
            "total_characters": total_chars,
            "average_words_per_page": (
                round(total_words / successful_pages.count(), 2)
                if successful_pages.count() > 0
                else 0
            ),
            "average_chars_per_page": (
                round(total_chars / successful_pages.count(), 2)
                if successful_pages.count() > 0
                else 0
            ),
        },
        "performance_stats": {
            "average_response_time": round(avg_response_time, 3),
            "total_crawl_time": round(sum(response_times), 3),
            "fastest_page": round(min(response_times), 3) if response_times else 0,
            "slowest_page": round(max(response_times), 3) if response_times else 0,
        },
        "domain_distribution": dict(
            sorted(domains.items(), key=lambda x: x[1], reverse=True)[:10]
        ),
        "content_type_distribution": dict(
            sorted(content_types.items(), key=lambda x: x[1], reverse=True)[:10]
        ),
        "language_distribution": dict(
            sorted(languages.items(), key=lambda x: x[1], reverse=True)[:10]
        ),
        "recent_pages": [
            {
                "id": str(page.id),
                "url": page.url,
                "title": page.title,
                "success": page.success,
                "word_count": page.word_count,
                "crawled_at": page.crawled_at.isoformat(),
                "response_time": page.response_time,
            }
            for page in recent_pages
        ],
    }

    return Response(stats)
