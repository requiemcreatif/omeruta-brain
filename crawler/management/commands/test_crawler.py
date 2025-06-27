import asyncio
from django.core.management.base import BaseCommand, CommandError
from django.contrib.auth.models import User
from crawler.models import CrawlJob
from crawler.services import CrawlerService


class Command(BaseCommand):
    """Management command to test the crawler functionality"""

    help = "Test the crawler with a sample URL"

    def add_arguments(self, parser):
        parser.add_argument("url", type=str, help="URL to crawl for testing")
        parser.add_argument(
            "--user",
            type=str,
            default="admin",
            help="Username to associate with the crawl job (default: admin)",
        )
        parser.add_argument(
            "--strategy",
            type=str,
            choices=["single", "deep_bfs", "deep_dfs"],
            default="single",
            help="Crawl strategy to use (default: single)",
        )
        parser.add_argument(
            "--max-pages",
            type=int,
            default=5,
            help="Maximum pages to crawl for deep strategies (default: 5)",
        )
        parser.add_argument(
            "--max-depth",
            type=int,
            default=2,
            help="Maximum depth for deep crawling (default: 2)",
        )

    def handle(self, *args, **options):
        url = options["url"]
        username = options["user"]
        strategy = options["strategy"]
        max_pages = options["max_pages"]
        max_depth = options["max_depth"]

        self.stdout.write(f"Testing crawler with URL: {url}")
        self.stdout.write(f"Strategy: {strategy}")

        try:
            # Get or create user
            user, created = User.objects.get_or_create(
                username=username,
                defaults={
                    "email": f"{username}@example.com",
                    "is_staff": True,
                    "is_superuser": True,
                },
            )

            if created:
                user.set_password("testpassword123")
                user.save()
                self.stdout.write(f"Created user: {username}")
            else:
                self.stdout.write(f"Using existing user: {username}")

            # Create crawl job
            crawl_job = CrawlJob.objects.create(
                created_by=user,
                strategy=strategy,
                start_urls=[url],
                max_pages=max_pages,
                max_depth=max_depth,
                delay_between_requests=0.5,
                respect_robots_txt=True,
                use_cache=True,
            )

            self.stdout.write(f"Created crawl job: {crawl_job.id}")

            # Execute the crawl
            self.stdout.write("Starting crawl...")
            asyncio.run(self._run_crawler(crawl_job))

            # Refresh from database
            crawl_job.refresh_from_db()

            # Display results
            self.stdout.write(self.style.SUCCESS("Crawl completed!"))
            self.stdout.write(f"Status: {crawl_job.status}")
            self.stdout.write(f"Successful pages: {crawl_job.successful_crawls}")
            self.stdout.write(f"Failed pages: {crawl_job.failed_crawls}")
            self.stdout.write(
                f"Total content size: {crawl_job.total_content_size} bytes"
            )

            if crawl_job.error_message:
                self.stdout.write(self.style.ERROR(f"Error: {crawl_job.error_message}"))

            # Show some page results
            pages = crawl_job.pages.all()[:3]  # Show first 3 pages
            if pages:
                self.stdout.write("\nSample pages:")
                for page in pages:
                    self.stdout.write(f"  - {page.url}")
                    self.stdout.write(f"    Title: {page.title or 'No title'}")
                    self.stdout.write(f"    Words: {page.word_count}")
                    self.stdout.write(f"    Success: {page.success}")
                    if page.clean_markdown:
                        preview = (
                            page.clean_markdown[:200] + "..."
                            if len(page.clean_markdown) > 200
                            else page.clean_markdown
                        )
                        self.stdout.write(f"    Content preview: {preview}")
                    self.stdout.write("")

        except Exception as e:
            raise CommandError(f"Crawler test failed: {e}")

    async def _run_crawler(self, crawl_job):
        """Run the crawler asynchronously"""
        crawler_service = CrawlerService()
        await crawler_service.execute_crawl_job(crawl_job)
