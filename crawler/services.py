import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, Union
from urllib.parse import urljoin, urlparse
import hashlib
import re
from datetime import datetime

from django.conf import settings
from django.utils import timezone
from django.db import transaction
from asgiref.sync import sync_to_async
from .models import CrawlStatistics

# Crawl4AI imports
try:
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
    from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
    from crawl4ai.content_filter_strategy import PruningContentFilter, BM25ContentFilter

    CRAWL4AI_AVAILABLE = True
except ImportError:
    CRAWL4AI_AVAILABLE = False

from .models import CrawlJob, CrawledPage, CrawlStatistics

# Import vectorization task for automatic processing
try:
    from knowledge_base.tasks import generate_embeddings_for_page

    VECTORIZATION_AVAILABLE = True
except ImportError:
    VECTORIZATION_AVAILABLE = False

logger = logging.getLogger(__name__)


class CrawlerService:
    """Main service for web crawling using Crawl4AI"""

    def __init__(self, auto_vectorize: bool = True):
        if not CRAWL4AI_AVAILABLE:
            raise ImportError(
                "Crawl4AI is not installed. Please install it with: pip install crawl4ai"
            )

        self.crawler = None
        self.browser_config = None
        self.auto_vectorize = auto_vectorize and VECTORIZATION_AVAILABLE
        self.setup_browser_config()

    def setup_browser_config(self):
        """Configure browser settings for optimal crawling"""
        self.browser_config = BrowserConfig(
            headless=True,
            verbose=False,
            user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 OmerutaBrain/1.0",
            java_script_enabled=True,
            viewport_width=1920,
            viewport_height=1080,
            ignore_https_errors=True,
            browser_type="chromium",
        )

    async def start_crawler(self):
        """Initialize the crawler instance"""
        if self.crawler is None:
            self.crawler = AsyncWebCrawler(config=self.browser_config)
            await self.crawler.start()

    async def stop_crawler(self):
        """Clean up crawler resources"""
        if self.crawler:
            await self.crawler.close()
            self.crawler = None

    def create_crawler_config(
        self,
        use_content_filter: bool = True,
        filter_type: str = "pruning",
        user_query: Optional[str] = None,
        cache_mode: str = "enabled",
    ) -> CrawlerRunConfig:
        """Create optimized crawler configuration for clean markdown extraction"""

        # Setup content filter for clean markdown
        content_filter = None
        if use_content_filter:
            if filter_type == "bm25" and user_query:
                content_filter = BM25ContentFilter(
                    user_query=user_query, bm25_threshold=1.0
                )
            else:
                # Default to pruning filter for general content cleaning
                content_filter = PruningContentFilter(
                    threshold=0.48, threshold_type="fixed", min_word_threshold=10
                )

        # Setup markdown generator with content filter
        markdown_generator = DefaultMarkdownGenerator(
            content_filter=content_filter,
            options={
                "ignore_links": False,  # Keep links for reference
                "ignore_images": True,  # Skip images for vector DB
                "escape_html": True,
                "body_width": 0,  # No line wrapping
                "skip_internal_links": False,
            },
        )

        # Map cache mode string to enum
        cache_mode_map = {
            "enabled": CacheMode.ENABLED,
            "disabled": CacheMode.DISABLED,
            "bypass": CacheMode.BYPASS,
        }

        return CrawlerRunConfig(
            cache_mode=cache_mode_map.get(cache_mode, CacheMode.ENABLED),
            markdown_generator=markdown_generator,
            page_timeout=60000,  # Page timeout in ms
            delay_before_return_html=2.0,  # Wait 2 seconds for dynamic content
        )

    async def crawl_single_url(
        self, url: str, crawl_job: CrawlJob, config_options: Optional[Dict] = None
    ) -> CrawledPage:
        """Crawl a single URL and extract clean markdown content"""

        start_time = time.time()
        page = CrawledPage(crawl_job=crawl_job, url=url, final_url=url, depth=0)

        try:
            # Create crawler config with optional customizations
            config_opts = config_options or {}
            crawler_config = self.create_crawler_config(**config_opts)

            # Perform the crawl
            result = await self.crawler.arun(url=url, config=crawler_config)

            if result.success:
                # Extract basic metadata
                page.final_url = result.url
                page.status_code = result.status_code
                page.success = True
                page.response_time = time.time() - start_time

                # Extract content
                if result.markdown:
                    # Get the full markdown content
                    raw_content = str(result.markdown)

                    # Apply our own cleaning for vector DB
                    page.clean_markdown = clean_markdown_for_vector_db(raw_content)

                # Extract metadata
                if result.metadata:
                    metadata = result.metadata
                    page.title = (metadata.get("title") or "")[:500]
                    page.meta_description = metadata.get("description") or ""
                    page.meta_keywords = metadata.get("keywords") or ""
                    page.author = metadata.get("author") or ""
                    page.language = metadata.get("language") or ""
                    page.content_type = metadata.get("content-type") or ""

                # Extract links
                if result.links:
                    page.internal_links = result.links.get("internal", [])
                    page.external_links = result.links.get("external", [])

                # Content analysis
                content = page.get_content_for_vector_db()
                if content:
                    page.content_length = len(content)
                    page.word_count = len(content.split())

                logger.info(f"Successfully crawled: {url} - {page.word_count} words")

            else:
                page.success = False
                page.error_message = result.error_message or "Unknown crawl error"
                page.response_time = time.time() - start_time
                logger.warning(f"Failed to crawl {url}: {page.error_message}")

        except Exception as e:
            page.success = False
            page.error_message = str(e)
            page.response_time = time.time() - start_time
            logger.error(f"Exception crawling {url}: {e}", exc_info=True)

        return page

    async def crawl_multiple_urls(
        self,
        urls: List[str],
        crawl_job: CrawlJob,
        config_options: Optional[Dict] = None,
        max_concurrent: int = 5,
    ) -> List[CrawledPage]:
        """Crawl multiple URLs concurrently"""

        pages = []
        config_opts = config_options or {}

        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)

        async def crawl_with_semaphore(url: str) -> CrawledPage:
            async with semaphore:
                return await self.crawl_single_url(url, crawl_job, config_opts)

        # Execute crawls concurrently
        tasks = [crawl_with_semaphore(url) for url in urls]
        pages = await asyncio.gather(*tasks, return_exceptions=False)

        return pages

    async def deep_crawl(
        self,
        start_url: str,
        crawl_job: CrawlJob,
        strategy: str = "bfs",
        max_pages: int = 10,
        max_depth: int = 3,
        config_options: Optional[Dict] = None,
    ) -> List[CrawledPage]:
        """Perform deep crawling with link discovery and following"""

        pages = []
        visited_urls = set()
        url_queue = [(start_url, 0)]  # (url, depth)
        config_opts = config_options or {}

        # Compile regex patterns
        include_patterns = []
        exclude_patterns = []

        if crawl_job.include_patterns:
            include_patterns = [
                re.compile(pattern) for pattern in crawl_job.include_patterns
            ]

        if crawl_job.exclude_patterns:
            exclude_patterns = [
                re.compile(pattern) for pattern in crawl_job.exclude_patterns
            ]

        def should_crawl_url(url: str, depth: int) -> bool:
            """Check if URL should be crawled based on patterns and depth"""
            if depth > max_depth:
                return False

            if url in visited_urls:
                return False

            # Check exclude patterns first
            for pattern in exclude_patterns:
                if pattern.search(url):
                    return False

            # If include patterns exist, URL must match at least one
            if include_patterns:
                return any(pattern.search(url) for pattern in include_patterns)

            return True

        def extract_links_from_page(page: CrawledPage, current_depth: int) -> List[str]:
            """Extract and filter links from a crawled page"""
            new_urls = []
            base_domain = urlparse(start_url).netloc

            # Process internal links
            for link in page.internal_links:
                try:
                    # Resolve relative URLs
                    absolute_url = urljoin(page.final_url, link)
                    link_domain = urlparse(absolute_url).netloc

                    # Check domain scope based on strategy
                    if link_domain == base_domain or link_domain.endswith(
                        f".{base_domain}"
                    ):
                        if should_crawl_url(absolute_url, current_depth + 1):
                            new_urls.append(absolute_url)

                except Exception as e:
                    logger.warning(f"Error processing link {link}: {e}")

            return new_urls

        # Main crawling loop
        while url_queue and len(pages) < max_pages:
            # Get next URL based on strategy
            if strategy == "dfs":
                current_url, current_depth = url_queue.pop()  # LIFO for DFS
            else:  # BFS or best-first
                current_url, current_depth = url_queue.pop(0)  # FIFO for BFS

            if current_url in visited_urls:
                continue

            visited_urls.add(current_url)

            # Crawl the current URL
            logger.info(
                f"Deep crawling [{len(pages)+1}/{max_pages}] depth {current_depth}: {current_url}"
            )

            page = await self.crawl_single_url(current_url, crawl_job, config_opts)
            page.depth = current_depth
            pages.append(page)

            # Update job progress
            crawl_job.progress = len(pages)
            crawl_job.total_found = len(visited_urls) + len(url_queue)

            # Extract new links if crawl was successful and we haven't reached max depth
            if page.success and current_depth < max_depth:
                new_links = extract_links_from_page(page, current_depth)

                # Add new URLs to queue
                for link in new_links:
                    if link not in visited_urls:
                        url_queue.append((link, current_depth + 1))

                # Update total found count
                crawl_job.total_found = len(visited_urls) + len(url_queue)

            # Add delay between requests if configured
            if crawl_job.delay_between_requests > 0:
                await asyncio.sleep(crawl_job.delay_between_requests)

        return pages

    async def execute_crawl_job(self, crawl_job: CrawlJob) -> None:
        """Execute a complete crawl job based on its configuration"""

        start_time = time.time()

        try:
            # Update job status
            crawl_job.status = "running"
            await sync_to_async(crawl_job.save)()

            # Start crawler
            await self.start_crawler()

            pages = []
            config_options = {
                "cache_mode": "enabled" if crawl_job.use_cache else "bypass",
                "use_content_filter": True,
                "filter_type": "pruning",
            }

            # Execute crawl based on strategy
            if crawl_job.strategy == "single":
                if crawl_job.start_urls:
                    page = await self.crawl_single_url(
                        crawl_job.start_urls[0], crawl_job, config_options
                    )
                    pages = [page]

            elif crawl_job.strategy == "multi":
                pages = await self.crawl_multiple_urls(
                    crawl_job.start_urls, crawl_job, config_options
                )

            elif crawl_job.strategy.startswith("deep_"):
                strategy_name = crawl_job.strategy.replace("deep_", "")
                if crawl_job.start_urls:
                    pages = await self.deep_crawl(
                        crawl_job.start_urls[0],
                        crawl_job,
                        strategy=strategy_name,
                        max_pages=crawl_job.max_pages,
                        max_depth=crawl_job.max_depth,
                        config_options=config_options,
                    )

            # Save all pages to database
            @sync_to_async
            def save_pages_sync():
                with transaction.atomic():
                    for page in pages:
                        page.save()

            await save_pages_sync()

            # Trigger automatic vectorization for successful pages
            successful_pages = [p for p in pages if p.success]
            if self.auto_vectorize and successful_pages:
                await self._trigger_vectorization(successful_pages)

            # Update job statistics
            failed_pages = [p for p in pages if not p.success]

            crawl_job.successful_crawls = len(successful_pages)
            crawl_job.failed_crawls = len(failed_pages)
            crawl_job.total_content_size = sum(
                len(p.get_content_for_vector_db()) for p in successful_pages
            )
            crawl_job.status = "completed"
            crawl_job.progress = len(pages)

        except Exception as e:
            crawl_job.status = "failed"
            crawl_job.error_message = str(e)
            logger.error(f"Crawl job {crawl_job.id} failed: {e}", exc_info=True)

        finally:
            # Calculate total crawl time
            total_time = time.time() - start_time

            # Create or update statistics
            @sync_to_async
            def create_stats_sync():

                stats, created = getattr(CrawlStatistics, "objects").get_or_create(
                    crawl_job=crawl_job, defaults={"total_crawl_time": total_time}
                )
                if not created:
                    stats.total_crawl_time = total_time
                    stats.save()
                stats.calculate_statistics()
                return stats

            await create_stats_sync()

            # Save final job state
            await sync_to_async(crawl_job.save)()

            # Clean up crawler
            await self.stop_crawler()

            logger.info(
                f"Crawl job {crawl_job.id} completed in {total_time:.2f}s: "
                f"{crawl_job.successful_crawls} successful, {crawl_job.failed_crawls} failed"
            )

    async def _trigger_vectorization(self, successful_pages: List[CrawledPage]):
        """Trigger automatic vectorization for successfully crawled pages"""

        vectorization_count = 0

        for page in successful_pages:
            # Only vectorize pages with content that haven't been processed
            if (
                page.clean_markdown
                and len(page.clean_markdown.strip()) > 100
                and not page.is_processed_for_embeddings
            ):

                try:
                    # Trigger async vectorization task
                    generate_embeddings_for_page.delay(str(page.id))
                    vectorization_count += 1
                    logger.info(f"🧠 Triggered vectorization for page: {page.url}")
                except Exception as e:
                    logger.warning(
                        f"Failed to trigger vectorization for {page.url}: {e}"
                    )

        if vectorization_count > 0:
            logger.info(f"✅ Triggered vectorization for {vectorization_count} pages")
        else:
            logger.info("ℹ️ No pages require vectorization")


# Utility functions for content processing
def chunk_content_for_embeddings(
    content: str, chunk_size: int = 1000, overlap: int = 100
) -> List[str]:
    """Split content into overlapping chunks for vector embeddings"""
    if not content:
        return []

    words = content.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i : i + chunk_size]
        chunk = " ".join(chunk_words)
        chunks.append(chunk)

        # Stop if we've reached the end
        if i + chunk_size >= len(words):
            break

    return chunks


def clean_markdown_for_vector_db(markdown_content: str) -> str:
    """Clean markdown content for LLM consumption - remove links, navigation, and formatting"""
    if not markdown_content:
        return ""

    content = markdown_content

    # Remove entire sections that are just noise for RAG
    noisy_sections = [
        "See also",
        "References",
        "External links",
        "Further reading",
        "Notes",
        "Sources",
        "Bibliography",
    ]

    for section_title in noisy_sections:
        # Regex to find a section header and all content until the next header
        # Handles both "## Title ##" and "## Title"
        pattern = re.compile(
            rf"(^|\n)##?\s*{re.escape(section_title)}\s*##?.*?(?=(\n##?\s*|$))",
            re.IGNORECASE | re.DOTALL,
        )
        content = pattern.sub("", content)

    # Remove malformed link/citation artifacts like [D](I) or [1]
    # This specifically targets single-character or numeric "links"
    content = re.sub(r"\[([a-zA-Z0-9]{1,2})\]\([^\)]+\)", "", content)

    # Remove all markdown links but keep the text content
    # Pattern: [text](url) -> text
    content = re.sub(r"\[([^\]]*)\]\([^\)]+\)", r"\1", content)

    # Remove empty links (images, empty anchors)
    content = re.sub(r"\[\]\([^\)]+\)", "", content)

    # Remove standalone URLs
    content = re.sub(r"https?://[^\s\)]+", "", content)

    # Remove email links (keep email text)
    content = re.sub(r"mailto:", "", content)

    # Remove navigation patterns
    navigation_patterns = [
        r"Skip to content",
        r"Toggle Navigation",
        r"Go to Top",
        r"Page load link",
        r"×",  # Close buttons
        r"Subscribe\s*$",  # Subscribe buttons
        r"Continue reading",
    ]
    for pattern in navigation_patterns:
        content = re.sub(pattern, "", content, flags=re.IGNORECASE)

    # Remove common navigation lists (bullet points that are clearly nav)
    # Remove lines that are just bullet points with single words/short phrases
    lines = content.split("\n")
    cleaned_lines = []

    for line in lines:
        # Skip lines that are likely navigation (bullet points with short content)
        if re.match(r"^\s*[\*\-]\s*[A-Za-z\s]{1,30}$", line.strip()):
            continue
        # Skip lines that are just whitespace or single characters
        if len(line.strip()) <= 2:
            continue
        # Skip lines that look like breadcrumbs or navigation
        if (
            re.match(r"^\s*[A-Za-z\s]{1,20}\s*$", line.strip())
            and len(line.strip().split()) <= 3
        ):
            continue

        cleaned_lines.append(line)

    content = "\n".join(cleaned_lines)

    # Remove markdown formatting but keep semantic content
    content = re.sub(r"\*\*([^*]+)\*\*", r"\1", content)  # Bold
    content = re.sub(r"\*([^*]+)\*", r"\1", content)  # Italic
    content = re.sub(r"`([^`]+)`", r"\1", content)  # Inline code
    content = re.sub(r"~~([^~]+)~~", r"\1", content)  # Strikethrough

    # Clean up headers - keep the text but remove markdown symbols
    content = re.sub(r"^#{1,6}\s*", "", content, flags=re.MULTILINE)

    # Remove bullet points and list markers
    content = re.sub(r"^\s*[\*\-\+]\s*", "", content, flags=re.MULTILINE)
    content = re.sub(r"^\s*\d+\.\s*", "", content, flags=re.MULTILINE)

    # Clean up excessive whitespace
    content = re.sub(r"\n\s*\n\s*\n+", "\n\n", content)
    content = re.sub(r"^\s+", "", content, flags=re.MULTILINE)

    # Remove lines that are just timestamps or meta info
    content = re.sub(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[^\n]*", "", content)

    # Remove copyright and footer text
    content = re.sub(r"©[^\n]*", "", content)
    content = re.sub(r"Copyright[^\n]*", "", content, flags=re.IGNORECASE)
    content = re.sub(r"All Rights Reserved[^\n]*", "", content, flags=re.IGNORECASE)

    # Remove any remaining lines that look like section headers
    content = re.sub(r"^\n##?.*##?\n", "\n", content, flags=re.MULTILINE)

    # Split into sentences and filter out very short ones (likely navigation/metadata)
    sentences = []
    for paragraph in content.split("\n\n"):
        paragraph = paragraph.strip()
        if len(paragraph) > 20 and not re.match(r"^[A-Z\s]{1,30}$", paragraph):
            sentences.append(paragraph)

    content = "\n\n".join(sentences)

    # Final cleanup
    content = content.strip()

    return content
