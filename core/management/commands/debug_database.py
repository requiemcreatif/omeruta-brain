from django.core.management.base import BaseCommand
from crawler.models import CrawledPage
from knowledge_base.models import KnowledgeEmbedding


class Command(BaseCommand):
    help = "Debug database contents - show what pages and embeddings exist"

    def handle(self, *args, **options):
        self.stdout.write("🔍 Database Debug Information")
        self.stdout.write("=" * 50)

        # Check crawled pages
        pages = CrawledPage.objects.all()
        self.stdout.write(f"\n📄 Crawled Pages ({pages.count()}):")

        if pages.exists():
            for page in pages:
                self.stdout.write(f"  • {page.url}")
                self.stdout.write(f"    Title: {page.title or 'No title'}")
                self.stdout.write(f"    Success: {page.success}")
                self.stdout.write(f"    Word count: {page.word_count}")
                self.stdout.write(
                    f"    Processed for embeddings: {page.is_processed_for_embeddings}"
                )
                self.stdout.write("")
        else:
            self.stdout.write("  No pages found in database!")

        # Check embeddings
        embeddings = KnowledgeEmbedding.objects.all()
        self.stdout.write(f"\n🧠 Knowledge Embeddings ({embeddings.count()}):")

        if embeddings.exists():
            # Group by page
            pages_with_embeddings = {}
            for embedding in embeddings:
                page_url = embedding.page.url
                if page_url not in pages_with_embeddings:
                    pages_with_embeddings[page_url] = []
                pages_with_embeddings[page_url].append(embedding)

            for page_url, page_embeddings in pages_with_embeddings.items():
                self.stdout.write(f"  • {page_url}")
                self.stdout.write(f"    Chunks: {len(page_embeddings)}")
                avg_quality = sum(
                    e.content_quality_score for e in page_embeddings
                ) / len(page_embeddings)
                self.stdout.write(f"    Avg Quality: {avg_quality:.3f}")

                # Show first chunk preview
                if page_embeddings:
                    first_chunk = page_embeddings[0].chunk_text[:100] + "..."
                    self.stdout.write(f"    Preview: {first_chunk}")
                self.stdout.write("")
        else:
            self.stdout.write("  No embeddings found in database!")

        # Summary
        self.stdout.write(f"\n📊 Summary:")
        self.stdout.write(f"  Total pages: {pages.count()}")
        self.stdout.write(f"  Successful pages: {pages.filter(success=True).count()}")
        self.stdout.write(
            f"  Pages with embeddings: {pages.filter(is_processed_for_embeddings=True).count()}"
        )
        self.stdout.write(f"  Total embedding chunks: {embeddings.count()}")
