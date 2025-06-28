from django.core.management.base import BaseCommand
from django.db.models import Q
from crawler.models import CrawledPage
from knowledge_base.services.embedding_generator import EmbeddingGenerationService


class Command(BaseCommand):
    help = "Generate embeddings for crawled content"

    def add_arguments(self, parser):
        parser.add_argument(
            "--all", action="store_true", help="Process all unprocessed pages"
        )
        parser.add_argument(
            "--page-ids", nargs="+", help="Specific page IDs to process"
        )
        parser.add_argument(
            "--force", action="store_true", help="Force regenerate existing embeddings"
        )
        parser.add_argument(
            "--batch-size", type=int, default=50, help="Batch size for processing"
        )
        parser.add_argument(
            "--test-one", action="store_true", help="Process just one page for testing"
        )

    def handle(self, *args, **options):
        embedding_service = EmbeddingGenerationService()

        if options["test_one"]:
            self.stdout.write("🧪 Testing with one page...")

            # Get one unprocessed page
            test_page = (
                CrawledPage.objects.filter(
                    success=True,
                    is_processed_for_embeddings=False,
                    clean_markdown__isnull=False,
                )
                .exclude(clean_markdown="")
                .first()
            )

            if not test_page:
                self.stdout.write(self.style.WARNING("No unprocessed pages found"))
                return

            result = embedding_service.process_page(test_page, options["force"])
            self.stdout.write(f"Test result: {result}")
            return

        if options["all"]:
            self.stdout.write("Processing all unprocessed pages...")
            result = embedding_service.batch_process_pages(
                page_ids=None, force_regenerate=options["force"]
            )
            self._print_results(result)

        elif options["page_ids"]:
            self.stdout.write(
                f"Processing {len(options['page_ids'])} specific pages..."
            )
            result = embedding_service.batch_process_pages(
                page_ids=options["page_ids"], force_regenerate=options["force"]
            )
            self._print_results(result)

        else:
            # Show stats
            stats = embedding_service.get_embedding_stats()
            self.stdout.write("\n" + "=" * 50)
            self.stdout.write("📊 EMBEDDING STATISTICS")
            self.stdout.write("=" * 50)
            self.stdout.write(f"Total embeddings: {stats['total_embeddings']:,}")
            self.stdout.write(
                f"Pages with embeddings: {stats['pages_with_embeddings']:,}"
            )
            self.stdout.write(f"Total crawled pages: {stats['total_crawled_pages']:,}")
            self.stdout.write(
                f"Processing percentage: {stats['processing_percentage']:.1f}%"
            )

            if stats["avg_quality"]:
                self.stdout.write(f"Average quality score: {stats['avg_quality']:.3f}")
                self.stdout.write(
                    f"Average semantic density: {stats['avg_density']:.3f}"
                )
                self.stdout.write(f"Total tokens: {stats['total_tokens']:,}")

            unprocessed_count = (
                CrawledPage.objects.filter(
                    success=True,
                    is_processed_for_embeddings=False,
                    clean_markdown__isnull=False,
                )
                .exclude(clean_markdown="")
                .count()
            )

            self.stdout.write(f"\n📋 Unprocessed pages: {unprocessed_count:,}")

            if unprocessed_count > 0:
                self.stdout.write("\n💡 Usage examples:")
                self.stdout.write("  python manage.py generate_embeddings --all")
                self.stdout.write("  python manage.py generate_embeddings --test-one")
                self.stdout.write(
                    "  python manage.py generate_embeddings --page-ids <id1> <id2>"
                )

    def _print_results(self, result):
        self.stdout.write("\n" + "=" * 50)
        self.stdout.write("🎯 PROCESSING RESULTS")
        self.stdout.write("=" * 50)
        self.stdout.write(f"Total pages: {result['total_pages']}")
        self.stdout.write(f"✅ Processed: {result['processed']}")
        self.stdout.write(f"⏭️  Skipped: {result['skipped']}")
        self.stdout.write(f"❌ Errors: {result['errors']}")
        self.stdout.write(f"📝 Total embeddings created: {result['total_embeddings']}")

        if result["errors"] > 0:
            self.stdout.write("\n⚠️  Errors occurred:")
            for detail in result["details"]:
                if detail["result"]["status"] == "error":
                    self.stdout.write(f"  {detail['url']}: {detail['result']['error']}")

        # Show some successful processing examples
        if result["processed"] > 0:
            self.stdout.write(f"\n✅ Successfully processed examples:")
            success_count = 0
            for detail in result["details"]:
                if detail["result"]["status"] == "success" and success_count < 3:
                    chunks = detail["result"]["chunk_count"]
                    self.stdout.write(f"  📄 {detail['title']}: {chunks} embeddings")
                    success_count += 1
