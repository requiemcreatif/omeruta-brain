import time
from django.core.management.base import BaseCommand
from django.db import connection
from crawler.models import CrawlJob, CrawledPage, CrawlStatistics
from knowledge_base.models import KnowledgeEmbedding, QueryCache


# pylint: disable=no-member
class Command(BaseCommand):
    help = "Clear data from crawler and knowledge_base applications"

    def add_arguments(self, parser):
        parser.add_argument(
            "app",
            type=str,
            choices=["all", "crawler", "knowledge_base"],
            help="Specify which app's data to clear.",
        )
        parser.add_argument(
            "--no-confirm",
            action="store_true",
            help="Skip confirmation prompt before deleting data.",
        )

    def handle(self, *args, **options):
        app_to_clear = options["app"]
        no_confirm = options["no_confirm"]

        targets = []
        if app_to_clear in ["all", "knowledge_base"]:
            targets.extend([KnowledgeEmbedding, QueryCache])
        if app_to_clear in ["all", "crawler"]:
            targets.extend([CrawlStatistics, CrawledPage, CrawlJob])

        if not no_confirm:
            self.stdout.write(
                self.style.WARNING(
                    f"This will permanently delete all data from the following models in the '{app_to_clear}' scope:"
                )
            )
            for model in targets:
                self.stdout.write(f"- {model.__name__}")

            confirm = input("Are you sure you want to continue? (yes/no): ")
            if confirm.lower() != "yes":
                self.stdout.write(self.style.ERROR("Operation cancelled."))
                return

        start_time = time.time()
        self.stdout.write(f"🗑️  Starting data deletion for '{app_to_clear}'...")

        # Deletion order is important due to foreign key constraints
        deletion_order = [
            KnowledgeEmbedding,
            QueryCache,
            CrawlStatistics,
            CrawledPage,
            CrawlJob,
        ]

        for model in deletion_order:
            if model in targets:
                model_name = model.__name__
                self.stdout.write(f"  - Deleting all objects from {model_name}...")
                count, _ = model.objects.all().delete()
                self.stdout.write(
                    self.style.SUCCESS(
                        f"    Successfully deleted {count} objects from {model_name}."
                    )
                )

        total_time = time.time() - start_time
        self.stdout.write(
            self.style.SUCCESS(
                f"\n✅  Data deletion complete in {total_time:.2f} seconds."
            )
        )

        # Optional: Reset sequence for PostgreSQL if you want IDs to restart from 1
        with connection.cursor() as cursor:
            for model in targets:
                table_name = model._meta.db_table
                self.stdout.write(f"  - Resetting sequence for {table_name}...")
                try:
                    cursor.execute(
                        f"ALTER SEQUENCE {table_name}_id_seq RESTART WITH 1;"
                    )
                    self.stdout.write(
                        self.style.SUCCESS(f"    Sequence for {table_name} reset.")
                    )
                except Exception as e:
                    self.stdout.write(
                        self.style.WARNING(
                            f"    Could not reset sequence for {table_name}: {e} (this is often okay)."
                        )
                    )

        self.stdout.write(self.style.SUCCESS("\n✨  Operation finished."))
