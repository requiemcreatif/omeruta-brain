from django.contrib import admin
from django.utils.html import format_html
from django.urls import reverse
from .models import KnowledgeEmbedding, QueryCache
from .services.smart_chunker import SmartSemanticChunker
import csv
from django.http import HttpResponse
from django.utils import timezone
from datetime import timedelta


@admin.register(KnowledgeEmbedding)
class KnowledgeEmbeddingAdmin(admin.ModelAdmin):
    list_display = [
        "id_short",
        "page_title_link",
        "chunk_index",
        "chunk_tokens",
        "content_quality_score",
        "semantic_density",
        "created_at",
        "chunk_preview",
    ]

    list_filter = [
        "created_at",
        "content_quality_score",
        "semantic_density",
        "processing_version",
        "page__success",
    ]

    search_fields = ["chunk_text", "page__title", "page__url", "metadata"]

    readonly_fields = [
        "id",
        "embedding",
        "created_at",
        "updated_at",
        "embedding_dimensions",
        "chunk_preview_full",
    ]

    ordering = ["-created_at"]

    list_per_page = 50

    fieldsets = (
        (
            "Basic Information",
            {
                "fields": (
                    "id",
                    "page",
                    "chunk_index",
                    "chunk_tokens",
                    "processing_version",
                )
            },
        ),
        (
            "Content",
            {"fields": ("chunk_text", "chunk_preview_full"), "classes": ("collapse",)},
        ),
        ("Quality Metrics", {"fields": ("content_quality_score", "semantic_density")}),
        ("Metadata", {"fields": ("metadata",), "classes": ("collapse",)}),
        (
            "Vector Data",
            {"fields": ("embedding_dimensions",), "classes": ("collapse",)},
        ),
        (
            "Timestamps",
            {"fields": ("created_at", "updated_at"), "classes": ("collapse",)},
        ),
    )

    def id_short(self, obj):
        """Show shortened UUID"""
        return str(obj.id)[:8] + "..."

    id_short.short_description = "ID"

    def page_title_link(self, obj):
        """Show page title with link to page admin"""
        if obj.page:
            url = reverse("admin:crawler_crawledpage_change", args=[obj.page.id])
            return format_html('<a href="{}">{}</a>', url, obj.page.title[:50])
        return "-"

    page_title_link.short_description = "Page"
    page_title_link.admin_order_field = "page__title"

    def chunk_preview(self, obj):
        """Show chunk text preview"""
        preview = (
            obj.chunk_text[:100] + "..."
            if len(obj.chunk_text) > 100
            else obj.chunk_text
        )
        return format_html('<span title="{}">{}</span>', obj.chunk_text, preview)

    chunk_preview.short_description = "Content Preview"

    def chunk_preview_full(self, obj):
        """Show full chunk text in readonly field"""
        return format_html(
            '<div style="max-height: 200px; overflow-y: auto; padding: 10px; background: #f8f9fa; border: 1px solid #dee2e6;">{}</div>',
            obj.chunk_text,
        )

    chunk_preview_full.short_description = "Full Chunk Text"

    def embedding_dimensions(self, obj):
        """Show embedding vector information"""
        if obj.embedding:
            return f"Vector length: {len(obj.embedding)} dimensions"
        return "No embedding"

    embedding_dimensions.short_description = "Embedding Info"

    actions = ["regenerate_quality_scores", "export_selected_chunks"]

    def regenerate_quality_scores(self, request, queryset):
        """Regenerate quality scores for selected embeddings"""

        chunker = SmartSemanticChunker()

        updated_count = 0
        for embedding in queryset:
            new_quality = chunker._calculate_quality_score(embedding.chunk_text)
            new_density = chunker._calculate_semantic_density(embedding.chunk_text)

            embedding.content_quality_score = new_quality
            embedding.semantic_density = new_density
            embedding.save(update_fields=["content_quality_score", "semantic_density"])
            updated_count += 1

        self.message_user(
            request, f"Updated quality scores for {updated_count} embeddings."
        )

    regenerate_quality_scores.short_description = "Regenerate quality scores"

    def export_selected_chunks(self, request, queryset):
        """Export selected chunks as text"""

        response = HttpResponse(content_type="text/csv")
        response["Content-Disposition"] = 'attachment; filename="knowledge_chunks.csv"'

        writer = csv.writer(response)
        writer.writerow(
            ["ID", "Page Title", "Chunk Index", "Quality Score", "Chunk Text"]
        )

        for embedding in queryset:
            writer.writerow(
                [
                    str(embedding.id),
                    embedding.page.title if embedding.page else "",
                    embedding.chunk_index,
                    embedding.content_quality_score,
                    embedding.chunk_text,
                ]
            )

        return response

    export_selected_chunks.short_description = "Export selected chunks to CSV"


@admin.register(QueryCache)
class QueryCacheAdmin(admin.ModelAdmin):
    list_display = [
        "query_preview",
        "access_count",
        "retrieval_time_ms",
        "generation_time_ms",
        "user_rating",
        "created_at",
        "last_accessed",
    ]

    list_filter = ["created_at", "last_accessed", "user_rating", "access_count"]

    search_fields = ["original_query", "query_hash"]

    readonly_fields = [
        "query_hash",
        "created_at",
        "last_accessed",
        "expanded_queries_display",
        "relevant_chunks_display",
    ]

    ordering = ["-last_accessed"]

    list_per_page = 25

    fieldsets = (
        (
            "Query Information",
            {"fields": ("original_query", "query_hash", "expanded_queries_display")},
        ),
        (
            "Results",
            {
                "fields": ("relevant_chunks_display", "generated_response"),
                "classes": ("collapse",),
            },
        ),
        (
            "Performance Metrics",
            {"fields": ("retrieval_time_ms", "generation_time_ms", "user_rating")},
        ),
        ("Usage Stats", {"fields": ("access_count", "created_at", "last_accessed")}),
    )

    def query_preview(self, obj):
        """Show query preview"""
        preview = (
            obj.original_query[:80] + "..."
            if len(obj.original_query) > 80
            else obj.original_query
        )
        return format_html('<span title="{}">{}</span>', obj.original_query, preview)

    query_preview.short_description = "Query"
    query_preview.admin_order_field = "original_query"

    def expanded_queries_display(self, obj):
        """Show expanded queries in a nice format"""
        if obj.expanded_queries:
            queries = "<br>".join(f"• {q}" for q in obj.expanded_queries)
            return format_html(
                '<div style="max-height: 100px; overflow-y: auto;">{}</div>', queries
            )
        return "No expansions"

    expanded_queries_display.short_description = "Query Expansions"

    def relevant_chunks_display(self, obj):
        """Show relevant chunk IDs"""
        if obj.relevant_chunks:
            chunks = ", ".join(obj.relevant_chunks[:5])  # Show first 5 chunk IDs
            if len(obj.relevant_chunks) > 5:
                chunks += f" ... and {len(obj.relevant_chunks) - 5} more"
            return chunks
        return "No chunks"

    relevant_chunks_display.short_description = "Relevant Chunks"

    actions = ["clear_old_cache", "export_query_stats"]

    def clear_old_cache(self, request, queryset):
        """Clear old cache entries"""

        old_date = timezone.now() - timedelta(days=30)
        old_entries = queryset.filter(last_accessed__lt=old_date)
        count = old_entries.count()
        old_entries.delete()

        self.message_user(
            request, f"Cleared {count} old cache entries (older than 30 days)."
        )

    clear_old_cache.short_description = "Clear old cache entries"

    def export_query_stats(self, request, queryset):
        """Export query statistics"""

        response = HttpResponse(content_type="text/csv")
        response["Content-Disposition"] = 'attachment; filename="query_stats.csv"'

        writer = csv.writer(response)
        writer.writerow(
            [
                "Query",
                "Access Count",
                "Retrieval Time (ms)",
                "Generation Time (ms)",
                "User Rating",
                "Created",
                "Last Accessed",
            ]
        )

        for cache in queryset:
            writer.writerow(
                [
                    cache.original_query,
                    cache.access_count,
                    cache.retrieval_time_ms,
                    cache.generation_time_ms,
                    cache.user_rating or "",
                    cache.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                    cache.last_accessed.strftime("%Y-%m-%d %H:%M:%S"),
                ]
            )

        return response

    export_query_stats.short_description = "Export query statistics to CSV"


# Customize admin site header and title
admin.site.site_header = "Omeruta Brain Knowledge Base Administration"
admin.site.site_title = "Knowledge Base Admin"
admin.site.index_title = "Knowledge Base Management"
