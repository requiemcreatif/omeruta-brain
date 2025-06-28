import asyncio
from django.core.management.base import BaseCommand
from agents.services.live_research_agent import LiveResearchAgent
import json


class Command(BaseCommand):
    help = "Test the Live Research Agent with internet research capabilities"

    def add_arguments(self, parser):
        parser.add_argument(
            "--topic",
            type=str,
            help="Research topic to investigate",
            default="latest developments in quantum computing 2024",
        )
        parser.add_argument(
            "--max-sources",
            type=int,
            default=5,
            help="Maximum number of sources to research (default: 5)",
        )
        parser.add_argument(
            "--research-depth",
            type=str,
            choices=["surface", "comprehensive", "deep"],
            default="comprehensive",
            help="Research depth level (default: comprehensive)",
        )
        parser.add_argument(
            "--no-local-kb",
            action="store_true",
            help="Skip local knowledge base search",
        )
        parser.add_argument(
            "--no-live-research",
            action="store_true",
            help="Skip live internet research (local KB only)",
        )
        parser.add_argument(
            "--show-log", action="store_true", help="Show detailed research log"
        )

    def handle(self, *args, **options):
        topic = options["topic"]
        max_sources = options["max_sources"]
        research_depth = options["research_depth"]
        include_local_kb = not options["no_local_kb"]
        use_live_research = not options["no_live_research"]
        show_log = options["show_log"]

        self.stdout.write("🔬 Testing Live Research Agent")
        self.stdout.write("=" * 60)
        self.stdout.write(f"Research Topic: {topic}")
        self.stdout.write(f"Max Sources: {max_sources}")
        self.stdout.write(f"Research Depth: {research_depth}")
        self.stdout.write(f"Include Local KB: {include_local_kb}")
        self.stdout.write(f"Use Live Research: {use_live_research}")
        self.stdout.write("")

        try:
            # Initialize the live research agent
            agent = LiveResearchAgent()

            # Show research capabilities
            self.stdout.write("🛠️ Research Capabilities:")
            capabilities = agent.get_research_capabilities()
            for key, value in capabilities.items():
                if isinstance(value, list):
                    self.stdout.write(f"   {key}: {', '.join(value)}")
                else:
                    self.stdout.write(f"   {key}: {value}")
            self.stdout.write("")

            # Conduct research
            self.stdout.write("🚀 Starting Research...")
            self.stdout.write("")

            async def run_research():
                return await agent.enhanced_research_chat(
                    message=topic,
                    use_live_research=use_live_research,
                    max_sources=max_sources,
                    research_depth=research_depth,
                )

            # Run the research
            result = asyncio.run(run_research())

            # Display results
            self.stdout.write("📊 Research Results:")
            self.stdout.write("=" * 40)

            if result.get("research_conducted"):
                self.stdout.write(f"✅ Live Research Conducted: YES")
                self.stdout.write(
                    f"🌐 Live Sources Used: {result.get('live_sources_used', 0)}"
                )
                self.stdout.write(
                    f"📚 Local Sources Used: {result.get('local_sources_used', 0)}"
                )

                # Research methodology
                methodology = result.get("research_methodology", {})
                if methodology:
                    self.stdout.write(
                        f"🔍 Search Queries: {len(methodology.get('search_queries_used', []))}"
                    )
                    self.stdout.write(
                        f"🌐 URLs Attempted: {methodology.get('urls_attempted', 0)}"
                    )
                    self.stdout.write(
                        f"✅ Sources Crawled: {methodology.get('sources_crawled', 0)}"
                    )
                    self.stdout.write(
                        f"⏱️ Research Time: {methodology.get('research_time_seconds', 0):.2f}s"
                    )

                # Quality metrics
                quality = result.get("quality_metrics", {})
                if quality:
                    self.stdout.write(
                        f"📈 Source Diversity: {quality.get('source_diversity', 0)}"
                    )
                    self.stdout.write(
                        f"🔄 Content Freshness: {quality.get('content_freshness', 'unknown')}"
                    )
                    self.stdout.write(
                        f"📝 Total Content Analyzed: {quality.get('total_content_analyzed', 0)} words"
                    )
            else:
                self.stdout.write(
                    f"❌ Live Research Conducted: NO (using local knowledge base)"
                )

            self.stdout.write("")

            # Show research log if requested
            if show_log and result.get("research_log"):
                self.stdout.write("📋 Research Log:")
                self.stdout.write("-" * 30)
                for log_entry in result["research_log"]:
                    self.stdout.write(f"   {log_entry}")
                self.stdout.write("")

            # Show the research response
            self.stdout.write("🎯 Research Response:")
            self.stdout.write("=" * 50)
            response = result.get("response", "No response generated")
            self.stdout.write(response)
            self.stdout.write("=" * 50)

            # Show sources
            sources = result.get("sources", [])
            if sources:
                self.stdout.write(f"\n📚 Sources Used ({len(sources)}):")
                self.stdout.write("-" * 30)
                for i, source in enumerate(sources, 1):
                    source_type = (
                        "🌐" if source.get("source_type") == "live_web" else "📚"
                    )
                    self.stdout.write(
                        f"{i}. {source_type} {source.get('title', 'Unknown Title')}"
                    )
                    self.stdout.write(f"   URL: {source.get('source_url', 'N/A')}")
                    self.stdout.write(
                        f"   Relevance: {source.get('relevance_score', 0):.2f}"
                    )
                    self.stdout.write(f"   Words: {source.get('word_count', 0)}")
                    self.stdout.write("")

            # Performance summary
            self.stdout.write("⚡ Performance Summary:")
            self.stdout.write("-" * 25)
            model_used = result.get("model_used", "unknown")
            self.stdout.write(f"Model Used: {model_used}")

            if result.get("research_methodology"):
                total_time = result["research_methodology"].get(
                    "research_time_seconds", 0
                )
                total_sources = result["research_methodology"].get("total_sources", 0)
                self.stdout.write(f"Total Research Time: {total_time:.2f} seconds")
                self.stdout.write(
                    f"Sources per Second: {total_sources/total_time:.2f}"
                    if total_time > 0
                    else "Sources per Second: N/A"
                )

            self.stdout.write("\n✨ Live Research Agent test completed successfully!")

        except Exception as e:
            self.stdout.write(f"❌ Test failed: {e}")
            import traceback

            self.stdout.write(traceback.format_exc())

        # Usage examples
        self.stdout.write("\n💡 Usage Examples:")
        self.stdout.write("=" * 20)
        examples = [
            "python manage.py test_live_research --topic 'artificial intelligence trends 2024'",
            "python manage.py test_live_research --topic 'climate change solutions' --max-sources 10",
            "python manage.py test_live_research --topic 'quantum computing applications' --research-depth deep",
            "python manage.py test_live_research --topic 'blockchain technology' --no-local-kb",
            "python manage.py test_live_research --topic 'machine learning' --show-log",
        ]

        for example in examples:
            self.stdout.write(f"   {example}")
