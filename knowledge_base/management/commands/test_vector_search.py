from django.core.management.base import BaseCommand
from knowledge_base.services.pgvector_search import PgVectorSearchService
from knowledge_base.services.enhanced_rag import EnhancedRAGService
from agents.services.enhanced_tinyllama_agent import EnhancedTinyLlamaAgent
import time


class Command(BaseCommand):
    help = "Test vector search and RAG functionality"

    def add_arguments(self, parser):
        parser.add_argument(
            "--query", type=str, required=True, help="Search query to test"
        )
        parser.add_argument(
            "--top-k", type=int, default=5, help="Number of results to return"
        )
        parser.add_argument(
            "--show-details", action="store_true", help="Show detailed results"
        )
        parser.add_argument(
            "--test-rag", action="store_true", help="Test complete RAG pipeline"
        )
        parser.add_argument(
            "--test-agent",
            action="store_true",
            help="Test enhanced agent with TinyLlama",
        )
        parser.add_argument(
            "--agent-type",
            type=str,
            default="general",
            choices=["general", "research", "qa", "content_analyzer"],
            help="Agent type for testing",
        )

    def handle(self, *args, **options):
        query = options["query"]

        self.stdout.write(f"\n🔍 Testing search for: '{query}'")
        self.stdout.write("=" * 60)

        # Test 1: Vector Search
        self.stdout.write("\n1️⃣ Testing Vector Search...")
        search_service = PgVectorSearchService()
        search_result = search_service.enhanced_search(query)

        self.stdout.write(f"\n📊 Search Results:")
        self.stdout.write(f"  Retrieval time: {search_result['retrieval_time_ms']}ms")
        self.stdout.write(f"  Total candidates: {search_result['total_candidates']}")
        self.stdout.write(f"  Unique candidates: {search_result['unique_candidates']}")
        self.stdout.write(f"  Final results: {search_result['final_count']}")
        self.stdout.write(
            f"  Query expansions: {len(search_result['query_expansions'])}"
        )
        self.stdout.write(
            f"  Used cross-encoder: {search_result['used_cross_encoder']}"
        )

        # Show top results
        if search_result["results"]:
            self.stdout.write(
                f"\n🎯 Top {min(options['top_k'], len(search_result['results']))} Results:"
            )
            for i, result in enumerate(search_result["results"][: options["top_k"]]):
                score = result.get("combined_score", result["similarity"])
                self.stdout.write(f"\n  {i+1}. {result['page_title']}")
                self.stdout.write(
                    f"     Score: {score:.3f} | Quality: {result['quality_score']:.3f}"
                )
                self.stdout.write(f"     URL: {result['page_url']}")

                if options["show_details"]:
                    preview = (
                        result["text"][:200] + "..."
                        if len(result["text"]) > 200
                        else result["text"]
                    )
                    self.stdout.write(f"     Preview: {preview}")
        else:
            self.stdout.write(self.style.WARNING("\n  No results found"))

        # Test 2: RAG Pipeline
        if options["test_rag"] or options["test_agent"]:
            self.stdout.write(f"\n2️⃣ Testing RAG Pipeline...")
            rag_service = EnhancedRAGService()
            rag_result = rag_service.generate_response(query)

            self.stdout.write(
                f"  Total time: {rag_result['search_metadata']['total_time_ms']}ms"
            )
            self.stdout.write(
                f"  Sources used: {rag_result['search_metadata']['sources_used']}"
            )

            if options["show_details"]:
                self.stdout.write(f"\n📝 Generated Context Preview:")
                context_preview = (
                    rag_result["context"][:300] + "..."
                    if len(rag_result["context"]) > 300
                    else rag_result["context"]
                )
                self.stdout.write(context_preview)

                self.stdout.write(f"\n💬 Enhanced Prompt Preview:")
                prompt_preview = (
                    rag_result["enhanced_prompt"][:400] + "..."
                    if len(rag_result["enhanced_prompt"]) > 400
                    else rag_result["enhanced_prompt"]
                )
                self.stdout.write(prompt_preview)

        # Test 3: Complete Agent
        if options["test_agent"]:
            self.stdout.write(
                f"\n3️⃣ Testing Enhanced Agent ({options['agent_type']})..."
            )

            agent = EnhancedTinyLlamaAgent(agent_type=options["agent_type"])

            # Check if TinyLlama is available
            if not agent.llm_service.is_available():
                self.stdout.write(
                    self.style.WARNING(
                        "  TinyLlama model not available - testing context only"
                    )
                )

                # Test context generation only
                context_result = agent.get_conversation_context(query)
                if "error" not in context_result:
                    self.stdout.write(
                        f"  Context generated: {len(context_result['context'])} chars"
                    )
                    self.stdout.write(f"  Sources: {len(context_result['sources'])}")
                else:
                    self.stdout.write(f"  Context error: {context_result['error']}")
            else:
                self.stdout.write("  🚀 Generating complete response...")
                start_time = time.time()

                agent_result = agent.process_message(
                    query,
                    use_context=True,
                    response_config={
                        "style": "informative",
                        "max_length": "medium",
                        "max_tokens": 200,
                        "temperature": 0.7,
                    },
                )

                total_time = (time.time() - start_time) * 1000

                self.stdout.write(f"\n🎯 Agent Results:")
                self.stdout.write(f"  Status: {agent_result['status']}")
                self.stdout.write(f"  Used context: {agent_result['used_context']}")
                self.stdout.write(f"  Total processing time: {total_time:.0f}ms")
                self.stdout.write(f"  Sources: {len(agent_result['sources'])}")

                if agent_result["quality_scores"]:
                    self.stdout.write(f"\n📊 Quality Scores:")
                    for metric, score in agent_result["quality_scores"].items():
                        self.stdout.write(f"     {metric}: {score:.3f}")

                if agent_result["status"] == "success":
                    self.stdout.write(f"\n💬 Generated Response:")
                    self.stdout.write("=" * 50)
                    self.stdout.write(agent_result["response"])
                    self.stdout.write("=" * 50)
                else:
                    self.stdout.write(
                        f"  Error: {agent_result.get('error_message', 'Unknown error')}"
                    )

        # Test 4: Performance Summary
        self.stdout.write(f"\n4️⃣ Performance Summary:")
        self.stdout.write("=" * 40)

        if search_result["results"]:
            avg_score = sum(
                r.get("combined_score", r["similarity"])
                for r in search_result["results"]
            ) / len(search_result["results"])
            self.stdout.write(f"  Average relevance score: {avg_score:.3f}")

        self.stdout.write(
            f"  Vector search time: {search_result['retrieval_time_ms']}ms"
        )

        if options["test_rag"]:
            self.stdout.write(
                f"  RAG pipeline time: {rag_result['search_metadata']['total_time_ms']}ms"
            )

        # System status
        stats = search_service.get_search_stats()
        self.stdout.write(f"\n🔧 System Status:")
        self.stdout.write(
            f"  Embedding model: {'✅' if stats['embedding_model_available'] else '❌'}"
        )
        self.stdout.write(
            f"  Cross-encoder: {'✅' if stats['cross_encoder_available'] else '❌'}"
        )

        # Usage suggestions
        self.stdout.write(f"\n💡 Try these variations:")
        self.stdout.write(f"  --show-details   (detailed output)")
        self.stdout.write(f"  --test-rag       (test RAG pipeline)")
        self.stdout.write(f"  --test-agent     (test complete agent)")
        self.stdout.write(f"  --agent-type research  (different agent types)")
