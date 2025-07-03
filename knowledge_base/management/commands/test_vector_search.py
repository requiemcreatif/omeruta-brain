from django.core.management.base import BaseCommand
from knowledge_base.services.pgvector_search import PgVectorSearchService
from knowledge_base.services.enhanced_rag import EnhancedRAGService
from agents.services.enhanced_phi3_agent import EnhancedPhi3Agent
import time
import logging

# Set up logging to see debug info
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Test vector search and Phi3 general agent functionality"

    def add_arguments(self, parser):
        parser.add_argument(
            "--query",
            type=str,
            default="Hello, test the general agent",
            help="Search query to test (default: 'Hello, test the general agent')",
        )
        parser.add_argument(
            "--top-k", type=int, default=3, help="Number of results to return"
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
            default=True,
            help="Test enhanced Phi3 agent (default: True)",
        )
        parser.add_argument(
            "--agent-type",
            type=str,
            default="general",
            choices=["general", "research", "qa", "content_analyzer"],
            help="Agent type for testing (default: general)",
        )
        parser.add_argument(
            "--skip-context",
            action="store_true",
            help="Skip context retrieval to test model only",
        )
        parser.add_argument(
            "--max-tokens",
            type=int,
            default=100,
            help="Maximum tokens for response (default: 100)",
        )

    def handle(self, *args, **options):
        query = options["query"]
        agent_type = options["agent_type"]
        use_context = not options["skip_context"]

        self.stdout.write(f"\n🧠 Testing Phi3 General Agent")
        self.stdout.write(f"Query: '{query}'")
        self.stdout.write(f"Agent Type: {agent_type}")
        self.stdout.write(f"Use Context: {use_context}")
        self.stdout.write("=" * 60)

        # Test 0: Quick System Check
        self.stdout.write("\n0️⃣ System Health Check...")
        try:
            # Test database connection
            from django.db import connection

            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
            self.stdout.write("  ✅ Database connection: OK")
        except Exception as e:
            self.stdout.write(f"  ❌ Database connection: {e}")
            return

        # Test embedding service
        try:
            search_service = PgVectorSearchService()
            stats = search_service.get_search_stats()
            self.stdout.write(
                f"  ✅ Vector search: {stats['total_embeddings']} embeddings available"
            )
        except Exception as e:
            self.stdout.write(f"  ❌ Vector search setup: {e}")

        # Test 1: Agent Initialization
        self.stdout.write(f"\n1️⃣ Initializing Phi3 Agent ({agent_type})...")
        init_start = time.time()

        try:
            agent = EnhancedPhi3Agent(agent_type=agent_type)
            init_time = (time.time() - init_start) * 1000
            self.stdout.write(f"  ✅ Agent initialized in {init_time:.0f}ms")

            # Check model availability
            model_available = agent.llm_service.is_available()
            self.stdout.write(f"  Model loaded: {'✅' if model_available else '❌'}")

            if model_available:
                model_info = agent.llm_service.get_model_info()
                self.stdout.write(f"  Model: {model_info.get('name', 'Unknown')}")
                self.stdout.write(f"  Device: {model_info.get('device', 'Unknown')}")

                # Handle memory usage - it might be a string or number
                memory_usage = model_info.get("memory_usage", "Unknown")
                if isinstance(memory_usage, (int, float)):
                    self.stdout.write(f"  Memory: {memory_usage:.1f}GB")
                else:
                    self.stdout.write(f"  Memory: {memory_usage}")
            else:
                self.stdout.write(
                    "  ❌ Model not available - initialization may have failed"
                )
                return

        except Exception as e:
            self.stdout.write(f"  ❌ Agent initialization failed: {e}")
            return

        # Test 2: Context Retrieval (if enabled)
        context_time = 0
        context_sources = 0

        if use_context:
            self.stdout.write(f"\n2️⃣ Testing Context Retrieval...")
            context_start = time.time()

            try:
                search_result = search_service.enhanced_search(query)
                context_time = (time.time() - context_start) * 1000
                context_sources = len(search_result.get("results", []))

                self.stdout.write(
                    f"  ✅ Context search completed in {context_time:.0f}ms"
                )
                self.stdout.write(f"  📚 Found {context_sources} relevant sources")

                if options["show_details"] and context_sources > 0:
                    self.stdout.write(f"\n  📖 Top sources:")
                    for i, result in enumerate(search_result["results"][:3]):
                        score = result.get(
                            "combined_score", result.get("similarity", 0)
                        )
                        self.stdout.write(
                            f"    {i+1}. {result['page_title']} (score: {score:.3f})"
                        )

            except Exception as e:
                self.stdout.write(f"  ❌ Context retrieval failed: {e}")
                use_context = False  # Fallback to no context

        # Test 3: Complete Agent Response
        self.stdout.write(f"\n3️⃣ Testing Complete Agent Response...")
        response_start = time.time()

        try:
            # Configure response
            response_config = {
                "max_tokens": options["max_tokens"],
                "temperature": 0.7,
            }

            # Process message
            result = agent.process_message(
                message=query,
                use_context=use_context,
                response_config=response_config,
            )

            total_time = (time.time() - response_start) * 1000

            # Display results
            self.stdout.write(f"\n📊 Results:")
            self.stdout.write(f"  Status: {result.get('status', 'unknown')}")
            self.stdout.write(f"  Total time: {total_time:.0f}ms")
            self.stdout.write(
                f"  Processing time: {result.get('processing_time_ms', 0):.0f}ms"
            )
            self.stdout.write(f"  Used context: {result.get('used_context', False)}")
            self.stdout.write(f"  Sources: {len(result.get('sources', []))}")
            self.stdout.write(f"  Model: {result.get('model_used', 'unknown')}")

            # Quality scores
            if result.get("quality_scores"):
                self.stdout.write(f"\n📈 Quality Scores:")
                for metric, score in result["quality_scores"].items():
                    self.stdout.write(f"     {metric}: {score:.3f}")

            # Response
            if result.get("status") == "success" and result.get("response"):
                self.stdout.write(f"\n💬 Generated Response:")
                self.stdout.write("=" * 50)
                self.stdout.write(result["response"])
                self.stdout.write("=" * 50)
            else:
                error_msg = result.get("error_message", "No response generated")
                self.stdout.write(f"  ❌ Error: {error_msg}")

        except Exception as e:
            self.stdout.write(f"  ❌ Agent processing failed: {e}")
            import traceback

            if options["show_details"]:
                self.stdout.write(f"  Full error: {traceback.format_exc()}")

        # Test 4: Performance Analysis
        self.stdout.write(f"\n4️⃣ Performance Analysis:")
        self.stdout.write("=" * 40)

        self.stdout.write(f"  Agent init: {init_time:.0f}ms")
        if use_context:
            self.stdout.write(f"  Context retrieval: {context_time:.0f}ms")
        self.stdout.write(f"  Response generation: {total_time:.0f}ms")

        # Performance recommendations
        self.stdout.write(f"\n💡 Performance Notes:")
        if init_time > 5000:
            self.stdout.write(
                f"  ⚠️  Slow initialization ({init_time:.0f}ms) - model loading issue?"
            )
        if context_time > 2000:
            self.stdout.write(
                f"  ⚠️  Slow context retrieval ({context_time:.0f}ms) - database performance?"
            )
        if total_time > 10000:
            self.stdout.write(
                f"  ⚠️  Slow response generation ({total_time:.0f}ms) - consider async processing"
            )

        if total_time < 5000:
            self.stdout.write(f"  ✅ Good performance overall ({total_time:.0f}ms)")

        # Test 5: Quick Commands for Testing
        self.stdout.write(f"\n5️⃣ Quick Test Commands:")
        self.stdout.write("=" * 40)
        self.stdout.write(f"  # Test without context (faster):")
        self.stdout.write(f"  python manage.py test_vector_search --skip-context")
        self.stdout.write(f"")
        self.stdout.write(f"  # Test different agent types:")
        self.stdout.write(
            f"  python manage.py test_vector_search --agent-type research"
        )
        self.stdout.write(f"  python manage.py test_vector_search --agent-type qa")
        self.stdout.write(f"")
        self.stdout.write(f"  # Test with custom query:")
        self.stdout.write(
            f'  python manage.py test_vector_search --query "What is AI?"'
        )
        self.stdout.write(f"")
        self.stdout.write(f"  # Detailed debugging:")
        self.stdout.write(f"  python manage.py test_vector_search --show-details")

        self.stdout.write(f"\n✅ Test completed!")
