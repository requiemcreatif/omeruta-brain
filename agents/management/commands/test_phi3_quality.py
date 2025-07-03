from django.core.management.base import BaseCommand
from agents.services.enhanced_phi3_agent import EnhancedPhi3Agent
import time
import json


class Command(BaseCommand):
    help = "Test the enhanced Phi-3 agent with quality improvements"

    def add_arguments(self, parser):
        parser.add_argument("--query", type=str, help="Specific query to test")
        parser.add_argument(
            "--agent-type", type=str, default="general", help="Agent type to use"
        )
        parser.add_argument(
            "--no-context", action="store_true", help="Disable context search"
        )
        parser.add_argument(
            "--verbose", action="store_true", help="Show detailed output"
        )

    def handle(self, *args, **options):
        self.stdout.write(
            self.style.SUCCESS(
                "🧪 Testing Enhanced Phi-3 Agent with Quality Improvements"
            )
        )

        agent = EnhancedPhi3Agent(agent_type=options["agent_type"])

        # Test specific query if provided
        if options["query"]:
            self.test_query(agent, options["query"], options)
            return

        # Test suite of different query types
        test_queries = [
            # Simple factual questions (should be concise but complete)
            {
                "query": "Who is the US president in 2025?",
                "expected_type": "simple_factual",
                "expected_style": "concise",
            },
            {
                "query": "What is machine learning?",
                "expected_type": "definition",
                "expected_style": "precise",
            },
            {
                "query": "When was the Declaration of Independence signed?",
                "expected_type": "simple_factual",
                "expected_style": "concise",
            },
            # Analytical questions (should be comprehensive)
            {
                "query": "Compare the advantages and disadvantages of renewable energy",
                "expected_type": "analytical",
                "expected_style": "comprehensive",
            },
            {
                "query": "Explain how neural networks work",
                "expected_type": "analytical",
                "expected_style": "comprehensive",
            },
            # Procedural questions (should be structured)
            {
                "query": "How to set up a Django project",
                "expected_type": "procedural",
                "expected_style": "structured",
            },
            # List-based questions (should be organized)
            {
                "query": "List the main programming languages used for web development",
                "expected_type": "list_based",
                "expected_style": "organized",
            },
        ]

        self.stdout.write(f"\n🔍 Testing {len(test_queries)} different query types...")

        results = []
        for i, test_case in enumerate(test_queries, 1):
            self.stdout.write(f"\n{'='*80}")
            self.stdout.write(f"Test {i}/{len(test_queries)}: {test_case['query']}")
            self.stdout.write(
                f"Expected: {test_case['expected_type']} ({test_case['expected_style']})"
            )

            result = self.test_query(agent, test_case["query"], options)
            result.update(test_case)
            results.append(result)

        # Summary
        self.stdout.write(f"\n{'='*80}")
        self.stdout.write(self.style.SUCCESS("📊 TEST SUMMARY"))
        self.stdout.write(f"{'='*80}")

        total_time = sum(r["processing_time_ms"] for r in results)
        avg_time = total_time / len(results)

        context_usage = sum(1 for r in results if r["used_context"])
        quality_issues = sum(1 for r in results if r.get("quality_issues", 0) > 0)

        self.stdout.write(f"Total tests: {len(results)}")
        self.stdout.write(f"Average response time: {avg_time:.0f}ms")
        self.stdout.write(f"Context usage: {context_usage}/{len(results)} tests")
        self.stdout.write(f"Quality issues detected: {quality_issues}")

        # Response length analysis
        lengths = [r["response_length"] for r in results]
        self.stdout.write(f"\nResponse lengths:")
        self.stdout.write(f"  Min: {min(lengths)} chars")
        self.stdout.write(f"  Max: {max(lengths)} chars")
        self.stdout.write(f"  Avg: {sum(lengths)/len(lengths):.0f} chars")

        # Question type analysis
        type_counts = {}
        for result in results:
            question_type = result.get("question_type", "unknown")
            type_counts[question_type] = type_counts.get(question_type, 0) + 1

        self.stdout.write(f"\nQuestion type classification:")
        for qtype, count in type_counts.items():
            self.stdout.write(f"  {qtype}: {count}")

        if options["verbose"]:
            self.stdout.write(f"\n📋 Detailed Results:")
            for i, result in enumerate(results, 1):
                self.stdout.write(f"\n{i}. {result['query'][:50]}...")
                self.stdout.write(f"   Type: {result.get('question_type', 'unknown')}")
                self.stdout.write(f"   Length: {result['response_length']} chars")
                self.stdout.write(f"   Time: {result['processing_time_ms']}ms")
                self.stdout.write(f"   Context: {result['used_context']}")

    def test_query(self, agent, query, options):
        """Test a single query and return results"""
        self.stdout.write(f"\n🔍 Query: {query}")

        start_time = time.time()

        # Test with context (default)
        use_context = not options.get("no_context", False)
        result = agent.process_message(query, use_context=use_context)

        end_time = time.time()
        wall_time = (end_time - start_time) * 1000

        # Extract key metrics
        response = result.get("response", "")
        processing_time = result.get("processing_time_ms", 0)
        used_context = result.get("used_context", False)
        sources_used = result.get("sources_used", 0)
        question_type = result.get("question_type", "unknown")

        # Analyze response quality
        quality_issues = self._analyze_response_quality(response)

        # Display results
        self.stdout.write(f"\n📝 Response ({len(response)} chars):")
        self.stdout.write(f"   {response}")

        self.stdout.write(f"\n📊 Metrics:")
        self.stdout.write(f"   Processing time: {processing_time}ms")
        self.stdout.write(f"   Wall time: {wall_time:.0f}ms")
        self.stdout.write(f"   Question type: {question_type}")
        self.stdout.write(f"   Used context: {used_context}")
        self.stdout.write(f"   Sources used: {sources_used}")
        self.stdout.write(f"   Response length: {len(response)} chars")
        self.stdout.write(f"   Word count: {len(response.split())} words")

        if quality_issues:
            self.stdout.write(f"\n⚠️  Quality Issues Detected:")
            for issue in quality_issues:
                self.stdout.write(f"   - {issue}")
        else:
            self.stdout.write(f"\n✅ Response quality: Good")

        # Show optimization info
        optimization = result.get("optimization", "unknown")
        framework = result.get("framework", "unknown")
        self.stdout.write(f"\n🚀 Optimization: {optimization} ({framework})")

        return {
            "query": query,
            "response": response,
            "response_length": len(response),
            "processing_time_ms": processing_time,
            "wall_time_ms": wall_time,
            "used_context": used_context,
            "sources_used": sources_used,
            "question_type": question_type,
            "quality_issues": len(quality_issues),
            "optimization": optimization,
            "framework": framework,
        }

    def _analyze_response_quality(self, response):
        """Analyze response for quality issues"""
        issues = []

        if not response:
            issues.append("Empty response")
            return issues

        response_lower = response.lower()

        # Check for repetitive uncertainty
        uncertainty_phrases = [
            "i don't have",
            "i cannot",
            "i'm sorry",
            "unfortunately",
            "i apologize",
            "i regret",
            "i must clarify",
        ]
        uncertainty_count = sum(
            1 for phrase in uncertainty_phrases if phrase in response_lower
        )
        if uncertainty_count >= 2:
            issues.append(f"Repetitive uncertainty ({uncertainty_count} phrases)")

        # Check for sentence repetition
        sentences = [s.strip() for s in response.split(".") if s.strip()]
        if len(sentences) > 3:
            unique_starts = set()
            for sentence in sentences:
                if len(sentence.strip()) > 10:
                    start = sentence.strip()[:20].lower()
                    if start in unique_starts:
                        issues.append("Sentence repetition detected")
                        break
                    unique_starts.add(start)

        # Check for very short responses
        if len(response) < 30:
            issues.append("Response too short")

        # Check for very long responses (potential rambling)
        if len(response) > 1000:
            issues.append("Response very long (potential rambling)")

        # Check for proper sentence structure
        if response and not response.strip().endswith((".", "!", "?")):
            issues.append("Improper sentence ending")

        return issues
