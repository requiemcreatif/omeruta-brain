from django.core.management.base import BaseCommand
from agents.services.tinyllama_agent import TinyLlamaAgent
from crawler.models import CrawledPage


class Command(BaseCommand):
    help = "Test TinyLlama integration with crawled data"

    def add_arguments(self, parser):
        parser.add_argument(
            "--question",
            type=str,
            help="Question to ask the agent",
            default="What information do you have?",
        )

    def handle(self, *args, **options):
        self.stdout.write("🤖 Testing TinyLlama Agent...\n")

        # Initialize agent
        agent = TinyLlamaAgent(agent_type="general")

        # Show knowledge base stats
        stats = agent.get_available_knowledge_stats()
        self.stdout.write(f"📊 Knowledge Base Stats:")
        for key, value in stats.items():
            self.stdout.write(f"   {key}: {value}")

        # Test with sample questions
        test_questions = [
            options["question"],
            "What topics are covered in the crawled content?",
            "Summarize what you know",
        ]

        for question in test_questions:
            self.stdout.write(f"\n❓ Question: {question}")

            result = agent.process_message(question)

            self.stdout.write(f"🤖 Response: {result['response']}")
            self.stdout.write(f"📋 Model: {result['model_used']}")
            self.stdout.write(f"🔍 Context used: {result['context_used']}")

            if "context_sources" in result:
                self.stdout.write(f"📚 Sources: {result['context_sources']}")

            if "error" in result:
                self.stdout.write(self.style.ERROR(f"❌ Error: {result['error']}"))

            self.stdout.write("-" * 50)
