from django.core.management.base import BaseCommand
from agents.services.tinyllama_agent import TinyLlamaAgent
from agents.services.conversation_memory import ConversationMemory
from crawler.models import CrawledPage


class Command(BaseCommand):
    help = "Test TinyLlama integration with enhanced features"

    def add_arguments(self, parser):
        parser.add_argument(
            "--question",
            type=str,
            help="Question to ask the agent",
            default="What is cryptocurrency and how does it work?",
        )
        parser.add_argument(
            "--test-conversation",
            action="store_true",
            help="Test conversation memory features",
        )

    def handle(self, *args, **options):
        self.stdout.write("🤖 Testing Enhanced TinyLlama Agent...\n")

        # Initialize agent
        agent = TinyLlamaAgent(agent_type="general")

        # Show knowledge base stats
        stats = agent.get_available_knowledge_stats()
        self.stdout.write(f"📊 Knowledge Base Stats:")
        for key, value in stats.items():
            self.stdout.write(f"   {key}: {value}")

        # Test question classification
        self.stdout.write(f"\n🧠 Testing Question Classification:")
        test_questions = [
            "What is Bitcoin?",  # factual
            "How to invest in cryptocurrency?",  # procedural
            "Compare Bitcoin and Ethereum",  # analytical
            "What do you think about AI?",  # opinion
        ]

        for question in test_questions:
            question_type = agent._classify_question_type(question)
            needs_context = agent._needs_context(question)
            self.stdout.write(
                f"   '{question[:30]}...' -> Type: {question_type}, Needs Context: {needs_context}"
            )

        # Test with different agent types
        self.stdout.write(f"\n🎭 Testing Different Agent Types:")
        agent_types = ["general", "research", "qa", "content_analyzer"]

        for agent_type in agent_types:
            self.stdout.write(f"\n--- Testing {agent_type.upper()} Agent ---")
            test_agent = TinyLlamaAgent(agent_type=agent_type)

            result = test_agent.process_message(options["question"])

            self.stdout.write(f"🤖 Response: {result['response'][:200]}...")
            self.stdout.write(f"📋 Model: {result['model_used']}")
            self.stdout.write(f"🔍 Context used: {result['context_used']}")
            self.stdout.write(
                f"🏷️ Question type: {result.get('question_type', 'unknown')}"
            )

            if "context_sources" in result:
                self.stdout.write(f"📚 Sources: {result['context_sources']}")

        # Test conversation memory if requested
        if options["test_conversation"]:
            self.stdout.write(f"\n💾 Testing Conversation Memory:")

            # Create conversation memory
            memory = ConversationMemory()
            self.stdout.write(f"   Created conversation: {memory.conversation_id}")

            # Simulate a conversation
            conversation_questions = [
                "What is Bitcoin?",
                "How does it differ from traditional currency?",
                "What are the risks of investing in it?",
            ]

            for i, question in enumerate(conversation_questions):
                self.stdout.write(f"\n   Exchange {i+1}:")
                self.stdout.write(f"   Q: {question}")

                # Get conversation context
                context = memory.get_conversation_context()
                if context:
                    enhanced_question = (
                        f"Previous context:\n{context}\n\nCurrent question: {question}"
                    )
                else:
                    enhanced_question = question

                # Process with agent
                result = agent.process_message(enhanced_question)
                response = result["response"]

                self.stdout.write(f"   A: {response[:150]}...")

                # Add to memory
                memory.add_exchange(
                    question,
                    response,
                    {
                        "question_type": result.get("question_type"),
                        "context_used": result.get("context_used"),
                    },
                )

            # Show conversation summary
            summary = memory.get_conversation_summary()
            self.stdout.write(f"\n📈 Conversation Summary:")
            self.stdout.write(f"   Total exchanges: {summary['total_exchanges']}")
            self.stdout.write(f"   Topics discussed: {summary['topics_discussed']}")

            # Clear memory
            memory.clear_conversation()
            self.stdout.write(f"   ✅ Conversation memory cleared")

        self.stdout.write(f"\n✅ Enhanced TinyLlama Agent testing complete!")
        self.stdout.write("-" * 60)
