import os
from django.core.management.base import BaseCommand
from agents.services.phi3_model_service import Phi3ModelService


class Command(BaseCommand):
    help = "Test Phi-3 model functionality"

    def add_arguments(self, parser):
        parser.add_argument(
            "--prompt",
            type=str,
            default="What are the advantages of Phi-3 for RAG applications?",
            help="Test prompt to use",
        )
        parser.add_argument(
            "--max-tokens",
            type=int,
            default=300,
            help="Maximum tokens to generate",
        )
        parser.add_argument(
            "--force-cpu",
            action="store_true",
            help="Force CPU usage",
        )

    def handle(self, *args, **options):
        if options["force_cpu"]:
            os.environ["FORCE_CPU_ONLY"] = "true"

        self.stdout.write("🧪 Testing Phi-3 Model")
        self.stdout.write("=" * 40)

        try:
            # Initialize service
            service = Phi3ModelService()
            self.stdout.write(f"📱 Device selected: {service.device}")

            # Test model initialization
            self.stdout.write("\n📥 Initializing Phi-3 model...")
            if service.initialize_model():
                self.stdout.write(
                    self.style.SUCCESS("✅ Model initialized successfully!")
                )

                # Get model info
                model_info = service.get_model_info()
                self.stdout.write(f"📊 Model info: {model_info}")

                # Test generation
                self.stdout.write(f"\n🧠 Testing generation with prompt:")
                self.stdout.write(f"   '{options['prompt']}'")

                response = service.generate_response(
                    prompt=options["prompt"],
                    max_tokens=options["max_tokens"],
                    system_prompt="You are a helpful AI assistant specialized in explaining AI technologies.",
                )

                if response:
                    self.stdout.write(self.style.SUCCESS("\n✅ Generation successful!"))
                    self.stdout.write("\n📝 Response:")
                    self.stdout.write("-" * 40)
                    self.stdout.write(response)
                    self.stdout.write("-" * 40)

                    # Test context stats
                    context_stats = service.get_context_stats()
                    self.stdout.write(f"\n📊 Context capabilities: {context_stats}")

                else:
                    self.stdout.write(
                        self.style.ERROR("❌ Generation failed - no response")
                    )

            else:
                self.stdout.write(self.style.ERROR("❌ Failed to initialize model"))

        except Exception as e:
            self.stdout.write(self.style.ERROR(f"❌ Error during testing: {e}"))

        finally:
            # Cleanup
            try:
                service.cleanup_model()
                self.stdout.write("\n🧹 Model cleaned up")
            except:
                pass

        self.stdout.write("\n🎯 Test completed!")
