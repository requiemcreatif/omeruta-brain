import os
from django.core.management.base import BaseCommand
from agents.services.local_model_service import TinyLlamaService


class Command(BaseCommand):
    help = "Test TinyLlama model loading with CPU-only configuration"

    def handle(self, *args, **options):
        # Force CPU-only for this test
        os.environ["FORCE_CPU_ONLY"] = "true"
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

        self.stdout.write("🧪 Testing TinyLlama model with CPU-only configuration...")

        try:
            # Initialize service
            service = TinyLlamaService()
            self.stdout.write(f"📱 Device selected: {service.device}")

            # Test model loading
            self.stdout.write("⏳ Loading model...")
            if service.initialize_model():
                self.stdout.write(self.style.SUCCESS("✅ Model loaded successfully!"))

                # Test generation
                self.stdout.write("⏳ Testing text generation...")
                response = service.generate_response(
                    prompt="What is AI?",
                    max_tokens=50,
                    system_prompt="You are a helpful assistant.",
                )

                if response:
                    self.stdout.write(
                        self.style.SUCCESS("✅ Text generation successful!")
                    )
                    self.stdout.write(f"📝 Sample response: {response[:100]}...")
                else:
                    self.stdout.write(self.style.ERROR("❌ Text generation failed"))

                # Get model info
                info = service.get_model_info()
                self.stdout.write(f"📊 Model info: {info}")

                # Cleanup
                service.cleanup_model()
                self.stdout.write("🧹 Model cleaned up")

            else:
                self.stdout.write(self.style.ERROR("❌ Model loading failed"))

        except Exception as e:
            self.stdout.write(self.style.ERROR(f"❌ Test failed with error: {e}"))
            import traceback

            traceback.print_exc()

        self.stdout.write("�� Test completed")
