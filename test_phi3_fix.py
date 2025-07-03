#!/usr/bin/env python3
"""
Quick test script for Phi-3 fix
"""

import os
import sys
import django
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Setup Django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "omeruta_brain_project.settings")
django.setup()

from agents.services.phi3_model_service import Phi3ModelService


def test_phi3_fix():
    print("🔧 Testing Phi-3 Fix")
    print("=" * 30)

    try:
        # Initialize service
        service = Phi3ModelService()
        print(f"📱 Device: {service.device}")

        # Initialize model
        print("📥 Loading model...")
        if service.initialize_model():
            print("✅ Model loaded!")

            # Test with simple prompt
            print("\n🧪 Testing generation...")
            response = service.generate_response(
                prompt="Hello, how are you?",
                max_tokens=50,
                temperature=0.7,
                system_prompt="You are a helpful assistant.",
            )

            if response:
                print("✅ Generation successful!")
                print(f"📝 Response: {response}")
                return True
            else:
                print("❌ Generation failed")
                return False
        else:
            print("❌ Model loading failed")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    finally:
        try:
            service.cleanup_model()
            print("🧹 Cleaned up")
        except:
            pass


if __name__ == "__main__":
    success = test_phi3_fix()
    print(f"\n{'✅ Success!' if success else '❌ Failed!'}")
    sys.exit(0 if success else 1)
