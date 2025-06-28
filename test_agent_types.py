#!/usr/bin/env python3
"""
Comprehensive test script for AI Assistant agent types and functionalities
"""

import os
import sys
import django
import requests
import json
import time
from datetime import datetime

# Setup Django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "omeruta_brain_project.settings")
django.setup()

from django.contrib.auth import authenticate
from django.test import Client
from django.contrib.auth.models import User


class AgentTester:
    """Test all agent types and enhanced RAG functionality"""

    def __init__(self, base_url="http://localhost:8000", username=None, password=None):
        self.base_url = base_url
        self.session = requests.Session()
        self.client = Client()
        self.auth_user = None

        if username and password:
            self.authenticate(username, password)

    def authenticate(self, username, password):
        """Authenticate and get session"""
        print(f"🔐 Authenticating as {username}...")

        try:
            # Try Django authentication first
            user = authenticate(username=username, password=password)
            if user:
                self.auth_user = user
                print(f"✅ Django authentication successful for {user.username}")

                # Get CSRF token for requests
                csrf_response = self.session.get(f"{self.base_url}/ai/")
                if csrf_response.status_code == 200:
                    # Extract CSRF token from cookies
                    csrf_token = csrf_response.cookies.get("csrftoken")
                    if csrf_token:
                        self.session.headers.update(
                            {"X-CSRFToken": csrf_token, "Referer": self.base_url}
                        )
                        print(f"✅ CSRF token obtained: {csrf_token[:10]}...")

                # Try to login via session
                login_data = {
                    "username": username,
                    "password": password,
                    "csrfmiddlewaretoken": csrf_token,
                }

                login_response = self.session.post(
                    f"{self.base_url}/auth/login/", data=login_data
                )
                print(f"Login response status: {login_response.status_code}")

                return True
            else:
                print(f"❌ Django authentication failed")
                return False

        except Exception as e:
            print(f"❌ Authentication error: {e}")
            return False

    def test_system_status(self):
        """Test system status and enhanced stats"""
        print("\n" + "=" * 60)
        print("🔍 TESTING SYSTEM STATUS")
        print("=" * 60)

        try:
            # Test enhanced stats endpoint
            response = self.session.get(
                f"{self.base_url}/ai/api/agents/tinyllama/enhanced_stats/"
            )
            print(f"Enhanced stats status: {response.status_code}")

            if response.status_code == 200:
                stats = response.json()
                print("✅ Enhanced stats retrieved successfully:")
                print(f"   Total embeddings: {stats.get('total_embeddings', 0):,}")
                print(
                    f"   Pages with embeddings: {stats.get('pages_with_embeddings', 0):,}"
                )
                print(
                    f"   Processing percentage: {stats.get('processing_percentage', 0):.1f}%"
                )
                print(f"   LLM available: {stats.get('llm_available', False)}")
                print(
                    f"   Embedding model available: {stats.get('embedding_model_available', False)}"
                )
                return stats
            else:
                print(f"❌ Enhanced stats failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return None

        except Exception as e:
            print(f"❌ System status test failed: {e}")
            return None

    def test_agent_type(self, agent_type, test_message, expected_behavior):
        """Test a specific agent type"""
        print(f"\n🤖 Testing Agent Type: {agent_type.upper()}")
        print(f"   Test message: {test_message}")
        print(f"   Expected: {expected_behavior}")
        print("-" * 50)

        test_data = {
            "message": test_message,
            "agent_type": agent_type,
            "use_context": True,
            "max_tokens": 300,
            "style": "informative",
            "max_length": "medium",
            "temperature": 0.7,
            "include_sources": True,
            "min_quality": 0.3,
        }

        try:
            start_time = time.time()
            response = self.session.post(
                f"{self.base_url}/ai/api/agents/tinyllama/enhanced_chat/",
                headers={"Content-Type": "application/json"},
                data=json.dumps(test_data),
            )
            end_time = time.time()

            print(f"   Response status: {response.status_code}")
            print(f"   Response time: {(end_time - start_time):.2f}s")

            if response.status_code == 200:
                result = response.json()

                print(f"✅ Agent {agent_type} responded successfully:")
                print(f"   Model used: {result.get('model_used', 'unknown')}")
                print(f"   Agent type: {result.get('agent_type', 'unknown')}")
                print(f"   Used context: {result.get('used_context', False)}")
                print(f"   Processing time: {result.get('processing_time_ms', 0)}ms")
                print(f"   Sources: {len(result.get('sources', []))}")

                # Display response
                response_text = result.get("response", "")
                print(
                    f"   Response: {response_text[:200]}{'...' if len(response_text) > 200 else ''}"
                )

                # Check quality scores if available
                if result.get("quality_scores"):
                    scores = result["quality_scores"]
                    print(f"   Quality scores:")
                    for metric, score in scores.items():
                        print(f"     {metric}: {score:.3f}")

                return {
                    "success": True,
                    "result": result,
                    "response_time": end_time - start_time,
                }
            else:
                print(f"❌ Agent {agent_type} failed:")
                print(f"   Status: {response.status_code}")
                print(f"   Error: {response.text}")
                return {"success": False, "error": response.text}

        except Exception as e:
            print(f"❌ Agent {agent_type} test error: {e}")
            return {"success": False, "error": str(e)}

    def test_knowledge_search(self, query):
        """Test direct knowledge base search"""
        print(f"\n🔍 Testing Knowledge Search: {query}")
        print("-" * 50)

        try:
            search_data = {"query": query, "filters": {"min_quality": 0.3}}

            start_time = time.time()
            response = self.session.post(
                f"{self.base_url}/ai/api/agents/tinyllama/search_knowledge/",
                headers={"Content-Type": "application/json"},
                data=json.dumps(search_data),
            )
            end_time = time.time()

            if response.status_code == 200:
                result = response.json()
                print(f"✅ Knowledge search successful:")
                print(f"   Search time: {result.get('retrieval_time_ms', 0)}ms")
                print(f"   Total candidates: {result.get('total_candidates', 0)}")
                print(f"   Final results: {result.get('final_count', 0)}")
                print(
                    f"   Used cross-encoder: {result.get('used_cross_encoder', False)}"
                )

                # Show top results
                results = result.get("results", [])
                for i, res in enumerate(results[:3]):
                    print(
                        f"   Result {i+1}: {res.get('page_title', 'Unknown')} (score: {res.get('similarity', 0):.3f})"
                    )

                return result
            else:
                print(f"❌ Knowledge search failed: {response.status_code}")
                return None

        except Exception as e:
            print(f"❌ Knowledge search error: {e}")
            return None

    def run_comprehensive_test(self):
        """Run comprehensive test of all agent types and functionalities"""
        print("🚀 STARTING COMPREHENSIVE AI ASSISTANT TEST")
        print("=" * 80)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Test system status first
        stats = self.test_system_status()
        if not stats:
            print("❌ System status check failed. Aborting tests.")
            return

        # Define test cases for each agent type
        test_cases = [
            {
                "agent_type": "general",
                "message": "What is artificial intelligence and how does it work?",
                "expected": "General informative response about AI",
            },
            {
                "agent_type": "research",
                "message": "Analyze the current trends in machine learning research",
                "expected": "Research-focused analysis with insights and synthesis",
            },
            {
                "agent_type": "qa",
                "message": "What are the main benefits of using neural networks?",
                "expected": "Direct, concise answer focused on key benefits",
            },
            {
                "agent_type": "content_analyzer",
                "message": "Analyze the key themes in modern AI development",
                "expected": "Structured analysis of themes and concepts",
            },
        ]

        # Test each agent type
        results = {}
        for test_case in test_cases:
            result = self.test_agent_type(
                test_case["agent_type"], test_case["message"], test_case["expected"]
            )
            results[test_case["agent_type"]] = result
            time.sleep(2)  # Brief pause between tests

        # Test knowledge search
        search_result = self.test_knowledge_search("machine learning algorithms")

        # Test different response styles
        self.test_response_styles()

        # Test various settings
        self.test_enhanced_settings()

        # Summary
        self.print_test_summary(results)

    def test_response_styles(self):
        """Test different response styles"""
        print(f"\n🎨 Testing Response Styles")
        print("=" * 50)

        styles = ["informative", "concise", "detailed", "analytical"]
        base_message = "What is deep learning?"

        for style in styles:
            print(f"\n   Testing style: {style}")
            test_data = {
                "message": base_message,
                "agent_type": "general",
                "use_context": True,
                "style": style,
                "include_sources": True,
            }

            try:
                response = self.session.post(
                    f"{self.base_url}/ai/api/agents/tinyllama/enhanced_chat/",
                    headers={"Content-Type": "application/json"},
                    data=json.dumps(test_data),
                )

                if response.status_code == 200:
                    result = response.json()
                    response_text = result.get("response", "")
                    word_count = len(response_text.split())
                    print(
                        f"   ✅ Style {style}: {word_count} words, {len(response_text)} chars"
                    )
                else:
                    print(f"   ❌ Style {style} failed: {response.status_code}")

            except Exception as e:
                print(f"   ❌ Style {style} error: {e}")

    def test_enhanced_settings(self):
        """Test enhanced RAG settings"""
        print(f"\n⚙️ Testing Enhanced Settings")
        print("=" * 50)

        # Test with different quality thresholds
        quality_thresholds = [0.1, 0.3, 0.7]
        base_message = "Tell me about neural networks"

        for threshold in quality_thresholds:
            print(f"\n   Testing quality threshold: {threshold}")
            test_data = {
                "message": base_message,
                "agent_type": "general",
                "use_context": True,
                "min_quality": threshold,
                "include_sources": True,
            }

            try:
                response = self.session.post(
                    f"{self.base_url}/ai/api/agents/tinyllama/enhanced_chat/",
                    headers={"Content-Type": "application/json"},
                    data=json.dumps(test_data),
                )

                if response.status_code == 200:
                    result = response.json()
                    sources_count = len(result.get("sources", []))
                    used_context = result.get("used_context", False)
                    print(
                        f"   ✅ Threshold {threshold}: {sources_count} sources, context: {used_context}"
                    )
                else:
                    print(f"   ❌ Threshold {threshold} failed: {response.status_code}")

            except Exception as e:
                print(f"   ❌ Threshold {threshold} error: {e}")

    def print_test_summary(self, results):
        """Print comprehensive test summary"""
        print("\n" + "=" * 80)
        print("📊 TEST SUMMARY")
        print("=" * 80)

        total_tests = len(results)
        successful_tests = sum(1 for r in results.values() if r.get("success"))

        print(f"Total agent types tested: {total_tests}")
        print(f"Successful tests: {successful_tests}")
        print(f"Failed tests: {total_tests - successful_tests}")
        print(f"Success rate: {(successful_tests/total_tests*100):.1f}%")

        print(f"\n📈 Performance Summary:")
        for agent_type, result in results.items():
            if result.get("success"):
                response_time = result.get("response_time", 0)
                print(f"   {agent_type}: {response_time:.2f}s")

        print(f"\n🎯 Recommendations:")
        if successful_tests == total_tests:
            print("   ✅ All agent types are working correctly!")
            print("   ✅ Enhanced RAG system is functional")
            print("   ✅ Ready for production use")
        else:
            print("   ⚠️  Some agent types need attention")
            print("   🔧 Check failed tests above for details")


def main():
    """Main test function"""
    print("🧪 AI Assistant Agent Testing Tool")
    print("=" * 50)

    # Initialize tester
    tester = AgentTester(
        base_url="http://localhost:8000",
        username="requiemcreatif@gmail.com",
        password="fucxaj-kymvYk-rakwa4",
    )

    # Run comprehensive tests
    tester.run_comprehensive_test()


if __name__ == "__main__":
    main()
