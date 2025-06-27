from django.core.cache import cache
from django.utils import timezone
from typing import List, Dict, Any
import uuid
import logging

logger = logging.getLogger(__name__)


class ConversationMemory:
    """Simple conversation memory for better context"""

    def __init__(self, conversation_id: str = None):
        self.conversation_id = conversation_id or str(uuid.uuid4())
        self.max_messages = 10  # Keep last 10 exchanges

    def add_exchange(
        self, user_message: str, agent_response: str, metadata: Dict = None
    ):
        """Add a user-agent exchange to memory"""
        key = f"conversation:{self.conversation_id}"

        exchange = {
            "user_message": user_message,
            "agent_response": agent_response,
            "timestamp": timezone.now().isoformat(),
            "metadata": metadata or {},
        }

        # Get existing conversation
        conversation = cache.get(key, [])
        conversation.append(exchange)

        # Keep only recent exchanges
        if len(conversation) > self.max_messages:
            conversation = conversation[-self.max_messages :]

        # Store back with 1 hour expiry
        cache.set(key, conversation, timeout=3600)
        logger.info(f"Added conversation exchange to {self.conversation_id}")

    def get_conversation_context(self) -> str:
        """Get conversation context for the model"""
        key = f"conversation:{self.conversation_id}"
        conversation = cache.get(key, [])

        if not conversation:
            return ""

        context_parts = []
        for exchange in conversation[-3:]:  # Last 3 exchanges
            context_parts.append(f"Previous Q: {exchange['user_message'][:100]}")
            context_parts.append(f"Previous A: {exchange['agent_response'][:100]}")

        return "\n".join(context_parts)

    def get_full_conversation(self) -> List[Dict[str, Any]]:
        """Get full conversation history"""
        key = f"conversation:{self.conversation_id}"
        return cache.get(key, [])

    def clear_conversation(self):
        """Clear conversation memory"""
        key = f"conversation:{self.conversation_id}"
        cache.delete(key)
        logger.info(f"Cleared conversation {self.conversation_id}")

    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the conversation"""
        conversation = self.get_full_conversation()

        if not conversation:
            return {
                "total_exchanges": 0,
                "conversation_started": None,
                "last_activity": None,
                "topics_discussed": [],
            }

        # Extract topics from recent messages (simple keyword extraction)
        recent_messages = [ex["user_message"] for ex in conversation[-5:]]
        topics = []
        for message in recent_messages:
            # Simple topic extraction - can be enhanced
            words = message.lower().split()
            topic_keywords = [
                "bitcoin",
                "cryptocurrency",
                "blockchain",
                "technology",
                "ai",
                "data",
            ]
            found_topics = [word for word in words if word in topic_keywords]
            topics.extend(found_topics)

        return {
            "total_exchanges": len(conversation),
            "conversation_started": (
                conversation[0]["timestamp"] if conversation else None
            ),
            "last_activity": conversation[-1]["timestamp"] if conversation else None,
            "topics_discussed": list(set(topics)),
        }
