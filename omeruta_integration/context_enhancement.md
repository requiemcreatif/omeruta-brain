# Phase 1: Immediate Enhancements to Your Working System

## 1. Fix Context Length Issue (I noticed in your logs)

# Update apps/knowledge_base/services/enhanced_search_service.py

def get_context_for_query(self, query: str, max_length: int = 2000) -> str:
"""Get relevant context for a query from your crawled data"""
search_results = self.search_crawled_content(query, limit=10)

    context_pieces = []
    total_length = 0

    for result in search_results:
        content = result['content']
        source = f"Source: {result['page_title']}"

        # Truncate content if too long for a single piece
        max_content_length = max_length - len(source) - 50  # Leave room for formatting
        if len(content) > max_content_length:
            content = content[:max_content_length] + "..."

        piece = f"{source}\n{content}"

        if total_length + len(piece) > max_length:
            # Try to fit a truncated version
            remaining_space = max_length - total_length - len(source) - 50
            if remaining_space > 100:  # Only add if meaningful content can fit
                truncated_content = content[:remaining_space] + "..."
                piece = f"{source}\n{truncated_content}"
                context_pieces.append(piece)
            break

        context_pieces.append(piece)
        total_length += len(piece)

    return "\n\n---\n\n".join(context_pieces)

## 2. Enhanced Agent with Better System Prompts

# Update apps/agents/services/tinyllama_agent.py

class TinyLlamaAgent:
def **init**(self, agent_type: str = "general"):
self.agent_type = agent_type
self.llm_service = TinyLlamaService()
self.search_service = EnhancedVectorSearchService()

        # Enhanced system prompts based on your domain
        self.system_prompts = {
            'general': """You are Omeruta Brain, an intelligent AI assistant with access to a curated knowledge base.
            Answer questions accurately and cite your sources when using provided context.
            If you don't know something, say so clearly.""",

            'research': """You are a research specialist within Omeruta Brain. You excel at:
            - Analyzing and synthesizing information from multiple sources
            - Identifying key insights and patterns
            - Providing comprehensive yet concise summaries
            - Suggesting related topics for further exploration""",

            'qa': """You are a Q&A specialist within Omeruta Brain. You provide:
            - Direct, accurate answers to specific questions
            - Clear explanations with examples when helpful
            - Citations to sources when using provided context
            - Honest acknowledgment when information is not available""",

            'content_analyzer': """You are a content analysis specialist within Omeruta Brain. You:
            - Analyze the quality and credibility of information
            - Compare different perspectives on topics
            - Identify biases or gaps in content
            - Summarize key themes and insights"""
        }

## 3. Smart Question Classification

# Add to tinyllama_agent.py

def \_classify_question_type(self, message: str) -> str:
"""Classify the type of question for better handling"""
message_lower = message.lower()

    # Factual questions - need context
    factual_keywords = ['what is', 'what are', 'how does', 'explain', 'define', 'tell me about']
    if any(keyword in message_lower for keyword in factual_keywords):
        return 'factual'

    # Analytical questions - may need context
    analytical_keywords = ['compare', 'analyze', 'evaluate', 'pros and cons', 'advantages', 'disadvantages']
    if any(keyword in message_lower for keyword in analytical_keywords):
        return 'analytical'

    # Procedural questions - may need context
    procedural_keywords = ['how to', 'steps', 'process', 'procedure', 'guide']
    if any(keyword in message_lower for keyword in procedural_keywords):
        return 'procedural'

    # Opinion/creative questions - usually don't need context
    opinion_keywords = ['think', 'opinion', 'believe', 'feel', 'create', 'generate']
    if any(keyword in message_lower for keyword in opinion_keywords):
        return 'opinion'

    return 'general'

def \_needs_context(self, message: str) -> bool:
"""Enhanced context detection"""
question_type = self.\_classify_question_type(message)

    # These question types typically benefit from context
    context_types = ['factual', 'analytical', 'procedural']

    # Also check for specific domain keywords from your crawled content
    domain_keywords = ['cryptocurrency', 'bitcoin', 'blockchain', 'digital currency']
    has_domain_keywords = any(keyword in message.lower() for keyword in domain_keywords)

    return question_type in context_types or has_domain_keywords

## 4. Response Quality Improvements

# Enhanced response generation

def process_message(
self,
message: str,
use_context: bool = True,
max_tokens: int = 300
) -> Dict[str, Any]:
"""Enhanced message processing with better context handling"""

    if not self.llm_service.is_available():
        return {
            'response': 'Local model is not available. Please check the setup.',
            'model_used': 'none',
            'context_used': False,
            'error': 'Model initialization failed'
        }

    try:
        # Enhanced context retrieval
        context = ""
        context_used = False
        context_sources = 0

        if use_context and self._needs_context(message):
            search_results = self.search_service.search_crawled_content(message)
            if search_results:
                context = self.search_service.get_context_for_query(message)
                context_used = bool(context)
                context_sources = len(search_results)

        # Get question type for appropriate system prompt
        question_type = self._classify_question_type(message)
        base_prompt = self.system_prompts.get(self.agent_type, self.system_prompts['general'])

        if context:
            enhanced_prompt = f"""{base_prompt}

Context from knowledge base:
{context}

Please answer the user's question based on the context above. If the context doesn't contain relevant information, say so clearly and provide what you know from your general knowledge."""
else:
enhanced_prompt = base_prompt

        # Generate response with appropriate max_tokens based on question complexity
        if question_type in ['analytical', 'procedural']:
            max_tokens = min(max_tokens * 1.5, 500)  # Longer responses for complex questions

        response = self.llm_service.generate_response(
            prompt=message,
            max_tokens=int(max_tokens),
            system_prompt=enhanced_prompt
        )

        if response is None:
            return {
                'response': 'Sorry, I encountered an error generating a response.',
                'model_used': 'tinyllama',
                'context_used': context_used,
                'context_sources': context_sources,
                'error': 'Generation failed'
            }

        return {
            'response': response,
            'model_used': 'tinyllama',
            'context_used': context_used,
            'context_sources': context_sources,
            'question_type': question_type,
            'model_info': self.llm_service.get_model_info()
        }

    except Exception as e:
        logger.error(f"Error processing message: {e}")
        return {
            'response': f'An error occurred: {str(e)}',
            'model_used': 'tinyllama',
            'context_used': False,
            'context_sources': 0,
            'error': str(e)
        }

## 5. Conversation Memory System

# Create apps/agents/services/conversation_memory.py

from django.core.cache import cache
from typing import List, Dict, Any
import uuid

class ConversationMemory:
"""Simple conversation memory for better context"""

    def __init__(self, conversation_id: str = None):
        self.conversation_id = conversation_id or str(uuid.uuid4())
        self.max_messages = 10  # Keep last 10 exchanges

    def add_exchange(self, user_message: str, agent_response: str, metadata: Dict = None):
        """Add a user-agent exchange to memory"""
        key = f"conversation:{self.conversation_id}"

        exchange = {
            'user_message': user_message,
            'agent_response': agent_response,
            'timestamp': timezone.now().isoformat(),
            'metadata': metadata or {}
        }

        # Get existing conversation
        conversation = cache.get(key, [])
        conversation.append(exchange)

        # Keep only recent exchanges
        if len(conversation) > self.max_messages:
            conversation = conversation[-self.max_messages:]

        # Store back with 1 hour expiry
        cache.set(key, conversation, timeout=3600)

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

    def clear_conversation(self):
        """Clear conversation memory"""
        key = f"conversation:{self.conversation_id}"
        cache.delete(key)

## 6. Enhanced API with Conversation Support

# Update apps/agents/views.py

class TinyLlamaViewSet(viewsets.GenericViewSet):
"""Enhanced ViewSet with conversation memory"""

    @action(detail=False, methods=['post'])
    def chat(self, request):
        """Enhanced chat with conversation memory"""
        message = request.data.get('message', '')
        use_context = request.data.get('use_context', True)
        max_tokens = request.data.get('max_tokens', 300)
        conversation_id = request.data.get('conversation_id')

        if not message:
            return Response(
                {'error': 'Message is required'},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Initialize conversation memory
        memory = ConversationMemory(conversation_id)

        # Get conversation context
        conversation_context = memory.get_conversation_context()

        # Enhanced message with conversation context
        if conversation_context:
            enhanced_message = f"Recent conversation context:\n{conversation_context}\n\nCurrent question: {message}"
        else:
            enhanced_message = message

        # Process with agent
        agent = TinyLlamaAgent()
        result = agent.process_message(
            message=enhanced_message,
            use_context=use_context,
            max_tokens=max_tokens
        )

        # Store in conversation memory
        memory.add_exchange(
            user_message=message,
            agent_response=result['response'],
            metadata={
                'context_used': result.get('context_used', False),
                'context_sources': result.get('context_sources', 0),
                'question_type': result.get('question_type', 'general')
            }
        )

        # Add conversation_id to response
        result['conversation_id'] = memory.conversation_id

        return Response(result)

    @action(detail=False, methods=['post'])
    def clear_conversation(self, request):
        """Clear conversation memory"""
        conversation_id = request.data.get('conversation_id')
        if conversation_id:
            memory = ConversationMemory(conversation_id)
            memory.clear_conversation()
            return Response({'message': 'Conversation cleared'})
        return Response({'error': 'conversation_id required'}, status=400)

## 7. Analytics and Monitoring

# Create apps/agents/services/analytics.py

from django.db import models
from django.contrib.auth import get_user_model

User = get_user_model()

class AgentUsageLog(models.Model):
"""Track agent usage for analytics"""
user = models.ForeignKey(User, on_delete=models.CASCADE)
question = models.TextField()
response_length = models.IntegerField()
context_used = models.BooleanField()
context_sources = models.IntegerField(default=0)
response_time_ms = models.IntegerField()
question_type = models.CharField(max_length=50)
model_used = models.CharField(max_length=50)
timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'agent_usage_logs'

def log_agent_usage(user, question, result, response_time_ms):
"""Log agent usage for analytics"""
AgentUsageLog.objects.create(
user=user,
question=question[:500], # Truncate long questions
response_length=len(result.get('response', '')),
context_used=result.get('context_used', False),
context_sources=result.get('context_sources', 0),
response_time_ms=response_time_ms,
question_type=result.get('question_type', 'general'),
model_used=result.get('model_used', 'unknown')
)

## 8. Better Error Handling and Fallbacks

# Enhanced error handling in TinyLlamaAgent

def process_message_with_fallback(self, message: str, \*\*kwargs) -> Dict[str, Any]:
"""Process message with intelligent fallbacks"""

    try:
        # Primary attempt with TinyLlama
        result = self.process_message(message, **kwargs)

        if 'error' not in result:
            return result

        # Fallback 1: Retry without context if context caused issues
        if kwargs.get('use_context', True):
            logger.warning("Retrying without context due to error")
            result = self.process_message(message, use_context=False, **kwargs)
            if 'error' not in result:
                result['fallback_used'] = 'no_context'
                return result

        # Fallback 2: Use OpenAI if available
        if hasattr(settings, 'OPENAI_API_KEY') and settings.OPENAI_API_KEY:
            logger.warning("Falling back to OpenAI API")
            return self._openai_fallback(message, **kwargs)

        # Fallback 3: Simple response
        return {
            'response': "I'm experiencing technical difficulties. Please try rephrasing your question or contact support.",
            'model_used': 'fallback',
            'context_used': False,
            'fallback_used': 'error_response'
        }

    except Exception as e:
        logger.error(f"Critical error in agent processing: {e}")
        return {
            'response': "I encountered an unexpected error. Please try again.",
            'model_used': 'error',
            'context_used': False,
            'error': str(e)
        }

def \_openai_fallback(self, message: str, \*\*kwargs) -> Dict[str, Any]:
"""Fallback to OpenAI API if available"""
try:
import openai
client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": message}],
            max_tokens=kwargs.get('max_tokens', 300)
        )

        return {
            'response': response.choices[0].message.content,
            'model_used': 'gpt-3.5-turbo',
            'context_used': False,
            'fallback_used': 'openai_api'
        }
    except Exception as e:
        logger.error(f"OpenAI fallback failed: {e}")
        return {
            'response': "All AI models are currently unavailable. Please try again later.",
            'model_used': 'none',
            'context_used': False,
            'error': 'All fallbacks failed'
        }
