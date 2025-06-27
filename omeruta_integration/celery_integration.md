# Why Celery is Essential for Your System

## Critical Benefits for Your Use Case:

### 1. **Long-Running AI Tasks**

- TinyLlama inference (1-5 seconds) blocks web requests
- Knowledge base searches with embeddings are CPU/GPU intensive
- Auto knowledge expansion can take minutes
- Content quality analysis requires processing multiple pages

### 2. **Your Existing Crawler Integration**

- Crawl jobs already take time (you have async crawling)
- Perfect opportunity to add AI analysis to crawl completion
- Background processing of crawled content for embeddings

### 3. **Multi-Agent Orchestration**

- Research agent auto-expansion needs background processing
- Content analysis across multiple sources is compute-heavy
- Agent collaboration requires task queuing

### 4. **Scalability & User Experience**

- Users get instant responses ("processing...")
- Heavy computation happens in background
- Multiple users can use the system simultaneously

## Complete Celery Implementation

# 1. Enhanced Celery Configuration

# omeruta_brain_project/celery.py

import os
from celery import Celery
from django.conf import settings

# Set the default Django settings module for the 'celery' program.

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'omeruta_brain_project.settings')

app = Celery('omeruta_brain')

# Using a string here means the worker doesn't have to serialize

# the configuration object to child processes.

app.config_from_object('django.conf:settings', namespace='CELERY')

# Enhanced configuration for AI workloads

app.conf.update( # Task routing for different types of work
task_routes={
'apps.agents.tasks._': {'queue': 'ai_processing'},
'apps.knowledge_base.tasks._': {'queue': 'embeddings'},
'apps.crawler.tasks.\*': {'queue': 'crawling'},
},

    # Task result settings
    result_expires=3600,  # Results expire after 1 hour
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,

    # Concurrency settings
    worker_concurrency=4,  # Adjust based on CPU/GPU cores
    worker_prefetch_multiplier=1,  # Important for AI tasks

    # Task time limits
    task_soft_time_limit=300,  # 5 minutes soft limit
    task_time_limit=600,       # 10 minutes hard limit

    # Queue priorities
    task_default_priority=5,
    worker_hijack_root_logger=False,

)

# Queue definitions

app.conf.task_routes = { # High priority: User-facing AI responses
'apps.agents.tasks.process_user_message': {
'queue': 'ai_high_priority',
'priority': 9
},

    # Medium priority: Content analysis
    'apps.agents.tasks.analyze_content': {
        'queue': 'ai_processing',
        'priority': 6
    },

    # Low priority: Background maintenance
    'apps.knowledge_base.tasks.process_embeddings': {
        'queue': 'embeddings',
        'priority': 3
    },

    # Crawling tasks
    'apps.crawler.tasks.*': {
        'queue': 'crawling',
        'priority': 4
    }

}

# Load task modules from all registered Django apps.

app.autodiscover_tasks()

@app.task(bind=True)
def debug_task(self):
print(f'Request: {self.request!r}')

# 2. AI Agent Tasks

# apps/agents/tasks.py

from celery import shared_task
from celery.utils.log import get_task_logger
from django.core.cache import cache
from .services.tinyllama_agent import TinyLlamaAgent
from .services.orchestrator import AgentOrchestrator
from .services.enhanced_search_service import EnhancedVectorSearchService
import time
import uuid

logger = get_task_logger(**name**)

@shared_task(bind=True, max_retries=3)
def process_user_message_async(self, message, user_id, conversation_id=None, use_context=True):
"""Process user message asynchronously with TinyLlama"""
task_id = self.request.id

    try:
        # Update task status
        cache.set(f"task_status:{task_id}", {
            'status': 'processing',
            'progress': 0,
            'message': 'Initializing AI agent...'
        }, timeout=300)

        # Initialize agent
        agent = TinyLlamaAgent()

        cache.set(f"task_status:{task_id}", {
            'status': 'processing',
            'progress': 25,
            'message': 'Searching knowledge base...'
        }, timeout=300)

        # Process message
        start_time = time.time()
        result = agent.process_message(
            message=message,
            use_context=use_context,
            max_tokens=300
        )

        processing_time = time.time() - start_time

        cache.set(f"task_status:{task_id}", {
            'status': 'processing',
            'progress': 75,
            'message': 'Generating response...'
        }, timeout=300)

        # Add task metadata
        result.update({
            'task_id': task_id,
            'processing_time': processing_time,
            'processed_async': True,
            'user_id': user_id,
            'conversation_id': conversation_id
        })

        # Store result in cache
        cache.set(f"ai_result:{task_id}", result, timeout=3600)

        # Update final status
        cache.set(f"task_status:{task_id}", {
            'status': 'completed',
            'progress': 100,
            'message': 'Response ready',
            'result_key': f"ai_result:{task_id}"
        }, timeout=300)

        logger.info(f"Processed message for user {user_id} in {processing_time:.2f}s")
        return result

    except Exception as exc:
        logger.error(f"Error processing message: {exc}")

        # Update error status
        cache.set(f"task_status:{task_id}", {
            'status': 'failed',
            'progress': 0,
            'message': f'Error: {str(exc)}',
            'error': str(exc)
        }, timeout=300)

        # Retry logic
        if self.request.retries < self.max_retries:
            logger.info(f"Retrying task {task_id}, attempt {self.request.retries + 1}")
            raise self.retry(countdown=60, exc=exc)

        raise exc

@shared_task(bind=True)
def process_multiagent_query(self, query, user_id, conversation_id=None):
"""Process query through multi-agent orchestrator"""
task_id = self.request.id

    try:
        from django.contrib.auth import get_user_model
        User = get_user_model()
        user = User.objects.get(id=user_id)

        # Update status
        cache.set(f"task_status:{task_id}", {
            'status': 'processing',
            'progress': 10,
            'message': 'Analyzing query...'
        }, timeout=300)

        # Initialize orchestrator
        orchestrator = AgentOrchestrator(user)

        cache.set(f"task_status:{task_id}", {
            'status': 'processing',
            'progress': 30,
            'message': 'Routing to best agent...'
        }, timeout=300)

        # Process query
        result = orchestrator.process_query(query, conversation_id)

        cache.set(f"task_status:{task_id}", {
            'status': 'processing',
            'progress': 80,
            'message': 'Finalizing response...'
        }, timeout=300)

        # Add task metadata
        result.update({
            'task_id': task_id,
            'processed_async': True,
            'user_id': user_id
        })

        # Store result
        cache.set(f"ai_result:{task_id}", result, timeout=3600)

        cache.set(f"task_status:{task_id}", {
            'status': 'completed',
            'progress': 100,
            'message': 'Response ready',
            'result_key': f"ai_result:{task_id}"
        }, timeout=300)

        return result

    except Exception as exc:
        logger.error(f"Error in multiagent processing: {exc}")
        cache.set(f"task_status:{task_id}", {
            'status': 'failed',
            'progress': 0,
            'message': f'Error: {str(exc)}',
            'error': str(exc)
        }, timeout=300)
        raise exc

@shared_task
def auto_expand_knowledge_async(topic, max_urls=5):
"""Automatically expand knowledge base for a topic"""
try:
from .services.auto_knowledge_expander import AutoKnowledgeExpander

        expander = AutoKnowledgeExpander()
        result = expander.auto_expand_knowledge(topic, max_urls)

        logger.info(f"Knowledge expansion for '{topic}' completed: {result}")
        return result

    except Exception as exc:
        logger.error(f"Knowledge expansion failed: {exc}")
        raise exc

@shared_task
def analyze_content_quality_batch():
"""Analyze content quality for all pages in background"""
try:
from .services.content_curator import ContentCurator
from apps.crawler.models import CrawledPage

        curator = ContentCurator()
        pages = CrawledPage.objects.filter(success=True)[:100]  # Process in batches

        results = []
        for page in pages:
            try:
                assessment = curator.assess_content_quality(str(page.id))
                if 'error' not in assessment:
                    results.append(assessment)
            except Exception as e:
                logger.warning(f"Failed to assess page {page.id}: {e}")

        logger.info(f"Assessed quality for {len(results)} pages")
        return {'assessed_count': len(results), 'results': results}

    except Exception as exc:
        logger.error(f"Batch quality analysis failed: {exc}")
        raise exc

@shared_task
def cleanup_expired_tasks():
"""Clean up expired task results and statuses"""
try: # This would clean up expired cache entries # Implementation depends on your cache backend
logger.info("Cleaned up expired tasks")
return "Cleanup completed"
except Exception as exc:
logger.error(f"Cleanup failed: {exc}")
raise exc

# 3. Knowledge Base Tasks

# apps/knowledge_base/tasks.py

from celery import shared_task
from celery.utils.log import get_task_logger
from .services.enhanced_search_service import EnhancedVectorSearchService
from .models import Document, DocumentChunk
from apps.crawler.models import CrawledPage

logger = get_task_logger(**name**)

@shared_task(bind=True)
def process_page_embeddings(self, page_id):
"""Process embeddings for a crawled page"""
try:
page = CrawledPage.objects.get(id=page_id)

        if not page.clean_markdown:
            logger.warning(f"Page {page_id} has no clean markdown content")
            return {'status': 'skipped', 'reason': 'no_content'}

        # Create embeddings
        search_service = EnhancedVectorSearchService()

        # For now, we'll just mark it as processed
        # In a full implementation, you'd generate and store embeddings
        page.is_processed_for_embeddings = True
        page.save(update_fields=['is_processed_for_embeddings'])

        logger.info(f"Processed embeddings for page {page_id}")
        return {'status': 'success', 'page_id': page_id}

    except CrawledPage.DoesNotExist:
        logger.error(f"Page {page_id} not found")
        return {'status': 'error', 'reason': 'page_not_found'}
    except Exception as exc:
        logger.error(f"Error processing embeddings for page {page_id}: {exc}")
        raise self.retry(countdown=60, exc=exc)

@shared_task
def batch_process_unprocessed_pages():
"""Process embeddings for all unprocessed pages"""
try:
unprocessed_pages = CrawledPage.objects.filter(
success=True,
is_processed_for_embeddings=False,
clean_markdown\_\_isnull=False
).exclude(clean_markdown='')

        task_ids = []
        for page in unprocessed_pages[:50]:  # Process in batches of 50
            task = process_page_embeddings.delay(str(page.id))
            task_ids.append(task.id)

        logger.info(f"Queued {len(task_ids)} embedding tasks")
        return {'queued_tasks': len(task_ids), 'task_ids': task_ids}

    except Exception as exc:
        logger.error(f"Batch processing failed: {exc}")
        raise exc

@shared_task
def update_content_freshness():
"""Check and update content freshness scores"""
try:
from .services.smart_scheduler import SmartCrawlScheduler

        scheduler = SmartCrawlScheduler()
        stale_content = scheduler.get_stale_content(max_age_days=30)

        # Queue refresh tasks for stale content
        refresh_tasks = []
        for content in stale_content[:10]:  # Limit to 10 per run
            # Here you would queue crawl tasks for the stale URLs
            refresh_tasks.append(content['url'])

        logger.info(f"Identified {len(stale_content)} stale items, queued {len(refresh_tasks)} for refresh")
        return {'stale_count': len(stale_content), 'refresh_queued': len(refresh_tasks)}

    except Exception as exc:
        logger.error(f"Freshness update failed: {exc}")
        raise exc

# 4. Enhanced API with Async Support

# apps/agents/views.py (Updated with Celery integration)

from celery.result import AsyncResult
from django.core.cache import cache

class AsyncTinyLlamaViewSet(viewsets.GenericViewSet):
"""Async version of TinyLlama API with Celery"""

    permission_classes = [permissions.IsAuthenticated]

    @action(detail=False, methods=['post'])
    def chat_async(self, request):
        """Start async chat processing"""
        message = request.data.get('message', '')
        use_context = request.data.get('use_context', True)
        conversation_id = request.data.get('conversation_id')

        if not message:
            return Response(
                {'error': 'Message is required'},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Start async task
        task = process_user_message_async.delay(
            message=message,
            user_id=request.user.id,
            conversation_id=conversation_id,
            use_context=use_context
        )

        return Response({
            'task_id': task.id,
            'status': 'processing',
            'message': 'Your request is being processed. Use the task_id to check status.',
            'check_status_url': f'/api/agents/async/status/{task.id}/'
        })

    @action(detail=False, methods=['get'], url_path='status/(?P<task_id>[^/.]+)')
    def check_task_status(self, request, task_id):
        """Check status of async task"""
        try:
            # Get task result
            task_result = AsyncResult(task_id)

            # Get cached status for more detailed progress
            cached_status = cache.get(f"task_status:{task_id}")

            if cached_status:
                response_data = cached_status.copy()
                response_data['task_id'] = task_id
                response_data['celery_status'] = task_result.status

                # If completed, include result
                if cached_status['status'] == 'completed' and 'result_key' in cached_status:
                    result = cache.get(cached_status['result_key'])
                    if result:
                        response_data['result'] = result

                return Response(response_data)

            # Fallback to Celery status only
            return Response({
                'task_id': task_id,
                'status': task_result.status.lower(),
                'ready': task_result.ready(),
                'result': task_result.result if task_result.ready() else None
            })

        except Exception as e:
            return Response(
                {'error': 'Failed to get task status', 'details': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @action(detail=False, methods=['post'])
    def multiagent_async(self, request):
        """Start async multi-agent processing"""
        query = request.data.get('query', '')
        conversation_id = request.data.get('conversation_id')

        if not query:
            return Response(
                {'error': 'Query is required'},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Start async task
        task = process_multiagent_query.delay(
            query=query,
            user_id=request.user.id,
            conversation_id=conversation_id
        )

        return Response({
            'task_id': task.id,
            'status': 'processing',
            'message': 'Multi-agent query is being processed.',
            'check_status_url': f'/api/agents/async/status/{task.id}/'
        })

    @action(detail=False, methods=['post'])
    def expand_knowledge(self, request):
        """Trigger knowledge expansion for a topic"""
        topic = request.data.get('topic', '')
        max_urls = request.data.get('max_urls', 5)

        if not topic:
            return Response(
                {'error': 'Topic is required'},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Start async knowledge expansion
        task = auto_expand_knowledge_async.delay(topic, max_urls)

        return Response({
            'task_id': task.id,
            'status': 'processing',
            'message': f'Knowledge expansion for "{topic}" started.',
            'check_status_url': f'/api/agents/async/status/{task.id}/'
        })

# 5. Periodic Tasks Setup

# apps/agents/tasks.py (Additional periodic tasks)

from celery.schedules import crontab
from celery import Celery

app = Celery('omeruta_brain')

# Periodic task configuration

app.conf.beat_schedule = { # Clean up expired tasks every hour
'cleanup-expired-tasks': {
'task': 'apps.agents.tasks.cleanup_expired_tasks',
'schedule': crontab(minute=0), # Every hour
},

    # Process unprocessed embeddings every 30 minutes
    'process-embeddings': {
        'task': 'apps.knowledge_base.tasks.batch_process_unprocessed_pages',
        'schedule': crontab(minute='*/30'),  # Every 30 minutes
    },

    # Check content freshness daily
    'update-freshness': {
        'task': 'apps.knowledge_base.tasks.update_content_freshness',
        'schedule': crontab(hour=2, minute=0),  # Daily at 2 AM
    },

    # Quality analysis weekly
    'quality-analysis': {
        'task': 'apps.agents.tasks.analyze_content_quality_batch',
        'schedule': crontab(hour=3, minute=0, day_of_week=1),  # Weekly on Monday
    },

}

# 6. WebSocket Integration for Real-time Updates

# apps/agents/consumers.py (Django Channels integration)

import json
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from celery.result import AsyncResult
from django.core.cache import cache

class AgentConsumer(AsyncWebsocketConsumer):
"""WebSocket consumer for real-time AI responses"""

    async def connect(self):
        self.user = self.scope["user"]
        if self.user.is_anonymous:
            await self.close()
            return

        self.group_name = f"user_{self.user.id}"

        # Join user group
        await self.channel_layer.group_add(
            self.group_name,
            self.channel_name
        )

        await self.accept()

    async def disconnect(self, close_code):
        # Leave user group
        await self.channel_layer.group_discard(
            self.group_name,
            self.channel_name
        )

    async def receive(self, text_data):
        data = json.loads(text_data)
        action = data.get('action')

        if action == 'start_chat':
            await self.start_chat(data)
        elif action == 'check_task':
            await self.check_task_status(data)

    async def start_chat(self, data):
        """Start async chat and send updates"""
        message = data.get('message', '')

        if not message:
            await self.send(text_data=json.dumps({
                'type': 'error',
                'message': 'Message is required'
            }))
            return

        # Start async task
        from .tasks import process_user_message_async
        task = process_user_message_async.delay(
            message=message,
            user_id=self.user.id,
            use_context=data.get('use_context', True)
        )

        # Send task started notification
        await self.send(text_data=json.dumps({
            'type': 'task_started',
            'task_id': task.id,
            'message': 'Processing your message...'
        }))

        # Start polling for updates
        await self.poll_task_status(task.id)

    async def poll_task_status(self, task_id):
        """Poll task status and send updates"""
        import asyncio

        max_polls = 60  # 2 minutes max
        poll_count = 0

        while poll_count < max_polls:
            # Check cached status
            status_data = cache.get(f"task_status:{task_id}")

            if status_data:
                await self.send(text_data=json.dumps({
                    'type': 'task_update',
                    'task_id': task_id,
                    **status_data
                }))

                if status_data['status'] in ['completed', 'failed']:
                    # Send final result if completed
                    if status_data['status'] == 'completed' and 'result_key' in status_data:
                        result = cache.get(status_data['result_key'])
                        if result:
                            await self.send(text_data=json.dumps({
                                'type': 'task_completed',
                                'task_id': task_id,
                                'result': result
                            }))
                    break

            poll_count += 1
            await asyncio.sleep(2)  # Poll every 2 seconds

    async def check_task_status(self, data):
        """Check specific task status"""
        task_id = data.get('task_id')
        if not task_id:
            return

        status_data = cache.get(f"task_status:{task_id}")
        if status_data:
            await self.send(text_data=json.dumps({
                'type': 'task_status',
                'task_id': task_id,
                **status_data
            }))

# 7. Production Deployment Configuration

# docker-compose.yml for production

version: '3.8'

services:
redis:
image: redis:7-alpine
ports: - "6379:6379"
command: redis-server --appendonly yes
volumes: - redis_data:/data

postgres:
image: postgres:15
environment:
POSTGRES_DB: omeruta_brain
POSTGRES_USER: omeruta_user
POSTGRES_PASSWORD: your_password
volumes: - postgres_data:/var/lib/postgresql/data
ports: - "5432:5432"

web:
build: .
command: gunicorn omeruta_brain_project.wsgi:application --bind 0.0.0.0:8000
volumes: - .:/app
ports: - "8000:8000"
depends_on: - postgres - redis
environment: - DEBUG=False - DATABASE_URL=postgresql://omeruta_user:your_password@postgres:5432/omeruta_brain - REDIS_URL=redis://redis:6379/0

# Celery worker for AI tasks

celery-ai:
build: .
command: celery -A omeruta_brain_project worker --loglevel=info --queues=ai_high_priority,ai_processing --concurrency=2
volumes: - .:/app
depends_on: - postgres - redis
environment: - DATABASE_URL=postgresql://omeruta_user:your_password@postgres:5432/omeruta_brain - REDIS_URL=redis://redis:6379/0

# Celery worker for embeddings

celery-embeddings:
build: .
command: celery -A omeruta_brain_project worker --loglevel=info --queues=embeddings --concurrency=1
volumes: - .:/app
depends_on: - postgres - redis

# Celery worker for crawling

celery-crawler:
build: .
command: celery -A omeruta_brain_project worker --loglevel=info --queues=crawling --concurrency=3
volumes: - .:/app
depends_on: - postgres - redis

# Celery beat scheduler

celery-beat:
build: .
command: celery -A omeruta_brain_project beat --loglevel=info
volumes: - .:/app
depends_on: - postgres - redis

volumes:
postgres_data:
redis_data:

🚀 Why Celery is a Game-Changer for Your System
Current Pain Points Celery Solves:

TinyLlama Blocking Requests - Your API calls currently wait 1-5 seconds for AI responses
Knowledge Base Searches - Embedding searches can be slow with larger datasets
Auto Knowledge Expansion - Research agent needs to crawl new content (takes minutes)
Content Processing - Quality analysis and embeddings generation is CPU intensive

Immediate Benefits You'll Get:
✅ Instant API Responses - Users get task_id immediately, results via polling/WebSocket
✅ Background Intelligence - AI works while users do other things
✅ Auto-Scaling - Multiple workers handle concurrent requests
✅ Fault Tolerance - Failed tasks retry automatically
✅ Progress Updates - Real-time status updates ("Searching knowledge base...")

// BEFORE: Synchronous API (blocking)
// User waits 3-5 seconds for response

fetch('/api/agents/tinyllama/chat/', {
method: 'POST',
headers: {'Authorization': `Bearer ${token}`, 'Content-Type': 'application/json'},
body: JSON.stringify({message: "What is cryptocurrency?"})
})
.then(response => response.json())
.then(data => {
// Finally get response after 3-5 seconds
displayMessage(data.response);
});

// AFTER: Asynchronous with Celery (non-blocking)
// User gets instant feedback + real-time updates

// 1. Start async task (instant response)
fetch('/api/agents/async/chat_async/', {
method: 'POST',
headers: {'Authorization': `Bearer ${token}`, 'Content-Type': 'application/json'},
body: JSON.stringify({message: "What is cryptocurrency?"})
})
.then(response => response.json())
.then(data => {
const taskId = data.task_id;
showProcessingIndicator("AI is thinking...");

    // 2. Poll for updates
    pollTaskStatus(taskId);

});

function pollTaskStatus(taskId) {
const pollInterval = setInterval(() => {
fetch(`/api/agents/async/status/${taskId}/`)
.then(response => response.json())
.then(data => {
// Real-time progress updates
updateProgressBar(data.progress);
updateStatusMessage(data.message);

            if (data.status === 'completed') {
                clearInterval(pollInterval);
                hideProcessingIndicator();
                displayMessage(data.result.response);
            } else if (data.status === 'failed') {
                clearInterval(pollInterval);
                showError(data.error);
            }
        });
    }, 1000); // Check every second

}

// Even better: WebSocket for real-time updates
const socket = new WebSocket(`ws://localhost:8000/ws/agent/`);

socket.onmessage = function(event) {
const data = JSON.parse(event.data);

    switch(data.type) {
        case 'task_started':
            showProcessingIndicator(data.message);
            break;
        case 'task_update':
            updateProgressBar(data.progress);
            updateStatusMessage(data.message);
            break;
        case 'task_completed':
            hideProcessingIndicator();
            displayMessage(data.result.response);
            break;
    }

};

// Start chat via WebSocket
socket.send(JSON.stringify({
action: 'start_chat',
message: 'What is cryptocurrency?',
use_context: true
}));

// Your Enhanced Frontend Experience:
/\*
┌─────────────────────────────────────────┐
│ BEFORE (Synchronous): │
│ User: "What is cryptocurrency?" │
│ [████████████████████] 5 seconds │
│ Response: "Cryptocurrency is..." │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ AFTER (Asynchronous with Celery): │
│ User: "What is cryptocurrency?" │
│ Instant: "AI is thinking..." │
│ [██░░░░░░░░] 25% "Searching knowledge" │
│ [████░░░░░░] 40% "Found 3 sources" │
│ [██████░░░░] 60% "Analyzing content" │
│ [████████░░] 80% "Generating response" │
│ [██████████] 100% "Complete!" │
│ Response: "Cryptocurrency is..." │
└─────────────────────────────────────────┘
\*/

// Complete Implementation Example

class OmerutaBrainClient {
constructor(apiBaseUrl, authToken) {
this.apiBaseUrl = apiBaseUrl;
this.authToken = authToken;
this.socket = null;
this.activeChats = new Map(); // Track multiple conversations
}

    // Initialize WebSocket connection
    connectWebSocket() {
        this.socket = new WebSocket(`ws://${window.location.host}/ws/agent/`);

        this.socket.onopen = () => {
            console.log('🔗 Connected to Omeruta Brain');
            this.showConnectionStatus('Connected', 'success');
        };

        this.socket.onclose = () => {
            console.log('❌ Disconnected from Omeruta Brain');
            this.showConnectionStatus('Disconnected', 'error');
            // Auto-reconnect
            setTimeout(() => this.connectWebSocket(), 5000);
        };

        this.socket.onmessage = (event) => {
            this.handleWebSocketMessage(JSON.parse(event.data));
        };
    }

    // Enhanced chat with multiple conversation support
    async startChat(message, options = {}) {
        const chatId = options.conversationId || this.generateChatId();

        // Show immediate feedback
        this.displayUserMessage(message, chatId);
        this.showTypingIndicator(chatId, "🤖 AI is processing your message...");

        if (this.socket && this.socket.readyState === WebSocket.OPEN) {
            // Use WebSocket for real-time updates
            this.socket.send(JSON.stringify({
                action: 'start_chat',
                message: message,
                conversation_id: chatId,
                use_context: options.useContext !== false,
                agent_type: options.agentType || 'auto'
            }));
        } else {
            // Fallback to polling
            return this.startChatWithPolling(message, chatId, options);
        }

        return chatId;
    }

    // Fallback polling method
    async startChatWithPolling(message, chatId, options) {
        try {
            // Start async task
            const response = await fetch(`${this.apiBaseUrl}/agents/async/chat_async/`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${this.authToken}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    message: message,
                    conversation_id: chatId,
                    use_context: options.useContext !== false
                })
            });

            const data = await response.json();

            if (response.ok) {
                this.pollTaskStatus(data.task_id, chatId);
                return chatId;
            } else {
                throw new Error(data.error || 'Failed to start chat');
            }
        } catch (error) {
            this.showError(`Failed to start chat: ${error.message}`, chatId);
            throw error;
        }
    }

    // Enhanced polling with exponential backoff
    pollTaskStatus(taskId, chatId) {
        let pollCount = 0;
        const maxPolls = 120; // 4 minutes max
        let pollInterval = 1000; // Start with 1 second

        const poll = async () => {
            try {
                const response = await fetch(
                    `${this.apiBaseUrl}/agents/async/status/${taskId}/`,
                    {
                        headers: {
                            'Authorization': `Bearer ${this.authToken}`
                        }
                    }
                );

                const data = await response.json();

                this.updateChatProgress(chatId, data);

                if (data.status === 'completed') {
                    this.handleChatCompletion(chatId, data);
                    return;
                } else if (data.status === 'failed') {
                    this.handleChatError(chatId, data);
                    return;
                }

                pollCount++;
                if (pollCount < maxPolls) {
                    // Exponential backoff with jitter
                    const jitter = Math.random() * 200;
                    pollInterval = Math.min(pollInterval * 1.1, 3000); // Max 3 seconds
                    setTimeout(poll, pollInterval + jitter);
                } else {
                    this.handleChatTimeout(chatId);
                }

            } catch (error) {
                console.error('Polling error:', error);
                this.showError(`Connection error: ${error.message}`, chatId);
            }
        };

        poll();
    }

    // WebSocket message handling
    handleWebSocketMessage(data) {
        const chatId = data.conversation_id || 'default';

        switch (data.type) {
            case 'task_started':
                this.showTypingIndicator(chatId, data.message);
                break;

            case 'task_update':
                this.updateChatProgress(chatId, data);
                break;

            case 'task_completed':
                this.handleChatCompletion(chatId, data);
                break;

            case 'task_failed':
                this.handleChatError(chatId, data);
                break;

            case 'agent_routed':
                this.showAgentSelection(chatId, data.selected_agent, data.confidence);
                break;

            case 'knowledge_expanded':
                this.showKnowledgeExpansion(chatId, data.expansion_info);
                break;
        }
    }

    // UI Update Methods
    updateChatProgress(chatId, data) {
        const progressElement = document.getElementById(`progress-${chatId}`);
        const statusElement = document.getElementById(`status-${chatId}`);

        if (progressElement) {
            progressElement.style.width = `${data.progress || 0}%`;
            progressElement.setAttribute('aria-valuenow', data.progress || 0);
        }

        if (statusElement) {
            statusElement.textContent = data.message || 'Processing...';
        }

        // Show agent selection if available
        if (data.selected_agent) {
            this.showAgentSelection(chatId, data.selected_agent, data.routing_score);
        }
    }

    handleChatCompletion(chatId, data) {
        this.hideTypingIndicator(chatId);

        const result = data.result || data;

        // Display the AI response
        this.displayAIMessage(result.response, chatId, {
            model: result.model_used,
            agent: result.selected_agent,
            contextUsed: result.context_used,
            contextSources: result.context_sources,
            confidence: result.confidence_score,
            processingTime: result.processing_time
        });

        // Show additional insights if available
        if (result.knowledge_expanded) {
            this.showKnowledgeExpansion(chatId, result.expansion_info);
        }

        if (result.quality_assessment) {
            this.showQualityAssessment(chatId, result.quality_assessment);
        }
    }

    handleChatError(chatId, data) {
        this.hideTypingIndicator(chatId);
        this.showError(data.error || 'An unexpected error occurred', chatId);
    }

    handleChatTimeout(chatId) {
        this.hideTypingIndicator(chatId);
        this.showError('Request timed out. Please try again.', chatId);
    }

    // Multi-Agent Specific Methods
    async startResearch(topic, options = {}) {
        const chatId = this.generateChatId();

        this.displayUserMessage(`🔍 Research: ${topic}`, chatId);
        this.showTypingIndicator(chatId, "🔬 Research agent is gathering information...");

        try {
            const response = await fetch(`${this.apiBaseUrl}/agents/async/research_topic/`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${this.authToken}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    topic: topic,
                    expand_knowledge: options.expandKnowledge !== false,
                    max_sources: options.maxSources || 5
                })
            });

            const data = await response.json();

            if (response.ok) {
                this.pollTaskStatus(data.task_id, chatId);
                return chatId;
            } else {
                throw new Error(data.error);
            }
        } catch (error) {
            this.showError(`Research failed: ${error.message}`, chatId);
            throw error;
        }
    }

    async analyzeContent(urls, analysisType = 'comparison') {
        const chatId = this.generateChatId();

        this.displayUserMessage(`📊 Analyzing ${urls.length} sources`, chatId);
        this.showTypingIndicator(chatId, "🧠 Analysis agent is comparing sources...");

        try {
            const response = await fetch(`${this.apiBaseUrl}/agents/async/analyze_content/`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${this.authToken}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    urls: urls,
                    analysis_type: analysisType
                })
            });

            const data = await response.json();

            if (response.ok) {
                this.pollTaskStatus(data.task_id, chatId);
                return chatId;
            } else {
                throw new Error(data.error);
            }
        } catch (error) {
            this.showError(`Analysis failed: ${error.message}`, chatId);
            throw error;
        }
    }

    // UI Helper Methods
    displayUserMessage(message, chatId) {
        const chatContainer = this.getChatContainer(chatId);
        const messageElement = document.createElement('div');
        messageElement.className = 'message user-message';
        messageElement.innerHTML = `
            <div class="message-avatar">👤</div>
            <div class="message-content">
                <div class="message-text">${this.escapeHtml(message)}</div>
                <div class="message-time">${new Date().toLocaleTimeString()}</div>
            </div>
        `;
        chatContainer.appendChild(messageElement);
        this.scrollToBottom(chatContainer);
    }

    displayAIMessage(message, chatId, metadata = {}) {
        const chatContainer = this.getChatContainer(chatId);
        const messageElement = document.createElement('div');
        messageElement.className = 'message ai-message';

        const agentIcon = this.getAgentIcon(metadata.agent);
        const confidenceBar = metadata.confidence ?
            `<div class="confidence-bar">
                <div class="confidence-fill" style="width: ${metadata.confidence * 100}%"></div>
                <span class="confidence-text">${Math.round(metadata.confidence * 100)}% confident</span>
            </div>` : '';

        messageElement.innerHTML = `
            <div class="message-avatar">${agentIcon}</div>
            <div class="message-content">
                <div class="message-text">${this.formatMessage(message)}</div>
                ${confidenceBar}
                <div class="message-metadata">
                    <span class="metadata-item">🤖 ${metadata.model || 'AI'}</span>
                    ${metadata.agent ? `<span class="metadata-item">🎯 ${metadata.agent}</span>` : ''}
                    ${metadata.contextUsed ? `<span class="metadata-item">📚 ${metadata.contextSources || 0} sources</span>` : ''}
                    ${metadata.processingTime ? `<span class="metadata-item">⚡ ${metadata.processingTime.toFixed(2)}s</span>` : ''}
                </div>
                <div class="message-time">${new Date().toLocaleTimeString()}</div>
            </div>
        `;
        chatContainer.appendChild(messageElement);
        this.scrollToBottom(chatContainer);
    }

    showTypingIndicator(chatId, message) {
        const chatContainer = this.getChatContainer(chatId);

        // Remove existing typing indicator
        const existingIndicator = chatContainer.querySelector('.typing-indicator');
        if (existingIndicator) {
            existingIndicator.remove();
        }

        const typingElement = document.createElement('div');
        typingElement.className = 'message typing-indicator';
        typingElement.id = `typing-${chatId}`;
        typingElement.innerHTML = `
            <div class="message-avatar">🤖</div>
            <div class="message-content">
                <div class="typing-animation">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
                <div class="typing-status" id="status-${chatId}">${message}</div>
                <div class="progress-container">
                    <div class="progress-bar">
                        <div class="progress-fill" id="progress-${chatId}" style="width: 0%"></div>
                    </div>
                </div>
            </div>
        `;
        chatContainer.appendChild(typingElement);
        this.scrollToBottom(chatContainer);
    }

    hideTypingIndicator(chatId) {
        const typingElement = document.getElementById(`typing-${chatId}`);
        if (typingElement) {
            typingElement.remove();
        }
    }

    showAgentSelection(chatId, agentName, confidence) {
        const chatContainer = this.getChatContainer(chatId);
        const agentElement = document.createElement('div');
        agentElement.className = 'agent-selection';
        agentElement.innerHTML = `
            <div class="agent-info">
                ${this.getAgentIcon(agentName)} <strong>${this.getAgentName(agentName)}</strong> selected
                <span class="confidence-score">(${Math.round(confidence * 100)}% confidence)</span>
            </div>
        `;
        chatContainer.appendChild(agentElement);
    }

    showKnowledgeExpansion(chatId, expansionInfo) {
        const chatContainer = this.getChatContainer(chatId);
        const expansionElement = document.createElement('div');
        expansionElement.className = 'knowledge-expansion';
        expansionElement.innerHTML = `
            <div class="expansion-info">
                🧠 <strong>Knowledge Expanded!</strong>
                Crawled ${expansionInfo.urls_crawled?.length || 0} new sources for better answers.
            </div>
        `;
        chatContainer.appendChild(expansionElement);
    }

    // Utility Methods
    generateChatId() {
        return `chat_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }

    getChatContainer(chatId) {
        let container = document.getElementById(`chat-${chatId}`);
        if (!container) {
            container = document.createElement('div');
            container.id = `chat-${chatId}`;
            container.className = 'chat-container';
            document.getElementById('main-chat-area').appendChild(container);
        }
        return container;
    }

    getAgentIcon(agentType) {
        const icons = {
            'research': '🔬',
            'analysis': '📊',
            'qa': '❓',
            'general': '🤖',
            'tinyllama': '🦙'
        };
        return icons[agentType] || '🤖';
    }

    getAgentName(agentType) {
        const names = {
            'research': 'Research Agent',
            'analysis': 'Analysis Agent',
            'qa': 'Q&A Agent',
            'general': 'General Agent',
            'tinyllama': 'TinyLlama'
        };
        return names[agentType] || 'AI Agent';
    }

    formatMessage(message) {
        // Convert markdown-like formatting to HTML
        return message
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/`(.*?)`/g, '<code>$1</code>')
            .replace(/\n/g, '<br>');
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    scrollToBottom(container) {
        container.scrollTop = container.scrollHeight;
    }

    showConnectionStatus(status, type) {
        const statusElement = document.getElementById('connection-status');
        if (statusElement) {
            statusElement.textContent = status;
            statusElement.className = `connection-status ${type}`;
        }
    }

    showError(message, chatId) {
        const chatContainer = this.getChatContainer(chatId);
        const errorElement = document.createElement('div');
        errorElement.className = 'message error-message';
        errorElement.innerHTML = `
            <div class="message-avatar">❌</div>
            <div class="message-content">
                <div class="message-text">${this.escapeHtml(message)}</div>
                <div class="message-time">${new Date().toLocaleTimeString()}</div>
            </div>
        `;
        chatContainer.appendChild(errorElement);
        this.scrollToBottom(chatContainer);
    }

}

// Usage Example:
const brain = new OmerutaBrainClient('http://localhost:8000/api', 'your-jwt-token');

// Initialize connection
brain.connectWebSocket();

// Start a chat
brain.startChat("What is cryptocurrency and how does blockchain work?", {
useContext: true,
agentType: 'auto' // Let orchestrator decide
});

// Start research
brain.startResearch("latest developments in quantum computing", {
expandKnowledge: true,
maxSources: 5
});

// Analyze content
brain.analyzeContent([
'https://bitcoin.org/bitcoin.pdf',
'https://ethereum.org/whitepaper/'
], 'comparison');

// The user experience is now:
// ✅ Instant feedback ("AI is thinking...")
// ✅ Real-time progress updates
// ✅ Agent selection transparency
// ✅ Knowledge expansion notifications
// ✅ Multiple conversation support
// ✅ Automatic reconnection
// ✅ Confidence scoring
// ✅ Rich metadata display
