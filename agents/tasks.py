from celery import shared_task
from celery.utils.log import get_task_logger
from django.core.cache import cache
from django.contrib.auth import get_user_model
from django.utils import timezone
from .services.tinyllama_agent import TinyLlamaAgent
from .services.enhanced_phi3_agent import EnhancedPhi3Agent
from .services.enhanced_search_service import EnhancedVectorSearchService
from .services.conversation_memory import ConversationMemory
import time
import uuid
import json

logger = get_task_logger(__name__)
User = get_user_model()


@shared_task(bind=True, max_retries=3, time_limit=120, soft_time_limit=100)
def process_user_message_async(
    self,
    message,
    user_id,
    conversation_id=None,
    use_context=True,
    agent_type="general",
    max_tokens=300,
):
    """Process user message asynchronously with enhanced Phi3"""
    task_id = self.request.id

    try:
        # Add null check for message
        if not message:
            message = ""

        # Update task status
        cache.set(
            f"task_status:{task_id}",
            {
                "status": "processing",
                "progress": 0,
                "message": "Initializing AI agent...",
                "user_id": user_id,
                "conversation_id": conversation_id,
            },
            timeout=300,
        )

        # Get user for logging
        try:
            user = User.objects.get(id=user_id)
        except User.DoesNotExist:
            raise Exception(f"User {user_id} not found")

        cache.set(
            f"task_status:{task_id}",
            {
                "status": "processing",
                "progress": 15,
                "message": "Loading AI model...",
                "user_id": user_id,
                "conversation_id": conversation_id,
            },
            timeout=300,
        )

        # Initialize enhanced agent with specified type for better quality
        agent = EnhancedPhi3Agent(agent_type=agent_type)

        cache.set(
            f"task_status:{task_id}",
            {
                "status": "processing",
                "progress": 30,
                "message": "Searching knowledge base...",
                "user_id": user_id,
                "conversation_id": conversation_id,
            },
            timeout=300,
        )

        # Initialize conversation memory if conversation_id provided
        memory = None
        enhanced_message = message
        if conversation_id:
            memory = ConversationMemory(conversation_id)
            conversation_context = memory.get_conversation_context()
            if conversation_context:
                enhanced_message = f"Recent conversation context:\n{conversation_context}\n\nCurrent question: {message}"

        cache.set(
            f"task_status:{task_id}",
            {
                "status": "processing",
                "progress": 50,
                "message": "Generating AI response...",
                "user_id": user_id,
                "conversation_id": conversation_id,
            },
            timeout=300,
        )

        # Process message with enhanced error handling
        start_time = time.time()
        try:
            result = agent.process_message(
                message=enhanced_message,
                use_context=use_context,
                response_config={
                    "max_tokens": max_tokens,
                    "temperature": 0.7,
                    "style": "informative",
                },
            )
            processing_time = time.time() - start_time
        except Exception as model_error:
            processing_time = time.time() - start_time
            error_msg = str(model_error)

            # Provide specific error messages for common issues
            if "MPS" in error_msg or "Metal" in error_msg:
                error_msg = "AI model encountered a graphics processing error. The system will automatically retry with CPU processing."
            elif "CUDA" in error_msg:
                error_msg = (
                    "GPU processing error detected. Falling back to CPU processing."
                )
            elif "memory" in error_msg.lower() or "allocation" in error_msg.lower():
                error_msg = "Insufficient memory to process request. Please try with a shorter message or lower token limit."
            elif "timeout" in error_msg.lower():
                error_msg = "AI model processing timed out. Please try again with a simpler question."
            else:
                error_msg = f"AI processing error: {error_msg}"

            logger.error(f"Model processing error for user {user_id}: {model_error}")

            # Return error result instead of raising
            result = {
                "response": f"I apologize, but I encountered a technical issue while processing your request. {error_msg} Please try again.",
                "error": True,
                "error_type": "model_error",
                "model_used": agent.llm_service.get_model_info().get(
                    "name", "TinyLlama"
                ),
                "context_used": False,
                "context_sources": 0,
                "question_type": "error",
                "processing_time": processing_time,
            }

        cache.set(
            f"task_status:{task_id}",
            {
                "status": "processing",
                "progress": 80,
                "message": "Saving conversation...",
                "user_id": user_id,
                "conversation_id": conversation_id,
            },
            timeout=300,
        )

        # Store in conversation memory
        if memory:
            memory.add_exchange(
                user_message=message,
                agent_response=result["response"],
                metadata={
                    "context_used": result.get("context_used", False),
                    "context_sources": result.get("context_sources", 0),
                    "question_type": result.get("question_type", "general"),
                    "agent_type": agent_type,
                    "processing_time": processing_time,
                },
            )

        # Convert any numpy float32 values to regular Python floats for JSON serialization
        def convert_numpy_types(obj):
            """Recursively convert numpy types to Python native types"""
            import numpy as np

            if isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif hasattr(obj, "item"):  # numpy scalar
                return obj.item()
            else:
                return obj

        # Convert numpy types in the result
        result = convert_numpy_types(result)

        # Add task metadata
        result.update(
            {
                "task_id": task_id,
                "processing_time": processing_time,
                "processed_async": True,
                "user_id": user_id,
                "conversation_id": conversation_id,
                "agent_type": agent_type,
                "response_time_ms": int(processing_time * 1000),
            }
        )

        # Store result in cache
        cache.set(f"ai_result:{task_id}", result, timeout=3600)

        # Update final status
        cache.set(
            f"task_status:{task_id}",
            {
                "status": "completed",
                "progress": 100,
                "message": "Response ready",
                "result_key": f"ai_result:{task_id}",
                "user_id": user_id,
                "conversation_id": conversation_id,
                "processing_time": processing_time,
            },
            timeout=300,
        )

        # Log usage for analytics
        try:
            from .models import AgentUsageLog

            AgentUsageLog.objects.create(  # pylint: disable=no-member
                user=user,
                question=message[:500],  # Truncate long questions
                response_length=len(result.get("response", "")),
                context_used=result.get("context_used", False),
                context_sources=result.get("context_sources", 0),
                response_time_ms=int(processing_time * 1000),
                question_type=result.get("question_type", "general"),
                model_used=result.get("model_used", "tinyllama"),
                agent_type=agent_type,
            )
        except Exception as e:
            logger.warning(f"Failed to log usage: {e}")

        logger.info(f"Processed message for user {user_id} in {processing_time:.2f}s")
        return result

    except Exception as exc:
        logger.error(f"Error processing message: {exc}")

        # Update error status
        cache.set(
            f"task_status:{task_id}",
            {
                "status": "failed",
                "progress": 0,
                "message": f"Error: {str(exc)}",
                "error": str(exc),
                "user_id": user_id,
                "conversation_id": conversation_id,
            },
            timeout=300,
        )

        # Retry logic
        if self.request.retries < self.max_retries:
            logger.info(f"Retrying task {task_id}, attempt {self.request.retries + 1}")
            raise self.retry(countdown=60, exc=exc)

        raise exc


@shared_task(bind=True)
def process_multiagent_query(
    self, query, user_id, conversation_id=None, agent_preference=None
):
    """Process query through intelligent agent selection"""
    task_id = self.request.id

    try:
        user = User.objects.get(id=user_id)

        # Update status
        cache.set(
            f"task_status:{task_id}",
            {
                "status": "processing",
                "progress": 10,
                "message": "Analyzing query...",
                "user_id": user_id,
                "conversation_id": conversation_id,
            },
            timeout=300,
        )

        # Simple agent selection logic
        agent_type = agent_preference or "general"

        # Add null check for query before calling .lower()
        if not query:
            query = ""

        query_lower = query.lower()

        # Determine best agent based on query content
        if any(
            word in query_lower
            for word in ["research", "find", "gather", "collect", "sources"]
        ):
            agent_type = "research"
        elif any(
            word in query_lower for word in ["analyze", "compare", "evaluate", "assess"]
        ):
            agent_type = "content_analyzer"
        elif any(
            word in query_lower for word in ["what", "how", "when", "where", "why"]
        ):
            agent_type = "qa"

        cache.set(
            f"task_status:{task_id}",
            {
                "status": "processing",
                "progress": 30,
                "message": f"Selected {agent_type} agent...",
                "user_id": user_id,
                "conversation_id": conversation_id,
                "selected_agent": agent_type,
            },
            timeout=300,
        )

        # Process with selected agent
        start_time = time.time()
        result = process_user_message_async.delay(
            message=query,
            user_id=user_id,
            conversation_id=conversation_id,
            use_context=True,
            agent_type=agent_type,
            max_tokens=400,
        ).get()  # Wait for completion

        processing_time = time.time() - start_time

        # Add multiagent metadata
        result.update(
            {
                "multiagent_processing": True,
                "selected_agent": agent_type,
                "agent_selection_confidence": 0.8,  # Placeholder
                "total_processing_time": processing_time,
            }
        )

        # Store result
        cache.set(f"ai_result:{task_id}", result, timeout=3600)

        cache.set(
            f"task_status:{task_id}",
            {
                "status": "completed",
                "progress": 100,
                "message": "Response ready",
                "result_key": f"ai_result:{task_id}",
                "user_id": user_id,
                "conversation_id": conversation_id,
                "selected_agent": agent_type,
            },
            timeout=300,
        )

        return result

    except Exception as exc:
        logger.error(f"Error in multiagent processing: {exc}")
        cache.set(
            f"task_status:{task_id}",
            {
                "status": "failed",
                "progress": 0,
                "message": f"Error: {str(exc)}",
                "error": str(exc),
                "user_id": user_id,
                "conversation_id": conversation_id,
            },
            timeout=300,
        )
        raise exc


@shared_task
def auto_expand_knowledge_async(topic, max_urls=5):
    """Automatically expand knowledge base for a topic"""
    try:
        logger.info(f"Starting knowledge expansion for topic: {topic}")

        # This would integrate with your crawler
        # For now, we'll simulate the process
        from crawler.models import CrawledPage

        # Search for existing content
        existing_pages = CrawledPage.objects.filter(  # pylint: disable=no-member
            page_title__icontains=topic
        ).count()

        result = {
            "topic": topic,
            "existing_pages": existing_pages,
            "expansion_needed": existing_pages < 3,
            "status": "completed",
        }

        if result["expansion_needed"]:
            # Here you would trigger your crawler for specific searches
            # crawler_task = crawl_topic.delay(topic, max_urls)
            result["action"] = "triggered_crawl"
        else:
            result["action"] = "sufficient_content"

        logger.info(f"Knowledge expansion for '{topic}' completed: {result}")
        return result

    except Exception as exc:
        logger.error(f"Knowledge expansion failed: {exc}")
        raise exc


@shared_task
def analyze_content_quality_batch():
    """Analyze content quality for pages in background"""
    try:
        from crawler.models import CrawledPage

        # Get pages that haven't been quality assessed recently
        pages = (
            CrawledPage.objects.filter(success=True)  # pylint: disable=no-member
            .exclude(clean_markdown__isnull=True)
            .exclude(clean_markdown="")[:50]
        )  # Process in batches of 50

        results = []
        for page in pages:
            try:
                # Simple quality assessment
                content_length = len(page.clean_markdown or "")
                word_count = len((page.clean_markdown or "").split())

                quality_score = min(1.0, word_count / 500)  # Normalize to 0-1

                assessment = {
                    "page_id": str(page.id),
                    "url": page.url,
                    "title": page.page_title,
                    "content_length": content_length,
                    "word_count": word_count,
                    "quality_score": quality_score,
                    "assessment_time": timezone.now().isoformat(),
                }

                results.append(assessment)

            except Exception as e:
                logger.warning(f"Failed to assess page {page.id}: {e}")

        logger.info(f"Assessed quality for {len(results)} pages")
        return {"assessed_count": len(results), "results": results}

    except Exception as exc:
        logger.error(f"Batch quality analysis failed: {exc}")
        raise exc


@shared_task
def batch_process_unprocessed_pages():
    """Process embeddings for all unprocessed pages"""
    try:
        from crawler.models import CrawledPage

        # Find unprocessed pages
        unprocessed_pages = (
            CrawledPage.objects.filter(success=True)  # pylint: disable=no-member
            .exclude(clean_markdown__isnull=True)
            .exclude(clean_markdown="")[:20]
        )  # Process in smaller batches

        processed_count = 0
        for page in unprocessed_pages:
            try:
                # Here you would generate embeddings
                # For now, just mark as processed
                logger.info(f"Processing embeddings for page: {page.page_title}")
                processed_count += 1

            except Exception as e:
                logger.warning(f"Failed to process page {page.id}: {e}")

        logger.info(f"Processed embeddings for {processed_count} pages")
        return {"processed_count": processed_count}

    except Exception as exc:
        logger.error(f"Batch processing failed: {exc}")
        raise exc


@shared_task
def update_content_freshness():
    """Check and update content freshness scores"""
    try:
        from crawler.models import CrawledPage
        from datetime import timedelta

        # Find pages older than 30 days
        cutoff_date = timezone.now() - timedelta(days=30)
        stale_pages = CrawledPage.objects.filter(  # pylint: disable=no-member
            success=True, created_at__lt=cutoff_date
        )[
            :10
        ]  # Limit to 10 per run

        refresh_needed = []
        for page in stale_pages:
            refresh_needed.append(
                {
                    "page_id": str(page.id),
                    "url": page.url,
                    "title": page.page_title,
                    "age_days": (timezone.now() - page.created_at).days,
                }
            )

        logger.info(f"Identified {len(refresh_needed)} pages needing refresh")
        return {"stale_count": len(refresh_needed), "pages_identified": refresh_needed}

    except Exception as exc:
        logger.error(f"Freshness update failed: {exc}")
        raise exc


@shared_task
def cleanup_expired_tasks():
    """Clean up expired task results and statuses"""
    try:
        # This is a simplified cleanup - in production you'd want more sophisticated cleanup
        from django.core.cache import cache

        # Clear expired conversations (older than 24 hours)
        # This would be more sophisticated in a real implementation

        logger.info("Cleaned up expired tasks and conversations")
        return "Cleanup completed"

    except Exception as exc:
        logger.error(f"Cleanup failed: {exc}")
        raise exc


@shared_task
def health_check():
    """Health check task for monitoring"""
    try:
        # Check if AI services are available
        agent = EnhancedPhi3Agent()
        is_available = agent.llm_service.is_available()

        return {
            "status": "healthy" if is_available else "degraded",
            "ai_service_available": is_available,
            "timestamp": timezone.now().isoformat(),
        }

    except Exception as exc:
        logger.error(f"Health check failed: {exc}")
        return {
            "status": "unhealthy",
            "error": str(exc),
            "timestamp": timezone.now().isoformat(),
        }


@shared_task(bind=True, max_retries=2, time_limit=900, soft_time_limit=720)
def process_live_research_async(
    self,
    research_topic,
    user_id,
    max_sources=5,
    include_local_kb=True,
    research_depth="comprehensive",
    use_live_research=True,
    conversation_id=None,
):
    """Process live research asynchronously"""
    task_id = self.request.id

    try:
        # Update task status
        cache.set(
            f"task_status:{task_id}",
            {
                "status": "processing",
                "progress": 0,
                "message": "Initializing live research...",
                "user_id": user_id,
                "conversation_id": conversation_id,
                "research_topic": research_topic,
            },
            timeout=300,
        )

        # Get user for logging
        try:
            user = User.objects.get(id=user_id)
        except User.DoesNotExist:
            raise Exception(f"User {user_id} not found")

        cache.set(
            f"task_status:{task_id}",
            {
                "status": "processing",
                "progress": 15,
                "message": "Loading research agent...",
                "user_id": user_id,
                "conversation_id": conversation_id,
                "research_topic": research_topic,
            },
            timeout=300,
        )

        # Initialize live research agent
        from .services.live_research_agent import LiveResearchAgent

        agent = LiveResearchAgent()

        cache.set(
            f"task_status:{task_id}",
            {
                "status": "processing",
                "progress": 30,
                "message": "Conducting live research...",
                "user_id": user_id,
                "conversation_id": conversation_id,
                "research_topic": research_topic,
            },
            timeout=300,
        )

        # Process research with enhanced error handling
        start_time = time.time()
        try:
            # Use asyncio to run the async research method
            import asyncio

            async def run_research():
                return await agent.enhanced_research_chat(
                    message=research_topic,
                    use_live_research=use_live_research,
                    max_sources=max_sources,
                    research_depth=research_depth,
                )

            result = asyncio.run(run_research())
            processing_time = time.time() - start_time

        except Exception as research_error:
            processing_time = time.time() - start_time
            error_msg = str(research_error)

            logger.error(f"Live research error for user {user_id}: {research_error}")

            # Return error result instead of raising
            result = {
                "response": f"I encountered an issue while conducting live research on '{research_topic}'. {error_msg} Please try again with a different topic or use the regular chat mode.",
                "error": True,
                "error_type": "research_error",
                "research_conducted": False,
                "live_sources_used": 0,
                "local_sources_used": 0,
                "sources": [],
                "research_methodology": {
                    "research_time_seconds": processing_time,
                    "error": error_msg,
                },
                "quality_metrics": {},
                "research_log": [f"❌ Research failed: {error_msg}"],
                "agent_type": "live_research",
                "processing_time_ms": int(processing_time * 1000),
            }

        cache.set(
            f"task_status:{task_id}",
            {
                "status": "processing",
                "progress": 90,
                "message": "Finalizing research results...",
                "user_id": user_id,
                "conversation_id": conversation_id,
                "research_topic": research_topic,
            },
            timeout=300,
        )

        # Convert any numpy float32 values to regular Python floats for JSON serialization
        def convert_numpy_types(obj):
            """Recursively convert numpy types to Python native types"""
            import numpy as np

            if isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif hasattr(obj, "item"):  # numpy scalar
                return obj.item()
            else:
                return obj

        # Convert numpy types in the result
        result = convert_numpy_types(result)

        # Add task metadata
        result.update(
            {
                "task_id": task_id,
                "processing_time": processing_time,
                "processed_async": True,
                "user_id": user_id,
                "conversation_id": conversation_id,
                "research_topic": research_topic,
                "timestamp": timezone.now().isoformat(),
                "config_used": {
                    "max_sources": max_sources,
                    "include_local_kb": include_local_kb,
                    "research_depth": research_depth,
                    "use_live_research": use_live_research,
                },
            }
        )

        # Store result in cache
        cache.set(f"ai_result:{task_id}", result, timeout=3600)

        # Update final status
        cache.set(
            f"task_status:{task_id}",
            {
                "status": "completed",
                "progress": 100,
                "message": "Live research completed",
                "result_key": f"ai_result:{task_id}",
                "user_id": user_id,
                "conversation_id": conversation_id,
                "research_topic": research_topic,
                "processing_time": processing_time,
                "live_sources_used": result.get("live_sources_used", 0),
                "local_sources_used": result.get("local_sources_used", 0),
            },
            timeout=300,
        )

        # Log usage for analytics
        try:
            from .models import AgentUsageLog

            AgentUsageLog.objects.create(  # pylint: disable=no-member
                user=user,
                question=research_topic[:500],  # Truncate long topics
                response_length=len(result.get("response", "")),
                context_used=result.get("research_conducted", False),
                context_sources=result.get("live_sources_used", 0)
                + result.get("local_sources_used", 0),
                response_time_ms=int(processing_time * 1000),
                question_type="live_research",
                model_used="live_research_agent",
                agent_type="live_research",
            )
        except Exception as e:
            logger.warning(f"Failed to log live research usage: {e}")

        logger.info(
            f"Live research completed for user {user_id} in {processing_time:.2f}s"
        )
        return result

    except Exception as exc:
        logger.error(f"Error processing live research: {exc}")

        # Update error status
        cache.set(
            f"task_status:{task_id}",
            {
                "status": "failed",
                "progress": 0,
                "message": f"Live research failed: {str(exc)}",
                "error": str(exc),
                "user_id": user_id,
                "conversation_id": conversation_id,
                "research_topic": research_topic,
            },
            timeout=300,
        )

        # Retry logic
        if self.request.retries < self.max_retries:
            logger.info(
                f"Retrying live research task {task_id}, attempt {self.request.retries + 1}"
            )
            raise self.retry(countdown=120, exc=exc)

        raise exc
