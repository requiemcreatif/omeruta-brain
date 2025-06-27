import json
import asyncio
import logging
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from celery.result import AsyncResult
from django.core.cache import cache
from django.contrib.auth import get_user_model
from django.contrib.auth.models import AnonymousUser

logger = logging.getLogger(__name__)
User = get_user_model()


class AgentConsumer(AsyncWebsocketConsumer):
    """WebSocket consumer for real-time AI responses and task updates"""

    async def connect(self):
        """Handle WebSocket connection"""
        self.user = self.scope.get("user")

        # Check authentication
        if self.user is None or isinstance(self.user, AnonymousUser):
            logger.warning("Unauthenticated WebSocket connection attempt")
            await self.close(code=4001)  # Unauthorized
            return

        self.user_id = str(self.user.id)
        self.group_name = f"user_{self.user_id}"
        self.active_tasks = set()  # Track active tasks for this connection

        # Join user-specific group
        await self.channel_layer.group_add(self.group_name, self.channel_name)

        await self.accept()

        # Send connection confirmation
        await self.send(
            text_data=json.dumps(
                {
                    "type": "connection_established",
                    "message": "Connected to Omeruta Brain WebSocket",
                    "user_id": self.user_id,
                    "timestamp": self._get_timestamp(),
                }
            )
        )

        logger.info(f"WebSocket connected for user {self.user_id}")

    async def disconnect(self, close_code):
        """Handle WebSocket disconnection"""
        if hasattr(self, "group_name"):
            # Leave user group
            await self.channel_layer.group_discard(self.group_name, self.channel_name)

            # Cancel any active polling tasks
            for task_id in self.active_tasks:
                # Here you could implement task cancellation if needed
                pass

        logger.info(
            f"WebSocket disconnected for user {getattr(self, 'user_id', 'unknown')} with code {close_code}"
        )

    async def receive(self, text_data):
        """Handle messages from WebSocket"""
        try:
            data = json.loads(text_data)
            action = data.get("action")

            logger.debug(
                f"WebSocket received action: {action} from user {self.user_id}"
            )

            if action == "start_chat":
                await self.start_chat(data)
            elif action == "check_task":
                await self.check_task_status(data)
            elif action == "cancel_task":
                await self.cancel_task(data)
            elif action == "ping":
                await self.send(
                    text_data=json.dumps(
                        {"type": "pong", "timestamp": self._get_timestamp()}
                    )
                )
            else:
                await self.send_error(f"Unknown action: {action}")

        except json.JSONDecodeError:
            await self.send_error("Invalid JSON data")
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")
            await self.send_error("Internal server error")

    async def start_chat(self, data):
        """Start async chat and send real-time updates"""
        message = data.get("message", "").strip()

        if not message:
            await self.send_error("Message is required")
            return

        try:
            # Import here to avoid circular imports
            from .tasks import process_user_message_async

            # Extract parameters
            conversation_id = data.get("conversation_id")
            use_context = data.get("use_context", True)
            agent_type = data.get("agent_type", "general")
            max_tokens = data.get("max_tokens", 300)

            # Validate agent type
            valid_agent_types = ["general", "research", "qa", "content_analyzer"]
            if agent_type not in valid_agent_types:
                agent_type = "general"

            # Start async task
            task = process_user_message_async.delay(
                message=message,
                user_id=self.user.id,
                conversation_id=conversation_id,
                use_context=use_context,
                agent_type=agent_type,
                max_tokens=max_tokens,
            )

            # Track this task
            self.active_tasks.add(task.id)

            # Send task started notification
            await self.send(
                text_data=json.dumps(
                    {
                        "type": "task_started",
                        "task_id": task.id,
                        "message": "Processing your message...",
                        "agent_type": agent_type,
                        "conversation_id": conversation_id,
                        "timestamp": self._get_timestamp(),
                    }
                )
            )

            # Start polling for updates in background
            asyncio.create_task(self.poll_task_status(task.id, conversation_id))

        except Exception as e:
            logger.error(f"Error starting chat: {e}")
            await self.send_error(f"Failed to start chat: {str(e)}")

    async def poll_task_status(self, task_id, conversation_id=None):
        """Poll task status and send real-time updates"""
        max_polls = 120  # 4 minutes max (120 * 2 seconds)
        poll_count = 0

        try:
            while poll_count < max_polls:
                # Check if connection is still active
                if task_id not in self.active_tasks:
                    break

                # Get cached status from our task
                status_data = cache.get(f"task_status:{task_id}")

                if status_data:
                    # Send progress update
                    await self.send(
                        text_data=json.dumps(
                            {
                                "type": "task_update",
                                "task_id": task_id,
                                "conversation_id": conversation_id,
                                "timestamp": self._get_timestamp(),
                                **status_data,
                            }
                        )
                    )

                    # Check if task is completed or failed
                    if status_data["status"] in ["completed", "failed"]:
                        if (
                            status_data["status"] == "completed"
                            and "result_key" in status_data
                        ):
                            # Get the final result
                            result = cache.get(status_data["result_key"])
                            if result:
                                await self.send(
                                    text_data=json.dumps(
                                        {
                                            "type": "task_completed",
                                            "task_id": task_id,
                                            "conversation_id": conversation_id,
                                            "result": result,
                                            "timestamp": self._get_timestamp(),
                                        }
                                    )
                                )
                        elif status_data["status"] == "failed":
                            await self.send(
                                text_data=json.dumps(
                                    {
                                        "type": "task_failed",
                                        "task_id": task_id,
                                        "conversation_id": conversation_id,
                                        "error": status_data.get(
                                            "error", "Unknown error"
                                        ),
                                        "timestamp": self._get_timestamp(),
                                    }
                                )
                            )

                        # Remove from active tasks
                        self.active_tasks.discard(task_id)
                        break

                poll_count += 1
                await asyncio.sleep(2)  # Poll every 2 seconds

            else:
                # Timeout reached
                await self.send(
                    text_data=json.dumps(
                        {
                            "type": "task_timeout",
                            "task_id": task_id,
                            "conversation_id": conversation_id,
                            "message": "Task processing timeout",
                            "timestamp": self._get_timestamp(),
                        }
                    )
                )
                self.active_tasks.discard(task_id)

        except Exception as e:
            logger.error(f"Error polling task status: {e}")
            await self.send_error(f"Error monitoring task: {str(e)}")
            self.active_tasks.discard(task_id)

    async def check_task_status(self, data):
        """Check specific task status on demand"""
        task_id = data.get("task_id")
        if not task_id:
            await self.send_error("task_id is required")
            return

        try:
            # Get cached status
            status_data = cache.get(f"task_status:{task_id}")

            if status_data:
                await self.send(
                    text_data=json.dumps(
                        {
                            "type": "task_status",
                            "task_id": task_id,
                            "timestamp": self._get_timestamp(),
                            **status_data,
                        }
                    )
                )
            else:
                # Fallback to Celery result
                task_result = AsyncResult(task_id)
                await self.send(
                    text_data=json.dumps(
                        {
                            "type": "task_status",
                            "task_id": task_id,
                            "status": task_result.status.lower(),
                            "ready": task_result.ready(),
                            "timestamp": self._get_timestamp(),
                        }
                    )
                )

        except Exception as e:
            logger.error(f"Error checking task status: {e}")
            await self.send_error(f"Failed to check task status: {str(e)}")

    async def cancel_task(self, data):
        """Cancel an active task"""
        task_id = data.get("task_id")
        if not task_id:
            await self.send_error("task_id is required")
            return

        try:
            # Remove from active tasks
            self.active_tasks.discard(task_id)

            # Try to revoke the Celery task
            from celery import current_app

            current_app.control.revoke(task_id, terminate=True)

            await self.send(
                text_data=json.dumps(
                    {
                        "type": "task_cancelled",
                        "task_id": task_id,
                        "message": "Task cancellation requested",
                        "timestamp": self._get_timestamp(),
                    }
                )
            )

        except Exception as e:
            logger.error(f"Error cancelling task: {e}")
            await self.send_error(f"Failed to cancel task: {str(e)}")

    async def send_error(self, message):
        """Send error message to client"""
        await self.send(
            text_data=json.dumps(
                {
                    "type": "error",
                    "message": message,
                    "timestamp": self._get_timestamp(),
                }
            )
        )

    def _get_timestamp(self):
        """Get current timestamp in ISO format"""
        from django.utils import timezone

        return timezone.now().isoformat()

    # Group message handlers (for broadcasting to user groups)
    async def task_update(self, event):
        """Handle task update broadcast to group"""
        await self.send(text_data=json.dumps(event))

    async def task_completed(self, event):
        """Handle task completion broadcast to group"""
        await self.send(text_data=json.dumps(event))

    async def system_notification(self, event):
        """Handle system notifications"""
        await self.send(text_data=json.dumps(event))
