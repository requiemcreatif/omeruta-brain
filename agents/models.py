from django.db import models
from django.contrib.auth import get_user_model

User = get_user_model()


class AgentUsageLog(models.Model):
    """Track agent usage for analytics and performance monitoring"""

    user = models.ForeignKey(User, on_delete=models.CASCADE)
    question = models.TextField(help_text="User's question (truncated if long)")
    response_length = models.IntegerField(help_text="Length of AI response")
    context_used = models.BooleanField(
        default=False, help_text="Whether knowledge base context was used"
    )
    context_sources = models.IntegerField(
        default=0, help_text="Number of context sources used"
    )
    response_time_ms = models.IntegerField(help_text="Response time in milliseconds")
    question_type = models.CharField(
        max_length=50,
        default="general",
        help_text="Type of question (factual, analytical, procedural, opinion)",
    )
    model_used = models.CharField(
        max_length=50, default="phi3", help_text="AI model used for response"
    )
    agent_type = models.CharField(
        max_length=50,
        default="general",
        help_text="Type of agent used (general, research, qa, content_analyzer)",
    )
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "agent_usage_logs"
        ordering = ["-timestamp"]
        indexes = [
            models.Index(fields=["user", "timestamp"]),
            models.Index(fields=["agent_type", "timestamp"]),
            models.Index(fields=["question_type", "timestamp"]),
        ]

    def __str__(self):
        return f"{self.user.username} - {self.agent_type} - {self.timestamp.strftime('%Y-%m-%d %H:%M')}"
