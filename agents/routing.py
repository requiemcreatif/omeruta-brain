from django.urls import re_path
from . import consumers

websocket_urlpatterns = [
    re_path(r"^ws/agent/$", consumers.AgentConsumer.as_asgi()),
    re_path(r"^ws/agent/(?P<user_id>\w+)/$", consumers.AgentConsumer.as_asgi()),
]
