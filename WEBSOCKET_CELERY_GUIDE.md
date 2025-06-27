# 🧠 Omeruta Brain WebSocket & Celery Integration Guide

## 🎯 **What's New - Real-Time AI System**

Your Omeruta Brain now has **real-time capabilities** with:

✅ **Instant Response Feedback** - No more waiting 3-5 seconds in silence  
✅ **Live Progress Updates** - See "Searching knowledge base...", "Generating response..."  
✅ **WebSocket Real-Time** - Instant bi-directional communication  
✅ **Async Task Processing** - AI works in background, UI stays responsive  
✅ **Multiple Agent Support** - Research, Q&A, Analysis agents with live switching  
✅ **Conversation Memory** - Contextual conversations across sessions  
✅ **Auto-Reconnection** - Fault-tolerant connection management

---

## 🏃 **Quick Start - Testing Your System**

### **1. Check All Services Are Running**

```bash
# Terminal 1: Redis (required for Celery & WebSocket)
redis-server

# Terminal 2: Django with WebSocket support
cd omeruta_brain
source venv/bin/activate
python manage.py start_websocket_server

# Terminal 3: Celery Worker (AI processing)
source venv/bin/activate
python manage.py start_celery_worker --queue ai_high_priority,ai_processing

# Terminal 4: Optional - Celery Beat (periodic tasks)
source venv/bin/activate
python manage.py start_celery_beat
```

### **2. Test WebSocket Demo**

Open your browser to: **http://localhost:8000/static/websocket-demo.html**

**Quick Test Steps:**

1. Click "Connect WebSocket" (should show 🟢 Connected)
2. Type: "What is cryptocurrency?"
3. Watch real-time progress: `🤖 AI is thinking... → 25% Searching knowledge → 75% Generating response → Complete!`
4. Try different agents: Research, Q&A, Content Analyzer

---

## 🔧 **API Endpoints - Before & After**

### **BEFORE: Synchronous (Blocking)**

```javascript
// User waits 3-5 seconds with no feedback
fetch("/api/agents/tinyllama/chat/", {
  method: "POST",
  headers: {
    Authorization: `Bearer ${token}`,
    "Content-Type": "application/json",
  },
  body: JSON.stringify({ message: "What is Bitcoin?" }),
})
  .then((response) => response.json())
  .then((data) => {
    // Finally get response after long wait
    console.log(data.response);
  });
```

### **AFTER: Asynchronous with Real-Time Updates**

```javascript
// 🚀 NEW: Instant feedback + live progress
fetch("/api/agents/async/chat_async/", {
  method: "POST",
  headers: {
    Authorization: `Bearer ${token}`,
    "Content-Type": "application/json",
  },
  body: JSON.stringify({ message: "What is Bitcoin?" }),
})
  .then((response) => response.json())
  .then((data) => {
    console.log(`Task started: ${data.task_id}`);
    // Poll for updates or use WebSocket for real-time
    pollTaskStatus(data.task_id);
  });
```

### **🔥 Even Better: WebSocket Real-Time**

```javascript
const socket = new WebSocket("ws://localhost:8000/ws/agent/");

socket.onmessage = function (event) {
  const data = JSON.parse(event.data);

  switch (data.type) {
    case "task_started":
      showProgress("🤖 AI is thinking...");
      break;
    case "task_update":
      updateProgress(data.progress, data.message);
      break;
    case "task_completed":
      showResponse(data.result.response);
      break;
  }
};

// Start chat via WebSocket
socket.send(
  JSON.stringify({
    action: "start_chat",
    message: "What is Bitcoin?",
    agent_type: "research",
  })
);
```

---

## 🎮 **New API Endpoints**

### **Async Chat Processing**

```bash
# Start async chat
curl -X POST http://localhost:8000/api/agents/async/chat_async/ \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Explain blockchain technology",
    "agent_type": "research",
    "use_context": true,
    "max_tokens": 300
  }'

# Response: {"task_id": "abc123", "status": "processing", "check_status_url": "..."}
```

### **Check Task Status**

```bash
curl -X GET "http://localhost:8000/api/agents/async/status/abc123/" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"

# Real-time status updates:
# {"status": "processing", "progress": 25, "message": "Searching knowledge base..."}
# {"status": "processing", "progress": 75, "message": "Generating response..."}
# {"status": "completed", "result": {"response": "Blockchain is...", ...}}
```

### **Multi-Agent Processing**

```bash
# Route to best agent automatically
curl -X POST http://localhost:8000/api/agents/async/multiagent_async/ \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Compare Bitcoin and Ethereum for investment",
    "conversation_id": "session_123"
  }'
```

### **Knowledge Expansion**

```bash
# Auto-expand knowledge base
curl -X POST http://localhost:8000/api/agents/async/expand_knowledge/ \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "quantum computing",
    "max_urls": 5
  }'
```

---

## 🌟 **User Experience Transformation**

### **Before vs After Comparison**

```
┌─────────────────────────────────────────┐
│ BEFORE (Synchronous):                   │
│ User: "What is cryptocurrency?"         │
│ [████████████████████] 5 seconds       │
│ Response: "Cryptocurrency is..."        │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ AFTER (Asynchronous with Real-Time):   │
│ User: "What is cryptocurrency?"         │
│ Instant: "🤖 AI is thinking..."        │
│ [██░░░░░░░░] 25% "Searching knowledge"  │
│ [████░░░░░░] 40% "Found 3 sources"      │
│ [██████░░░░] 60% "Analyzing content"    │
│ [████████░░] 80% "Generating response"  │
│ [██████████] 100% "Complete!"           │
│ Response: "Cryptocurrency is..."        │
│ + Metadata: Agent type, sources, time   │
└─────────────────────────────────────────┘
```

---

## 🧪 **Testing Scenarios**

### **Test 1: Basic WebSocket Connection**

```bash
# 1. Open browser console at http://localhost:8000/static/websocket-demo.html
# 2. Run:
setAuthToken('YOUR_JWT_TOKEN_HERE')  # Get from login endpoint
# 3. Refresh page
# 4. Click "Connect WebSocket" → Should show 🟢 Connected
```

### **Test 2: Real-Time AI Chat**

```bash
# In the demo:
# 1. Type: "What is the future of cryptocurrency?"
# 2. Watch for real-time updates:
#    - "🤖 AI is processing your message..."
#    - Progress bar: 0% → 25% → 50% → 75% → 100%
#    - Status: "Searching knowledge base..." → "Generating response..."
#    - Final response with metadata (sources, time, agent type)
```

### **Test 3: Agent Type Switching**

```bash
# 1. Select "Research Agent" from dropdown
# 2. Ask: "Research the latest developments in AI"
# 3. Switch to "Q&A Agent"
# 4. Ask: "What is machine learning?"
# 5. Notice different response styles and processing approaches
```

### **Test 4: Fallback to Polling**

```bash
# 1. Disconnect WebSocket (close browser dev tools network tab)
# 2. Send message → Should automatically fall back to polling mode
# 3. Still get progress updates, just via HTTP polling instead of WebSocket
```

---

## 🔧 **Troubleshooting**

### **Common Issues & Solutions**

#### **1. WebSocket Connection Failed**

```bash
# Check if Redis is running
redis-cli ping  # Should return "PONG"

# Check if ASGI server is running with WebSocket support
python manage.py start_websocket_server --verbosity 2

# Check WebSocket URL in browser console:
# ws://localhost:8000/ws/agent/ should be accessible
```

#### **2. Tasks Not Processing**

```bash
# Check Celery worker is running
python manage.py start_celery_worker --queue ai_high_priority --verbosity 2

# Check Redis for queued tasks
redis-cli
> KEYS celery*
> LLEN celery  # Should show pending tasks
```

#### **3. Authentication Issues**

```bash
# Get a fresh JWT token:
curl -X POST http://localhost:8000/api/authentication/login/ \
  -H "Content-Type: application/json" \
  -d '{"username": "your_username", "password": "your_password"}'

# Use the access token in your requests
```

#### **4. Model Loading Errors**

```bash
# Check if TinyLlama can load on your system:
python manage.py test_tinyllama

# If MPS errors, the model will fall back to CPU automatically
```

---

## 🎯 **Production Deployment**

### **Docker Compose Setup**

```yaml
# docker-compose.yml
version: "3.8"

services:
  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]

  web:
    build: .
    command: daphne -b 0.0.0.0 -p 8000 omeruta_brain_project.asgi:application
    ports: ["8000:8000"]
    depends_on: [redis]

  celery-ai:
    build: .
    command: celery -A omeruta_brain_project worker --loglevel=info --queues=ai_high_priority,ai_processing
    depends_on: [redis]

  celery-beat:
    build: .
    command: celery -A omeruta_brain_project beat --loglevel=info
    depends_on: [redis]
```

### **Environment Variables**

```bash
# .env
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0
```

---

## 🚀 **Performance Benefits**

### **Metrics You'll See**

| Metric               | Before               | After                    |
| -------------------- | -------------------- | ------------------------ |
| **User Feedback**    | 0-5 seconds (silent) | Instant + live updates   |
| **Concurrent Users** | Limited (blocking)   | Unlimited (async)        |
| **Task Failures**    | Lost forever         | Auto-retry + monitoring  |
| **User Experience**  | Frustrating waits    | Engaging real-time       |
| **Scalability**      | Single-threaded      | Multi-worker distributed |

### **Real Performance Example**

```bash
# Before: 1 user blocks everyone for 5 seconds
User A: [██████████████████████] 5 sec wait
User B: [██████████████████████] 5 sec wait (queued)
User C: [██████████████████████] 5 sec wait (queued)

# After: 3 users get instant feedback, process in parallel
User A: ✅ Instant "Processing..." → Result in 5 sec
User B: ✅ Instant "Processing..." → Result in 5 sec
User C: ✅ Instant "Processing..." → Result in 5 sec
All running simultaneously with live progress!
```

---

## 📊 **Monitoring & Analytics**

### **Built-in Usage Tracking**

Your system now automatically tracks:

- ✅ Response times per agent type
- ✅ Context usage statistics
- ✅ Question type classification
- ✅ User engagement patterns
- ✅ Task success/failure rates

### **Access Analytics**

```bash
# View recent usage logs
python manage.py shell
>>> from agents.models import AgentUsageLog
>>> AgentUsageLog.objects.all()[:10]

# Average response times
>>> from django.db.models import Avg
>>> AgentUsageLog.objects.aggregate(Avg('response_time_ms'))
```

---

## 🎉 **Next Steps - Advanced Features**

Your system is now ready for:

1. **🔄 Auto-Knowledge Expansion** - Research agent automatically crawls new sources
2. **🤖 Multi-Agent Orchestration** - Different agents collaborate on complex queries
3. **📱 Mobile App Integration** - WebSocket API ready for React Native/Flutter
4. **🔍 Advanced Analytics** - User behavior insights and performance optimization
5. **🌐 Distributed Processing** - Scale across multiple servers

---

## 🏆 **Congratulations!**

You've successfully upgraded from a **synchronous, blocking AI system** to a **real-time, scalable, enterprise-ready platform**!

**Your users now experience:**

- ⚡ Instant feedback instead of silent waits
- 📊 Live progress updates with transparency
- 🔄 Automatic reconnection and fault tolerance
- 🎯 Smart agent selection and routing
- 💬 Contextual conversations with memory
- 📈 Professional-grade performance monitoring

**Ready to test? Open: http://localhost:8000/static/websocket-demo.html** 🚀
