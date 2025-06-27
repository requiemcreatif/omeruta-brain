/**
 * Omeruta Brain WebSocket Client
 * Provides real-time AI communication with fallback support
 */
class OmerutaBrainClient {
  constructor(apiBaseUrl, authToken) {
    this.apiBaseUrl = apiBaseUrl;
    this.authToken = authToken;
    this.socket = null;
    this.activeChats = new Map(); // Track multiple conversations
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;
    this.reconnectDelay = 1000;
    this.isConnected = false;
  }

  // Initialize WebSocket connection
  connectWebSocket() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/ws/agent/`;

    console.log("🔗 Connecting to Omeruta Brain WebSocket...");
    this.socket = new WebSocket(wsUrl);

    this.socket.onopen = () => {
      console.log("✅ Connected to Omeruta Brain");
      this.isConnected = true;
      this.reconnectAttempts = 0;
      this.showConnectionStatus("Connected", "success");

      // Send authentication if needed
      if (this.authToken) {
        this.socket.send(
          JSON.stringify({
            action: "authenticate",
            token: this.authToken,
          })
        );
      }
    };

    this.socket.onclose = (event) => {
      console.log("❌ Disconnected from Omeruta Brain");
      this.isConnected = false;
      this.showConnectionStatus("Disconnected", "error");

      // Auto-reconnect with exponential backoff
      if (this.reconnectAttempts < this.maxReconnectAttempts) {
        const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts);
        console.log(
          `🔄 Reconnecting in ${delay}ms... (attempt ${
            this.reconnectAttempts + 1
          })`
        );

        setTimeout(() => {
          this.reconnectAttempts++;
          this.connectWebSocket();
        }, delay);
      } else {
        this.showConnectionStatus("Connection failed", "error");
        console.error("❌ Max reconnection attempts reached");
      }
    };

    this.socket.onmessage = (event) => {
      this.handleWebSocketMessage(JSON.parse(event.data));
    };

    this.socket.onerror = (error) => {
      console.error("WebSocket error:", error);
      this.showConnectionStatus("Connection error", "error");
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
      this.socket.send(
        JSON.stringify({
          action: "start_chat",
          message: message,
          conversation_id: chatId,
          use_context: options.useContext !== false,
          agent_type: options.agentType || "general",
          max_tokens: options.maxTokens || 300,
        })
      );
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
      const response = await fetch(
        `${this.apiBaseUrl}/agents/async/chat_async/`,
        {
          method: "POST",
          headers: {
            Authorization: `Bearer ${this.authToken}`,
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            message: message,
            conversation_id: chatId,
            use_context: options.useContext !== false,
            agent_type: options.agentType || "general",
            max_tokens: options.maxTokens || 300,
          }),
        }
      );

      const data = await response.json();

      if (response.ok) {
        this.pollTaskStatus(data.task_id, chatId);
        return chatId;
      } else {
        throw new Error(data.error || "Failed to start chat");
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
              Authorization: `Bearer ${this.authToken}`,
            },
          }
        );

        const data = await response.json();

        this.updateChatProgress(chatId, data);

        if (data.status === "completed") {
          this.handleChatCompletion(chatId, data);
          return;
        } else if (data.status === "failed") {
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
        console.error("Polling error:", error);
        this.showError(`Connection error: ${error.message}`, chatId);
      }
    };

    poll();
  }

  // WebSocket message handling
  handleWebSocketMessage(data) {
    const chatId = data.conversation_id || "default";

    switch (data.type) {
      case "connection_established":
        console.log("✅ WebSocket connection established");
        break;

      case "task_started":
        this.showTypingIndicator(chatId, data.message);
        this.setTaskId(chatId, data.task_id);
        break;

      case "task_update":
        this.updateChatProgress(chatId, data);
        break;

      case "task_completed":
        this.handleChatCompletion(chatId, data);
        break;

      case "task_failed":
        this.handleChatError(chatId, data);
        break;

      case "task_timeout":
        this.handleChatTimeout(chatId);
        break;

      case "error":
        this.showError(data.message, chatId);
        break;

      case "pong":
        // Handle ping/pong for connection keep-alive
        break;

      default:
        console.log("Unknown WebSocket message type:", data.type);
    }
  }

  // UI Update Methods
  updateChatProgress(chatId, data) {
    const progressElement = document.getElementById(`progress-${chatId}`);
    const statusElement = document.getElementById(`status-${chatId}`);

    if (progressElement) {
      const progress = data.progress || 0;
      progressElement.style.width = `${progress}%`;
      progressElement.setAttribute("aria-valuenow", progress);

      // Add visual feedback
      if (progress > 75) {
        progressElement.className = "progress-fill progress-success";
      } else if (progress > 50) {
        progressElement.className = "progress-fill progress-warning";
      } else {
        progressElement.className = "progress-fill progress-info";
      }
    }

    if (statusElement) {
      statusElement.textContent = data.message || "Processing...";
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
      agent: result.selected_agent || result.agent_type,
      contextUsed: result.context_used,
      contextSources: result.context_sources,
      confidence: result.confidence_score,
      processingTime: result.processing_time || result.response_time_ms,
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
    this.showError(data.error || "An unexpected error occurred", chatId);
  }

  handleChatTimeout(chatId) {
    this.hideTypingIndicator(chatId);
    this.showError("Request timed out. Please try again.", chatId);
  }

  // UI Helper Methods
  displayUserMessage(message, chatId) {
    const chatContainer = this.getChatContainer(chatId);
    const messageElement = document.createElement("div");
    messageElement.className = "message user-message";
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
    const messageElement = document.createElement("div");
    messageElement.className = "message ai-message";

    const agentIcon = this.getAgentIcon(metadata.agent);
    const confidenceBar = metadata.confidence
      ? `<div class="confidence-bar">
                <div class="confidence-fill" style="width: ${
                  metadata.confidence * 100
                }%"></div>
                <span class="confidence-text">${Math.round(
                  metadata.confidence * 100
                )}% confident</span>
            </div>`
      : "";

    const processingTime = metadata.processingTime
      ? metadata.processingTime > 1000
        ? `${(metadata.processingTime / 1000).toFixed(2)}s`
        : `${metadata.processingTime}ms`
      : "";

    messageElement.innerHTML = `
            <div class="message-avatar">${agentIcon}</div>
            <div class="message-content">
                <div class="message-text">${this.formatMessage(message)}</div>
                ${confidenceBar}
                <div class="message-metadata">
                    <span class="metadata-item">🤖 ${
                      metadata.model || "AI"
                    }</span>
                    ${
                      metadata.agent
                        ? `<span class="metadata-item">🎯 ${this.getAgentName(
                            metadata.agent
                          )}</span>`
                        : ""
                    }
                    ${
                      metadata.contextUsed
                        ? `<span class="metadata-item">📚 ${
                            metadata.contextSources || 0
                          } sources</span>`
                        : ""
                    }
                    ${
                      processingTime
                        ? `<span class="metadata-item">⚡ ${processingTime}</span>`
                        : ""
                    }
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
    const existingIndicator = chatContainer.querySelector(".typing-indicator");
    if (existingIndicator) {
      existingIndicator.remove();
    }

    const typingElement = document.createElement("div");
    typingElement.className = "message typing-indicator";
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
    const agentElement = document.createElement("div");
    agentElement.className = "agent-selection";
    agentElement.innerHTML = `
            <div class="agent-info">
                ${this.getAgentIcon(agentName)} <strong>${this.getAgentName(
      agentName
    )}</strong> selected
                ${
                  confidence
                    ? `<span class="confidence-score">(${Math.round(
                        confidence * 100
                      )}% confidence)</span>`
                    : ""
                }
            </div>
        `;
    chatContainer.appendChild(agentElement);
  }

  showKnowledgeExpansion(chatId, expansionInfo) {
    const chatContainer = this.getChatContainer(chatId);
    const expansionElement = document.createElement("div");
    expansionElement.className = "knowledge-expansion";
    expansionElement.innerHTML = `
            <div class="expansion-info">
                🧠 <strong>Knowledge Expanded!</strong>
                Crawled ${
                  expansionInfo.urls_crawled?.length || 0
                } new sources for better answers.
            </div>
        `;
    chatContainer.appendChild(expansionElement);
  }

  // Connection Management
  showConnectionStatus(status, type) {
    const statusElement = document.getElementById("connection-status");
    if (statusElement) {
      statusElement.textContent = status;
      statusElement.className = `connection-status ${type}`;
    }

    // Show toast notification
    this.showToast(`Connection: ${status}`, type);
  }

  showToast(message, type = "info") {
    const toast = document.createElement("div");
    toast.className = `toast toast-${type}`;
    toast.textContent = message;

    document.body.appendChild(toast);

    // Auto-remove after 3 seconds
    setTimeout(() => {
      if (toast.parentNode) {
        toast.parentNode.removeChild(toast);
      }
    }, 3000);
  }

  // Utility Methods
  generateChatId() {
    return `chat_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  setTaskId(chatId, taskId) {
    // Store task ID for potential cancellation
    const chatData = this.activeChats.get(chatId) || {};
    chatData.taskId = taskId;
    this.activeChats.set(chatId, chatData);
  }

  getChatContainer(chatId) {
    let container = document.getElementById(`chat-${chatId}`);
    if (!container) {
      container = document.createElement("div");
      container.id = `chat-${chatId}`;
      container.className = "chat-container";

      const mainChatArea = document.getElementById("main-chat-area");
      if (mainChatArea) {
        mainChatArea.appendChild(container);
      }
    }
    return container;
  }

  getAgentIcon(agentType) {
    const icons = {
      research: "🔬",
      content_analyzer: "📊",
      qa: "❓",
      general: "🤖",
      tinyllama: "🦙",
    };
    return icons[agentType] || "🤖";
  }

  getAgentName(agentType) {
    const names = {
      research: "Research Agent",
      content_analyzer: "Analysis Agent",
      qa: "Q&A Agent",
      general: "General Agent",
      tinyllama: "TinyLlama",
    };
    return names[agentType] || "AI Agent";
  }

  formatMessage(message) {
    // Convert markdown-like formatting to HTML
    return message
      .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
      .replace(/\*(.*?)\*/g, "<em>$1</em>")
      .replace(/`(.*?)`/g, "<code>$1</code>")
      .replace(/\n/g, "<br>");
  }

  escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }

  scrollToBottom(container) {
    container.scrollTop = container.scrollHeight;
  }

  showError(message, chatId) {
    const chatContainer = this.getChatContainer(chatId);
    const errorElement = document.createElement("div");
    errorElement.className = "message error-message";
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

  // Advanced Features
  cancelTask(chatId) {
    const chatData = this.activeChats.get(chatId);
    if (chatData && chatData.taskId) {
      if (this.socket && this.socket.readyState === WebSocket.OPEN) {
        this.socket.send(
          JSON.stringify({
            action: "cancel_task",
            task_id: chatData.taskId,
          })
        );
      }
    }
  }

  ping() {
    if (this.socket && this.socket.readyState === WebSocket.OPEN) {
      this.socket.send(
        JSON.stringify({
          action: "ping",
        })
      );
    }
  }

  // Keep connection alive
  startKeepAlive() {
    setInterval(() => {
      if (this.isConnected) {
        this.ping();
      }
    }, 30000); // Ping every 30 seconds
  }

  // Disconnect
  disconnect() {
    if (this.socket) {
      this.socket.close();
      this.socket = null;
      this.isConnected = false;
    }
  }
}

// CSS Styles for the chat interface
const chatStyles = `
<style>
.chat-container {
    max-width: 800px;
    margin: 0 auto;
    padding: 20px;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

.message {
    display: flex;
    margin-bottom: 16px;
    animation: fadeIn 0.3s ease-in;
}

.message-avatar {
    width: 40px;
    height: 40px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 20px;
    margin-right: 12px;
}

.message-content {
    flex: 1;
    max-width: calc(100% - 52px);
}

.message-text {
    background: #f1f1f1;
    padding: 12px 16px;
    border-radius: 18px;
    margin-bottom: 4px;
    word-wrap: break-word;
}

.user-message .message-text {
    background: #007AFF;
    color: white;
}

.ai-message .message-text {
    background: #e9ecef;
    color: #333;
}

.error-message .message-text {
    background: #ff4444;
    color: white;
}

.message-time {
    font-size: 12px;
    color: #666;
    margin-left: 16px;
}

.message-metadata {
    margin: 8px 16px 0;
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}

.metadata-item {
    font-size: 12px;
    background: #f8f9fa;
    padding: 2px 6px;
    border-radius: 8px;
    color: #666;
}

.typing-indicator {
    opacity: 0.8;
}

.typing-animation {
    display: flex;
    gap: 4px;
    margin-bottom: 8px;
}

.typing-dot {
    width: 8px;
    height: 8px;
    background: #007AFF;
    border-radius: 50%;
    animation: typing 1.5s infinite;
}

.typing-dot:nth-child(2) { animation-delay: 0.2s; }
.typing-dot:nth-child(3) { animation-delay: 0.4s; }

.progress-container {
    margin: 8px 0;
}

.progress-bar {
    width: 100%;
    height: 4px;
    background: #e9ecef;
    border-radius: 2px;
    overflow: hidden;
}

.progress-fill {
    height: 100%;
    background: #007AFF;
    border-radius: 2px;
    transition: width 0.3s ease;
}

.progress-success { background: #28a745; }
.progress-warning { background: #ffc107; }
.progress-info { background: #17a2b8; }

.agent-selection {
    margin: 8px 0;
    padding: 8px 12px;
    background: #f8f9fa;
    border-left: 3px solid #007AFF;
    border-radius: 4px;
    font-size: 14px;
}

.confidence-bar {
    margin: 8px 0;
    height: 6px;
    background: #e9ecef;
    border-radius: 3px;
    overflow: hidden;
    position: relative;
}

.confidence-fill {
    height: 100%;
    background: linear-gradient(90deg, #ff4444, #ffc107, #28a745);
    border-radius: 3px;
    transition: width 0.3s ease;
}

.confidence-text {
    position: absolute;
    right: 4px;
    top: -20px;
    font-size: 11px;
    color: #666;
}

.connection-status {
    position: fixed;
    top: 20px;
    right: 20px;
    padding: 8px 12px;
    border-radius: 4px;
    font-size: 12px;
    font-weight: bold;
    z-index: 1000;
}

.connection-status.success {
    background: #d4edda;
    color: #155724;
    border: 1px solid #c3e6cb;
}

.connection-status.error {
    background: #f8d7da;
    color: #721c24;
    border: 1px solid #f5c6cb;
}

.toast {
    position: fixed;
    bottom: 20px;
    right: 20px;
    padding: 12px 16px;
    border-radius: 4px;
    color: white;
    font-weight: bold;
    z-index: 1001;
    animation: slideIn 0.3s ease-out;
}

.toast-success { background: #28a745; }
.toast-error { background: #dc3545; }
.toast-info { background: #17a2b8; }

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes typing {
    0%, 60%, 100% { transform: translateY(0); }
    30% { transform: translateY(-10px); }
}

@keyframes slideIn {
    from { transform: translateX(100%); }
    to { transform: translateX(0); }
}
</style>
`;

// Inject styles
document.head.insertAdjacentHTML("beforeend", chatStyles);

// Export for use
window.OmerutaBrainClient = OmerutaBrainClient;
