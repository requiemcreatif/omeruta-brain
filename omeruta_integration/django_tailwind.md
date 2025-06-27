<div class="w-8 h-8 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg flex items-center justify-center flex-shrink-0">
                <i class="fas fa-brain text-white text-sm"></i>
            </div>
            <div class="flex-1">
                <div class="bg-gray-800 rounded-2xl rounded-tl-sm px-4 py-3 max-w-md">
                    <p class="text-gray-100">👋 Chat cleared. How can I help you today?</p>
                </div>
            </div>
        </div>
    `;
    conversationId = null;
    messageCount = 0;
    totalResponseTime = 0;
    contextUsageCount = 0;
    updateStatisticsDisplay();
}

function updateStatistics(responseTime, contextUsed) {
totalResponseTime += responseTime;
if (contextUsed) contextUsageCount++;
updateStatisticsDisplay();
}

function updateStatisticsDisplay() {
document.getElementById('message-count').textContent = messageCount;

    const avgTime = messageCount > 0 ? totalResponseTime / messageCount : 0;
    document.getElementById('avg-response-time').textContent = `${avgTime.toFixed(1)}s`;

    const contextPercentage = messageCount > 0 ? (contextUsageCount / messageCount) * 100 : 0;
    document.getElementById('context-usage').textContent = `${contextPercentage.toFixed(0)}%`;

}

function exportChat() {
const messages = document.querySelectorAll('#chat-messages > div');
let chatText = 'Omeruta Brain Chat Export\n';
chatText += '='.repeat(30) + '\n\n';

    messages.forEach(message => {
        const isUser = message.querySelector('.fa-user');
        const content = message.querySelector('p').textContent.trim();

        chatText += `${isUser ? 'User' : 'AI'}: ${content}\n\n`;
    });

    const blob = new Blob([chatText], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `omeruta-chat-${new Date().toISOString().split('T')[0]}.txt`;
    a.click();
    URL.revokeObjectURL(url);

}

// Enter key to send message (but allow Shift+Enter for new lines)
messageInput.addEventListener('keypress', function(e) {
if (e.key === 'Enter' && !e.shiftKey) {
e.preventDefault();
sendMessage(e);
}
});

// Auto-focus message input
messageInput.focus();
</script>
{% endblock %}

# 6. Beautiful Dashboard with Tailwind

# templates/core/dashboard.html

{% extends 'base.html' %}
{% load humanize %}

{% block title %}Dashboard - Omeruta Brain{% endblock %}

{% block content %}

<div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
    <!-- Header -->
    <div class="mb-8">
        <h1 class="text-3xl font-bold text-white mb-2">Dashboard</h1>
        <p class="text-gray-400">Welcome back to your AI-powered knowledge management system</p>
    </div>
    
    <!-- Stats Grid -->
    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <!-- Total Crawl Jobs -->
        <div class="bg-gradient-to-br from-blue-900/50 to-blue-800/30 rounded-xl p-6 border border-blue-500/20">
            <div class="flex items-center justify-between">
                <div>
                    <p class="text-blue-300 text-sm font-medium">Total Crawl Jobs</p>
                    <p class="text-3xl font-bold text-white">{{ stats.total_crawl_jobs }}</p>
                    <p class="text-blue-400 text-xs">+{{ stats.recent_jobs }} this week</p>
                </div>
                <div class="w-12 h-12 bg-blue-600/20 rounded-lg flex items-center justify-center">
                    <i class="fas fa-spider text-blue-400 text-xl"></i>
                </div>
            </div>
        </div>
        
        <!-- Knowledge Pages -->
        <div class="bg-gradient-to-br from-green-900/50 to-green-800/30 rounded-xl p-6 border border-green-500/20">
            <div class="flex items-center justify-between">
                <div>
                    <p class="text-green-300 text-sm font-medium">Knowledge Pages</p>
                    <p class="text-3xl font-bold text-white">{{ stats.total_pages }}</p>
                    <p class="text-green-400 text-xs">{{ ai_status.pages_with_content|default:0 }} with content</p>
                </div>
                <div class="w-12 h-12 bg-green-600/20 rounded-lg flex items-center justify-center">
                    <i class="fas fa-database text-green-400 text-xl"></i>
                </div>
            </div>
        </div>
        
        <!-- Total Words -->
        <div class="bg-gradient-to-br from-purple-900/50 to-purple-800/30 rounded-xl p-6 border border-purple-500/20">
            <div class="flex items-center justify-between">
                <div>
                    <p class="text-purple-300 text-sm font-medium">Total Words</p>
                    <p class="text-3xl font-bold text-white">{{ stats.total_words|floatformat:0|intcomma }}</p>
                    <p class="text-purple-400 text-xs">Extracted content</p>
                </div>
                <div class="w-12 h-12 bg-purple-600/20 rounded-lg flex items-center justify-center">
                    <i class="fas fa-file-text text-purple-400 text-xl"></i>
                </div>
            </div>
        </div>
        
        <!-- AI Status -->
        <div class="bg-gradient-to-br from-yellow-900/50 to-yellow-800/30 rounded-xl p-6 border border-yellow-500/20">
            <div class="flex items-center justify-between">
                <div>
                    <p class="text-yellow-300 text-sm font-medium">AI Assistant</p>
                    <p class="text-3xl font-bold text-white">{{ ai_available|yesno:"Online,Offline" }}</p>
                    <p class="text-yellow-400 text-xs">TinyLlama ready</p>
                </div>
                <div class="w-12 h-12 bg-yellow-600/20 rounded-lg flex items-center justify-center">
                    <i class="fas fa-brain text-yellow-400 text-xl"></i>
                </div>
            </div>
        </div>
    </div>
    
    <!-- Main Content Grid -->
    <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <!-- Recent Activity -->
        <div class="lg:col-span-2">
            <div class="bg-gray-900 rounded-xl border border-gray-800 p-6">
                <div class="flex items-center justify-between mb-6">
                    <h2 class="text-xl font-semibold text-white">Recent Crawl Jobs</h2>
                    <a href="{% url 'crawler:job_create' %}" 
                       class="px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded-lg text-sm transition-colors">
                        <i class="fas fa-plus mr-2"></i>New Crawl
                    </a>
                </div>
                
                {% if recent_crawl_jobs %}
                    <div class="space-y-4">
                        {% for job in recent_crawl_jobs %}
                        <div class="flex items-center justify-between p-4 bg-gray-800 rounded-lg border border-gray-700">
                            <div class="flex items-center space-x-4">
                                <div class="w-10 h-10 bg-gray-700 rounded-lg flex items-center justify-center">
                                    <i class="fas fa-spider text-gray-400"></i>
                                </div>
                                <div>
                                    <h3 class="text-white font-medium">
                                        <code class="text-xs bg-gray-700 px-2 py-1 rounded">{{ job.id|slice:":8" }}</code>
                                    </h3>
                                    <p class="text-sm text-gray-400">{{ job.strategy|title }} • {{ job.start_urls|length }} URL{{ job.start_urls|length|pluralize }}</p>
                                </div>
                            </div>
                            
                            <div class="flex items-center space-x-4">
                                <div class="text-right">
                                    <div class="flex items-center space-x-2">
                                        {% if job.status == 'completed' %}
                                            <span class="w-2 h-2 bg-green-500 rounded-full"></span>
                                            <span class="text-green-400 text-sm">Completed</span>
                                        {% elif job.status == 'failed' %}
                                            <span class="w-2 h-2 bg-red-500 rounded-full"></span>
                                            <span class="text-red-400 text-sm">Failed</span>
                                        {% elif job.status == 'running' %}
                                            <span class="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></span>
                                            <span class="text-blue-400 text-sm">Running</span>
                                        {% else %}
                                            <span class="w-2 h-2 bg-gray-500 rounded-full"></span>
                                            <span class="text-gray-400 text-sm">{{ job.status|title }}</span>
                                        {% endif %}
                                    </div>
                                    <p class="text-xs text-gray-500">{{ job.created_at|timesince }} ago</p>
                                </div>
                                
                                <div class="flex space-x-2">
                                    <a href="{% url 'crawler:job_detail' job.id %}" 
                                       class="p-2 bg-gray-700 hover:bg-gray-600 text-gray-300 rounded-lg transition-colors">
                                        <i class="fas fa-eye text-xs"></i>
                                    </a>
                                    {% if job.status == 'completed' %}
                                        <button onclick="analyzeWithAI('{{ job.id }}')" 
                                                class="p-2 bg-blue-600 hover:bg-blue-500 text-white rounded-lg transition-colors">
                                            <i class="fas fa-brain text-xs"></i>
                                        </button>
                                    {% endif %}
                                </div>
                            </div>
                        </div>
                        {% endfor %}
                    </div>
                    
                    <div class="mt-6 text-center">
                        <a href="{% url 'crawler:job_list' %}" 
                           class="text-blue-400 hover:text-blue-300 text-sm transition-colors">
                            View all crawl jobs <i class="fas fa-arrow-right ml-1"></i>
                        </a>
                    </div>
                {% else %}
                    <div class="text-center py-12">
                        <i class="fas fa-spider text-4xl text-gray-600 mb-4"></i>
                        <h3 class="text-lg font-medium text-gray-400 mb-2">No crawl jobs yet</h3>
                        <p class="text-gray-500 mb-6">Start building your knowledge base by creating your first crawl job.</p>
                        <a href="{% url 'crawler:job_create' %}" 
                           class="inline-flex items-center px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded-lg transition-colors">
                            <i class="fas fa-plus mr-2"></i>
                            Create First Crawl Job
                        </a>
                    </div>
                {% endif %}
            </div>
        </div>
        
        <!-- Quick Actions & Stats -->
        <div class="space-y-6">
            <!-- Quick Actions -->
            <div class="bg-gray-900 rounded-xl border border-gray-800 p-6">
                <h2 class="text-xl font-semibold text-white mb-4">Quick Actions</h2>
                
                <div class="space-y-3">
                    <a href="{% url 'chat' %}" 
                       class="w-full flex items-center space-x-3 p-3 bg-gradient-to-r from-blue-600 to-purple-600 
                              hover:from-blue-500 hover:to-purple-500 text-white rounded-lg transition-all duration-200">
                        <i class="fas fa-comments"></i>
                        <span>Chat with AI</span>
                    </a>
                    
                    <a href="{% url 'crawler:job_create' %}" 
                       class="w-full flex items-center space-x-3 p-3 bg-gray-800 hover:bg-gray-700 
                              text-gray-300 rounded-lg transition-colors">
                        <i class="fas fa-spider"></i>
                        <span>New Crawl Job</span>
                    </a>
                    
                    <a href="{% url 'knowledge_base' %}" 
                       class="w-full flex items-center space-x-3 p-3 bg-gray-800 hover:bg-gray-700 
                              text-gray-300 rounded-lg transition-colors">
                        <i class="fas fa-database"></i>
                        <span>Browse Knowledge</span>
                    </a>
                    
                    <button onclick="generateSystemReport()" 
                            class="w-full flex items-center space-x-3 p-3 bg-gray-800 hover:bg-gray-700 
                                   text-gray-300 rounded-lg transition-colors">
                        <i class="fas fa-chart-line"></i>
                        <span>System Report</span>
                    </button>
                </div>
            </div>
            
            <!-- AI Insights -->
            <div class="bg-gray-900 rounded-xl border border-gray-800 p-6">
                <h2 class="text-xl font-semibold text-white mb-4">AI Insights</h2>
                
                <div id="ai-insights">
                    <div class="space-y-3">
                        <div class="p-3 bg-gray-800 rounded-lg">
                            <p class="text-sm text-gray-300">Your knowledge base contains information about 
                               <span class="text-blue-400 font-medium">cryptocurrency and blockchain technology</span>.</p>
                        </div>
                        
                        <div class="p-3 bg-gray-800 rounded-lg">
                            <p class="text-sm text-gray-300">Consider crawling more sources about 
                               <span class="text-green-400 font-medium">AI and machine learning</span> to expand coverage.</p>
                        </div>
                        
                        <button onclick="generateInsights()" 
                                class="w-full mt-4 px-4 py-2 bg-purple-600 hover:bg-purple-500 text-white 
                                       rounded-lg text-sm transition-colors">
                            <i class="fas fa-magic mr-2"></i>Generate New Insights
                        </button>
                    </div>
                </div>
            </div>
            
            <!-- System Health -->
            <div class="bg-gray-900 rounded-xl border border-gray-800 p-6">
                <h2 class="text-xl font-semibold text-white mb-4">System Health</h2>
                
                <div class="space-y-4">
                    <div class="flex items-center justify-between">
                        <span class="text-gray-400">AI Model</span>
                        <div class="flex items-center space-x-2">
                            <span class="w-2 h-2 bg-green-500 rounded-full"></span>
                            <span class="text-green-400 text-sm">Online</span>
                        </div>
                    </div>
                    
                    <div class="flex items-center justify-between">
                        <span class="text-gray-400">Knowledge Base</span>
                        <div class="flex items-center space-x-2">
                            <span class="w-2 h-2 bg-green-500 rounded-full"></span>
                            <span class="text-green-400 text-sm">Ready</span>
                        </div>
                    </div>
                    
                    <div class="flex items-center justify-between">
                        <span class="text-gray-400">Crawler Service</span>
                        <div class="flex items-center space-x-2">
                            <span class="w-2 h-2 bg-green-500 rounded-full"></span>
                            <span class="text-green-400 text-sm">Available</span>
                        </div>
                    </div>
                    
                    <div class="flex items-center justify-between">
                        <span class="text-gray-400">Background Tasks</span>
                        <div class="flex items-center space-x-2">
                            <span class="w-2 h-2 bg-yellow-500 rounded-full"></span>
                            <span class="text-yellow-400 text-sm">Processing</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>

<!-- AI Analysis Modal -->
<div id="aiAnalysisModal" class="fixed inset-0 bg-black bg-opacity-50 hidden items-center justify-center z-50">
    <div class="bg-gray-900 rounded-xl border border-gray-800 p-6 max-w-2xl w-full mx-4 max-h-[80vh] overflow-y-auto">
        <div class="flex items-center justify-between mb-4">
            <h3 class="text-xl font-semibold text-white">
                <i class="fas fa-brain mr-2 text-blue-400"></i>AI Analysis
            </h3>
            <button onclick="closeModal('aiAnalysisModal')" 
                    class="text-gray-400 hover:text-white transition-colors">
                <i class="fas fa-times"></i>
            </button>
        </div>
        
        <div id="analysis-content">
            <div class="text-center py-8">
                <div class="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
                <p class="mt-4 text-gray-400">AI is analyzing the crawl results...</p>
            </div>
        </div>
    </div>
</div>
{% endblock %}

{% block extra_js %}

<script>
function showModal(modalId) {
    document.getElementById(modalId).classList.remove('hidden');
    document.getElementById(modalId).classList.add('flex');
}

function closeModal(modalId) {
    document.getElementById(modalId).classList.add('hidden');
    document.getElementById(modalId).classList.remove('flex');
}

async function analyzeWithAI(jobId) {
    showModal('aiAnalysisModal');
    
    try {
        const response = await fetch('/api/agents/crawler-agent/analyze_crawl_job/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': document.querySelector('[name=csrfmiddlewaretoken]').value
            },
            body: JSON.stringify({ job_id: jobId })
        });
        
        const data = await response.json();
        
        if (data.error) {
            document.getElementById('analysis-content').innerHTML = `
                <div class="bg-red-900/20 border border-red-500/30 rounded-lg p-4">
                    <div class="flex items-center space-x-2">
                        <i class="fas fa-exclamation-triangle text-red-400"></i>
                        <span class="text-red-300">${data.error}</span>
                    </div>
                </div>
            `;
        } else {
            document.getElementById('analysis-content').innerHTML = `
                <div class="space-y-4">
                    <div class="bg-gray-800 rounded-lg p-4">
                        <h4 class="text-white font-medium mb-3">Crawl Statistics</h4>
                        <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
                            <div class="text-center">
                                <div class="text-2xl font-bold text-blue-400">${data.crawl_stats.total_pages}</div>
                                <div class="text-xs text-gray-400">Pages</div>
                            </div>
                            <div class="text-center">
                                <div class="text-2xl font-bold text-green-400">${data.crawl_stats.total_words.toLocaleString()}</div>
                                <div class="text-xs text-gray-400">Words</div>
                            </div>
                            <div class="text-center">
                                <div class="text-2xl font-bold text-purple-400">${data.crawl_stats.urls.length}</div>
                                <div class="text-xs text-gray-400">URLs</div>
                            </div>
                            <div class="text-center">
                                <div class="text-2xl font-bold text-yellow-400">${data.model_used}</div>
                                <div class="text-xs text-gray-400">AI Model</div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="bg-gray-800 rounded-lg p-4">
                        <h4 class="text-white font-medium mb-3">AI Analysis</h4>
                        <div class="text-gray-300 leading-relaxed">
                            ${data.ai_analysis.replace(/\n/g, '<br>')}
                        </div>
                    </div>
                </div>
            `;
        }
    } catch (error) {
        document.getElementById('analysis-content').innerHTML = `
            <div class="bg-red-900/20 border border-red-500/30 rounded-lg p-4">
                <div class="flex items-center space-x-2">
                    <i class="fas fa-exclamation-triangle text-red-400"></i>
                    <span class="text-red-300">Error: ${error.message}</span>
                </div>
            </div>
        `;
    }
}

async function generateInsights() {
    const insightsDiv = document.getElementById('ai-insights');
    
    insightsDiv.innerHTML = `
        <div class="text-center py-8">
            <div class="inline-block animate-spin rounded-full h-6 w-6 border-b-2 border-purple-500"></div>
            <p class="mt-2 text-gray-400 text-sm">AI is analyzing your system...</p>
        </div>
    `;
    
    try {
        const response = await fetch('/api/chat/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': document.querySelector('[name=csrfmiddlewaretoken]').value
            },
            body: JSON.stringify({
                message: 'Analyze my knowledge base and provide 3 key insights about content coverage, quality, and recommendations for improvement. Keep it concise.',
                use_context: true
            })
        });
        
        const data = await response.json();
        
        if (data.response) {
            insightsDiv.innerHTML = `
                <div class="space-y-3">
                    <div class="p-3 bg-gray-800 rounded-lg">
                        <div class="text-sm text-gray-300">${data.response.replace(/\n/g, '<br>')}</div>
                        <div class="mt-2 text-xs text-gray-500 flex items-center space-x-2">
                            <span>🤖 ${data.model_used}</span>
                            ${data.context_used ? `<span>• 📚 ${data.context_sources || 0} sources</span>` : ''}
                        </div>
                    </div>
                    
                    <button onclick="generateInsights()" 
                            class="w-full mt-4 px-4 py-2 bg-purple-600 hover:bg-purple-500 text-white 
                                   rounded-lg text-sm transition-colors">
                        <i class="fas fa-refresh mr-2"></i>Refresh Insights
                    </button>
                </div>
            `;
        } else {
            throw new Error('No response from AI');
        }
    } catch (error) {
        insightsDiv.innerHTML = `
            <div class="bg-red-900/20 border border-red-500/30 rounded-lg p-3">
                <div class="text-sm text-red-300">Error generating insights: ${error.message}</div>
            </div>
        `;
    }
}

async function generateSystemReport() {
    // Implementation for system report generation
    alert('System report generation coming soon!');
}
</script>

{% endblock %}

# 7. Tailwind Theme Customization

# theme/static_src/tailwind.config.js

module.exports = {
content: [
'../templates/**/*.html',
'../../templates/**/*.html',
'../../**/templates/**/*.html',
'../../**/*.py',
],
darkMode: 'class',
theme: {
extend: {
colors: {
// Custom color palette for Omeruta Brain
'omeruta': {
50: '#f0f9ff',
100: '#e0f2fe',
500: '#3b82f6',
600: '#2563eb',
700: '#1d4ed8',
800: '#1e40af',
900: '#1e3a8a',
},
// Dark theme colors
'dark': {
50: '#f8fafc',
100: '#f1f5f9',
200: '#e2e8f0',
300: '#cbd5e1',
400: '#94a3b8',
500: '#64748b',
600: '#475569',
700: '#334155',
800: '#1e293b',
900: '#0f172a',
950: '#020617',
}
},
fontFamily: {
'sans': ['Inter', 'system-ui', 'sans-serif'],
'mono': ['JetBrains Mono', 'Fira Code', 'monospace'],
},
animation: {
'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
'bounce-slow': 'bounce 2s infinite',
'gradient': 'gradient 15s ease infinite',
},
keyframes: {
gradient: {
'0%, 100%': {
'background-size': '200% 200%',
'background-position': 'left center'
},
'50%': {
'background-size': '200% 200%',
'background-position': 'right center'
},
},
},
backdropBlur: {
xs: '2px',
},
},
},
plugins: [
require('@tailwindcss/forms'),
require('@tailwindcss/typography'),
require('@tailwindcss/aspect-ratio'),
],
}

# 8. Custom CSS for Advanced Styling

# theme/static_src/src/styles.css

@import "tailwindcss/base";
@import "tailwindcss/components";
@import "tailwindcss/utilities";

/_ Custom components for Omeruta Brain _/
@layer components {
.glass-effect {
@apply bg-white/10 backdrop-blur-md border border-white/20;
}

.gradient-text {
@apply bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent;
}

.chat-bubble-user {
@apply bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-2xl rounded-tr-sm px-4 py-3 max-w-md ml-auto;
}

.chat-bubble-ai {
@apply bg-gray-800 text-gray-100 rounded-2xl rounded-tl-sm px-4 py-3 max-w-md;
}

.btn-primary {
@apply bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-500 hover:to-purple-500 text-white font-medium py-2 px-4 rounded-lg transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-gray-900;
}

.btn-secondary {
@apply bg-gray-800 hover:bg-gray-700 text-gray-300 hover:text-white font-medium py-2 px-4 rounded-lg transition-colors focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-offset-2 focus:ring-offset-gray-900;
}

.card {
@apply bg-gray-900 border border-gray-800 rounded-xl shadow-xl;
}

.input-field {
@apply bg-gray-800 border border-gray-700 text-white placeholder-gray-400 rounded-lg px-4 py-3 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent;
}

.status-indicator {
@apply w-2 h-2 rounded-full;
}

.status-online {
@apply bg-green-500 animate-pulse;
}

.status-offline {
@apply bg-red-500;
}

.status-processing {
@apply bg-yellow-500 animate-bounce;
}
}

/_ Custom animations _/
@layer utilities {# Django + Tailwind CSS Implementation for Omeruta Brain

## Why Django + Tailwind is Perfect for Your Project:

### ✅ **Modern Design System**

- Beautiful, consistent components
- Dark mode support (perfect for AI tools)
- Responsive by default
- Professional look that rivals Next.js apps

### ✅ **Developer Experience**

- Utility-first CSS (faster than writing custom CSS)
- Live reload during development
- Built-in purging (small CSS files)
- Easy customization with theme files

### ✅ **AI Tool Aesthetic**

- Perfect for dashboards and admin interfaces
- Great typography for chat interfaces
- Excellent form styling
- Beautiful data visualization containers

## Complete Setup Guide

# 1. Installation

# Add to requirements.txt

django-tailwind==3.8.0
django-browser-reload==1.12.1 # For hot reload during development

# Install

pip install django-tailwind django-browser-reload

# 2. Django Settings Configuration

# omeruta_brain_project/settings.py

INSTALLED_APPS = [
# ... your existing apps
'tailwind',
'theme', # This will be created
'django_browser_reload', # For development hot reload
]

MIDDLEWARE = [
# ... your existing middleware
'django_browser_reload.middleware.BrowserReloadMiddleware', # Development only
]

# Tailwind configuration

TAILWIND_APP_NAME = 'theme'

# For development

INTERNAL_IPS = [
"127.0.0.1",
]

# Static files (make sure these are set)

STATIC_URL = '/static/'
STATICFILES_DIRS = [BASE_DIR / 'static']
STATIC_ROOT = BASE_DIR / 'staticfiles'

# 3. Initialize Tailwind

# Run these commands:

python manage.py tailwind init
python manage.py tailwind install
python manage.py tailwind build

# For development with live reload:

python manage.py tailwind start

# 4. Base Template with Tailwind

# templates/base.html

<!DOCTYPE html>
<html lang="en" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}Omeruta Brain{% endblock %}</title>
    
    {% load static tailwind_tags %}
    {% tailwind_css %}
    
    <!-- Font Awesome for icons -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    
    {% block extra_css %}{% endblock %}
</head>
<body class="bg-gray-950 text-gray-100 min-h-screen">
    <!-- Navigation -->
    <nav class="bg-gray-900 border-b border-gray-800 sticky top-0 z-50">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div class="flex justify-between h-16">
                <!-- Logo and main nav -->
                <div class="flex items-center space-x-8">
                    <a href="{% url 'dashboard' %}" class="flex items-center space-x-3">
                        <div class="w-8 h-8 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
                            <i class="fas fa-brain text-white text-sm"></i>
                        </div>
                        <span class="text-xl font-bold bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
                            Omeruta Brain
                        </span>
                    </a>
                    
                    <!-- Desktop Navigation -->
                    <div class="hidden md:flex space-x-6">
                        <a href="{% url 'dashboard' %}" 
                           class="flex items-center space-x-2 px-3 py-2 rounded-lg text-sm font-medium transition-colors
                                  {% if request.resolver_match.url_name == 'dashboard' %}bg-gray-800 text-blue-400{% else %}text-gray-300 hover:text-white hover:bg-gray-800{% endif %}">
                            <i class="fas fa-tachometer-alt text-xs"></i>
                            <span>Dashboard</span>
                        </a>
                        
                        <a href="{% url 'chat' %}" 
                           class="flex items-center space-x-2 px-3 py-2 rounded-lg text-sm font-medium transition-colors
                                  {% if request.resolver_match.url_name == 'chat' %}bg-gray-800 text-blue-400{% else %}text-gray-300 hover:text-white hover:bg-gray-800{% endif %}">
                            <i class="fas fa-comments text-xs"></i>
                            <span>AI Chat</span>
                        </a>
                        
                        <a href="{% url 'crawler:job_list' %}" 
                           class="flex items-center space-x-2 px-3 py-2 rounded-lg text-sm font-medium transition-colors
                                  {% if 'crawler' in request.resolver_match.url_name %}bg-gray-800 text-blue-400{% else %}text-gray-300 hover:text-white hover:bg-gray-800{% endif %}">
                            <i class="fas fa-spider text-xs"></i>
                            <span>Crawler</span>
                        </a>
                        
                        <a href="{% url 'knowledge_base' %}" 
                           class="flex items-center space-x-2 px-3 py-2 rounded-lg text-sm font-medium transition-colors
                                  {% if request.resolver_match.url_name == 'knowledge_base' %}bg-gray-800 text-blue-400{% else %}text-gray-300 hover:text-white hover:bg-gray-800{% endif %}">
                            <i class="fas fa-database text-xs"></i>
                            <span>Knowledge</span>
                        </a>
                    </div>
                </div>
                
                <!-- Right side -->
                <div class="flex items-center space-x-4">
                    <!-- AI Status -->
                    <div class="hidden sm:flex items-center space-x-2 text-xs">
                        <div class="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                        <span class="text-gray-400">AI Online</span>
                    </div>
                    
                    <!-- User Menu -->
                    <div class="relative" x-data="{ open: false }">
                        <button @click="open = !open" 
                                class="flex items-center space-x-3 p-2 rounded-lg hover:bg-gray-800 transition-colors">
                            <div class="w-8 h-8 bg-gradient-to-r from-blue-500 to-purple-600 rounded-full flex items-center justify-center">
                                <span class="text-white text-sm font-medium">
                                    {{ user.first_name|first|default:user.email|first|upper }}
                                </span>
                            </div>
                            <div class="hidden sm:block text-left">
                                <div class="text-sm font-medium text-white">{{ user.get_full_name|default:user.username }}</div>
                                <div class="text-xs text-gray-400">{{ user.email }}</div>
                            </div>
                            <i class="fas fa-chevron-down text-xs text-gray-400"></i>
                        </button>
                        
                        <!-- Dropdown -->
                        <div x-show="open" @click.away="open = false"
                             x-transition:enter="transition ease-out duration-100"
                             x-transition:enter-start="transform opacity-0 scale-95"
                             x-transition:enter-end="transform opacity-100 scale-100"
                             x-transition:leave="transition ease-in duration-75"
                             x-transition:leave-start="transform opacity-100 scale-100"
                             x-transition:leave-end="transform opacity-0 scale-95"
                             class="absolute right-0 mt-2 w-48 bg-gray-800 rounded-lg shadow-lg border border-gray-700 py-1 z-50">
                            <a href="#" class="block px-4 py-2 text-sm text-gray-300 hover:bg-gray-700 hover:text-white">
                                <i class="fas fa-user w-4"></i> Profile
                            </a>
                            <a href="#" class="block px-4 py-2 text-sm text-gray-300 hover:bg-gray-700 hover:text-white">
                                <i class="fas fa-cog w-4"></i> Settings
                            </a>
                            <div class="border-t border-gray-700 my-1"></div>
                            <a href="{% url 'logout' %}" class="block px-4 py-2 text-sm text-red-400 hover:bg-gray-700">
                                <i class="fas fa-sign-out-alt w-4"></i> Logout
                            </a>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </nav>

    <!-- Main Content -->
    <main class="flex-1">
        <!-- Messages -->
        {% if messages %}
            <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-4">
                {% for message in messages %}
                    <div class="mb-4 p-4 rounded-lg border
                                {% if message.tags == 'success' %}bg-green-900/20 border-green-500/30 text-green-400
                                {% elif message.tags == 'error' %}bg-red-900/20 border-red-500/30 text-red-400
                                {% elif message.tags == 'warning' %}bg-yellow-900/20 border-yellow-500/30 text-yellow-400
                                {% else %}bg-blue-900/20 border-blue-500/30 text-blue-400{% endif %}">
                        {{ message }}
                    </div>
                {% endfor %}
            </div>
        {% endif %}

        {% block content %}{% endblock %}
    </main>

    <!-- Footer -->
    <footer class="bg-gray-900 border-t border-gray-800 mt-auto">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
            <div class="flex justify-between items-center text-sm text-gray-400">
                <div class="flex items-center space-x-4">
                    <span>&copy; 2025 Omeruta Brain v1.0</span>
                    <div class="flex items-center space-x-1">
                        <div class="w-2 h-2 bg-green-500 rounded-full"></div>
                        <span>Backend Healthy</span>
                    </div>
                </div>
                <div class="flex items-center space-x-4">
                    <span>{{ total_pages|default:0 }} pages in knowledge base</span>
                </div>
            </div>
        </div>
    </footer>

    <!-- Alpine.js for dropdown interactions -->
    <script src="https://unpkg.com/alpinejs@3.x.x/dist/cdn.min.js" defer></script>

    <!-- HTMX for dynamic interactions -->
    <script src="https://unpkg.com/htmx.org@1.9.6"></script>

    {% if settings.DEBUG %}
        {% load django_browser_reload %}
        {% django_browser_reload_script %}
    {% endif %}

    {% block extra_js %}{% endblock %}

</body>
</html>

# 5. Modern AI Chat Interface

# templates/core/chat.html

{% extends 'base.html' %}
{% load static %}

{% block title %}AI Chat - Omeruta Brain{% endblock %}

{% block content %}

<div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
    <div class="grid grid-cols-1 lg:grid-cols-4 gap-6">
        <!-- Chat Interface -->
        <div class="lg:col-span-3">
            <div class="bg-gray-900 rounded-xl border border-gray-800 shadow-xl overflow-hidden">
                <!-- Chat Header -->
                <div class="px-6 py-4 border-b border-gray-800 bg-gradient-to-r from-gray-900 to-gray-800">
                    <div class="flex items-center justify-between">
                        <div class="flex items-center space-x-3">
                            <div class="w-10 h-10 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
                                <i class="fas fa-robot text-white"></i>
                            </div>
                            <div>
                                <h1 class="text-lg font-semibold text-white">AI Assistant</h1>
                                <p class="text-sm text-gray-400">Powered by TinyLlama with your knowledge base</p>
                            </div>
                        </div>
                        
                        <div class="flex items-center space-x-2">
                            <button onclick="clearChat()" 
                                    class="px-3 py-1 text-xs bg-gray-700 hover:bg-gray-600 text-gray-300 rounded-lg transition-colors">
                                <i class="fas fa-trash mr-1"></i> Clear
                            </button>
                            <button onclick="exportChat()" 
                                    class="px-3 py-1 text-xs bg-blue-600 hover:bg-blue-500 text-white rounded-lg transition-colors">
                                <i class="fas fa-download mr-1"></i> Export
                            </button>
                        </div>
                    </div>
                </div>
                
                <!-- Chat Messages -->
                <div id="chat-messages" class="h-96 lg:h-[500px] overflow-y-auto p-6 space-y-4 bg-gray-950">
                    <!-- Welcome Message -->
                    <div class="flex items-start space-x-3">
                        <div class="w-8 h-8 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg flex items-center justify-center flex-shrink-0">
                            <i class="fas fa-brain text-white text-sm"></i>
                        </div>
                        <div class="flex-1">
                            <div class="bg-gray-800 rounded-2xl rounded-tl-sm px-4 py-3 max-w-md">
                                <p class="text-gray-100">
                                    👋 Hello! I'm your AI assistant with access to your knowledge base. 
                                    I can help you find information, analyze content, and answer questions about your crawled data.
                                </p>
                                <div class="mt-2 text-xs text-gray-400 flex items-center space-x-2">
                                    <span>🤖 TinyLlama</span>
                                    <span>•</span>
                                    <span>📚 {{ ai_status.total_crawled_pages|default:0 }} pages available</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Typing Indicator -->
                <div id="typing-indicator" class="hidden px-6 py-3 border-t border-gray-800">
                    <div class="flex items-center space-x-3">
                        <div class="w-8 h-8 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
                            <div class="flex space-x-1">
                                <div class="w-1 h-1 bg-white rounded-full animate-bounce"></div>
                                <div class="w-1 h-1 bg-white rounded-full animate-bounce" style="animation-delay: 0.1s"></div>
                                <div class="w-1 h-1 bg-white rounded-full animate-bounce" style="animation-delay: 0.2s"></div>
                            </div>
                        </div>
                        <div class="flex-1">
                            <span id="typing-text" class="text-sm text-gray-400">AI is thinking...</span>
                            <div id="progress-container" class="hidden mt-2">
                                <div class="w-full bg-gray-700 rounded-full h-1">
                                    <div id="progress-bar" class="bg-gradient-to-r from-blue-500 to-purple-600 h-1 rounded-full transition-all duration-300" style="width: 0%"></div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Chat Input -->
                <div class="px-6 py-4 border-t border-gray-800 bg-gray-900">
                    <form id="chat-form" onsubmit="sendMessage(event)" class="space-y-3">
                        <div class="flex space-x-3">
                            <div class="flex-1">
                                <textarea 
                                    id="message-input"
                                    rows="1"
                                    class="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-3 text-white placeholder-gray-400 
                                           focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
                                    placeholder="Ask me anything about your knowledge base..."
                                    maxlength="1000"></textarea>
                            </div>
                            <button type="submit" 
                                    id="send-button"
                                    class="px-6 py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-lg 
                                           hover:from-blue-500 hover:to-purple-500 focus:outline-none focus:ring-2 focus:ring-blue-500 
                                           disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200">
                                <i class="fas fa-paper-plane"></i>
                            </button>
                        </div>
                        
                        <!-- Options -->
                        <div class="flex items-center justify-between">
                            <div class="flex items-center space-x-4">
                                <label class="flex items-center space-x-2">
                                    <input type="checkbox" id="use-context" checked 
                                           class="w-4 h-4 text-blue-600 bg-gray-700 border-gray-600 rounded focus:ring-blue-500">
                                    <span class="text-sm text-gray-400">Use knowledge base context</span>
                                </label>
                            </div>
                            <div class="text-xs text-gray-500">
                                <span id="char-count">0</span>/1000
                            </div>
                        </div>
                    </form>
                </div>
            </div>
        </div>
        
        <!-- Sidebar -->
        <div class="space-y-6">
            <!-- Knowledge Base Stats -->
            <div class="bg-gray-900 rounded-xl border border-gray-800 p-6">
                <h2 class="text-lg font-semibold text-white mb-4 flex items-center">
                    <i class="fas fa-database mr-2 text-blue-400"></i>
                    Knowledge Base
                </h2>
                
                <div class="grid grid-cols-2 gap-4 mb-4">
                    <div class="text-center">
                        <div class="text-2xl font-bold text-blue-400">{{ ai_status.total_crawled_pages|default:0 }}</div>
                        <div class="text-xs text-gray-400">Total Pages</div>
                    </div>
                    <div class="text-center">
                        <div class="text-2xl font-bold text-green-400">{{ ai_status.pages_with_content|default:0 }}</div>
                        <div class="text-xs text-gray-400">With Content</div>
                    </div>
                </div>
                
                <a href="{% url 'knowledge_base' %}" 
                   class="w-full bg-gray-800 hover:bg-gray-700 text-gray-300 py-2 px-4 rounded-lg text-sm 
                          transition-colors flex items-center justify-center">
                    <i class="fas fa-eye mr-2"></i>
                    View Knowledge Base
                </a>
            </div>
            
            <!-- Quick Actions -->
            <div class="bg-gray-900 rounded-xl border border-gray-800 p-6">
                <h2 class="text-lg font-semibold text-white mb-4 flex items-center">
                    <i class="fas fa-bolt mr-2 text-yellow-400"></i>
                    Quick Actions
                </h2>
                
                <div class="space-y-2">
                    <button onclick="quickAsk('What topics are covered in the knowledge base?')" 
                            class="w-full text-left px-3 py-2 bg-gray-800 hover:bg-gray-700 text-gray-300 rounded-lg text-sm transition-colors">
                        📊 Knowledge Summary
                    </button>
                    <button onclick="quickAsk('What are the latest additions to the knowledge base?')" 
                            class="w-full text-left px-3 py-2 bg-gray-800 hover:bg-gray-700 text-gray-300 rounded-lg text-sm transition-colors">
                        🆕 Recent Updates
                    </button>
                    <button onclick="quickAsk('Analyze the quality of the content in the knowledge base')" 
                            class="w-full text-left px-3 py-2 bg-gray-800 hover:bg-gray-700 text-gray-300 rounded-lg text-sm transition-colors">
                        🔍 Quality Analysis
                    </button>
                </div>
            </div>
            
            <!-- Session Stats -->
            <div class="bg-gray-900 rounded-xl border border-gray-800 p-6">
                <h2 class="text-lg font-semibold text-white mb-4 flex items-center">
                    <i class="fas fa-chart-line mr-2 text-purple-400"></i>
                    Session Stats
                </h2>
                
                <div class="grid grid-cols-1 gap-3">
                    <div class="flex justify-between items-center">
                        <span class="text-sm text-gray-400">Messages</span>
                        <span id="message-count" class="text-lg font-semibold text-white">0</span>
                    </div>
                    <div class="flex justify-between items-center">
                        <span class="text-sm text-gray-400">Avg Response</span>
                        <span id="avg-response-time" class="text-lg font-semibold text-white">0s</span>
                    </div>
                    <div class="flex justify-between items-center">
                        <span class="text-sm text-gray-400">Context Usage</span>
                        <span id="context-usage" class="text-lg font-semibold text-white">0%</span>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>
{% endblock %}

{% block extra_js %}

<script>
let conversationId = null;
let messageCount = 0;
let totalResponseTime = 0;
let contextUsageCount = 0;

// Auto-resize textarea
const messageInput = document.getElementById('message-input');
messageInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = Math.min(this.scrollHeight, 150) + 'px';
    
    // Update character count
    document.getElementById('char-count').textContent = this.value.length;
});

async function sendMessage(event) {
    event.preventDefault();
    
    const message = messageInput.value.trim();
    if (!message) return;
    
    // Add user message
    addMessage(message, 'user');
    messageInput.value = '';
    messageInput.style.height = 'auto';
    document.getElementById('char-count').textContent = '0';
    
    // Show typing indicator
    showTypingIndicator('AI is processing your message...');
    
    // Disable send button
    const sendButton = document.getElementById('send-button');
    sendButton.disabled = true;
    
    const useContext = document.getElementById('use-context').checked;
    
    try {
        const startTime = Date.now();
        
        const response = await fetch('/api/chat/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': document.querySelector('[name=csrfmiddlewaretoken]').value,
            },
            body: JSON.stringify({
                message: message,
                use_context: useContext,
                conversation_id: conversationId
            })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            const responseTime = (Date.now() - startTime) / 1000;
            
            if (data.conversation_id) {
                conversationId = data.conversation_id;
            }
            
            addMessage(data.response, 'ai', {
                model: data.model_used,
                agent: data.selected_agent,
                contextUsed: data.context_used,
                contextSources: data.context_sources,
                responseTime: responseTime
            });
            
            updateStatistics(responseTime, data.context_used);
        } else {
            addMessage(`Error: ${data.error || 'Something went wrong'}`, 'ai', {error: true});
        }
        
    } catch (error) {
        addMessage(`Error: ${error.message}`, 'ai', {error: true});
    } finally {
        hideTypingIndicator();
        sendButton.disabled = false;
        messageInput.focus();
    }
}

function addMessage(content, sender, metadata = {}) {
    const chatMessages = document.getElementById('chat-messages');
    const messageDiv = document.createElement('div');
    
    if (sender === 'user') {
        messageDiv.className = 'flex items-start space-x-3 justify-end';
        messageDiv.innerHTML = `
            <div class="flex-1 max-w-md">
                <div class="bg-gradient-to-r from-blue-600 to-purple-600 rounded-2xl rounded-tr-sm px-4 py-3">
                    <p class="text-white">${formatMessage(content)}</p>
                </div>
            </div>
            <div class="w-8 h-8 bg-gray-700 rounded-lg flex items-center justify-center flex-shrink-0">
                <i class="fas fa-user text-gray-300 text-sm"></i>
            </div>
        `;
        messageCount++;
    } else {
        const isError = metadata.error;
        const bgClass = isError ? 'bg-red-900/50 border border-red-500/30' : 'bg-gray-800';
        const textClass = isError ? 'text-red-200' : 'text-gray-100';
        
        let agentInfo = '';
        if (metadata.model && !isError) {
            const contextInfo = metadata.contextUsed ? 
                `📚 ${metadata.contextSources || 0} sources` : '🧠 General knowledge';
            
            agentInfo = `
                <div class="mt-2 text-xs text-gray-400 flex items-center space-x-2">
                    <span>🤖 ${metadata.model}</span>
                    <span>•</span>
                    <span>${contextInfo}</span>
                    ${metadata.responseTime ? `<span>• ⚡ ${metadata.responseTime.toFixed(2)}s</span>` : ''}
                </div>
            `;
        }
        
        messageDiv.className = 'flex items-start space-x-3';
        messageDiv.innerHTML = `
            <div class="w-8 h-8 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg flex items-center justify-center flex-shrink-0">
                <i class="fas fa-${isError ? 'exclamation-triangle' : 'brain'} text-white text-sm"></i>
            </div>
            <div class="flex-1 max-w-md">
                <div class="${bgClass} rounded-2xl rounded-tl-sm px-4 py-3">
                    <p class="${textClass}">${formatMessage(content)}</p>
                    ${agentInfo}
                </div>
            </div>
        `;
    }
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function formatMessage(content) {
    return content
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/`(.*?)`/g, '<code class="bg-gray-700 px-1 rounded">$1</code>')
        .replace(/\n/g, '<br>');
}

function showTypingIndicator(text) {
    const indicator = document.getElementById('typing-indicator');
    const typingText = document.getElementById('typing-text');
    
    typingText.textContent = text;
    indicator.classList.remove('hidden');
    
    const chatMessages = document.getElementById('chat-messages');
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function hideTypingIndicator() {
    document.getElementById('typing-indicator').classList.add('hidden');
}

function quickAsk(question) {
    messageInput.value = question;
    sendMessage(new Event('submit'));
}

function clearChat() {
    const chatMessages = document.getElementById('chat-messages');
    chatMessages.innerHTML = `
        <div class="flex items-start space-x-3">
            <div class="w-8 h-8 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg flex items-center justify-center flex-shrink-0">
                <i class="fas fa-brain text-white
