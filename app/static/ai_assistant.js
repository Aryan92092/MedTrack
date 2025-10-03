// AI Assistant JavaScript
class AIAssistant {
    constructor() {
        this.isOpen = false;
        this.isTyping = false;
        this.currentCSV = null;
        this.currentCSVFilename = null;
        this.quickAnswers = {};
        
        this.initializeElements();
        this.bindEvents();
        this.loadQuickAnswers();
        this.loadChatHistory();
    }
    
    initializeElements() {
        // Main elements
        this.widget = document.getElementById('ai-assistant-widget');
        this.toggleBtn = document.getElementById('ai-toggle-btn');
        this.chatWindow = document.getElementById('ai-chat-window');
        this.chatMessages = document.getElementById('ai-chat-messages');
        this.messageInput = document.getElementById('ai-message-input');
        this.sendBtn = document.getElementById('ai-send-btn');
        this.typingIndicator = document.getElementById('ai-typing-indicator');
        this.badge = document.getElementById('ai-badge');
        
        // Control buttons
        this.minimizeBtn = document.getElementById('ai-minimize-btn');
        this.closeBtn = document.getElementById('ai-close-btn');
        this.clearBtn = document.getElementById('ai-clear-btn');
        this.csvBtn = document.getElementById('ai-csv-btn');
        
        // CSV upload elements
        this.csvUpload = document.getElementById('ai-csv-upload');
        this.csvFile = document.getElementById('ai-csv-file');
        this.csvSelect = document.getElementById('ai-csv-select');
        this.csvClose = document.getElementById('ai-csv-close');
        this.csvInfo = document.getElementById('ai-csv-info');
        
        // Quick actions
        this.quickActions = document.getElementById('ai-quick-actions');
        this.quickButtons = document.getElementById('ai-quick-buttons');
        this.quickHeader = this.quickActions ? this.quickActions.querySelector('h5') : null;
        this.quickCollapsed = true;
    }
    
    bindEvents() {
        // Toggle chat window
        this.toggleBtn.addEventListener('click', () => this.toggleChat());
        this.minimizeBtn.addEventListener('click', () => this.toggleChat());
        this.closeBtn.addEventListener('click', () => this.toggleChat());
        
        // Message input
        this.messageInput.addEventListener('input', () => this.handleInputChange());
        this.messageInput.addEventListener('keydown', (e) => this.handleKeyDown(e));
        this.sendBtn.addEventListener('click', () => this.sendMessage());
        
        // CSV functionality
        this.csvBtn.addEventListener('click', () => this.toggleCSVUpload());
        this.csvSelect.addEventListener('click', () => this.csvFile.click());
        this.csvFile.addEventListener('change', (e) => this.handleCSVUpload(e));
        this.csvClose.addEventListener('click', () => this.toggleCSVUpload());
        
        // Clear chat
        this.clearBtn.addEventListener('click', () => this.clearChat());
        
        // Auto-resize textarea
        this.messageInput.addEventListener('input', () => this.autoResizeTextarea());
        // Collapse/expand quick actions by clicking the header
        if (this.quickHeader) {
            this.quickHeader.style.cursor = 'pointer';
            this.quickHeader.title = 'Show/Hide quick questions';
            this.quickHeader.addEventListener('click', () => this.toggleQuickActions());
        }
    }
    
    toggleChat() {
        this.isOpen = !this.isOpen;
        this.chatWindow.style.display = this.isOpen ? 'flex' : 'none';
        
        if (this.isOpen) {
            this.messageInput.focus();
            this.scrollToBottom();
            this.hideBadge();
            // Collapse quick actions by default to maximize space for messages
            this.setQuickActionsCollapsed(true);
        }
    }
    
    toggleCSVUpload() {
        const isVisible = this.csvUpload.style.display !== 'none';
        this.csvUpload.style.display = isVisible ? 'none' : 'block';
    }
    
    handleInputChange() {
        const hasText = this.messageInput.value.trim().length > 0;
        this.sendBtn.disabled = !hasText;
    }
    
    handleKeyDown(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            this.sendMessage();
        }
    }
    
    autoResizeTextarea() {
        this.messageInput.style.height = 'auto';
        this.messageInput.style.height = Math.min(this.messageInput.scrollHeight, 100) + 'px';
    }
    
    async sendMessage() {
        const message = this.messageInput.value.trim();
        if (!message || this.isTyping) return;
        
        // Add user message to chat
        this.addMessage('user', message);
        this.messageInput.value = '';
        this.sendBtn.disabled = true;
        this.autoResizeTextarea();
        
        // Show typing indicator
        this.showTyping();
        // Ensure quick actions are collapsed while conversing
        this.setQuickActionsCollapsed(true);
        
        try {
            // Determine which endpoint to use
            const endpoint = this.currentCSV ? '/ai/chat-with-csv' : '/ai/chat';
            const payload = {
                message: message
            };
            
            if (this.currentCSV) {
                payload.csv_content = this.currentCSV;
                payload.csv_filename = this.currentCSVFilename;
            }
            
            const response = await fetch(endpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(payload)
            });
            
            const data = await response.json();

            if (data && data.success && data.response) {
                this.addMessage('assistant', data.response);
            } else if (data && data.error) {
                this.addMessage('assistant', `Sorry, I encountered an error: ${data.error}`);
            } else {
                this.addMessage('assistant', 'Sorry, I did not receive a response.');
            }
        } catch (error) {
            this.addMessage('assistant', 'Sorry, I\'m having trouble connecting right now. Please try again later.');
            console.error('AI Assistant Error:', error);
        } finally {
            this.hideTyping();
        }
    }

    setQuickActionsCollapsed(collapsed) {
        this.quickCollapsed = collapsed;
        if (!this.quickActions || !this.quickButtons) return;
        if (collapsed) {
            this.quickButtons.style.display = 'none';
            this.quickActions.classList.add('collapsed');
        } else {
            this.quickButtons.style.display = 'flex';
            this.quickActions.classList.remove('collapsed');
        }
    }

    toggleQuickActions() {
        this.setQuickActionsCollapsed(!this.quickCollapsed);
    }
    
    addMessage(role, content) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `ai-message ai-message-${role}`;
        
        const avatar = document.createElement('div');
        avatar.className = 'ai-message-avatar';
        
        if (role === 'assistant') {
            avatar.innerHTML = `
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"/>
                </svg>
            `;
        } else {
            avatar.innerHTML = `
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
                    <circle cx="12" cy="7" r="4"></circle>
                </svg>
            `;
        }
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'ai-message-content';
        contentDiv.innerHTML = this.formatMessage(content);
        
        messageDiv.appendChild(avatar);
        messageDiv.appendChild(contentDiv);
        
        this.chatMessages.appendChild(messageDiv);
        this.scrollToBottom();
    }
    
    formatMessage(content) {
        // Convert line breaks to <br> and handle basic formatting
        return content
            .replace(/\n/g, '<br>')
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>');
    }
    
    showTyping() {
        this.isTyping = true;
        this.typingIndicator.style.display = 'flex';
        this.scrollToBottom();
    }
    
    hideTyping() {
        this.isTyping = false;
        this.typingIndicator.style.display = 'none';
    }
    
    scrollToBottom() {
        setTimeout(() => {
            this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
        }, 100);
    }
    
    showBadge() {
        this.badge.style.display = 'flex';
    }
    
    hideBadge() {
        this.badge.style.display = 'none';
    }
    
    async loadQuickAnswers() {
        try {
            const response = await fetch('/ai/quick-answers');
            const data = await response.json();
            
            if (data.success) {
                this.quickAnswers = data.quick_answers;
                this.renderQuickButtons();
            }
        } catch (error) {
            console.error('Failed to load quick answers:', error);
        }
    }
    
    renderQuickButtons() {
        this.quickButtons.innerHTML = '';
        
        Object.keys(this.quickAnswers).forEach(question => {
            const button = document.createElement('button');
            button.className = 'ai-quick-btn';
            button.textContent = question;
            button.addEventListener('click', () => {
                this.messageInput.value = question;
                this.handleInputChange();
                this.sendMessage();
            });
            this.quickButtons.appendChild(button);
        });
    }
    
    async loadChatHistory() {
        try {
            const response = await fetch('/ai/history');
            const data = await response.json();
            
            if (data.success && data.history.length > 0) {
                // Clear the initial welcome message
                this.chatMessages.innerHTML = '';
                
                // Load conversation history
                data.history.forEach(msg => {
                    this.addMessage(msg.role, msg.content);
                });
                
                // Show badge if there are unread messages
                if (data.history.length > 0) {
                    this.showBadge();
                }
            }
        } catch (error) {
            console.error('Failed to load chat history:', error);
        }
    }
    
    async clearChat() {
        if (confirm('Are you sure you want to clear the chat history?')) {
            try {
                const response = await fetch('/ai/clear-history', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    }
                });
                
                const data = await response.json();
                
                if (data.success) {
                    // Clear the chat messages
                    this.chatMessages.innerHTML = '';
                    
                    // Add welcome message back
                    this.addMessage('assistant', `Hello! I'm your AI assistant for MediTrack. I can help you with:

• Stock and inventory management
• CSV data analysis  
• Medicine optimization tips
• Expiry date management

How can I assist you today?`);
                }
            } catch (error) {
                console.error('Failed to clear chat history:', error);
            }
        }
    }
    
    handleCSVUpload(event) {
        const file = event.target.files[0];
        if (!file) return;
        
        if (!file.name.toLowerCase().endsWith('.csv')) {
            alert('Please select a CSV file.');
            return;
        }
        
        const reader = new FileReader();
        reader.onload = (e) => {
            this.currentCSV = e.target.result;
            this.currentCSVFilename = file.name;
            
            // Show CSV info
            this.csvInfo.innerHTML = `
                <strong>File loaded:</strong> ${file.name}<br>
                <strong>Size:</strong> ${(file.size / 1024).toFixed(1)} KB<br>
                <em>You can now ask questions about this CSV data!</em>
            `;
            this.csvInfo.style.display = 'block';
            
            // Hide CSV upload area
            this.toggleCSVUpload();
            
            // Add a message about CSV upload
            this.addMessage('assistant', `Great! I've loaded your CSV file "${file.name}". You can now ask me questions about the data, request analysis, or get insights about your inventory. What would you like to know?`);
        };
        
        reader.readAsText(file);
    }
}

// Initialize AI Assistant when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.aiAssistant = new AIAssistant();
});

// Handle page visibility changes to show badge for new messages
document.addEventListener('visibilitychange', () => {
    if (document.hidden && window.aiAssistant && !window.aiAssistant.isOpen) {
        // Page is hidden and chat is closed - could show badge for new messages
        // This is a placeholder for future notification functionality
    }
});
