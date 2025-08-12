document.addEventListener('DOMContentLoaded', function() {
    const chatForm = document.getElementById('chatForm');
    const messageInput = document.getElementById('messageInput');
    const sendBtn = document.getElementById('sendBtn');
    const chatMessages = document.getElementById('chatMessages');
    const typingIndicator = document.getElementById('typingIndicator');

    chatForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const message = messageInput.value.trim();
        if (!message) return;

        // Disable input and show loading
        messageInput.disabled = true;
        sendBtn.disabled = true;
        typingIndicator.style.display = 'block';

        // Add user message to chat
        addMessageToChat('user', message);
        messageInput.value = '';

        // Scroll to bottom
        scrollToBottom();

        try {
            // Send message to server
            const formData = new FormData();
            formData.append('message', message);

            const response = await fetch('/chat/message', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.success) {
                // Add AI response to chat
                addMessageToChat('assistant', data.response, data.citations);
            } else {
                addMessageToChat('error', data.message || 'Sorry, something went wrong. Please try again.');
            }
        } catch (error) {
            console.error('Chat error:', error);
            addMessageToChat('error', 'Network error. Please check your connection and try again.');
        } finally {
            // Re-enable input and hide loading
            messageInput.disabled = false;
            sendBtn.disabled = false;
            typingIndicator.style.display = 'none';
            messageInput.focus();
            scrollToBottom();
        }
    });

    function addMessageToChat(type, content, citations = []) {
        // Clear welcome message on first message
        const existingWelcome = chatMessages.querySelector('.text-center.text-muted');
        if (existingWelcome) {
            existingWelcome.remove();
        }

        const messageDiv = document.createElement('div');
        messageDiv.className = `mb-3 d-flex ${type === 'user' ? 'justify-content-end' : 'justify-content-start'}`;

        let messageContent = '';
        let badgeClass = '';
        let icon = '';

        switch (type) {
            case 'user':
                badgeClass = 'bg-primary';
                icon = '<i class="fas fa-user me-2"></i>';
                break;
            case 'assistant':
                badgeClass = 'bg-success';
                icon = '<i class="fas fa-robot me-2"></i>';
                break;
            case 'error':
                badgeClass = 'bg-danger';
                icon = '<i class="fas fa-exclamation-triangle me-2"></i>';
                break;
        }

        messageContent = `
            <div class="card border-0 shadow-sm" style="max-width: 80%;">
                <div class="card-header py-2 ${badgeClass} text-white">
                    ${icon}${type === 'user' ? 'You' : type === 'assistant' ? 'AI Assistant' : 'Error'}
                </div>
                <div class="card-body">
                    <div class="message-content">${formatMessage(content)}</div>
                    ${type === 'assistant' && citations && citations.length > 0 ? formatCitations(citations) : ''}
                </div>
            </div>
        `;

        messageDiv.innerHTML = messageContent;
        chatMessages.appendChild(messageDiv);
    }

    function formatMessage(content) {
        // Convert line breaks to HTML and escape HTML
        return content
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/\n/g, '<br>');
    }

    function formatCitations(citations) {
        if (!citations || citations.length === 0) return '';

        let citationHtml = '<hr><div class="citations mt-3"><small class="text-muted"><strong>Sources:</strong></small><br>';
        
        citations.forEach((citation, index) => {
            citationHtml += `
                <button type="button" class="btn btn-outline-info btn-sm me-2 mb-1 citation-btn" 
                        data-citation-index="${index}">
                    <i class="fas fa-file-pdf me-1"></i>
                    ${citation.document_filename} (Page ${citation.page_number})
                </button>
            `;
        });
        
        citationHtml += '</div>';
        return citationHtml;
    }

    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Handle citation clicks
    chatMessages.addEventListener('click', function(e) {
        if (e.target.closest('.citation-btn')) {
            const btn = e.target.closest('.citation-btn');
            const citationIndex = btn.getAttribute('data-citation-index');
            
            // Find the citations from the message
            const messageCard = btn.closest('.card');
            const citations = getCitationsFromMessage(messageCard);
            
            if (citations && citations[citationIndex]) {
                showCitationModal(citations[citationIndex]);
            }
        }
    });

    function getCitationsFromMessage(messageCard) {
        // This is a simple approach - in a real app, you'd store citations data more robustly
        // For now, we'll extract from the most recent assistant message with citations
        const allMessages = chatMessages.querySelectorAll('.card');
        for (let i = allMessages.length - 1; i >= 0; i--) {
            const message = allMessages[i];
            if (message.querySelector('.citations') && message === messageCard) {
                // Return mock citation data - this would be stored properly in a real app
                return [{
                    document_filename: btn.textContent.split(' (')[0].replace(/.*\s/, ''),
                    page_number: btn.textContent.match(/Page (\d+)/)[1],
                    content_preview: 'Citation content would be displayed here...'
                }];
            }
        }
        return null;
    }

    function showCitationModal(citation) {
        const modal = new bootstrap.Modal(document.getElementById('citationModal'));
        const content = document.getElementById('citationContent');
        
        content.innerHTML = `
            <h6><i class="fas fa-file-pdf me-2 text-danger"></i>${citation.document_filename}</h6>
            <p><strong>Page:</strong> ${citation.page_number}</p>
            <div class="bg-light p-3 rounded">
                <small class="text-muted">Content preview:</small><br>
                ${citation.content_preview}
            </div>
        `;
        
        modal.show();
    }

    // Focus on input when page loads
    messageInput.focus();

    // Auto-resize chat area
    function resizeChatArea() {
        const navbar = document.querySelector('.navbar');
        const chatCard = document.querySelector('.card');
        const windowHeight = window.innerHeight;
        const navbarHeight = navbar ? navbar.offsetHeight : 0;
        const padding = 100; // Account for margins and other elements
        
        const maxHeight = windowHeight - navbarHeight - padding;
        chatCard.style.minHeight = maxHeight + 'px';
    }

    // Initial resize and on window resize
    resizeChatArea();
    window.addEventListener('resize', resizeChatArea);
});
