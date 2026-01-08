// API Configuration - Use relative URLs to work with any hostname
const API_BASE = ''; // Empty string means same origin as the page
const API_CHAT_ENDPOINT = `${API_BASE}/api/chat`;
const API_UPLOAD_ENDPOINT = `${API_BASE}/api/upload`;
const API_LOAD_ALL_ENDPOINT = `${API_BASE}/api/load-all`;
const API_LIST_PDFS_ENDPOINT = `${API_BASE}/api/pdfs`;

// DOM Elements
const chatbotButton = document.getElementById('chatbot-button');
const chatbotModal = document.getElementById('chatbot-modal');
const closeChatbotBtn = document.getElementById('close-chatbot');
const chatMessages = document.getElementById('chat-messages');
const chatInput = document.getElementById('chat-input');
const sendButton = document.getElementById('send-button');
const statusBar = document.getElementById('status-bar');
const statusText = document.getElementById('status-text');
const uploadBtn = document.getElementById('upload-btn');
const fileInput = document.getElementById('file-input');
const loadAllBtn = document.getElementById('load-all-btn');
const documentList = document.getElementById('document-list');
const contentArea = document.getElementById('content-area');

// State
let isLoading = false;
let documents = [];

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    loadDocumentList();
    checkServerHealth();
});

// Event Listeners
function setupEventListeners() {
    // Chatbot popup
    chatbotButton.addEventListener('click', openChatbot);
    closeChatbotBtn.addEventListener('click', closeChatbot);
    chatbotModal.addEventListener('click', (e) => {
        if (e.target === chatbotModal) {
            closeChatbot();
        }
    });
    
    // Chat
    sendButton.addEventListener('click', sendMessage);
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    
    // Document upload
    uploadBtn.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileUpload);
    loadAllBtn.addEventListener('click', loadAllDocuments);
    
    // Close chatbot on Escape key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && chatbotModal.classList.contains('active')) {
            closeChatbot();
        }
    });
}

// Chatbot Popup Functions
function openChatbot() {
    chatbotModal.classList.add('active');
    chatInput.focus();
}

function closeChatbot() {
    chatbotModal.classList.remove('active');
}

// Update status bar
function updateStatus(message, isLoading = false) {
    statusText.innerHTML = isLoading 
        ? `<span class="status-loading"></span> ${message}`
        : message;
}

// Send message
async function sendMessage() {
    const message = chatInput.value.trim();
    
    if (!message || isLoading) {
        return;
    }
    
    // Clear input and disable
    chatInput.value = '';
    chatInput.disabled = true;
    sendButton.disabled = true;
    isLoading = true;
    
    // Add user message to chat
    addMessage(message, 'user');
    
    // Show typing indicator
    const typingIndicator = showTypingIndicator();
    
    // Update status
    updateStatus('Processing your question...', true);
    
    try {
        // Send request to API
        const response = await fetch(API_CHAT_ENDPOINT, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ query: message })
        });
        
        // Remove typing indicator
        typingIndicator.remove();
        
        if (response.ok) {
            const data = await response.json();
            addMessage(data.answer, 'bot');
            updateStatus('Ready - Type your question below');
        } else {
            const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
            addMessage(
                `Sorry, I encountered an error: ${error.detail || 'Unknown error'}. Please try again.`,
                'bot',
                true
            );
            updateStatus('Error occurred - Try again');
        }
    } catch (error) {
        typingIndicator.remove();
        console.error('Chat error:', error);
        addMessage(
            'Sorry, I couldn\'t connect to the server. Please make sure the server is running.',
            'bot',
            true
        );
        updateStatus('Connection error - Check if server is running');
    } finally {
        chatInput.disabled = false;
        sendButton.disabled = false;
        isLoading = false;
        chatInput.focus();
    }
}

// (Quiz functionality removed)

// Add message to chat
function addMessage(text, type, isError = false) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}-message${isError ? ' error-message' : ''}`;
    
    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';
    
    const p = document.createElement('p');
    p.innerHTML = formatMessage(text);
    messageContent.appendChild(p);
    
    messageDiv.appendChild(messageContent);
    chatMessages.appendChild(messageDiv);
    
    scrollToBottom();
    
    return messageDiv;
}

// Format message text
function formatMessage(text) {
    return text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/\n/g, '<br>');
}

// Show typing indicator
function showTypingIndicator() {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message bot-message';
    
    const typingDiv = document.createElement('div');
    typingDiv.className = 'typing-indicator';
    typingDiv.id = 'typing-indicator';
    
    for (let i = 0; i < 3; i++) {
        const span = document.createElement('span');
        typingDiv.appendChild(span);
    }
    
    messageDiv.appendChild(typingDiv);
    chatMessages.appendChild(messageDiv);
    
    scrollToBottom();
    
    return messageDiv;
}

// Scroll to bottom
function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// File Upload Functions
async function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    if (!file.name.endsWith('.pdf')) {
        alert('Please upload a PDF file');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        updateStatus('Uploading and processing document...', true);
        
        const response = await fetch(API_UPLOAD_ENDPOINT, {
            method: 'POST',
            body: formData
        });
        
        if (response.ok) {
            const data = await response.json();
            alert(`Document uploaded successfully! Processed ${data.chunks_created} chunks.`);
            loadDocumentList(); // Refresh PDF list
            // Clear file input
            fileInput.value = '';
        } else {
            const error = await response.json().catch(() => ({ detail: 'Upload failed' }));
            alert(`Error uploading file: ${error.detail || 'Unknown error'}`);
        }
    } catch (error) {
        console.error('Upload error:', error);
        alert('Failed to upload file. Please check if the server is running.');
    } finally {
        updateStatus('Ready');
    }
}

async function loadAllDocuments() {
    if (!confirm('Load the latest PDF from the data/pdf directory? This will process only the most recently modified PDF file.')) {
        return;
    }
    
    try {
        updateStatus('Loading latest PDF...', true);
        loadAllBtn.disabled = true;
        
        const response = await fetch(API_LOAD_ALL_ENDPOINT, {
            method: 'POST'
        });
        
        if (response.ok) {
            const data = await response.json();
            if (data.success) {
                const message = data.latest_file 
                    ? `Successfully processed latest PDF: ${data.latest_file}\n` +
                      `Modified: ${data.file_modified || 'N/A'}\n` +
                      `Created ${data.chunks_created} chunks.\n` +
                      (data.old_chunks_deleted > 0 ? `Deleted ${data.old_chunks_deleted} old chunks.` : '')
                    : `Successfully loaded ${data.files_processed} file(s)! Processed ${data.chunks_created} chunks.`;
                alert(message);
                loadDocumentList(); // Refresh PDF list
            } else {
                alert(data.message || 'No documents found');
            }
        } else {
            const error = await response.json().catch(() => ({ detail: 'Load failed' }));
            alert(`Error loading latest PDF: ${error.detail || 'Unknown error'}`);
        }
    } catch (error) {
        console.error('Load latest PDF error:', error);
        alert('Failed to load latest PDF. Please check if the server is running.');
    } finally {
        updateStatus('Ready');
        loadAllBtn.disabled = false;
    }
}

// Document List Functions
async function loadDocumentList() {
    try {
        const response = await fetch(API_LIST_PDFS_ENDPOINT);
        if (response.ok) {
            const data = await response.json();
            displayPDFList(data.pdfs);
        } else {
            console.error('Failed to load PDF list');
            documentList.innerHTML = '<p class="empty-message">Error loading PDFs</p>';
        }
    } catch (error) {
        console.error('Error loading PDF list:', error);
        documentList.innerHTML = '<p class="empty-message">Error loading PDFs</p>';
    }
}

function displayPDFList(pdfs) {
    if (!pdfs || pdfs.length === 0) {
        documentList.innerHTML = '<p class="empty-message">No PDFs found</p>';
        return;
    }
    
    documentList.innerHTML = '';
    
    pdfs.forEach((pdf, index) => {
        const pdfItem = document.createElement('div');
        pdfItem.className = 'document-item';
        pdfItem.dataset.filename = pdf.filename;
        
        pdfItem.innerHTML = `
            <div class="document-item-name">${pdf.filename}</div>
            <div class="document-item-size">${pdf.size}</div>
        `;
        
        // Add click handler
        pdfItem.addEventListener('click', () => {
            // Remove active class from all items
            document.querySelectorAll('.document-item').forEach(item => {
                item.classList.remove('active');
            });
            // Add active class to clicked item
            pdfItem.classList.add('active');
            // Display PDF
            console.log('Displaying PDF:', pdf.filename);
            displayPDF(pdf.filename);
        });
        
        documentList.appendChild(pdfItem);
    });
}

// PDF.js viewer state
let currentPdfDoc = null;
let currentPageNum = 1;
let totalPages = 0;

async function displayPDF(filename) {
    // Display PDF in content area using PDF.js
    const pdfPath = `/data/pdf/${encodeURIComponent(filename)}`;
    
    // Clear content area and show loading
    contentArea.innerHTML = `
        <div class="pdf-viewer">
            <div class="pdf-header">
                <h3>${filename}</h3>
                <div class="pdf-controls">
                    <button id="prev-page" class="pdf-nav-btn" disabled>← Previous</button>
                    <span id="page-info" class="page-info">Loading...</span>
                    <button id="next-page" class="pdf-nav-btn" disabled>Next →</button>
                </div>
            </div>
            <div class="pdf-container">
                <canvas id="pdf-canvas" class="pdf-canvas"></canvas>
                <div id="pdf-loading" class="pdf-loading">Loading PDF...</div>
            </div>
        </div>
    `;
    
    // Initialize PDF.js
    if (typeof pdfjsLib === 'undefined') {
        console.error('PDF.js library not loaded');
        document.getElementById('pdf-loading').textContent = 'Error: PDF.js library not loaded';
        return;
    }
    
    // Set worker source
    pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';
    
    // Load PDF
    try {
        const loadingTask = pdfjsLib.getDocument(pdfPath);
        currentPdfDoc = await loadingTask.promise;
        totalPages = currentPdfDoc.numPages;
        currentPageNum = 1;
        
        // Hide loading
        document.getElementById('pdf-loading').style.display = 'none';
        
        // Render first page
        await renderPage(currentPageNum);
        
        // Update page info
        updatePageInfo();
        
        // Setup navigation buttons
        setupPDFNavigation();
        
    } catch (error) {
        console.error('Error loading PDF:', error);
        document.getElementById('pdf-loading').textContent = 'Error loading PDF. Please try opening in a new tab.';
    }
}

async function renderPage(pageNum) {
    if (!currentPdfDoc) return;
    
    try {
        const page = await currentPdfDoc.getPage(pageNum);
        const canvas = document.getElementById('pdf-canvas');
        const context = canvas.getContext('2d');
        
        // Set zoom ratio to 120% (1.2)
        const zoomRatio = 1.6; // 160%
        const scaledViewport = page.getViewport({ scale: zoomRatio });
        
        // Set canvas dimensions
        canvas.height = scaledViewport.height;
        canvas.width = scaledViewport.width;
        
        // Render PDF page
        const renderContext = {
            canvasContext: context,
            viewport: scaledViewport
        };
        
        await page.render(renderContext).promise;
        
    } catch (error) {
        console.error('Error rendering page:', error);
    }
}

function updatePageInfo() {
    const pageInfo = document.getElementById('page-info');
    if (pageInfo) {
        pageInfo.textContent = `Page ${currentPageNum} of ${totalPages}`;
    }
    
    // Update navigation buttons
    const prevBtn = document.getElementById('prev-page');
    const nextBtn = document.getElementById('next-page');
    
    if (prevBtn) prevBtn.disabled = currentPageNum <= 1;
    if (nextBtn) nextBtn.disabled = currentPageNum >= totalPages;
}

function setupPDFNavigation() {
    const prevBtn = document.getElementById('prev-page');
    const nextBtn = document.getElementById('next-page');
    
    if (prevBtn) {
        prevBtn.addEventListener('click', async () => {
            if (currentPageNum > 1) {
                currentPageNum--;
                document.getElementById('pdf-canvas').style.opacity = '0.5';
                await renderPage(currentPageNum);
                updatePageInfo();
                document.getElementById('pdf-canvas').style.opacity = '1';
            }
        });
    }
    
    if (nextBtn) {
        nextBtn.addEventListener('click', async () => {
            if (currentPageNum < totalPages) {
                currentPageNum++;
                document.getElementById('pdf-canvas').style.opacity = '0.5';
                await renderPage(currentPageNum);
                updatePageInfo();
                document.getElementById('pdf-canvas').style.opacity = '1';
            }
        });
    }
}

// Check server health on load
async function checkServerHealth() {
    try {
        const response = await fetch('/api/health');
        if (response.ok) {
            const data = await response.json();
            if (!data.rag_system_loaded) {
                updateStatus('RAG system will initialize on first question (may take 10-30 seconds)');
            } else {
                updateStatus('Ready - RAG system is loaded');
            }
        }
    } catch (error) {
        updateStatus('Server not responding - Make sure server is running on port 3000');
    }
}

