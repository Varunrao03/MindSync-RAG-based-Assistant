// API Configuration
const API_BASE = 'http://localhost:3000';
const API_CHAT_ENDPOINT = `${API_BASE}/api/chat`;
const API_UPLOAD_ENDPOINT = `${API_BASE}/api/upload`;
const API_LOAD_ALL_ENDPOINT = `${API_BASE}/api/load-all`;
const API_DOCUMENT_COUNT_ENDPOINT = `${API_BASE}/api/document-count`;
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
const documentCount = document.getElementById('document-count');
const contentArea = document.getElementById('content-area');
const startCaptureBtn = document.getElementById('start-capture-btn');
const stopCaptureBtn = document.getElementById('stop-capture-btn');
const pauseCaptureBtn = document.getElementById('pause-capture-btn');
const resumeCaptureBtn = document.getElementById('resume-capture-btn');

// State
let isLoading = false;
let documents = [];

// Screen capture state
let captureIntervalRef = null;
let streamRef = null;
let isCapturePausedRef = false;
let captureStatus = 'stopped'; // 'stopped', 'running', 'paused'

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    updateDocumentCount();
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
    
    // Screen capture controls
    startCaptureBtn.addEventListener('click', handleStartCapture);
    stopCaptureBtn.addEventListener('click', handleStopCapture);
    pauseCaptureBtn.addEventListener('click', handlePauseCapture);
    resumeCaptureBtn.addEventListener('click', handleResumeCapture);
    
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
            'Sorry, I couldn\'t connect to the server. Please make sure the server is running on http://localhost:3000',
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
            updateDocumentCount();
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
    if (!confirm('Load all PDFs from the data/pdf directory?')) {
        return;
    }
    
    try {
        updateStatus('Loading all documents...', true);
        loadAllBtn.disabled = true;
        
        const response = await fetch(API_LOAD_ALL_ENDPOINT, {
            method: 'POST'
        });
        
        if (response.ok) {
            const data = await response.json();
            if (data.success) {
                alert(`Successfully loaded ${data.files_processed} files! Processed ${data.chunks_created} chunks.`);
                updateDocumentCount();
                loadDocumentList(); // Refresh PDF list
            } else {
                alert(data.message || 'No documents found');
            }
        } else {
            const error = await response.json().catch(() => ({ detail: 'Load failed' }));
            alert(`Error loading documents: ${error.detail || 'Unknown error'}`);
        }
    } catch (error) {
        console.error('Load all error:', error);
        alert('Failed to load documents. Please check if the server is running.');
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

async function updateDocumentCount() {
    try {
        const response = await fetch(API_DOCUMENT_COUNT_ENDPOINT);
        if (response.ok) {
            const data = await response.json();
            documentCount.innerHTML = `<span>Documents: ${data.document_count}</span>`;
        }
    } catch (error) {
        console.error('Error fetching document count:', error);
    }
}

// PDF.js viewer state
let currentPdfDoc = null;
let currentPageNum = 1;
let totalPages = 0;

async function displayPDF(filename) {
    // Display PDF in content area using PDF.js
    const pdfPath = `${API_BASE}/data/pdf/${encodeURIComponent(filename)}`;
    
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
        const response = await fetch(`${API_BASE}/api/health`);
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

// ============================================================================
// Screen Capture Functions (based on ReadingSession.js)
// ============================================================================

/**
 * Start continuous screen capture
 * Captures screenshot every 30 seconds and sends to backend
 */
async function startContinuousCapture() {
    try {
        // Debug: Check what's available
        console.log('🔍 Debug - navigator:', navigator);
        console.log('🔍 Debug - navigator.mediaDevices:', navigator.mediaDevices);
        console.log('🔍 Debug - window.isSecureContext:', window.isSecureContext);
        console.log('🔍 Debug - location:', window.location.href);
        
        // Check if we're in a secure context
        if (!window.isSecureContext) {
            const errorMsg = 'Screen sharing requires a secure context (HTTPS or localhost). Please access the app via http://localhost:3000';
            console.error('❌', errorMsg);
            updateCaptureStatus('Secure context required');
            alert(errorMsg);
            return;
        }
        
        // Check if browser supports screen sharing
        if (!navigator.mediaDevices) {
            const errorMsg = 'navigator.mediaDevices is not available. This might be due to:\n- Browser not supporting MediaDevices API\n- Page not served from a secure context\n- Browser security settings\n\nPlease try using Chrome, Firefox, Edge, or Safari (14.1+)';
            console.error('❌', errorMsg);
            console.error('🔍 navigator.mediaDevices is:', navigator.mediaDevices);
            updateCaptureStatus('MediaDevices not available');
            alert(errorMsg);
            return;
        }
        
        if (!navigator.mediaDevices.getDisplayMedia) {
            const errorMsg = 'getDisplayMedia is not supported in this browser. Please use Chrome 72+, Firefox 66+, Edge 79+, or Safari 14.1+';
            console.error('❌', errorMsg);
            console.error('🔍 Available methods:', Object.keys(navigator.mediaDevices));
            updateCaptureStatus('getDisplayMedia not supported');
            alert(errorMsg);
            return;
        }
        
        // Request screen sharing permission - browser will show dialog
        console.log('🖥️ Requesting screen share permission...');
        updateCaptureStatus('Requesting screen share...');
        
        const mediaStream = await navigator.mediaDevices.getDisplayMedia({
            video: { 
                mediaSource: "screen",
                width: { ideal: 1280 },
                height: { ideal: 720 },
                frameRate: { ideal: 1 }
            },
            audio: false  // We only need video for screenshots
        });

        console.log('✅ Screen sharing started');
        streamRef = mediaStream;
        const track = mediaStream.getVideoTracks()[0];
        const imageCapture = new ImageCapture(track);
        
        // Detect when user stops sharing via browser notification
        track.addEventListener('ended', () => {
            console.log('⚠️ Screen sharing stopped by user');
            stopCapture();
            updateCaptureStatus('Screen sharing stopped');
            // Don't show alert, just update status
        });

        // Helper to start or restart the capture interval if not paused
        const startInterval = () => {
            if (captureIntervalRef) return;
            captureIntervalRef = setInterval(async () => {
                try {
                    if (isCapturePausedRef) {
                        return;
                    }
                    const bitmap = await imageCapture.grabFrame();
                    const canvas = document.createElement("canvas");
                    canvas.width = bitmap.width;
                    canvas.height = bitmap.height;
                    const ctx = canvas.getContext("2d");
                    ctx.drawImage(bitmap, 0, 0);
                    
                    // Debug: Log what's being captured
                    console.log(`📸 Captured screenshot: ${bitmap.width}x${bitmap.height}`);
                    
                    const dataUrl = canvas.toDataURL("image/png");

                    sendScreenshotToBackend(dataUrl);
                } catch (err) {
                    console.error("Capture failed:", err);
                }
            }, 30000); // Capture every 30 seconds
        };

        startInterval();
        captureStatus = 'running';
        updateCaptureStatus();

    } catch (err) {
        console.error("Screen sharing error:", err);
        captureStatus = 'stopped';
        updateCaptureStatus();
        
        // Provide user-friendly error messages
        if (err.name === 'NotAllowedError') {
            alert("❌ Screen sharing permission denied.\n\nPlease click 'Allow' when the browser asks for permission to share your screen.");
        } else if (err.name === 'NotReadableError') {
            alert("❌ Cannot access screen capture.\n\nMake sure no other application is using screen capture.");
        } else if (err.name === 'NotFoundError') {
            alert("❌ Screen capture not available.\n\nYour browser may not support screen sharing, or no screens are available.");
        } else if (err.name === 'AbortError') {
            // User cancelled - don't show error, just update status
            console.log('User cancelled screen sharing');
        } else {
            alert(`❌ Error starting screen capture: ${err.message || 'Unknown error'}\n\nPlease try again.`);
        }
    }
}

/**
 * Stop screen capture
 */
function stopCapture() {
    if (captureIntervalRef) {
        clearInterval(captureIntervalRef);
        captureIntervalRef = null;
    }
    if (streamRef) {
        streamRef.getTracks().forEach(track => track.stop());
        streamRef = null;
    }
    isCapturePausedRef = false;
    captureStatus = 'stopped';
    updateCaptureStatus();
}

/**
 * Pause screen capture
 */
function pauseCapture() {
    isCapturePausedRef = true;
    if (captureIntervalRef) {
        clearInterval(captureIntervalRef);
        captureIntervalRef = null;
    }
    captureStatus = 'paused';
    updateCaptureStatus();
}

/**
 * Resume screen capture
 */
function resumeCapture() {
    isCapturePausedRef = false;
    // If we still have the stream, recreate the ImageCapture and interval
    if (streamRef) {
        const track = streamRef.getVideoTracks()[0];
        if (!track) return;
        const imageCapture = new ImageCapture(track);
        if (!captureIntervalRef) {
            captureIntervalRef = setInterval(async () => {
                try {
                    if (isCapturePausedRef) return;
                    const bitmap = await imageCapture.grabFrame();
                    const canvas = document.createElement("canvas");
                    canvas.width = bitmap.width;
                    canvas.height = bitmap.height;
                    const ctx = canvas.getContext("2d");
                    ctx.drawImage(bitmap, 0, 0);
                    const dataUrl = canvas.toDataURL("image/png");
                    sendScreenshotToBackend(dataUrl);
                } catch (err) {
                    console.error("Capture failed:", err);
                }
            }, 30000);
        }
    }
    captureStatus = 'running';
    updateCaptureStatus();
}

/**
 * Send screenshot to backend
 * @param {string} dataUrl - Base64 encoded image data URL
 */
async function sendScreenshotToBackend(dataUrl) {
    try {
        // Convert data URL to blob properly
        // dataUrl is like "data:image/png;base64,iVBORw0KGgo..."
        const base64Data = dataUrl.split(',')[1];
        const byteCharacters = atob(base64Data);
        const byteNumbers = new Array(byteCharacters.length);
        for (let i = 0; i < byteCharacters.length; i++) {
            byteNumbers[i] = byteCharacters.charCodeAt(i);
        }
        const byteArray = new Uint8Array(byteNumbers);
        const blob = new Blob([byteArray], { type: 'image/png' });
        
        // Create FormData to send as multipart/form-data
        const formData = new FormData();
        formData.append('file', blob, `screenshot_${Date.now()}.png`);
        
        // Send to backend
        const apiResponse = await fetch(`${API_BASE}/api/screenshot`, {
            method: 'POST',
            body: formData
        });
        
        if (apiResponse.ok) {
            const result = await apiResponse.json();
            console.log('✅ Screenshot saved:', result.filename, `(${result.size} bytes)`);
        } else {
            const errorText = await apiResponse.text();
            console.warn('⚠️ Failed to save screenshot:', apiResponse.status, errorText);
        }
    } catch (error) {
        console.error('❌ Error sending screenshot:', error);
        // Don't throw - we still want capture to continue even if sending fails
    }
}

/**
 * Update capture status UI indicator
 */
function updateCaptureStatus(customMessage = null) {
    const statusIndicator = document.getElementById('capture-status-indicator');
    if (statusIndicator) {
        const message = customMessage || captureStatus;
        statusIndicator.textContent = `Status: ${message}`;
        statusIndicator.className = `capture-status capture-status-${captureStatus}`;
    }
    
    // Update button visibility based on status
    if (captureStatus === 'stopped') {
        startCaptureBtn.style.display = 'block';
        stopCaptureBtn.style.display = 'none';
        pauseCaptureBtn.style.display = 'none';
        resumeCaptureBtn.style.display = 'none';
    } else if (captureStatus === 'running') {
        startCaptureBtn.style.display = 'none';
        stopCaptureBtn.style.display = 'block';
        pauseCaptureBtn.style.display = 'block';
        resumeCaptureBtn.style.display = 'none';
    } else if (captureStatus === 'paused') {
        startCaptureBtn.style.display = 'none';
        stopCaptureBtn.style.display = 'block';
        pauseCaptureBtn.style.display = 'none';
        resumeCaptureBtn.style.display = 'block';
    }
}

// Capture control handlers
async function handleStartCapture() {
    await startContinuousCapture();
}

function handleStopCapture() {
    stopCapture();
}

function handlePauseCapture() {
    pauseCapture();
}

function handleResumeCapture() {
    resumeCapture();
}

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    stopCapture();
});
