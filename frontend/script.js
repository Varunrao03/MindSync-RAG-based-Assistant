// API Configuration - Use relative URLs to work with any hostname
const API_BASE = ''; // Empty string means same origin as the page
const API_CHAT_ENDPOINT = `${API_BASE}/api/chat`;
const API_UPLOAD_ENDPOINT = `${API_BASE}/api/upload`;
const API_LOAD_ALL_ENDPOINT = `${API_BASE}/api/load-all`;
const API_LIST_PDFS_ENDPOINT = `${API_BASE}/api/pdfs`;
const API_DELETE_PDF_ENDPOINT = (filename) => `${API_BASE}/api/pdfs/${encodeURIComponent(filename)}`;
const API_GENERATE_QUIZ_ENDPOINT = `${API_BASE}/api/generate-quiz`;

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
const deletePdfBtn = document.getElementById('delete-pdf-btn');
const quizBtn = document.getElementById('quiz-btn');
const quizModal = document.getElementById('quiz-modal');
const closeQuizBtn = document.getElementById('close-quiz');
const quizModalBody = document.getElementById('quiz-modal-body');
const documentList = document.getElementById('document-list');
const contentArea = document.getElementById('content-area');

// State
let isLoading = false;
let documents = [];

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM loaded, initializing...');
    console.log('Quiz elements:', { quizBtn, quizModal, quizModalBody, closeQuizBtn });
    
    setupEventListeners();
    loadDocumentList();
    checkServerHealth();
    
    // Verify quiz elements exist
    if (!quizBtn) {
        console.error('❌ Quiz button not found in DOM!');
    } else {
        console.log('✅ Quiz button found');
    }
    if (!quizModal) {
        console.error('❌ Quiz modal not found in DOM!');
    } else {
        console.log('✅ Quiz modal found');
    }
    if (!quizModalBody) {
        console.error('❌ Quiz modal body not found in DOM!');
    } else {
        console.log('✅ Quiz modal body found');
    }
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
    
    // Quiz popup
    if (quizBtn) {
        quizBtn.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            console.log('Quiz button clicked!', { quizModal, quizModalBody });
            openQuizModal();
        });
        console.log('✅ Quiz button event listener attached');
    } else {
        console.error('❌ Quiz button not found when setting up event listeners!');
    }
    
    if (closeQuizBtn) {
        closeQuizBtn.addEventListener('click', closeQuizModal);
        console.log('✅ Close quiz button event listener attached');
    } else {
        console.error('❌ Close quiz button not found!');
    }
    
    if (quizModal) {
        quizModal.addEventListener('click', (e) => {
            if (e.target === quizModal) {
                closeQuizModal();
            }
        });
        console.log('✅ Quiz modal click handler attached');
    } else {
        console.error('❌ Quiz modal not found when setting up event listeners!');
    }
    
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
    deletePdfBtn.addEventListener('click', handleDeletePDF);
    
    // Close modals on Escape key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            if (chatbotModal.classList.contains('active')) {
                closeChatbot();
            }
            if (quizModal.classList.contains('active')) {
                closeQuizModal();
            }
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

function openQuizModal() {
    console.log('openQuizModal called', { quizModal, quizModalBody });
    if (!quizModal) {
        console.error('Quiz modal not found!');
        alert('Quiz modal not found. Please refresh the page.');
        return;
    }
    if (!quizModalBody) {
        console.error('Quiz modal body not found!');
        alert('Quiz modal body not found. Please refresh the page.');
        return;
    }
    
    console.log('Adding active class to quiz modal...');
    quizModal.classList.add('active');
    
    // Use setTimeout to ensure DOM is ready and CSS is applied
    setTimeout(() => {
        const computedDisplay = window.getComputedStyle(quizModal).display;
        console.log('Modal classes:', quizModal.className);
        console.log('Modal display style:', computedDisplay);
        
        if (computedDisplay !== 'flex') {
            console.warn('Modal display is not flex, forcing display...');
            quizModal.style.display = 'flex';
        }
        
        showQuizForm();
        console.log('Quiz form shown');
    }, 10);
}

function closeQuizModal() {
    if (quizModal) {
        quizModal.classList.remove('active');
    }
    currentQuiz = null;
    quizAnswers = {};
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
    
    const filename = file.name; // Store filename for later use
    
    try {
        updateStatus('Uploading document...', true);
        
        const response = await fetch(API_UPLOAD_ENDPOINT, {
            method: 'POST',
            body: formData
        });
        
        if (response.ok) {
            const data = await response.json();
            const uploadedFilename = data.filename || filename;
            
            // Refresh PDF list first, then display the uploaded PDF
            await loadDocumentList();
            
            // Automatically display the uploaded PDF in the viewer
            displayPDF(uploadedFilename);
            
            // Mark the PDF as active in the sidebar
            setTimeout(() => {
                const pdfItem = document.querySelector(`.document-item[data-filename="${uploadedFilename}"]`);
                if (pdfItem) {
                    // Remove active class from all items
                    document.querySelectorAll('.document-item').forEach(item => {
                        item.classList.remove('active');
                    });
                    // Add active class to uploaded PDF
                    pdfItem.classList.add('active');
                }
            }, 200);
            
            // Show success message in status bar
            updateStatus(`✅ Uploaded ${uploadedFilename} - Click "Load Latest PDF" to process it`, false);
            setTimeout(() => {
                updateStatus('Ready');
            }, 4000);
            
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
    if (!confirm('Process the latest PDF? This will create chunks and embeddings for the most recently uploaded PDF file.')) {
        return;
    }
    
    try {
        updateStatus('Processing latest PDF - creating chunks and embeddings...', true);
        loadAllBtn.disabled = true;
        
        const response = await fetch(API_LOAD_ALL_ENDPOINT, {
            method: 'POST'
        });
        
        if (response.ok) {
            const data = await response.json();
            if (data.success) {
                const message = data.latest_file 
                    ? `✅ Successfully processed PDF: ${data.latest_file}\n` +
                      `📦 Collection: ${data.collection_name}\n` +
                      `📄 Modified: ${data.file_modified || 'N/A'}\n` +
                      `🔢 Created ${data.chunks_created} chunks with embeddings`
                    : `Successfully processed ${data.files_processed} file(s)! Created ${data.chunks_created} chunks.`;
                alert(message);
                loadDocumentList(); // Refresh PDF list
                updateStatus(`✅ Processed ${data.latest_file} - ${data.chunks_created} chunks created`, false);
            } else {
                alert(data.message || 'No documents found');
                updateStatus('Ready');
            }
        } else {
            const error = await response.json().catch(() => ({ detail: 'Load failed' }));
            alert(`Error processing latest PDF: ${error.detail || 'Unknown error'}`);
            updateStatus('Ready');
        }
    } catch (error) {
        console.error('Load latest PDF error:', error);
        alert('Failed to load latest PDF. Please check if the server is running.');
        updateStatus('Ready');
    } finally {
        loadAllBtn.disabled = false;
        setTimeout(() => {
            if (updateStatus && typeof updateStatus === 'function') {
                updateStatus('Ready');
            }
        }, 3000);
    }
}

async function handleDeletePDF() {
    // Load current PDF list
    try {
        const response = await fetch(API_LIST_PDFS_ENDPOINT);
        if (!response.ok) {
            alert('Failed to load PDF list');
            return;
        }
        
        const data = await response.json();
        const pdfs = data.pdfs;
        
        if (!pdfs || pdfs.length === 0) {
            alert('No PDFs found to delete');
            return;
        }
        
        // Confirm deletion of all PDFs
        const pdfNames = pdfs.map(pdf => pdf.filename).join('\n');
        if (!confirm(`Are you sure you want to delete ALL ${pdfs.length} PDF(s)?\n\nThis will delete:\n${pdfNames}\n\nThis will also remove all associated chunks and collections from the vector store.\n\nThis action cannot be undone!`)) {
            return;
        }
        
        // Delete all PDFs
        deletePdfBtn.disabled = true;
        updateStatus(`Deleting all PDFs...`, true);
        
        let deletedCount = 0;
        let totalChunksDeleted = 0;
        const errors = [];
        
        // Delete each PDF one by one
        for (const pdf of pdfs) {
            try {
                const deleteResponse = await fetch(API_DELETE_PDF_ENDPOINT(pdf.filename), {
                    method: 'DELETE'
                });
                
                if (deleteResponse.ok) {
                    const result = await deleteResponse.json();
                    deletedCount++;
                    totalChunksDeleted += result.chunks_deleted || 0;
                    console.log(`✅ Deleted ${pdf.filename}`);
                } else {
                    const error = await deleteResponse.json().catch(() => ({ detail: 'Delete failed' }));
                    errors.push(`${pdf.filename}: ${error.detail || 'Unknown error'}`);
                }
            } catch (error) {
                errors.push(`${pdf.filename}: ${error.message}`);
            }
        }
        
        // Refresh PDF list
        await loadDocumentList();
        
        // Clear content area
        contentArea.innerHTML = `
            <div class="welcome-message">
                <h2>Welcome!</h2>
                <p>Upload PDF documents using the sidebar or click the chatbot button to ask questions about your documents.</p>
            </div>
        `;
        
        // Show results
        if (deletedCount === pdfs.length) {
            alert(`✅ Successfully deleted all ${deletedCount} PDF(s)!\n\nRemoved ${totalChunksDeleted} chunks from vector store.`);
            updateStatus(`✅ Deleted all ${deletedCount} PDF(s)`, false);
        } else {
            const errorMsg = errors.length > 0 ? `\n\nErrors:\n${errors.join('\n')}` : '';
            alert(`Deleted ${deletedCount} out of ${pdfs.length} PDF(s).${errorMsg}`);
            updateStatus(`Deleted ${deletedCount}/${pdfs.length} PDF(s)`, false);
        }
    } catch (error) {
        console.error('Delete error:', error);
        alert('Failed to delete PDFs. Please check if the server is running.');
        updateStatus('Ready');
    } finally {
        deletePdfBtn.disabled = false;
        setTimeout(() => {
            updateStatus('Ready');
        }, 3000);
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
    return true; // Return success for await compatibility
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
            <div class="document-item-content">
                <div class="document-item-name">${pdf.filename}</div>
                <div class="document-item-size">${pdf.size}</div>
            </div>
        `;
        
        // Add click handler for the item
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

// Quiz Functions
let currentQuiz = null;
let quizAnswers = {};

function showQuizForm() {
    if (!quizModalBody) {
        console.error('Quiz modal body not found!');
        return;
    }
    
    quizModalBody.innerHTML = `
        <div class="quiz-form-container">
            <div class="quiz-form-header">
                <p>Generate a quiz based on your uploaded documents. Enter a topic and we'll create questions for you!</p>
            </div>
            <div class="quiz-form">
                <div class="form-group">
                    <label for="quiz-topic">Topic / Query:</label>
                    <input type="text" id="quiz-topic" placeholder="e.g., machine learning, neural networks..." autocomplete="off">
                </div>
                <div class="form-row">
                    <div class="form-group">
                        <label for="quiz-questions">Number of Questions:</label>
                        <input type="number" id="quiz-questions" min="1" max="20" value="5">
                    </div>
                    <div class="form-group">
                        <label for="quiz-difficulty">Difficulty:</label>
                        <select id="quiz-difficulty">
                            <option value="easy">Easy</option>
                            <option value="medium" selected>Medium</option>
                            <option value="hard">Hard</option>
                        </select>
                    </div>
                </div>
                <button class="generate-quiz-btn" id="generate-quiz-btn">
                    <span>🚀 Generate Quiz</span>
                </button>
            </div>
        </div>
    `;
    
    // Focus on topic input and attach event listeners
    const topicInput = document.getElementById('quiz-topic');
    const generateBtn = document.getElementById('generate-quiz-btn');
    
    if (topicInput) {
        topicInput.focus();
        topicInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                handleGenerateQuiz();
            }
        });
    }
    
    if (generateBtn) {
        generateBtn.addEventListener('click', handleGenerateQuiz);
        console.log('✅ Generate quiz button event listener attached');
    } else {
        console.error('❌ Generate quiz button not found!');
    }
}

async function handleGenerateQuiz() {
    const topicInput = document.getElementById('quiz-topic');
    const questionsInput = document.getElementById('quiz-questions');
    const difficultyInput = document.getElementById('quiz-difficulty');
    const generateBtn = document.getElementById('generate-quiz-btn');
    
    if (!topicInput || !questionsInput || !difficultyInput) {
        console.error('Quiz form inputs not found!');
        return;
    }
    
    const query = topicInput.value.trim();
    if (!query) {
        alert('Please enter a topic for quiz generation');
        return;
    }
    
    const numQ = parseInt(questionsInput.value) || 5;
    if (numQ < 1 || numQ > 20) {
        alert('Number of questions must be between 1 and 20');
        return;
    }
    
    const diff = difficultyInput.value;
    
    try {
        if (generateBtn) {
            generateBtn.disabled = true;
            generateBtn.innerHTML = '<span>⏳ Generating...</span>';
        }
        
        quizModalBody.innerHTML = `
            <div class="quiz-loading">
                <div class="loading-spinner"></div>
                <p>Generating quiz on "${query}"...</p>
                <p class="loading-subtext">This may take 10-30 seconds</p>
            </div>
        `;
        
        const response = await fetch(API_GENERATE_QUIZ_ENDPOINT, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                query: query,
                num_questions: numQ,
                difficulty: diff,
                question_types: 'all'
            })
        });
        
        if (response.ok) {
            const data = await response.json();
            if (data.success && data.quiz) {
                currentQuiz = data.quiz;
                quizAnswers = {};
                displayQuizInModal(currentQuiz);
            } else {
                alert('Failed to generate quiz. Please try again.');
                showQuizForm();
            }
        } else {
            const error = await response.json().catch(() => ({ detail: 'Failed to generate quiz' }));
            alert(`Error: ${error.detail || 'Unknown error'}`);
            showQuizForm();
        }
    } catch (error) {
        console.error('Quiz generation error:', error);
        alert('Failed to generate quiz. Please check if the server is running.');
        showQuizForm();
    } finally {
        if (generateBtn) {
            generateBtn.disabled = false;
            generateBtn.innerHTML = '<span>🚀 Generate Quiz</span>';
        }
    }
}

function displayQuizInModal(quiz) {
    const questions = quiz.questions || [];
    
    let html = `
        <div class="quiz-container-modal">
            <div class="quiz-header-modal">
                <h2>📝 ${quiz.quiz_title || 'Quiz'}</h2>
                <div class="quiz-meta">
                    <span class="quiz-badge difficulty-${quiz.difficulty || 'medium'}">${(quiz.difficulty || 'medium').toUpperCase()}</span>
                    <span class="quiz-badge">${questions.length} Questions</span>
                    <span class="quiz-badge">Topic: ${quiz.topic || 'General'}</span>
                </div>
            </div>
            
            <div class="quiz-questions-modal">
    `;
    
    questions.forEach((q, index) => {
        const qNum = q.question_number || (index + 1);
        const qType = q.question_type || 'multiple_choice';
        
        html += `
            <div class="quiz-question" data-question-number="${qNum}" data-question-type="${qType}">
                <div class="question-header">
                    <span class="question-number">Question ${qNum}</span>
                    <span class="question-type-badge">${qType.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}</span>
                </div>
                <div class="question-text">${q.question || ''}</div>
        `;
        
        if (qType === 'multiple_choice' && q.options) {
            html += '<div class="options-container">';
            q.options.forEach((option, optIndex) => {
                const optionLetter = String.fromCharCode(65 + optIndex);
                html += `
                    <label class="option-label">
                        <input type="radio" name="question-${qNum}" value="${optionLetter}" onchange="updateQuizAnswer(${qNum}, '${optionLetter}')">
                        <span class="option-letter">${optionLetter}.</span>
                        <span class="option-text">${option}</span>
                    </label>
                `;
            });
            html += '</div>';
        } else if (qType === 'true_false') {
            html += `
                <div class="options-container">
                    <label class="option-label">
                        <input type="radio" name="question-${qNum}" value="True" onchange="updateQuizAnswer(${qNum}, 'True')">
                        <span class="option-letter">✓</span>
                        <span class="option-text">True</span>
                    </label>
                    <label class="option-label">
                        <input type="radio" name="question-${qNum}" value="False" onchange="updateQuizAnswer(${qNum}, 'False')">
                        <span class="option-letter">✗</span>
                        <span class="option-text">False</span>
                    </label>
                </div>
            `;
        } else if (qType === 'short_answer') {
            html += `
                <div class="short-answer-container">
                    <textarea 
                        class="short-answer-input" 
                        placeholder="Type your answer here..."
                        onchange="updateQuizAnswer(${qNum}, this.value)"
                        rows="3"
                    ></textarea>
                </div>
            `;
        }
        
        html += `
                <div class="question-result" id="result-${qNum}" style="display: none;"></div>
            </div>
        `;
    });
    
    html += `
            </div>
            <div class="quiz-actions">
                <button class="submit-quiz-btn" onclick="submitQuiz()">Submit Quiz</button>
                <button class="reset-quiz-btn" onclick="resetQuiz()">Reset Answers</button>
                <button class="new-quiz-btn" onclick="showQuizForm()">New Quiz</button>
            </div>
            <div class="quiz-score" id="quiz-score" style="display: none;"></div>
        </div>
    `;
    
    quizModalBody.innerHTML = html;
    quizModalBody.scrollTop = 0;
}

function displayQuiz(quiz) {
    const questions = quiz.questions || [];
    
    let html = `
        <div class="quiz-container">
            <div class="quiz-header">
                <h2>📝 ${quiz.quiz_title || 'Quiz'}</h2>
                <div class="quiz-meta">
                    <span class="quiz-badge difficulty-${quiz.difficulty || 'medium'}">${(quiz.difficulty || 'medium').toUpperCase()}</span>
                    <span class="quiz-badge">${questions.length} Questions</span>
                    <span class="quiz-badge">Topic: ${quiz.topic || 'General'}</span>
                </div>
            </div>
            
            <div class="quiz-questions">
    `;
    
    questions.forEach((q, index) => {
        const qNum = q.question_number || (index + 1);
        const qType = q.question_type || 'multiple_choice';
        
        html += `
            <div class="quiz-question" data-question-number="${qNum}" data-question-type="${qType}">
                <div class="question-header">
                    <span class="question-number">Question ${qNum}</span>
                    <span class="question-type-badge">${qType.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}</span>
                </div>
                <div class="question-text">${q.question || ''}</div>
        `;
        
        if (qType === 'multiple_choice' && q.options) {
            html += '<div class="options-container">';
            q.options.forEach((option, optIndex) => {
                const optionLetter = String.fromCharCode(65 + optIndex);
                html += `
                    <label class="option-label">
                        <input type="radio" name="question-${qNum}" value="${optionLetter}" onchange="updateQuizAnswer(${qNum}, '${optionLetter}')">
                        <span class="option-letter">${optionLetter}.</span>
                        <span class="option-text">${option}</span>
                    </label>
                `;
            });
            html += '</div>';
        } else if (qType === 'true_false') {
            html += `
                <div class="options-container">
                    <label class="option-label">
                        <input type="radio" name="question-${qNum}" value="True" onchange="updateQuizAnswer(${qNum}, 'True')">
                        <span class="option-letter">✓</span>
                        <span class="option-text">True</span>
                    </label>
                    <label class="option-label">
                        <input type="radio" name="question-${qNum}" value="False" onchange="updateQuizAnswer(${qNum}, 'False')">
                        <span class="option-letter">✗</span>
                        <span class="option-text">False</span>
                    </label>
                </div>
            `;
        } else if (qType === 'short_answer') {
            html += `
                <div class="short-answer-container">
                    <textarea 
                        class="short-answer-input" 
                        placeholder="Type your answer here..."
                        onchange="updateQuizAnswer(${qNum}, this.value)"
                        rows="3"
                    ></textarea>
                </div>
            `;
        }
        
        html += `
                <div class="question-result" id="result-${qNum}" style="display: none;"></div>
            </div>
        `;
    });
    
    html += `
            </div>
            <div class="quiz-actions">
                <button class="submit-quiz-btn" onclick="submitQuiz()">Submit Quiz</button>
                <button class="reset-quiz-btn" onclick="resetQuiz()">Reset Answers</button>
            </div>
            <div class="quiz-score" id="quiz-score" style="display: none;"></div>
        </div>
    `;
    
    contentArea.innerHTML = html;
}

function updateQuizAnswer(questionNumber, answer) {
    quizAnswers[questionNumber] = answer;
}

function submitQuiz() {
    if (!currentQuiz) return;
    
    const questions = currentQuiz.questions || [];
    let correctCount = 0;
    let totalQuestions = questions.length;
    
    questions.forEach((q) => {
        const qNum = q.question_number || 0;
        const userAnswer = quizAnswers[qNum] || '';
        const correctAnswer = q.correct_answer || '';
        const resultDiv = document.getElementById(`result-${qNum}`);
        
        if (!resultDiv) return;
        
        let isCorrect = false;
        if (q.question_type === 'multiple_choice' || q.question_type === 'true_false') {
            isCorrect = userAnswer.trim().toUpperCase() === String(correctAnswer).trim().toUpperCase();
        } else if (q.question_type === 'short_answer') {
            // For short answer, just show the model answer
            isCorrect = false; // Don't auto-grade short answers
        }
        
        if (isCorrect) {
            correctCount++;
        }
        
        resultDiv.style.display = 'block';
        resultDiv.className = `question-result ${isCorrect ? 'correct' : 'incorrect'}`;
        
        if (q.question_type === 'short_answer') {
            resultDiv.innerHTML = `
                <div class="result-header">Answer & Explanation</div>
                <div class="correct-answer"><strong>Model Answer:</strong> ${correctAnswer}</div>
                ${q.explanation ? `<div class="explanation"><strong>Explanation:</strong> ${q.explanation}</div>` : ''}
            `;
        } else {
            resultDiv.innerHTML = `
                <div class="result-header ${isCorrect ? 'correct' : 'incorrect'}">
                    ${isCorrect ? '✓ Correct!' : '✗ Incorrect'}
                </div>
                <div class="user-answer"><strong>Your Answer:</strong> ${userAnswer || 'Not answered'}</div>
                <div class="correct-answer"><strong>Correct Answer:</strong> ${correctAnswer}</div>
                ${q.explanation ? `<div class="explanation"><strong>Explanation:</strong> ${q.explanation}</div>` : ''}
            `;
        }
    });
    
    // Show score
    const scoreDiv = document.getElementById('quiz-score');
    if (scoreDiv) {
        const percentage = ((correctCount / totalQuestions) * 100).toFixed(1);
        scoreDiv.style.display = 'block';
        scoreDiv.className = `quiz-score score-${percentage >= 70 ? 'good' : percentage >= 50 ? 'medium' : 'poor'}`;
        scoreDiv.innerHTML = `
            <h3>Quiz Results</h3>
            <div class="score-value">${correctCount} / ${totalQuestions} (${percentage}%)</div>
            <div class="score-message">${percentage >= 70 ? '🎉 Great job!' : percentage >= 50 ? '👍 Good effort!' : '📚 Keep studying!'}</div>
        `;
    }
}

function resetQuiz() {
    if (!currentQuiz) return;
    quizAnswers = {};
    // Check if we're in modal or main content area
    if (quizModal && quizModal.classList.contains('active')) {
        displayQuizInModal(currentQuiz);
    } else {
        displayQuiz(currentQuiz);
    }
    const scoreDiv = document.getElementById('quiz-score');
    if (scoreDiv) {
        scoreDiv.style.display = 'none';
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

