const API_BASE_URL = 'http://localhost:5000';

// Status indicator (may not exist on all pages)
const statusIndicator = document.getElementById('status-indicator');

// Check API status
async function checkAPIStatus() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        if (response.ok) {
            const data = await response.json();
            if (statusIndicator) {
                statusIndicator.textContent = data.chatbot_enabled 
                    ? 'API Online • Chatbot Enabled' 
                    : 'API Online • Chatbot Disabled';
                statusIndicator.className = 'status-indicator status-online';
            }
            return true;
        }
    } catch (error) {
        if (statusIndicator) {
            statusIndicator.textContent = 'API Offline';
            statusIndicator.className = 'status-indicator status-offline';
        }
        return false;
    }
    if (statusIndicator) {
        statusIndicator.textContent = 'API Offline';
        statusIndicator.className = 'status-indicator status-offline';
    }
    return false;
}

// Load model information
async function loadModelInfo() {
    try {
        const response = await fetch(`${API_BASE_URL}/model-info`);
        if (!response.ok) throw new Error('Failed to load model info');

        const data = await response.json();

        const infoItems = document.querySelectorAll('.info-item');
        if (infoItems.length >= 4) {
            infoItems[0].querySelector('p').textContent = data.model_type || 'Random Forest';
            infoItems[1].querySelector('p').textContent = (typeof data.accuracy === 'number') ? `${data.accuracy}%` : 'N/A';
            infoItems[2].querySelector('p').textContent = data.features?.join(', ') || 'Age, Height, Weight';
            infoItems[3].querySelector('p').textContent = `${(data.classes && data.classes.length) ? data.classes.length : 6} Categories`;
        }

    } catch (error) {
        console.error('Error loading model info:', error);
    }
}

// Get BMI class styling
function getBMIClassStyle(className) {
    const classMap = {
        'Underweight': 'underweight',
        'Normal Weight': 'normal-weight',
        'Overweight': 'overweight',
        'Obese Class 1': 'obese-class-1',
        'Obese Class 2': 'obese-class-2',
        'Obese Class 3': 'obese-class-3'
    };
    return classMap[className] || '';
}

// Format results
function displayResults(data) {
    const resultsContent = document.getElementById('results-content');
    if (!resultsContent) return;

    const bmiValue = (typeof data.calculated_bmi === 'number') ? data.calculated_bmi.toFixed(2) : data.calculated_bmi;

    const topPreds = Array.isArray(data.top_predictions) ? data.top_predictions : [];

    resultsContent.innerHTML = `
        <div class="bmi-display">
            <div class="bmi-value">${bmiValue}</div>
            <div class="bmi-class ${getBMIClassStyle(data.predicted_class)}">${data.predicted_class}</div>
            <div class="confidence-score">Confidence: ${data.confidence}%</div>
        </div>
        
        <div class="result-card">
            <h3 style="margin-bottom: 15px; color: #555;">Input Summary</h3>
            <p><strong>Age:</strong> ${data.input.age} years</p>
            <p><strong>Height:</strong> ${data.input.height} m</p>
            <p><strong>Weight:</strong> ${data.input.weight} kg</p>
            <p><strong>Calculated BMI:</strong> ${bmiValue}</p>
        </div>
        
        <div class="predictions-list">
            <h3 style="margin-bottom: 15px; color: #555;">Top Predictions</h3>
            ${topPreds.length > 0 ? topPreds.map(([className, probability]) => `
                <div class="prediction-item">
                    <span class="prediction-class">${className}</span>
                    <span class="prediction-probability">${probability}%</span>
                </div>
            `).join('') : '<div>No prediction details available.</div>'}
        </div>
    `;
}

// Handle BMI form submission
const bmiForm = document.getElementById('bmi-form');
if (bmiForm) {
    bmiForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        const submitBtn = document.getElementById('predict-btn');
        const resultsContent = document.getElementById('results-content');

        const formData = new FormData(e.target);
        const data = {
            age: parseFloat(formData.get('age')),
            height: parseFloat(formData.get('height')),
            weight: parseFloat(formData.get('weight'))
        };

        if (isNaN(data.age) || isNaN(data.height) || isNaN(data.weight)) {
            if (resultsContent) {
                resultsContent.innerHTML = `<div class="error"><strong>Error:</strong> Please enter valid numbers for all fields.</div>`;
            }
            return;
        }

        if (submitBtn) {
            submitBtn.disabled = true;
            submitBtn.textContent = 'Predicting...';
        }
        if (resultsContent) {
            resultsContent.innerHTML = `<div class="loading"><div class="spinner"></div><span>Analyzing patient data...</span></div>`;
        }

        try {
            const response = await fetch(`${API_BASE_URL}/predict`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(data)
            });

            const result = await response.json();

            if (!response.ok) {
                throw new Error(result.error || 'Prediction failed');
            }

            displayResults(result);

        } catch (error) {
            if (resultsContent) {
                resultsContent.innerHTML = `<div class="error"><strong>Error:</strong> ${error.message}</div>`;
            }
        } finally {
            if (submitBtn) {
                submitBtn.disabled = false;
                submitBtn.textContent = 'Predict BMI Classification';
            }
        }
    });
}

// --- IMPROVED CHATBOT FUNCTIONALITY ---
const chatForm = document.getElementById('chat-form');
const chatInput = document.getElementById('chat-input');
const chatWindow = document.getElementById('chat-window');
const sendBtn = document.getElementById('send-btn');

// Function to append message to chat window (FIXED)
function appendMessage(sender, text) {
    if (!chatWindow) return;

    const msgDiv = document.createElement('div');
    msgDiv.className = (sender === 'User') ? 'chat-user' : 'chat-bot';

    const senderSpan = document.createElement('strong');
    senderSpan.textContent = sender;

    const textSpan = document.createElement('span');
    textSpan.textContent = text;
    textSpan.style.display = 'block'; // Ensure text is on new line

    msgDiv.appendChild(senderSpan);
    msgDiv.appendChild(textSpan);

    chatWindow.appendChild(msgDiv);
    chatWindow.scrollTop = chatWindow.scrollHeight; // Auto-scroll
}

// Handle chat form submission
if (chatForm) {
    chatForm.addEventListener('submit', async function(e) {
        e.preventDefault();

        const message = chatInput ? chatInput.value.trim() : '';
        if (message === "") return;

        appendMessage('User', message);
        if (chatInput) chatInput.value = '';

        // Get the button - could be sendBtn or the form button
        const button = sendBtn || chatForm.querySelector('button[type="submit"]');
        
        if (button) {
            button.disabled = true;
            const originalText = button.textContent;
            button.textContent = 'Sending...';

            try {
                const response = await fetch(`${API_BASE_URL}/chatbot`, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ message })
                });

                const data = await response.json();

                if (response.ok && data.success && data.reply) {
                    appendMessage('Bot', data.reply);
                } else if (data.error) {
                    appendMessage('Bot', `Error: ${data.error}`);
                } else {
                    appendMessage('Bot', 'Sorry, I received an unexpected response.');
                }

            } catch (error) {
                console.error('Chat error:', error);
                appendMessage('Bot', `Sorry, I encountered an error: ${error.message}`);
            } finally {
                if (button) {
                    button.disabled = false;
                    button.textContent = originalText;
                }
            }
        }
    });
}

// Allow Enter key to send message (Shift+Enter for new line)
if (chatInput) {
    chatInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (chatForm) {
                chatForm.dispatchEvent(new Event('submit'));
            }
        }
    });
}

// Initialize app
async function initializeApp() {
    console.log('Initializing BMI Prediction App...');

    const isOnline = await checkAPIStatus();

    if (isOnline) {
        console.log('API is online. Loading model info...');
        await loadModelInfo();

        // Add welcome message to chatbot
        if (chatWindow) {
            appendMessage('Bot', 'Hello! I\'m your BMI and health assistant. Ask me anything about BMI, nutrition, fitness, or general health topics!');
        }
    } else {
        console.warn('API is offline. Please start the Flask server.');

        const resultsContent = document.getElementById('results-content');
        if (resultsContent) {
            resultsContent.innerHTML = `<div class="error"><strong>API Offline:</strong> Please start the Flask server at ${API_BASE_URL}</div>`;
        }
        
        if (chatWindow) {
            appendMessage('Bot', 'API is currently offline. Please start the Flask server to use the chatbot.');
        }
    }
}

// Check API status periodically (every 30 seconds)
setInterval(checkAPIStatus, 30000);

// Initialize on page load
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeApp);
} else {
    initializeApp();
}