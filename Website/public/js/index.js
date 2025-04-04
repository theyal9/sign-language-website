// Practice functionality
let countdownInterval;
let currentQuestion;

// Sign interpreter - Model Functionality
let modelInterval;

let predictionActive = false;
let mediaRecorder; 
let recordingChunks = []; // Store recorded data
let cameraStream; // Store the camera stream
let currentPrediction; 

window.addEventListener('scroll', function() {
    const header = document.querySelector('header');
    const scrollPosition = window.scrollY;
    
    if (scrollPosition > 100) {
        header.classList.add('scrolled-nav');
        header.classList.remove('bg-transparent');
    } else {
        header.classList.remove('scrolled-nav');
        header.classList.add('bg-transparent');
    }
});

document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('nav a').forEach(link => {
        link.addEventListener('click', (e) => {
            document.getElementById('consent-modal').classList.add('hidden');
            e.preventDefault();
            const targetId = e.target.getAttribute('href');
            const targetSection = document.querySelector(targetId);
            
            if (targetSection) {
                // Show all main sections
                document.querySelectorAll('main, section').forEach(el => {
                    if(!['learning-content', 'practice-content', 'model-content'].includes(el.id)) {
                        el.classList.remove('hidden');
                    }
                });
                
                // Show navigation
                document.querySelector('nav').classList.remove('hidden');
                
                // Scroll to target
                targetSection.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // Hide learning/practice sections by default
    document.getElementById('learning-content').classList.add('hidden');
    document.getElementById('practice-content').classList.add('hidden');
    document.getElementById('model-content').classList.add('hidden');

    // Start Learning button
    document.getElementById('start-learning')?.addEventListener('click', async () => {
        await loadVideoCards();
        // Hide other sections
        document.querySelectorAll('main, section').forEach(el => {
            if(el.id !== 'learning-content') el.classList.add('hidden');
        });
        
        // Hide navigation
        document.querySelector('nav').classList.add('hidden');
        
        document.querySelectorAll('main, section').forEach(el => {
            if(el.id !== 'learning-content') el.classList.add('hidden');
        });
        document.querySelector('nav').classList.add('hidden');
        const learningContent = document.getElementById('learning-content');
        learningContent.classList.remove('hidden');
        learningContent.scrollIntoView({ behavior: 'smooth' });
    });

    document.getElementById('back-to-home')?.addEventListener('click', () => {
        // Show all main sections
        document.querySelectorAll('main, section').forEach(el => {
            if(el.id !== 'learning-content' && el.id !== 'practice-content') {
                el.classList.remove('hidden');
            }
        });
        
        // Show navigation
        document.querySelector('nav').classList.remove('hidden');
        
        // Hide components
        document.getElementById('learning-content').classList.add('hidden');
        document.getElementById('practice-content').classList.add('hidden');
        document.getElementById('model-content').classList.add('hidden');
        document.getElementById('home').scrollIntoView({ behavior: 'smooth' });
    });
});

async function loadVideoCards(page = 1) {
    try {
        const response = await fetch(`http://127.0.0.1:3000/api/videos?page=${page}`);

        if (!response.ok) { 
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json(); 
        
        const grid = document.querySelector('#video-grid');

        // Display video cards
        grid.innerHTML = data.videos.map(video => `
            <div class="bg-white rounded-lg shadow-xl overflow-hidden hover:transform hover:scale-105 transition-all duration-300">
                <video class="w-full h-50 object-cover" autoplay loop muted playsinline>
                    <source src="${video.path}" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
                <div class="p-4 text-indigo-600">
                    <h3 class="text-xl font-bold mb-2">${video.title}</h3>
                </div>
            </div>
        `).join('');

        // Remove existing pagination
        const existingPagination = document.querySelector('.pagination');
        if (existingPagination) existingPagination.remove();

        // Add new pagination 
        const paginationDiv = document.createElement('div');
        paginationDiv.className = 'pagination mt-8 flex justify-center gap-2';
        
        if(data.currentPage > 1) {
            paginationDiv.innerHTML += `
                <button onclick="loadVideoCards(${data.currentPage - 1})" 
                    class="bg-white text-indigo-600 px-4 py-2 rounded">
                    Previous
                </button>`;
        }
        
        if(data.currentPage < data.totalPages) {
            paginationDiv.innerHTML += `
                <button onclick="loadVideoCards(${data.currentPage + 1})" 
                    class="bg-white text-indigo-600 px-4 py-2 rounded">
                    Next
                </button>`;
        }
        
        grid.after(paginationDiv);
    } catch (error) {
        console.error('Error loading videos:', error);
        const errorMessage = document.getElementById('error-message');
        errorMessage.classList.remove('hidden');
        errorMessage.textContent = 'Failed to load videos. Please try again.';
    }
}

document.getElementById('start-practicing')?.addEventListener('click', async () => {
    // Hide other sections
    document.querySelectorAll('main, section').forEach(el => {
        if(el.id !== 'practice-content') el.classList.add('hidden');
    });
    document.querySelector('nav').classList.add('hidden');
    document.getElementById('practice-content').classList.remove('hidden');
    
    // Load first question
    await loadPracticeQuestion();
});

document.getElementById('stop-practice')?.addEventListener('click', () => {
    // Show all main sections
    document.querySelectorAll('main, section').forEach(el => {
        if(el.id !== 'learning-content' && el.id !== 'practice-content') {
            el.classList.remove('hidden');
        }
    });
    
    // Show navigation
    document.querySelector('nav').classList.remove('hidden');
    
    // Hide components
    document.getElementById('learning-content').classList.add('hidden');
    document.getElementById('practice-content').classList.add('hidden');
    document.getElementById('model-content').classList.add('hidden');
    document.getElementById('home').scrollIntoView({ behavior: 'smooth' });
    
    // Clear any ongoing countdown
    clearInterval(countdownInterval);
});

async function loadPracticeQuestion() {
    try {
        const response = await fetch('http://localhost:3000/api/practice-video');
        currentQuestion = await response.json();
        
        const video = document.getElementById('practice-video');
        video.src = currentQuestion.videoPath;
        video.load();
        
        const optionsContainer = document.getElementById('answer-options');
        optionsContainer.innerHTML = currentQuestion.options.map(option => `
            <button class="answer-option bg-white text-emerald-800 p-4 rounded-lg hover:bg-gray-100 
                           transition-all duration-300 text-lg font-medium" 
                    data-correct="${option === currentQuestion.correctAnswer}">
                ${option}
            </button>
        `).join('');
        
        document.getElementById('feedback').classList.add('hidden');
        optionsContainer.querySelectorAll('.answer-option').forEach(btn => {
            btn.addEventListener('click', handleAnswer);
        });
    } catch (error) {
        console.error('Error loading practice question:', error);
    }
}

function handleAnswer(e) {
    // Clear any existing intervals to prevent multiple countdowns
    clearInterval(countdownInterval);

    const isCorrect = e.target.dataset.correct === 'true';
    const feedback = document.getElementById('feedback');
    const feedbackText = document.getElementById('feedback-text');
    const correctAnswerElement = document.getElementById('correct-answer');

    // Show feedback and correct answer
    feedback.classList.remove('hidden');
    feedbackText.textContent = isCorrect ? '✓ Correct!' : '✗ Incorrect!';
    feedbackText.style.color = isCorrect ? '#10b981' : '#ef4444';
    correctAnswerElement.textContent = currentQuestion.correctAnswer;

    // Disable all buttons
    document.querySelectorAll('.answer-option').forEach(btn => {
        btn.disabled = true;
        btn.classList.remove('bg-green-100', 'bg-red-100');
        if(btn.dataset.correct === 'true') {
            btn.classList.add(isCorrect ? 'bg-green-100' : 'bg-red-100');
        }
    });

    // Start fresh countdown
    let countdown = 5;
    const countdownElement = document.getElementById('countdown');
    countdownElement.textContent = countdown;  // Reset display to 5
    
    countdownInterval = setInterval(() => {
        countdown--;
        countdownElement.textContent = countdown;
        
        if(countdown <= 0) {
            clearInterval(countdownInterval);
            loadPracticeQuestion();
        }
    }, 1000);
}

// Model start button
document.getElementById('start-model')?.addEventListener('click', (e) => {
    e.preventDefault();
    document.getElementById('consent-modal').classList.remove('hidden');
    document.getElementById('model-content').classList.remove('hidden');
});

document.getElementById('consent-no').addEventListener('click', () => {
    // Show all main sections
    document.querySelectorAll('main, section').forEach(el => {
        if(el.id !== 'learning-content' && el.id !== 'practice-content') {
            el.classList.remove('hidden');
        }
    });
    
    // Show navigation
    document.querySelector('nav').classList.remove('hidden');
    
    // Hide model content
    document.getElementById('model-content').classList.add('hidden');
    document.getElementById('home').scrollIntoView({ behavior: 'smooth' });
});

async function startCamera() {
    try {
        // Stop existing streams
        if (cameraStream) {
            cameraStream.getTracks().forEach(track => track.stop());
        }
        
        // Get new stream
        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480 }
        });
        
        const video = document.getElementById('camera-feed');
        video.srcObject = cameraStream;
    } catch (error) {
        console.error('Camera error:', error);
        alert('Camera access failed. Please enable permissions.');
    }
}

function startRecording() {
    
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        return;
    }

    recordingChunks = [];
    mediaRecorder = new MediaRecorder(cameraStream, { mimeType: 'video/mp4' });

    mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) recordingChunks.push(e.data);
    };

    mediaRecorder.start();
}

// stopRecording function
async function stopRecording() {
    return new Promise((resolve, reject) => {
        try {
            if (!mediaRecorder || mediaRecorder.state !== 'recording') {
                resolve(null);
                return;
            }

            mediaRecorder.onstop = () => {
                try {
                    const blob = new Blob(recordingChunks, { type: 'video/mp4' });
                    resolve(blob);
                } catch (error) {
                    reject(error);
                }
            };

            mediaRecorder.onerror = (event) => {
                reject(event.error);
            };

            mediaRecorder.stop();
        } catch (error) {
            reject(error);
        }
    });
}

document.getElementById('consent-yes').addEventListener('click', async () => {
    // Hide ALL other sections
    document.querySelectorAll('main, section').forEach(el => {
        el.classList.add('hidden');
    });
    
    // Show model-content and its interface
    document.getElementById('model-content').classList.remove('hidden');
    document.getElementById('model-interface').classList.remove('hidden');
    
    // Hide navigation and consent modal
    document.querySelector('nav').classList.add('hidden');
    document.getElementById('consent-modal').classList.add('hidden');

    try {
        await startCamera();
        startRecording();
        startPredictionCycle();
    } catch (error) {
        console.error('Camera access error:', error);
    }
});

let currentVideoBlob = null;

async function startPredictionCycle() {
    if (predictionActive) return;
    predictionActive = true;

    try {
        // Cleanup previous resources
        if (mediaRecorder?.state === 'recording') {
            await stopRecording();
        }
        if (cameraStream) {
            cameraStream.getTracks().forEach(track => track.stop());
        }

        // Stop any existing camera stream
        if (cameraStream) {
            cameraStream.getTracks().forEach(track => track.stop());
        }

        // Start fresh camera session
        await startCamera();
        
        // Show countdown and start recording
        document.getElementById('countdown-timer').textContent = '5s remaining';
        recordingChunks = [];
        startRecording();

        // Update countdown
        let countdown = 5;
        const countdownInterval = setInterval(() => {
            countdown--;
            document.getElementById('countdown-timer').textContent = `${countdown}s remaining`;
            if (countdown <= 0) clearInterval(countdownInterval);
        }, 1000);

        // Wait exactly 5 seconds
        await new Promise(resolve => setTimeout(resolve, 5000));
        
        // Stop recording and camera
        currentVideoBlob = await stopRecording();
        cameraStream.getTracks().forEach(track => track.stop());

        // Get prediction
        const formData = new FormData();
        formData.append('video', currentVideoBlob, 'gesture.mp4');
        const response = await fetch('http://localhost:3000/api/predict', {
            method: 'POST',
            body: formData,
        });
        
        if (!response.ok) throw new Error('Prediction API error');
        const { prediction, confidence } = await response.json();
        currentPrediction = prediction;

        // Show prediction UI
        showPredictionResult(prediction);
        await waitForUserFeedback();

    } catch (error) {
        console.error('Prediction cycle error:', error);
    } finally {
        predictionActive = false;
    }
}

let feedbackResolver = null;

function waitForUserFeedback() {
  return new Promise(resolve => {
    feedbackResolver = resolve; // Store resolver reference
    
    const handleFeedback = () => {
      // Cleanup listeners
      document.getElementById('correct-btn').removeEventListener('click', handleCorrect);
      document.getElementById('incorrect-btn').removeEventListener('click', handleIncorrect);
      resolve();
    };

    const handleCorrect = () => handleFeedback();
    const handleIncorrect = () => handleFeedback();

    document.getElementById('correct-btn').addEventListener('click', handleCorrect);
    document.getElementById('incorrect-btn').addEventListener('click', handleIncorrect);
  });
}

function showPredictionResult(prediction) {
    document.getElementById('prediction-text').textContent = prediction;
    document.getElementById('prediction-result').classList.remove('hidden');
}

// Handle prediction feedback
document.getElementById('correct-btn').addEventListener('click', async () => {
    // Restart cycle
    startPredictionCycle();
});

document.getElementById('incorrect-btn').addEventListener('click', () => {
    document.getElementById('correction-input').classList.remove('hidden');
    document.getElementById('prediction-result').classList.add('hidden');
});

// Correction handling
document.getElementById('submit-correction').addEventListener('click', async () => {
    const userInput = document.getElementById('correct-word').value.trim().toLowerCase();
    
    try {
        // Validate input
        const validation = await fetch(
            `http://localhost:3000/api/validate-word?word=${encodeURIComponent(userInput)}`
        );
        const { valid } = await validation.json();
        
        if (!valid) {
            alert('Please enter a valid English word');
            return;
        }

        // Submit correction
        const formData = new FormData();
        formData.append('video', currentVideoBlob, 'gesture.mp4');
        formData.append('correct_sign', userInput);
        formData.append('predicted_sign', currentPrediction);

        await fetch('http://localhost:3000/api/store-data', {
            method: 'POST',
            body: formData
        });

        // Clear UI
        document.getElementById('correction-input').classList.add('hidden');
        document.getElementById('correct-word').value = '';

        // Restart cycle
        startPredictionCycle();

    } catch (error) {
        console.error('Correction error:', error);
    }
});

function resetRecognitionCycle() {
    recordingChunks = []; 
    if (mediaRecorder?.state !== 'recording') {
        mediaRecorder.start();
    }
    document.getElementById('correction-input').classList.add('hidden');
    document.getElementById('prediction-result').classList.add('hidden');
    document.getElementById('correct-word').value = '';
    startPredictionCycle();
}

async function submitDataToServer(videoData, correctSign, predictedSign) {
    try {
        const formData = new FormData();
        formData.append('video', videoData, 'gesture.mp4');
        formData.append('correct_sign', correctSign);
        formData.append('predicted_sign', predictedSign);

        await fetch('http://localhost:3000/api/store-data', {
            method: 'POST',
            body: formData
        });
    } catch (error) {
        console.error('Error submitting data:', error);
    }
}

// Stop button handler
document.getElementById('stop-model').addEventListener('click', () => {
    predictionActive = false;
    
    if (mediaRecorder?.state === 'recording') {
        mediaRecorder.stop();
    }
    
    // Cleanup media streams
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
    }

    clearInterval(modelInterval);
    if (mediaRecorder?.state !== 'inactive') {
        mediaRecorder?.stop();
    }

    // Stop camera feed
    const video = document.getElementById('camera-feed');
    if (video.srcObject) {
        video.srcObject.getTracks().forEach(track => {
            track.stop();  // This actually stops the camera
        });
        video.srcObject = null;
    }

    // Show all main sections
    document.querySelectorAll('main, section').forEach(el => {
        if(el.id !== 'learning-content' && el.id !== 'practice-content') {
            el.classList.remove('hidden');
        }
    });
    
    // Show navigation
    document.querySelector('nav').classList.remove('hidden');
    
    // Hide learning content
    document.getElementById('model-content').classList.add('hidden');
    document.getElementById('home').scrollIntoView({ behavior: 'smooth' });
});
