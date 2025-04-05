// Javascript file for the website

// Global Variables 
// Practice functionality
let countdownInterval;
let currentQuestion; 

// Model Functionality
let predictionActive = false;
let mediaRecorder; 
// Store recorded video chunks
let recordingChunks = [];
let cameraStream;
let currentPrediction; 
let currentVideoBlob = null;

// Sticky Header on Scroll
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

// Navigation click event
document.addEventListener('DOMContentLoaded', () => {
    // Handle navigation links
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
                
                // Scroll to required section
                targetSection.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // Hide sub-sections on load
    document.getElementById('learning-content').classList.add('hidden');
    document.getElementById('practice-content').classList.add('hidden');
    document.getElementById('model-content').classList.add('hidden');

    // Start Learning button
    // It loads the video cards and hides other sections
    document.getElementById('start-learning')?.addEventListener('click', async () => {
        await loadVideoCards();
        // Hide other sections
        document.querySelectorAll('main, section').forEach(el => {
            if(el.id !== 'learning-content') el.classList.add('hidden');
        });
        
        // Hide navigation
        document.querySelector('nav').classList.add('hidden');
        
        const learningContent = document.getElementById('learning-content');
        learningContent.classList.remove('hidden');
        learningContent.scrollIntoView({ behavior: 'smooth' });
    });

    // Back to Home button event
    // Button is used to go back to home page from other sections
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

    // Interpreter Launch button event
    // It sends a request to the server to start the interpreter
    document.getElementById('launch-interpreter')?.addEventListener('click', async (e) => {
        e.preventDefault();
        console.log("Launching interpreter...");
        try {
            const response = await fetch('http://localhost:3000/api/start-interpreter');
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            alert('Interpreter window should appear! Check behind your browser.');
        } catch (error) {
            console.error('Error:', error);
            alert(`Failed to launch: ${error.message}`);
        }
    });
});

// Load video cards from the server
// It fetches the video data from the server and displays it in a grid format
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

// Start practicing button event
// It hides other sections and loads the first question for practice
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

// Stop practicing button event
// It hides the practice section and shows all other sections
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

// Load practice question function
// It fetches a random question from the server and displays it
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

// Handle answer selection
// It checks if the selected answer is correct and provides feedback
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
    
    // Start countdown
    countdownInterval = setInterval(() => {
        countdown--;
        countdownElement.textContent = countdown;
        
        if(countdown <= 0) {
            clearInterval(countdownInterval);
            loadPracticeQuestion();
        }
    }, 1000);
}

// Model start button event
// It shows the consent modal and hides other sections
document.getElementById('start-model')?.addEventListener('click', (e) => {
    e.preventDefault();
    document.getElementById('consent-modal').classList.remove('hidden');
    document.getElementById('model-content').classList.remove('hidden');
});

// Consent modal close button event
// It hides the consent modal and shows all other sections
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

// Start camera function
// It initializes the camera stream and handles camera feed errors
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

// Start recording function
// It initializes the MediaRecorder and starts recording the camera stream
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

// Stop recording function
// It stops the MediaRecorder and returns the recorded video as a Blob
async function stopRecording() {
    return new Promise((resolve, reject) => {
        try {
            // Check if mediaRecorder is defined and in the correct state
            if (!mediaRecorder || mediaRecorder.state !== 'recording') {
                resolve(null);
                return;
            }

            // Stop the MediaRecorder
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

// Consent modal event handler
// It handles the user's consent for camera access and model usage
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

// Start prediction cycle function
// It handles the entire prediction process including camera access, recording, and prediction API call
async function startPredictionCycle() {
    if (predictionActive) return;
    predictionActive = true;
    
    try {
        // Cleanup previous resources
        if (mediaRecorder?.state === 'recording') {
            await stopRecording();
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

// function waitForUserFeedback() {
//   return new Promise(resolve => {

//     const handleFeedback = () => {
//       // Cleanup listeners
//       document.getElementById('correct-btn').removeEventListener('click', handleCorrect);
//       document.getElementById('incorrect-btn').removeEventListener('click', handleIncorrect);
//       resolve();
//     };

//     const handleCorrect = () => handleFeedback();
//     const handleIncorrect = () => handleFeedback();

//     document.getElementById('correct-btn').addEventListener('click', handleCorrect);
//     document.getElementById('incorrect-btn').addEventListener('click', handleIncorrect);
//   });
// }

// Wait for user feedback function
// It handles the user's feedback on the prediction result and returns the feedback value
function waitForUserFeedback() {
    return new Promise(resolve => {
        const correctBtn = document.getElementById('correct-btn');
        const incorrectBtn = document.getElementById('incorrect-btn');
        const submitBtn = document.getElementById('submit-feedback-btn');
    
        let feedback = null;
        
        const handleCorrect = () => {
            feedback = true;
            submitBtn.disabled = false;
        };
    
        const handleIncorrect = () => {
            feedback = false;
            submitBtn.disabled = false;
        };
    
        const handleSubmit = async () => {
            if (feedback === null) return;
    
            // Cleanup event listeners
            correctBtn.removeEventListener('click', handleCorrect);
            incorrectBtn.removeEventListener('click', handleIncorrect);
            submitBtn.removeEventListener('click', handleSubmit);
    
            // Hide buttons individually
            correctBtn.classList.add('hidden');
            incorrectBtn.classList.add('hidden');
            submitBtn.classList.add('hidden');
        
            // Short delay to ensure UI updates
            await new Promise(r => setTimeout(r, 100));
            resolve(feedback);
        };
    
        // Add event listeners
        correctBtn.addEventListener('click', handleCorrect);
        incorrectBtn.addEventListener('click', handleIncorrect);
        submitBtn.addEventListener('click', handleSubmit);
    });
}
  
// Show prediction result function
// It updates the UI with the prediction result and confidence level
function showPredictionResult(prediction) {
    document.getElementById('prediction-text').textContent = prediction;
    document.getElementById('prediction-result').classList.remove('hidden');
}

// Handle prediction feedback
// It handles the user's feedback on the prediction result and updates the UI accordingly
document.getElementById('correct-btn').addEventListener('click', async () => {
    document.getElementById('prediction-result').classList.add('hidden');
    // Restart cycle
    startPredictionCycle();
});

// Handle incorrect prediction
// It shows the correction input field and hides the prediction result
document.getElementById('incorrect-btn').addEventListener('click', () => {
    document.getElementById('correction-input').classList.remove('hidden');
    document.getElementById('prediction-result').classList.add('hidden');
});

// Correction handling event
// It handles the user's correction input and submits it to the server
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

// Reset recognition cycle function
// It resets the recognition cycle by stopping the current recording and starting a new one
function resetRecognitionCycle() {
    // Stop current recording if active
    recordingChunks = []; 
    if (mediaRecorder?.state !== 'recording') {
        mediaRecorder.start();
    } 
    
    // Hide correction input and prediction result
    document.getElementById('correction-input').classList.add('hidden');
    document.getElementById('prediction-result').classList.add('hidden');
    document.getElementById('correct-word').value = '';
    startPredictionCycle();
}

// Submit data to server function
// It sends the recorded video and user feedback to the server for storage
async function submitDataToServer(videoData, correctSign, predictedSign) {
    try {
        // Validate input and send data to server
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

// Stop model button event
// It stops the model prediction and cleans up resources
document.getElementById('stop-model').addEventListener('click', () => {
    predictionActive = false;
    
    if (mediaRecorder?.state === 'recording') {
        mediaRecorder.stop();
    }
    
    // Cleanup media streams
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
    }

    if (mediaRecorder?.state !== 'inactive') {
        mediaRecorder?.stop();
    }

    // Stop camera feed
    const video = document.getElementById('camera-feed');
    if (video.srcObject) {
        video.srcObject.getTracks().forEach(track => {
            track.stop();  
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