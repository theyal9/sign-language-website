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
                // Hide learning/practice sections
                document.getElementById('learning-content').classList.add('hidden');
                document.getElementById('practice-content').classList.add('hidden');
                document.getElementById('model-content').classList.add('hidden');
                
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
    // Hide components
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
        const response = await fetch(`http://localhost:3000/api/videos?page=${page}`);
        const data = await response.json(); 
        
        const grid = document.querySelector('#video-grid');

        // Render video cards
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

        // Add new pagination controls
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

// Practice functionality
let countdownInterval;
let currentQuestion;

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
        if(btn.dataset.correct === 'true') {
            btn.classList.add(isCorrect ? 'bg-green-100' : 'bg-red-100');
        }
    });

    // Start countdown for next question
    let countdown = 5;
    const countdownElement = document.getElementById('countdown');
    
    countdownInterval = setInterval(() => {
        countdown--;
        countdownElement.textContent = countdown;
        
        if(countdown <= 0) {
            clearInterval(countdownInterval);
            loadPracticeQuestion();
        }
    }, 1000);
}

// let mediaRecorder;
// let recordedChunks = [];
let modelInterval;

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
    
    // Hide learning content
    document.getElementById('model-content').classList.add('hidden');
    document.getElementById('home').scrollIntoView({ behavior: 'smooth' });
});

let cameraStream;
let mediaRecorder;
let recordingChunks = [];

async function startCamera() {
    try {
        cameraStream = await navigator.mediaDevices.getUserMedia({ video: true });
        const video = document.getElementById('camera-feed');
        video.srcObject = cameraStream;
    } catch (error) {
        console.error('Error accessing camera:', error);
    }
}

function startRecording() {
    recordingChunks = [];
    mediaRecorder = new MediaRecorder(cameraStream, { mimeType: 'video/webm' });
    
    mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) recordingChunks.push(e.data);
    };
    
    mediaRecorder.start();
}

async function stopRecording() {
    return new Promise((resolve) => {
        mediaRecorder.onstop = async () => {
            const blob = new Blob(recordingChunks, { type: 'video/webm' });
            const buffer = await blob.arrayBuffer();
            const base64 = btoa(String.fromCharCode(...new Uint8Array(buffer)));
            resolve(base64);
        };
        mediaRecorder.stop();
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
        // const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        // const video = document.getElementById('camera-feed');
        // video.srcObject = stream;
        
        // // Setup recording
        // mediaRecorder = new MediaRecorder(stream, {
        //     mimeType: 'video/mp4'
        // });
        // mediaRecorder.ondataavailable = (e) => recordedChunks.push(e.data);
        // mediaRecorder.start();
        
        // startPredictionCycle();

        await startCamera();
        startRecording();
        startPredictionCycle();
    } catch (error) {
        console.error('Camera access error:', error);
    }
});

async function startPredictionCycle() {
    let seconds = 5;
    const timerElement = document.getElementById('countdown-timer');
    
    const countdown = setInterval(() => {
        seconds--;
        timerElement.textContent = `${seconds}s remaining`;
    }, 1000);

    // Wait for recording to complete
    await new Promise(resolve => setTimeout(resolve, 5000));
    
    clearInterval(countdown);
    const videoData = await stopRecording();
    
    // Get prediction
    const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ videoData })
    });
    
    const { prediction } = await response.json();
    showPredictionResult(prediction);
    
    // Restart camera
    await startCamera();
    startRecording();
    startPredictionCycle();
}
// function startPredictionCycle() {
//     let seconds = 5;
//     const timerElement = document.getElementById('countdown-timer');
    
//     modelInterval = setInterval(async () => {
//         seconds--;
//         timerElement.textContent = `${seconds}s remaining`;
        
//         if (seconds <= 0) {
//             clearInterval(modelInterval);
//             mediaRecorder.stop();
            
//             // Get prediction from model
//             const prediction = await getModelPrediction();
//             showPredictionResult(prediction);
//         }
//     }, 1000);
// }

// Modified prediction flow
async function getModelPrediction() {
    const videoBlob = new Blob(recordedChunks, { type: 'video/webm' });
    const reader = new FileReader();
    
    return new Promise((resolve) => {
        reader.onloadend = async () => {
            const base64Data = reader.result.split(',')[1];
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ videoData: base64Data })
            });
            const data = await response.json();
            resolve(data.prediction);
        };
        reader.readAsDataURL(videoBlob);
    });
}

function showPredictionResult(prediction) {
    document.getElementById('prediction-text').textContent = prediction;
    document.getElementById('prediction-result').classList.remove('hidden');
}

// Handle prediction feedback
document.getElementById('correct-btn').addEventListener('click', async () => {
    const videoBlob = new Blob(recordedChunks, { type: 'video/webm' });
    const reader = new FileReader();
    reader.readAsDataURL(videoBlob);
    
    reader.onloadend = () => {
        const base64Data = reader.result.split(',')[1];
        submitDataToServer(base64Data, prediction, prediction);
    };
    
    resetRecognitionCycle();
});

document.getElementById('incorrect-btn').addEventListener('click', () => {
    document.getElementById('correction-input').classList.remove('hidden');
});

// Modified correction handling
document.getElementById('submit-correction').addEventListener('click', async () => {
    const userInput = document.getElementById('correct-word').value.toLowerCase().trim();
    
    // Validate word
    const validation = await fetch(`/api/validate-word?word=${encodeURIComponent(userInput)}`);
    const { valid } = await validation.json();
    
    if (!valid) {
        alert('Please enter a valid English word from the dictionary');
        return;
    }

    // Proceed with storage
    const videoBlob = new Blob(recordedChunks, { type: 'video/webm' });
    const reader = new FileReader();
    
    reader.onloadend = () => {
        const base64Data = reader.result.split(',')[1];
        submitDataToServer(base64Data, userInput, prediction);
    };
    reader.readAsDataURL(videoBlob);

    resetRecognitionCycle();
});

function resetRecognitionCycle() {
    recordedChunks = [];
    mediaRecorder.start();
    document.getElementById('correction-input').classList.add('hidden');
    document.getElementById('prediction-result').classList.add('hidden');
    document.getElementById('correct-word').value = '';
    startPredictionCycle();
}

async function submitDataToServer(videoData, correctSign, predictedSign) {
    try {
        await fetch('/api/store-data', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ videoData, correctSign, predictedSign })
        });
    } catch (error) {
        console.error('Error submitting data:', error);
    }
}

// Stop button handler
document.getElementById('stop-model').addEventListener('click', () => {
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