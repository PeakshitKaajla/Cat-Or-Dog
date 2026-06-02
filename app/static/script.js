document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const errorMessage = document.getElementById('error-message');
    
    const uploadSection = document.querySelector('.upload-section');
    const loadingSection = document.getElementById('loading-section');
    const resultsSection = document.getElementById('results-section');
    
    const imagePreview = document.getElementById('image-preview');
    const winnerLabel = document.getElementById('winner-label');
    const catScore = document.getElementById('cat-score');
    const dogScore = document.getElementById('dog-score');
    const catProgress = document.getElementById('cat-progress');
    const dogProgress = document.getElementById('dog-progress');
    const resetButton = document.getElementById('reset-button');

    // Setup Drag & Drop
    dropZone.addEventListener('click', () => fileInput.click());

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    resetButton.addEventListener('click', resetUI);

    // File Handling
    function handleFile(file) {
        // Validate it's an image
        if (!file.type.startsWith('image/')) {
            showError("Please upload a valid image file (JPEG, PNG, etc).");
            return;
        }

        hideError();
        
        // Show local preview immediately
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            predictImage(file);
        };
        reader.readAsDataURL(file);
    }

    // API Interaction
    async function predictImage(file) {
        // Show Loading
        uploadSection.classList.add('hidden');
        loadingSection.classList.remove('hidden');

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || "Server error occurred");
            }

            const data = await response.json();
            showResults(data);

        } catch (error) {
            resetUI();
            showError(error.message);
        }
    }

    // UI Updates
    function showResults(data) {
        loadingSection.classList.add('hidden');
        resultsSection.classList.remove('hidden');

        // Winner Text
        winnerLabel.textContent = data.label;
        winnerLabel.className = `winner-label winner-${data.label.toLowerCase()}`;

        // Animate Bars
        // Small delay ensures CSS transitions trigger properly after display:none is removed
        setTimeout(() => {
            const catProb = data.class_probabilities.cat * 100;
            const dogProb = data.class_probabilities.dog * 100;

            catScore.textContent = `${catProb.toFixed(1)}%`;
            catProgress.style.width = `${catProb}%`;

            dogScore.textContent = `${dogProb.toFixed(1)}%`;
            dogProgress.style.width = `${dogProb}%`;
        }, 50);
    }

    function resetUI() {
        resultsSection.classList.add('hidden');
        loadingSection.classList.add('hidden');
        uploadSection.classList.remove('hidden');
        fileInput.value = "";
        
        // Reset bars
        catProgress.style.width = '0%';
        dogProgress.style.width = '0%';
        catScore.textContent = '0%';
        dogScore.textContent = '0%';
    }

    function showError(msg) {
        errorMessage.textContent = msg;
        errorMessage.classList.remove('hidden');
    }

    function hideError() {
        errorMessage.classList.add('hidden');
    }
});
