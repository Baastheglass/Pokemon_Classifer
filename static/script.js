document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const uploadButton = document.getElementById('uploadButton');
    const previewImage = document.getElementById('previewImage');
    const resultsContainer = document.querySelector('.results-container');
    const loadingElement = document.querySelector('.loading');
    
    // Handle drag and drop events
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => {
            dropZone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => {
            dropZone.classList.remove('dragover');
        });
    });

    // Handle file drop
    dropZone.addEventListener('drop', (e) => {
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) {
            handleFile(file);
        }
    });

    // Handle click to upload
    dropZone.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    function handleFile(file) {
        // Preview the image
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            previewImage.style.width = 'auto';
            previewImage.style.height = '200px';
        };
        reader.readAsDataURL(file);
        
        // Enable upload button
        uploadButton.disabled = false;
        uploadButton.onclick = () => uploadImage(file);
    }

    async function uploadImage(file) {
        // Show loading state
        loadingElement.style.display = 'block';
        resultsContainer.style.display = 'none';
        uploadButton.disabled = true;

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            
            if (response.ok) {
                displayResults(data);
            } else {
                throw new Error(data.error || 'Failed to process image');
            }
        } catch (error) {
            alert(error.message);
        } finally {
            loadingElement.style.display = 'none';
            uploadButton.disabled = false;
        }
    }

    function displayResults(data) {
        const topPrediction = data.all_results[0];
        const otherPredictions = data.all_results.slice(1);

        // Display top prediction
        document.querySelector('.pokemon-name').textContent = topPrediction.pokemon;
        document.querySelector('.confidence').textContent = `Confidence: ${topPrediction.confidence}`;

        // Display other predictions
        const predictionsList = document.querySelector('.predictions-list');
        predictionsList.innerHTML = otherPredictions
            .map(pred => `
                <li>
                    <span>${pred.pokemon}</span>
                    <span>${pred.confidence}</span>
                </li>
            `)
            .join('');

        // Show results
        resultsContainer.style.display = 'block';
    }
}); 