document.addEventListener('DOMContentLoaded', () => {
    const personDropzone = document.getElementById('person-dropzone');
    const garmentDropzone = document.getElementById('garment-dropzone');
    const personInput = document.getElementById('person-input');
    const garmentInput = document.getElementById('garment-input');
    const personPreview = document.getElementById('person-preview');
    const garmentPreview = document.getElementById('garment-preview');
    const tryonBtn = document.getElementById('tryon-btn');
    
    const resultSection = document.getElementById('result-section');
    const resultImg = document.getElementById('result-img');
    const loader = document.getElementById('loader');
    const errorMessage = document.getElementById('error-message');

    let personFile = null;
    let garmentFile = null;

    function setupDropzone(dropzone, input, preview, fileCallback) {
        dropzone.addEventListener('click', () => input.click());

        dropzone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropzone.classList.add('dragover');
        });

        dropzone.addEventListener('dragleave', () => {
            dropzone.classList.remove('dragover');
        });

        dropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropzone.classList.remove('dragover');
            if (e.dataTransfer.files.length > 0) {
                handleFile(e.dataTransfer.files[0], preview, fileCallback);
            }
        });

        input.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFile(e.target.files[0], preview, fileCallback);
            }
        });
    }

    function handleFile(file, previewElement, fileCallback) {
        if (!file.type.startsWith('image/')) return;
        
        fileCallback(file);
        
        const reader = new FileReader();
        reader.onload = (e) => {
            previewElement.src = e.target.result;
            previewElement.hidden = false;
        };
        reader.readAsDataURL(file);

        checkReadyStatus();
    }

    function checkReadyStatus() {
        if (personFile && garmentFile) {
            tryonBtn.disabled = false;
        }
    }

    setupDropzone(personDropzone, personInput, personPreview, (f) => personFile = f);
    setupDropzone(garmentDropzone, garmentInput, garmentPreview, (f) => garmentFile = f);

    tryonBtn.addEventListener('click', async () => {
        if (!personFile || !garmentFile) return;

        // UI Reset for loading
        tryonBtn.disabled = true;
        resultSection.hidden = false;
        resultImg.hidden = true;
        errorMessage.hidden = true;
        loader.hidden = false;

        const formData = new FormData();
        formData.append('person_image', personFile);
        formData.append('garment_image', garmentFile);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Failed to process image');
            }

            const blob = await response.blob();
            const imageUrl = URL.createObjectURL(blob);
            
            resultImg.src = imageUrl;
            loader.hidden = true;
            resultImg.hidden = false;

            // Optional: scroll to result
            resultSection.scrollIntoView({ behavior: 'smooth' });
            
        } catch (err) {
            loader.hidden = true;
            errorMessage.textContent = err.message || 'An error occurred during inference.';
            errorMessage.hidden = false;
        } finally {
            tryonBtn.disabled = false;
        }
    });
});
