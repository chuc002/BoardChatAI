document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('uploadForm');
    const uploadBtn = document.getElementById('uploadBtn');
    const uploadProgress = document.getElementById('uploadProgress');
    const uploadMessage = document.getElementById('uploadMessage');
    const progressBar = uploadProgress.querySelector('.progress-bar');

    uploadForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const fileInput = document.getElementById('pdfFile');
        const file = fileInput.files[0];
        
        if (!file) {
            showMessage('Please select a PDF file to upload.', 'danger');
            return;
        }

        // Validate file type
        if (file.type !== 'application/pdf') {
            showMessage('Please select a valid PDF file.', 'danger');
            return;
        }

        // Validate file size (16MB limit)
        const maxSize = 16 * 1024 * 1024; // 16MB in bytes
        if (file.size > maxSize) {
            showMessage('File size exceeds 16MB limit. Please choose a smaller file.', 'danger');
            return;
        }

        // Start upload
        uploadBtn.disabled = true;
        uploadBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Uploading...';
        uploadProgress.style.display = 'block';
        uploadMessage.innerHTML = '';

        // Simulate progress
        let progress = 0;
        const progressInterval = setInterval(() => {
            progress += Math.random() * 20;
            if (progress > 90) {
                progress = 90;
                clearInterval(progressInterval);
            }
            updateProgress(progress);
        }, 200);

        try {
            const formData = new FormData();
            formData.append('file', file);

            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            clearInterval(progressInterval);
            updateProgress(100);

            const data = await response.json();

            if (data.success) {
                showMessage(data.message, 'success');
                fileInput.value = ''; // Clear the file input
                
                // Reload page after successful upload to show new document
                setTimeout(() => {
                    window.location.reload();
                }, 1500);
            } else {
                showMessage(data.message || 'Upload failed. Please try again.', 'danger');
            }

        } catch (error) {
            clearInterval(progressInterval);
            console.error('Upload error:', error);
            showMessage('Network error. Please check your connection and try again.', 'danger');
        } finally {
            // Reset UI
            setTimeout(() => {
                uploadBtn.disabled = false;
                uploadBtn.innerHTML = '<i class="fas fa-upload me-2"></i>Upload';
                uploadProgress.style.display = 'none';
                updateProgress(0);
            }, 2000);
        }
    });

    function updateProgress(percent) {
        progressBar.style.width = percent + '%';
        progressBar.setAttribute('aria-valuenow', percent);
    }

    function showMessage(message, type) {
        uploadMessage.innerHTML = `
            <div class="alert alert-${type} alert-dismissible fade show" role="alert">
                ${message}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>
        `;
    }

    // Drag and drop functionality
    const uploadCard = document.querySelector('.card');
    
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadCard.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        uploadCard.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        uploadCard.addEventListener(eventName, unhighlight, false);
    });

    function highlight(e) {
        uploadCard.classList.add('border-primary');
    }

    function unhighlight(e) {
        uploadCard.classList.remove('border-primary');
    }

    uploadCard.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;

        if (files.length > 0) {
            const fileInput = document.getElementById('pdfFile');
            fileInput.files = files;
            
            // Trigger form submission if it's a PDF
            if (files[0].type === 'application/pdf') {
                showMessage('PDF file detected. Click upload to process.', 'info');
            } else {
                showMessage('Please drop a PDF file.', 'warning');
            }
        }
    }
});
