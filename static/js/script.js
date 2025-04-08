// For index.html (upload form)
if (document.getElementById('uploadForm')) {
    document.getElementById('uploadForm').addEventListener('submit', function(e) {
        e.preventDefault();
        
        const formData = new FormData(this);
        
        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            document.getElementById('message').textContent = data.message || data.error;
        })
        .catch(error => {
            document.getElementById('message').textContent = 'Error uploading image';
            console.error('Error:', error);
        });
    });
}

// For webcam.html (popup and attendance)
if (document.getElementById('videoFeed')) {
    const videoFeed = document.getElementById('videoFeed');
    const popup = document.getElementById('attendancePopup');
    const popupMessage = document.getElementById('popupMessage');
    const closePopup = document.getElementById('closePopup');
    let lastRecognized = null; // Track last recognized name to avoid repeated popups

    // Function to show popup
    function showPopup(name) {
        if (name !== lastRecognized) {
            popupMessage.textContent = `${name}, your attendance is marked. You can go now!`;
            popup.classList.remove('hidden');
            setTimeout(() => {
                if (!popup.classList.contains('hidden')) {
                    popup.classList.add('hidden');
                }
            }, 5000); // Auto-hide after 5 seconds
            lastRecognized = name;
        }
    }

    // Close popup manually
    closePopup.addEventListener('click', function() {
        popup.classList.add('hidden');
    });

    // Fetch video feed with headers
    let xhr = new XMLHttpRequest();
    xhr.open('GET', '/video_feed', true);
    xhr.onreadystatechange = function() {
        if (xhr.readyState >= 3) { // Receiving data
            const headers = xhr.getAllResponseHeaders();
            const recognizedNamesHeader = xhr.getResponseHeader('X-Recognized-Names');
            if (recognizedNamesHeader) {
                const names = recognizedNamesHeader.split(',').filter(name => name.trim());
                names.forEach(name => {
                    if (name && name !== 'Unknown') {
                        showPopup(name);
                    }
                });
            }
        }
    };
    xhr.send();

    // Ensure video feed updates
    videoFeed.src = '/video_feed?' + new Date().getTime(); // Prevent caching
}