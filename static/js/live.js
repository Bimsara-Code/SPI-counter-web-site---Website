const video = document.getElementById('videoElement');
const canvas = document.getElementById('canvas');
const photo = document.getElementById('photo');
const captureBtn = document.getElementById('capture-btn');

navigator.mediaDevices.getUserMedia({ video: true })
  .then(stream => {
    video.srcObject = stream;
  })
  .catch(error => {
    console.error('Error accessing webcam:', error);
  });

// capture a photo from the video stream
function capture() {
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
  const imageData = canvas.toDataURL();

  photo.src = canvas.toDataURL('image/png');
  photo.style.display = 'block';
  
  canvas.toBlob(blob => {
    const formData = new FormData();
    formData.append('image', blob, 'capture.jpg');

    fetch('/upload', {
      method: 'POST',
      body: formData
    })
    .then(response => {
      return response.text();
    })
    .then(data => {
      console.log(data);
      document.querySelector('.spi-count-text').textContent = data; // Update the spi-text element with the response
    })
    .catch(error => {
      console.error('Error uploading image:', error);
    });
  }, 'image/jpeg', 0.9);
};
