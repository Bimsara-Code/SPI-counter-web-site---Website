const dragArea = document.querySelector('.drag-area');
const dragText = document.querySelector('.header');

let button = document.querySelector('.button');
let input = document.querySelector('.click-bt');

let file;
let fileURL;

button.onclick = () =>{
    input.click();
};

input.addEventListener('change', function() {
    file = this.files[0];
    dragArea.classList.add('active');
    displayFile();
});

dragArea.addEventListener('dragover', (event)=>{
    event.preventDefault();
    dragText.textContent = 'Release to Upload';
    dragArea.classList.add('active');
    // console.log('File is inside the drag area');
});

dragArea.addEventListener('dragleave', ()=>{
    dragText.textContent = 'Drag & Drop';
    dragArea.classList.remove('active');
});

dragArea.addEventListener('drop', (event)=> {
    event.preventDefault();
    document.querySelector('.spi-count-text').textContent = ""
    file = event.dataTransfer.files[0];
    displayFile();
});


// --------------------------- functions---------------------------------

function displayFile(){
    let fileType = file.type;

    let valid_extentions = ['image/jpeg', 'image/jpg', 'image/png'];

    if(valid_extentions.includes(fileType)){
        let fileReader = new FileReader();

        fileReader.onload = () => {
            fileURL = fileReader.result;
            let imgTab = `<img src="${fileURL}" alt="uploaded image">`;
            dragArea.innerHTML = imgTab;
        }
        fileReader.readAsDataURL(file);
    }
    else{
        alert('This file is not an Image');
        dragArea.classList.remove('active');
    }
}

function send_image() {
    if (fileURL) {
        let formData = new FormData();
        formData.append('image', file);

        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            return response.text(); // Convert response to text
        })
        .then(data => {
            document.querySelector('.spi-count-text').textContent = data; // Update the spi-text element with the response
        })
        .catch(error => {
            alert('Error');
        });
    } else {
        alert('Upload an Image !!!');
    }
}

