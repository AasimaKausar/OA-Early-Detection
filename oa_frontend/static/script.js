
const xrayInput = document.getElementById("xray");
const imagePreview = document.getElementById("imagePreview");
const fileMessage = document.getElementById("fileMessage");

const uploadedImage = document.getElementById("uploadedImage");
const processingStatus = document.getElementById("processingStatus");
const klGrade = document.getElementById("klGrade");
const processButton = document.getElementById("processButton");

const uploadForm = xrayInput.closest("form");

let previewUrl = null;


// Display selected X-ray image
xrayInput.addEventListener("change", function () {

    const file = xrayInput.files[0];

    if (!file) {
        imagePreview.src = "";
        imagePreview.style.display = "none";
        return;
    }

    const allowedTypes = ["image/jpeg", "image/png"];

    if (!allowedTypes.includes(file.type)) {

        alert("Please select a JPG, JPEG, or PNG image.");

        xrayInput.value = "";
        imagePreview.src = "";
        imagePreview.style.display = "none";

        fileMessage.textContent = "No image selected yet.";

        return;
    }

    // Release previous preview
    if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
    }

    // Create new preview
    previewUrl = URL.createObjectURL(file);

    imagePreview.src = previewUrl;
    imagePreview.style.display = "block";

    fileMessage.textContent = "Image selected successfully.";

    uploadedImage.textContent = file.name;

    processingStatus.textContent = "Image ready for processing";

    klGrade.textContent = "Not available";

    processButton.disabled = false;
    processButton.textContent = "Analyse Image";

});


// Submit image to Flask for AI processing
uploadForm.addEventListener("submit", function (event) {

    if (!xrayInput.files.length) {

        event.preventDefault();

        alert("Please select a knee X-ray image first.");

        return;
    }

    processButton.disabled = true;

    processButton.textContent = "Processing...";

    processingStatus.textContent =
        "Processing image with the AI model...";

});


// Release temporary browser resources
window.addEventListener("beforeunload", function () {

    if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
    }

});


// ==========================================
// PATIENT REGISTRATION FORM
// ==========================================


function showPatientForm() {

    const patientForm = document.getElementById("patientForm");

    const registerButton = document.getElementById(
        "registerPatientButton"
    );

    const fields = patientForm.querySelectorAll("input, select");

    if (patientForm.style.display === "none") {

        patientForm.style.display = "block";

        fields.forEach(field => {
            field.disabled = false;
        });

        registerButton.innerHTML = "✕ Close Registration";

    } else {

        patientForm.style.display = "none";

        fields.forEach(field => {
            field.disabled = true;
        });

        registerButton.innerHTML = "＋ Register Patient";

    }

}


