document.addEventListener("DOMContentLoaded", () => {
  const detectButton = document.getElementById("detectButton");
  const imageInput = document.getElementById("imageUpload");

  detectButton.addEventListener("click", async () => {
    const file = imageInput.files[0];

    if (!file) {
      alert("Please select an image to upload.");
      return;
    }

    const formData = new FormData();
    formData.append("image", file);

    try {
      const response = await fetch("/upload", {
        method: "POST",
        body: formData
      });

      const result = await response.json();

      if (response.ok) {
        alert(`Image uploaded! Prediction: ${result.prediction || 'Success'}`);
      } else {
        alert(`Upload failed: ${result.error}`);
      }

    } catch (error) {
      console.error("Error:", error);
      alert("Something went wrong while uploading.");
    }
  });
});
