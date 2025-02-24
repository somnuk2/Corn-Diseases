import os
import requests

# Directory containing the images
# image_directory = "D:/CNN-With/data/CDM"
# image_directory = "D:/CNN-With/data/HT"
# image_directory = "D:/CNN-With/data/MDMV"
# image_directory = "D:/CNN-With/data/NCLB"
# image_directory = "D:/CNN-With/data/SCLB"
# image_directory = "D:/CNN-With/data/SCMV"
image_directory = "D:/CNN-With/data/SR"

# Loop through all the files in the directory
for filename in os.listdir(image_directory):
    if filename.endswith(".jpeg") or filename.endswith(".jpg"):  # Only process image files
        image_path = os.path.join(image_directory, filename)

        # Open the image file
        with open(image_path, "rb") as img_file:
            files = {"file": img_file}
            
            # Send the image to the FastAPI server
            response = requests.post("http://127.0.0.1:8000/predict_image", files=files)

            # Print the prediction result
            print(f"Prediction for {filename}: {response.json()}")
