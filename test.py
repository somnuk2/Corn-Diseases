import time
import os
import requests
from requests.exceptions import ConnectTimeout, RequestException

def make_request_with_retry(url, files, retries=3, delay=3, timeout=60):
    """
    Makes a POST request to the given URL with retries on failure.
    
    Parameters:
    - url: The URL to send the POST request to.
    - files: The files to send in the request.
    - retries: The number of retries on failure.
    - delay: The delay in seconds between each retry.
    - timeout: The timeout for each request in seconds.
    
    Returns:
    - The response object if the request is successful.
    - None if the request fails after retries.
    """
    for attempt in range(retries):
        try:
            response = requests.post(url, files=files, timeout=timeout)
            response.raise_for_status()  # Raises an error for bad HTTP status codes
            return response
        except ConnectTimeout:
            print(f"Attempt {attempt + 1} failed: Connection timed out. Retrying in {delay} seconds...")
        except RequestException as e:
            print(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay} seconds...")
        
        # Wait before retrying
        time.sleep(delay)
    
    # If all retries fail
    print(f"Request failed after {retries} attempts.")
    return None

# Directory containing the images
# image_directory = "D:/CNN-With/data/SR"  # Update this path as needed
image_directory = "D:/CNN-With/data/CDM"

# Loop through all the files in the directory
for filename in os.listdir(image_directory):
    if filename.endswith(".jpeg") or filename.endswith(".jpg"):  # Only process image files
        image_path = os.path.join(image_directory, filename)

        with open(image_path, "rb") as img_file:
            files = {"file": img_file}
            
            # Send the image to the FastAPI server with retry logic
            response = make_request_with_retry("https://corn-diseases.onrender.com/predict_image", files)

            if response:
                # Print the prediction result
                print(f"Prediction for {filename}: {response.json()}")
            else:
                print(f"Failed to get prediction for {filename}.")
                break
