from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse  # Import FileResponse
from fastapi.middleware.cors import CORSMiddleware  # Import CORSMiddleware
import io
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import uvicorn
import pandas as pd
import os

# Model parameters
img_height = 224
img_width = 224
# Class names for the images
class_names = ['CDM', 'HT', 'MDMV', 'NCLB', 'SCLB', 'SCMV', 'SR']
print("Class Names:", class_names)

model = None  # Set model to None initially

app = FastAPI()

# Add CORSMiddleware to allow requests from specific origins
origins = [
    "http://localhost:8080",  # Allow your local frontend
    # You can also add other origins here if needed, like:
    # "https://your-frontend-domain.com", 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows all origins listed in the 'origins' list
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

@app.on_event("startup")
async def load_model():
    global model
    # Load the model
    model = tf.keras.models.load_model("model.h5")
    # Print the model summary
    model.summary()
# Serve static files (including favicon.ico)
app.mount("/static", StaticFiles(directory="static"), name="static")
    
# Define the root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Image Prediction API!"}

# Define the predict_image endpoint
@app.post("/predict_image")
async def predict_image(file: UploadFile = File(...)):  # Ensure the file is required
    try:
        global model
        # Load model if it was not loaded
        if model is None:
            model = tf.keras.models.load_model("model.h5")  
            model.summary()
        else:    
            print("Model loaded successfully!")  # Check in logs
        # Read the uploaded image file
        contents = await file.read()
        # Load the image and resize it
        img = image.load_img(io.BytesIO(contents), target_size=(img_height, img_width))
        # Convert the image to a numpy array
        img_array = image.img_to_array(img)
        # Add batch dimension (since the model expects batches)
        img_array = tf.expand_dims(img_array, axis=0)
        #  Get the prediction (raw output from the model)
        
        prediction = model.predict(img_array)
        # If the model has logits as output, apply softmax
        if 'from_logits' in model.loss.get_config() and model.loss.get_config()['from_logits']:
            prediction = tf.nn.softmax(prediction).numpy()
        # Create a DataFrame to show the prediction probabilities
        pred_df = pd.DataFrame(prediction, columns=class_names)
        # Display the prediction results
        print("Prediction results for the single image:")
        print(pred_df)
        # In the absence of ground truth, we'll omit confusion matrix and metrics
        predicted_label = pred_df.idxmax(axis=1).iloc[0]  # Get the predicted label
        confidence = pred_df.max(axis=1).iloc[0]  # Get the confidence of the prediction
        print(f"Predicted Label: {predicted_label}")
        print(f"Prediction Confidence: {confidence:.4f}")

        return {"prediction": predicted_label}
    
    except Exception as e:
        return {"error": str(e)}
    
# You can also provide a route for favicon.ico if you want to customize it
@app.get("/favicon.ico")
async def favicon():
    return FileResponse("static/favicon.ico")
    
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0",  port=int(os.getenv("PORT", 8000)), reload=True)