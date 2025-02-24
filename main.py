from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import uvicorn
import pandas as pd

# Model parameters
img_height = 224
img_width = 224
# Class names for the images
class_names = ['CDM', 'HT', 'MDMV', 'NCLB', 'SCLB', 'SCMV', 'SR']
print("Class Names:", class_names)
# Load the model
model = tf.keras.models.load_model("Trained_Model_Cnn/model.h5")
# Print the model summary
model.summary()

app = FastAPI()

@app.post("/predict_image")
async def predict_image(file: UploadFile = File(...)):  # Ensure the file is required
    try:
        # Read the uploaded image file
        contents = await file.read()
        # img = Image.open(io.BytesIO(contents))
        # # Load the image and resize it
        img = image.load_img(io.BytesIO(contents), target_size=(img_height, img_width))
        # Convert the image to a numpy array
        img_array = image.img_to_array(img)
        # Add batch dimension (since the model expects batches)
        img_array = tf.expand_dims(img_array, axis=0)
        
        # # Ensure the image is in RGB mode (important if the model expects RGB)
        # if img.mode != 'RGB':
        #     img = img.convert('RGB')

        # # Preprocess the image (resize, normalize, etc.)
        # img = img.resize((224, 224))  # Resize to model input size, if necessary
        
        # # # Convert the image to a numpy array and normalize pixel values to [0, 1]
        # # img = np.array(img) / 255.0  # Normalize the pixel values

        # # If the image has only two dimensions (grayscale), convert it to RGB by repeating the channels
        # if img.ndim == 2:  # Grayscale image
        #     img = np.stack([img] * 3, axis=-1)

        # # Check the image's shape and ensure it has 3 color channels
        # if img.shape[-1] != 3:
        #     return {"error": "Image does not have 3 color channels (RGB)."}

        # # Reshape the image for the model (adding batch dimension)
        # img = img.reshape(1, 224, 224, 3)  # Assuming the model expects 224x224 RGB images

        # Get the prediction (raw output from the model)
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

if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)