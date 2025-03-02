# Use the official Python image from Docker Hub
FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt /app/
COPY model.h5 /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application files
COPY . /app/

# Expose the port that FastAPI will run on
EXPOSE 8000

# Start the app with Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
