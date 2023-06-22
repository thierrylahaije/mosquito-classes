# TensorFlow Docker Image
FROM tensorflow/tensorflow:2.12.0

# Copy your code into the container
COPY . /app

# Set the working directory
WORKDIR /app

# Install required libraries
RUN pip install numpy csvkit tensorflow pandas opencv-python google-cloud-storage

# Run the mosquito python script
CMD ["python", "mosquito.py"]



