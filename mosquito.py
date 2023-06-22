import tensorflow as tf
import csv
from functions import load_data, define_train_model, predict, accuracy
import pandas as pd
from google.cloud import storage
import os

def main():

    # Set the path to your credentials JSON file
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'mosquito-detection-92eb945af9f6.json'

    # Step 1: Load and prepare training data
    train_csv_path = 'train_data/train.csv'
    train_image_dir = 'train_data/train_images/'
    num_train_images = 1000

    # Create a client to interact with the Cloud Storage API
    storage_client = storage.Client()

    # Get a reference to the bucket
    bucket_name = 'mosquito-bucket'
    bucket = storage_client.bucket(bucket_name)

    # List all the blobs (files) in the bucket with the prefix 'train_images/'
    blobs = bucket.list_blobs(prefix='train_images/')

    # Create the directory structure within the local 'train_data' directory
    os.makedirs(train_image_dir, exist_ok=True)

    # Download the images from the bucket to the local directory
    for i, blob in enumerate(blobs):
        if i >= num_train_images:
            break
        
        # Get the filename of the blob
        filename = blob.name
        
        # Extract the relative path within the 'train_images/' folder
        relative_path = filename.replace('train_images/', '', 1)
        
        # Create the directories in the local 'train_data' directory
        destination_dir = os.path.join(train_image_dir, os.path.dirname(relative_path))
        os.makedirs(destination_dir, exist_ok=True)
        
        # Download the blob to the local directory
        destination_path = os.path.join(train_image_dir, relative_path)
        blob.download_to_filename(destination_path)

    # Load the training data using the updated image directory
    train_images, train_bboxes = load_data.load_train_data(train_csv_path, train_image_dir, num_train_images)

    print('Train data loaded.')

    # Step 2: Define the model architecture
    model = define_train_model.define_model()
    print('Model architecture defined.')

    # Step 3: Train the model
    batch_size = 8
    epochs = 5
    define_train_model.train_model(model, train_images, train_bboxes, batch_size, epochs)
    print('Model training completed.')

    # Step 4: Load and prepare test data
    test_csv_path = 'train_data/train.csv'
    test_image_dir = 'train_data/train_images/'
    num_test_images = 100

    test_csv, test_images = load_data.load_test_data(test_csv_path, test_image_dir, num_test_images)
    print('Test data loaded.')

    # Step 5: Make predictions
    predict.predict_bboxes(model, test_csv, test_images)
    print('Predictions completed.')

    # Step 6: Calculate accuracy
    predictions_df = pd.read_csv('predictions.csv')
    iou_threshold = 0.1
    acc_final = accuracy.calculate_accuracy(predictions_df, test_csv, iou_threshold)
    print('Accuracy:', acc_final)

if __name__ == '__main__':
    main()
