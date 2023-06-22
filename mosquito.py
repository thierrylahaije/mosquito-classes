import tensorflow as tf
import csv
from functions import load_data, define_train_model, predict, accuracy
import pandas as pd

def main():
    # Step 1: Load and prepare training data
    train_csv_path = 'train_data/train.csv'
    train_image_dir = 'train_data/train_images/'
    num_train_images = 1000

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
