import numpy as np
import csv

def predict_bboxes(model, test_csv, test_images):
    predictions = model.predict(test_images)
    output_file = 'predictions.csv'

    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['image_name', 'predicted_bbox']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(len(predictions)):
            row = test_csv.loc[i]
            image_name = row['img_fName']

            width_ratio = row['img_w'] / 350
            height_ratio = row['img_h'] / 350
            bbox = predictions[i] * np.array([width_ratio, height_ratio, width_ratio, height_ratio])

            writer.writerow({'image_name': image_name, 'predicted_bbox': bbox})

    print('Predictions have been written to', output_file)