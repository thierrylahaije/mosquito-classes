import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
import csv

# Stap 1: Data voorbereiden
train_csv_file = pd.read_csv('train_data/train.csv')
train_csv = train_csv_file[:6000]

# Lees de afbeeldingen en begrenzingskaders in voor de eerste 100 foto's
train_images = []
train_bboxes = []

for i in range(2000):
    row = train_csv.loc[i]
    image_path = 'train_data/train_images/' + row['img_fName']
    image = cv2.imread(image_path)
    image = cv2.resize(image, (350, 350))  # Verklein naar gewenst formaat
    train_images.append(image)
    
    # Pas de begrenzingskaders aan op basis van de verhouding van de resizingsfactor
    width_ratio = row['img_w'] / 350
    height_ratio = row['img_h'] / 350
    bbox = [row['bbx_xtl'] / width_ratio, row['bbx_ytl'] / height_ratio,
            row['bbx_xbr'] / width_ratio, row['bbx_ybr'] / height_ratio]
    train_bboxes.append(bbox)

print('Afbeeldingen Geladen')

train_images = np.array(train_images)
train_bboxes = np.array(train_bboxes)

# Stap 2: Modelarchitectuur definiëren
model = keras.Sequential([
    keras.layers.Input(shape=(350, 350, 3)),
    keras.applications.EfficientNetB0(input_shape=(350, 350, 3), include_top=False),
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(4)
])

# Stap 3: Model trainen
model.compile(optimizer='adam', loss='mse')
model.fit(train_images, train_bboxes, batch_size=16, epochs=20)

# Stap 4: Voorspellingen doen op de testdataset
test_csv = train_csv_file[7001:7203]
test_csv.reset_index(drop=True, inplace=True)


# Lees de afbeeldingen in voor de eerste 100 foto's in de testdataset
test_images = []
for i in range(100):
    row = test_csv.loc[i]
    print(row)
    image_path = 'train_data/train_images/' + row['img_fName']
    print(image_path)
    image = cv2.imread(image_path)
    image = cv2.resize(image, (350, 350))  # Verklein naar gewenst formaat
    test_images.append(image)

    print('Afbeelding gedaan ', image_path)

test_images = np.array(test_images)

predictions = model.predict(test_images)

# Stap 5: Schrijf de voorspellingen naar een CSV-bestand
output_file = 'predictions.csv'

with open(output_file, 'w', newline='') as csvfile:
    fieldnames = ['image_name', 'predicted_bbox']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for i in range(100):
        row = test_csv.loc[i]
        image_name = row['img_fName']

        # Pas de voorspelde begrenzingskaders aan op basis van de originele afmetingen
        width_ratio = row['img_w'] / 350
        height_ratio = row['img_h'] / 350
        bbox = predictions[i] * np.array([width_ratio, height_ratio, width_ratio, height_ratio])

        writer.writerow({'image_name': image_name, 'predicted_bbox': bbox})

print('Voorspellingen zijn geschreven naar', output_file)

# Importeer de voorspellingsresultaten uit het CSV-bestand
predictions_df = pd.read_csv('predictions.csv')

# Calculate IoU (Intersection over Union) between two bounding boxes
def calculate_iou(bbox_pred, bbox_true):
    x1_pred, y1_pred, x2_pred, y2_pred = bbox_pred
    x1_true, y1_true, x2_true, y2_true = bbox_true
    
    # Calculate intersection coordinates
    x1_inter = max(x1_pred, x1_true)
    y1_inter = max(y1_pred, y1_true)
    x2_inter = min(x2_pred, x2_true)
    y2_inter = min(y2_pred, y2_true)
    
    # Calculate intersection area
    inter_area = max(0, x2_inter - x1_inter + 1) * max(0, y2_inter - y1_inter + 1)
    
    # Calculate union area
    bbox_pred_area = (x2_pred - x1_pred + 1) * (y2_pred - y1_pred + 1)
    bbox_true_area = (x2_true - x1_true + 1) * (y2_true - y1_true + 1)
    union_area = bbox_pred_area + bbox_true_area - inter_area
    
    # Calculate IoU
    iou = inter_area / union_area
    return iou

# Calculate accuracy based on IoU threshold
iou_threshold = 0.5
total_images = 100
accurate_predictions = 0

for i, row in predictions_df.iterrows():
    image_name = row['image_name']
    # Read the predicted_bbox value from the CSV as a string
    predicted_bbox_str = row['predicted_bbox']

    # Split the string into individual coordinates
    predicted_bbox_list = predicted_bbox_str.strip('[]').split()

    # Convert each coordinate to a float and store them in a list
    predicted_bbox = [float(coordinate) for coordinate in predicted_bbox_list]
    
    # Find the corresponding true bounding box in the test dataset
    true_row = test_csv[test_csv['img_fName'] == image_name]
    true_bbox = [true_row['bbx_xtl'].item(), true_row['bbx_ytl'].item(),
                 true_row['bbx_xbr'].item(), true_row['bbx_ybr'].item()]
    
    # Calculate IoU between predicted and true bounding boxes
    iou = calculate_iou(predicted_bbox, true_bbox)
    
    print(iou)
    if iou >= iou_threshold:
        accurate_predictions += 1

accuracy = (accurate_predictions / total_images)*100
print('Accuracy:', accuracy)