import cv2
import pandas as pd

# Importeer de voorspellingsresultaten uit het CSV-bestand
predictions_df = pd.read_csv('predictions.csv')

# Lees de afbeeldingen in vanuit de CSV
for i, row in predictions_df.iterrows():
    image_name = row['image_name']
    bbox = row['predicted_bbox']
    
    # Haal de originele bestandsnaam zonder extensie op
    file_name = image_name.split('.')[0]
    
    # Construeer de nieuwe bestandsnaam
    new_file_name = file_name + '_kopie'
    
    # Lees de afbeelding in
    image_path = 'train_data/train_images/' + image_name
    image = cv2.imread(image_path)
    
    # Process the bounding box
    bbox_values = list(map(float, bbox.strip('[]').split()))
    x1, y1, x2, y2 = bbox_values
    
    # Draw a rectangle on the image
    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    
    # Save the processed image with the new filename
    output_path = 'output_images/' + new_file_name + '.jpg'
    cv2.imwrite(output_path, image)
    
    print('Image saved:', output_path)
