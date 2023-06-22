import pandas as pd
import numpy as np
import cv2

def load_train_data(csv_path, image_dir, num_images):
    train_csv_file = pd.read_csv(csv_path)
    train_csv = train_csv_file[:num_images]

    train_images = []
    train_bboxes = []

    for i in range(num_images):
        row = train_csv.loc[i]
        image_path = image_dir + row['img_fName']
        image = cv2.imread(image_path)
        image = cv2.resize(image, (350, 350))
        train_images.append(image)

        width_ratio = row['img_w'] / 350
        height_ratio = row['img_h'] / 350
        bbox = [
            row['bbx_xtl'] / width_ratio,
            row['bbx_ytl'] / height_ratio,
            row['bbx_xbr'] / width_ratio,
            row['bbx_ybr'] / height_ratio
        ]
        train_bboxes.append(bbox)

    train_images = np.array(train_images)
    train_bboxes = np.array(train_bboxes)

    return train_images, train_bboxes

def load_test_data(csv_path, image_dir, num_images):
    test_csv_file = pd.read_csv(csv_path)
    test_csv = test_csv_file[:num_images]
    
    test_images = []
    for i in range(num_images):
        row = test_csv.loc[i]
        image_path = image_dir + row['img_fName']
        image = cv2.imread(image_path)
        image = cv2.resize(image, (350, 350))
        test_images.append(image)

    test_images = np.array(test_images)

    return test_csv, test_images