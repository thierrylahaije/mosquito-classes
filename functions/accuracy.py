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

def calculate_accuracy(predictions_df, test_csv, iou_threshold):
    total_images = len(predictions_df)
    accurate_predictions = 0

    for _, row in predictions_df.iterrows():
        image_name = row['image_name']
        predicted_bbox_str = row['predicted_bbox']
        predicted_bbox_list = predicted_bbox_str.strip('[]').split()
        predicted_bbox = [float(coordinate) for coordinate in predicted_bbox_list]

        true_row = test_csv[test_csv['img_fName'] == image_name]
        true_bbox = [
            true_row['bbx_xtl'].item(),
            true_row['bbx_ytl'].item(),
            true_row['bbx_xbr'].item(),
            true_row['bbx_ybr'].item()
        ]

        iou = calculate_iou(predicted_bbox, true_bbox)

        if iou >= iou_threshold:
            accurate_predictions += 1

    accuracy = (accurate_predictions / total_images) * 100
    return accuracy