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