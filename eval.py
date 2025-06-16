import argparse
import os
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import yolov5
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import read_image
from sahi.utils.file import save_json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate object detection model')
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights')
    parser.add_argument('--data', type=str, required=True, help='Path to dataset config')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--use-sahi', action='store_true', help='Use SAHI for inference')
    parser.add_argument('--slice-size', type=int, default=640, help='Slice size for SAHI')
    parser.add_argument('--overlap-ratio', type=float, default=0.2, help='Overlap ratio for SAHI')
    return parser.parse_args()

def calculate_f1_score(precision, recall):
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)

def evaluate_with_sahi(model, image_path, conf_thres, slice_size, overlap_ratio):
    image = read_image(image_path)
    result = get_sliced_prediction(
        image=image,
        detection_model=model,
        slice_height=slice_size,
        slice_width=slice_size,
        overlap_height_ratio=overlap_ratio,
        overlap_width_ratio=overlap_ratio,
        conf_thres=conf_thres
    )
    return result

def main():
    args = parse_args()
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = yolov5.load(args.weights)
    model.conf = args.conf_thres
    model.iou = args.iou_thres
    model.to(device)
    
    # Load dataset
    coco_gt = COCO(args.data)
    
    # Initialize results list for COCO format
    results = []
    
    # Process images
    for img_id in tqdm(coco_gt.getImgIds()):
        img_info = coco_gt.loadImgs(img_id)[0]
        image_path = img_info['file_name']
        
        if args.use_sahi:
            # Use SAHI for inference
            result = evaluate_with_sahi(
                model, 
                image_path, 
                args.conf_thres,
                args.slice_size,
                args.overlap_ratio
            )
            predictions = result.object_prediction_list
        else:
            # Regular inference
            predictions = model(image_path)
            predictions = predictions.pred[0].cpu().numpy()
        
        # Convert predictions to COCO format
        for pred in predictions:
            if args.use_sahi:
                results.append({
                    'image_id': img_id,
                    'category_id': pred.category_id,
                    'bbox': pred.bbox,
                    'score': pred.score
                })
            else:
                x1, y1, x2, y2, score, class_id = pred
                results.append({
                    'image_id': img_id,
                    'category_id': int(class_id),
                    'bbox': [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                    'score': float(score)
                })
    
    # Save results
    save_json(results, 'predictions.json')
    
    # Initialize COCO detections
    coco_dt = coco_gt.loadRes('predictions.json')
    
    # Calculate metrics
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    # Calculate F1 score
    precision = coco_eval.stats[0]  # AP at IoU=0.50:0.95
    recall = coco_eval.stats[8]     # AR at IoU=0.50:0.95
    f1_score = calculate_f1_score(precision, recall)
    
    # Print results
    print(f"\nF1 Score: {f1_score:.4f}")
    print(f"mAP-small: {coco_eval.stats[3]:.4f}")  # AP for small objects
    print(f"mAP-medium: {coco_eval.stats[4]:.4f}")  # AP for medium objects
    print(f"mAP-large: {coco_eval.stats[5]:.4f}")   # AP for large objects

if __name__ == '__main__':
    main()
