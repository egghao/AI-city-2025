import os
import time
import argparse
import cv2
import json
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from pathlib import Path

from utils import get_model, preprocess_image, postprocess_result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', type=str, default='/data/Fisheye1K', help='Path to image folder')
    parser.add_argument('--model_path', type=str, default='yolov11.pt', help='Path to the model')
    parser.add_argument('--max_fps', type=float, default=25.0, help='Maximum FPS for evaluation')
    parser.add_argument('--output_json', type=str, default='results/output.json', help='Output JSON file for predictions')
    parser.add_argument('--framework', type=str, default='torch', choices=['torch', 'tensorrt'], help='Inference framework to use')
    parser.add_argument('--visualize', action='store_true', help='Save visualization of detections')
    parser.add_argument('--vis_dir', type=str, default='results/visualizations', help='Directory to save visualizations')
    args = parser.parse_args()

    # Create output directories if they don't exist
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    if args.visualize:
        os.makedirs(args.vis_dir, exist_ok=True)

    image_folder = args.image_folder
    model_path = args.model_path
    framework = args.framework

    model = get_model(model_path, framework=framework)

    image_files = sorted([
        os.path.join(image_folder, f)
        for f in os.listdir(image_folder)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    print(f"Found {len(image_files)} images.")

    all_predictions = []
    print('Prediction started')
    total_time = 0
    start_time = time.time()
    
    for image_path in tqdm(image_files, desc="Processing images", unit="img"):
        img = cv2.imread(image_path)
        if img is None:
            continue
            
        t0 = time.time()
        img_rgb = preprocess_image(img)
        results = model(img_rgb, verbose=False)
        predictions = postprocess_result(results, image_path)
        all_predictions.extend(predictions)
        
        if args.visualize:
            # Draw predictions on the image
            for pred in predictions:
                bbox = pred['bbox']
                score = pred['score']
                category_id = pred['category_id']
                
                # Convert bbox format from [x, y, w, h] to [x1, y1, x2, y2]
                x1, y1 = int(bbox[0]), int(bbox[1])
                x2, y2 = int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])
                
                # Draw bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add caption with class and confidence
                caption = f"Class {category_id}: {score:.2f}"
                cv2.putText(img, caption, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Save visualization
            vis_path = os.path.join(args.vis_dir, Path(image_path).name)
            cv2.imwrite(vis_path, img)
        
        t3 = time.time()
        total_time += (t3-t0)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Processed {len(image_files)} images in {elapsed_time:.2f} seconds.")
    print(f"Avg Processing Time: {total_time/len(image_files)*1000:.2f} ms")

    # Save predictions to JSON
    with open(args.output_json, 'w') as f:
        json.dump(all_predictions, f, indent=2)

    fps = len(image_files) / total_time
    normfps = min(fps, args.max_fps)/args.max_fps

    print(f"\n--- Evaluation Complete ---")
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"FPS: {fps:.2f}")
    print(f"Normalized FPS: {normfps:.4f}")

if __name__ == "__main__":
    main()