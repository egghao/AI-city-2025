from ultralytics import YOLO
import os
import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval_modified import COCOeval
import json

def f1_score(predictions_path, ground_truths_path):
    coco_gt = COCO(ground_truths_path)

    gt_image_ids = coco_gt.getImgIds()

    with open(predictions_path, 'r') as f:
        detection_data = json.load(f)
    filtered_detection_data = [
        item for item in detection_data if item['image_id'] in gt_image_ids]
    with open('./temp.json', 'w') as f:
        json.dump(filtered_detection_data, f)
    coco_dt = coco_gt.loadRes('./temp.json')
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    # Assuming the F1 score is at index 20 in the stats array
    return coco_eval.stats[20]  # Return the F1 score from the evaluation stats
    # return 0.85  # Simulated constant value for demo purposes

def get_model(model_path, framework='torch'):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} does not exist.")
    
    if framework == 'torch':
        # Use PyTorch model directly
        model = YOLO(model_path)
        return model
    elif framework == 'tensorrt':
        # Export and use TensorRT model
        model = YOLO(model_path)
        model.export(format="engine")  # creates '.engine' file
        engine_path = model_path.replace(".pt", ".engine")
        # Load the exported TensorRT model
        trt_model = YOLO(engine_path)
        return trt_model
    else:
        raise ValueError(f"Unsupported framework: {framework}. Choose from ['torch', 'tensorrt']")

def preprocess_image(img):
    if img is None:
        raise ValueError("Input image is None.")
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def postprocess_result(results, image_path):
    if not results or len(results) == 0:
        return []
    
    # Get image name from path
    img_name = os.path.basename(image_path)
    img_name = img_name.split('.')[0]
    
    # Get image ID using the provided format
    sceneList = ['M', 'A', 'E', 'N']
    cameraId = int(img_name.split('_')[0].split('camera')[1])
    sceneId = sceneList.index(img_name.split('_')[1])
    frameId = int(img_name.split('_')[2])
    imageId = int(str(cameraId) + str(sceneId) + str(frameId))
    
    # Get boxes, scores, and classes
    boxes = results[0].boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2 format
    scores = results[0].boxes.conf.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()
    
    # Convert to required format
    detections = []
    for box, score, cls in zip(boxes, scores, classes):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        
        detection = {
            "image_id": imageId,
            "category_id": int(cls),  # Convert to int to ensure JSON serializable
            "bbox": [float(x1), float(y1), float(width), float(height)],  # Convert to float for JSON
            "score": float(score)
        }
        detections.append(detection)
    
    return detections

def changeId(id):
    sceneList = ['M', 'A', 'E', 'N']
    cameraId = int(id.split('_')[0].split('camera')[1])
    sceneId = sceneList.index(id.split('_')[1])
    frameId = int(id.split('_')[2])
    imageId = int(str(cameraId)+str(sceneId)+str(frameId))
    return imageId