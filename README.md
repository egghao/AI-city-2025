# Fisheye Object Detection - AI City Challenge 2025

This repository contains the implementation of an object detection system for fisheye camera images, specifically designed for the AI City Challenge 2025.

## Project Structure

```
AI-city-2025/
├── data/
│   ├── Fisheye8K/
│   │   ├── train/
│   │   │   ├── images/
│   │   │   ├── labels/
│   │   │   └── annotations/
│   │   ├── test/
│   │   │   ├── images/
│   │   │   ├── labels/
│   │   │   └── annotations/
│   │   └── train.json
│   ├── Fisheye1K/
│   │   └── (test images)
│   └── Fisheye8k.yml
├── models/
│   ├── yolo11n.pt
│   ├── yolo11s.pt
│   └── yolo11m.pt
├── ifish_augmentation/
├── train.py
├── test.py
├── eval.py
├── utils.py
├── merge_datasets.py
├── visdrone2yolo.py
├── download_visdrone.py
├── cocoeval_modified.py
├── requirements.txt
└── README.md
```

## Object Classes

The model is trained to detect the following classes:
- Bus (0)
- Bike (1)
- Car (2)
- Pedestrain (3)
- Truck (4)

## Setup

1. Clone the repository:
```bash
git clone [repository-url]
cd AI-city-2025
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Linux/Mac
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Preparation

### Downloading VisDrone Dataset

1. Download the VisDrone dataset using the provided script:
```bash
python download_visdrone.py --output_dir data/VisDrone
```

This will download:
- VisDrone2019-DET-train.zip
- VisDrone2019-DET-val.zip
- VisDrone2019-DET-test-dev.zip
- VisDrone2019-DET-test-challenge.zip

### Converting VisDrone to YOLO Format

1. Convert the VisDrone annotations to YOLO format:
```bash
python visdrone2yolo.py --visdrone_dir data/VisDrone \
                        --output_dir data/visdrone_yolo \
                        --split train
```

The script will:
- Convert bounding box annotations to YOLO format
- Create train/val/test splits
- Generate dataset configuration file

### Applying Fisheye Augmentation

1. Apply fisheye distortion to create synthetic fisheye images:
```bash
python ifish_augmentation/convert_visdrone.py \
    --input_dir data/visdrone_yolo \
    --output_dir data/Synthetic_VisDrone \
    --distortion 0.5
```

The augmentation process:
- Applies fisheye distortion to regular images
- Adjusts bounding box coordinates to match the distorted images
- Creates a synthetic dataset with fisheye-like characteristics
- Maintains the same directory structure (train/val splits)

Parameters:
- `--distortion`: Fisheye distortion coefficient (0-1, default: 0.5)
- `--input_dir`: Directory containing YOLO-formatted VisDrone dataset
- `--output_dir`: Directory to save synthetic fisheye images and labels

### Merging Datasets (Optional)

If you want to combine VisDrone with other datasets:
```bash
python merge_datasets.py --dataset1 data/Fisheye8K \
                        --dataset2 data/Synthetic_VisDrone \
                        --output_dir data/merged_dataset
```

## Data Structure

### Training Data (Fisheye8K)
- Located in `data/Fisheye8K/train/`
- Contains training images, labels, and annotations
- Includes `train.json` with ground truth annotations

### Validation Data (Fisheye8K test)
- Located in `data/Fisheye8K/val/`
- Used for model validation during training

### Test Data (Fisheye1K)
- Located in `data/Fisheye1K/`
- Used for final evaluation and generating predictions

## Model Training

The model configuration is defined in `data/Fisheye8k.yml`. This YAML file specifies:
- Dataset paths
- Class names
- Training/validation split

To train the model:
```bash
python train.py --model yolo11n.pt --data data/Fisheye8k.yml --epochs 100 --batch-size 16
```

Available model sizes:
- YOLO11n (nano): 5.4MB
- YOLO11s (small): 18MB
- YOLO11m (medium): 39MB

## Data Processing and Visualization

### Dataset Conversion
- `visdrone2yolo.py`: Converts VisDrone dataset format to YOLO format
- `merge_datasets.py`: Merges multiple datasets for training

### Visualization Tools
- `visualize_labels.py`: Visualizes bounding box annotations on images
- Results are saved in the `visualizations/` directory

## Evaluation

Run the evaluation script using:
```bash
python test.py --image_folder /path/to/test/images \
               --model_path /path/to/model.pt \
               --max_fps 25.0 \
               --output_json predictions.json
```

### Parameters:
- `--image_folder`: Path to the test images directory
- `--model_path`: Path to the trained model weights
- `--max_fps`: Maximum FPS for evaluation (default: 25.0)
- `--output_json`: Output path for predictions JSON file

## Output Format

The evaluation script generates predictions in the following JSON format:
```json
[
  {
    "image_id": integer,
    "category_id": integer,
    "bbox": [x1, y1, width, height],
    "score": float
  },
  ...
]
```

### Image ID Format
Image IDs are generated using the following convention:
- Camera Index: Extracted from filename (e.g., "camera29" → 29)
- Scene Index: M=0, A=1, E=2, N=3
- Frame Index: Extracted from filename
- Final ID: Concatenation of these values

Example: "camera29_N_97.png" → ID = 29397

## Performance Metrics

The evaluation script reports:
- Processing time per image
- Overall FPS
- Normalized FPS (relative to max_fps)
- COCO-style metrics (mAP, precision, recall)

## Requirements

See `requirements.txt` for detailed package dependencies. Key requirements:
- Python 3.8+
- PyTorch 2.0+
- Ultralytics 8.0+
- OpenCV 4.8+
- CUDA-capable GPU (recommended)