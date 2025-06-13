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
├── utils.py
├── eval.py
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

## Data Structure

### Training Data (Fisheye8K)
- Located in `data/Fisheye8K/train/`
- Contains training images, labels, and annotations
- Includes `train.json` with ground truth annotations

### Validation Data (Fisheye8K test)
- Located in `data/Fisheye8K/test/`
- Used for model validation during training

### Test Data (Fisheye1K)
- Located in `data/Fisheye1K/`
- Used for final evaluation and generating predictions

## Model Training

The model configuration is defined in `data/Fisheye8k.yml`. This YAML file specifies:
- Dataset paths
- Class names
- Training/validation split

## Evaluation

Run the evaluation script using:
```bash
python eval.py --image_folder /path/to/test/images \
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

## Requirements

See `requirements.txt` for detailed package dependencies. Key requirements:
- Python 3.8+
- PyTorch 2.0+
- Ultralytics 8.0+
- OpenCV 4.8+
- CUDA-capable GPU (recommended)
