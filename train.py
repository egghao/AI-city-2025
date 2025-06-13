import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

from ultralytics import YOLO
import multiprocessing

def main():
    # Load a model
    model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(data="data/Fisheye8k.yaml",
                        epochs=100,
                        batch=16,
                        imgsz=640,
                        device="0",  # specify the device to use, e.g., "0" for GPU 0
                        project="models",  # changed from "/models/" to "models"
                        name="yolo11n_fisheye8k",  # specify the name of the model
                        save_period=10,  # save the model every 10 epochs
                        save=True,  # save the model after training
                        )  # train the model with the specified parameters

if __name__ == '__main__':
    multiprocessing.freeze_support()  # Required for Windows
    main()