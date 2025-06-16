import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

from ultralytics import YOLO
import multiprocessing

def main():
    # Load a model
    model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

    # Train the model with default augmentation
    results = model.train(data="data/Fisheye8K.yaml",
                        epochs=100,
                        batch=16,
                        imgsz=640,
                        device="0",  # specify the device to use, e.g., "0" for GPU 0
                        project="models",  # changed from "/models/" to "models"
                        name="yolo11n_fisheye8k_640",  # specify the name of the model
                        save_period=10,  # save the model every 10 epochs
                        save=True,  # save the model after training
                        )  # train the model with the specified parameters
    
    
    # Train the model with 1280 resolution
    # results = model.train(data="data/Fisheye8K.yaml",
    #                     epochs=100,
    #                     batch=16,
    #                     imgsz=1280,
    #                     device="0",  # specify the device to use, e.g., "0" for GPU 0
    #                     project="models",  # changed from "/models/" to "models"
    #                     name="yolo11n_fisheye8k_1280",  # specify the name of the model
    #                     save_period=10,  # save the model every 10 epochs
    #                     save=True,  # save the model after training
    #                     )  # train the model with the specified parameters
    
    # Train the model with merged dataset
    # results = model.train(data="data/Merged_Dataset.yaml",
    #                     epochs=100,
    #                     batch=16,
    #                     imgsz=640,
    #                     device="0",  # specify the device to use, e.g., "0" for GPU 0
    #                     project="models",  # changed from "/models/" to "models"
    #                     name="yolo11n_merged_dataset_640",  # specify the name of the model
    #                     save_period=10,  # save the model every 10 epochs
    #                     save=True,  # save the model after training
    #                     )  # train the model with the specified parameters
    
    # Train the model with merged dataset and 1280 resolution
    # results = model.train(data="data/Merged_Dataset.yaml",
    #                     epochs=100,
    #                     batch=16,
    #                     imgsz=1280,
    #                     device="0",  # specify the device to use, e.g., "0" for GPU 0
    #                     project="models",  # changed from "/models/" to "models"
    #                     name="yolo11n_merged_dataset_1280",  # specify the name of the model
    #                     save_period=10,  # save the model every 10 epochs
    #                     save=True,  # save the model after training
    #                     )  # train the model with the specified parameters
    

    # Train another model without augmentation
    # results_no_aug = model.train(data="data/Merged_Dataset.yaml",
    #                     epochs=100,
    #                     batch=16,
    #                     imgsz=640,
    #                     device="0",
    #                     project="models",
    #                     name="yolo11n_merged_dataset_no_aug",
    #                     save_period=10,
    #                     save=True,
    #                     augment=False)  # disable augmentation

if __name__ == '__main__':
    multiprocessing.freeze_support()  # Required for Windows
    main()