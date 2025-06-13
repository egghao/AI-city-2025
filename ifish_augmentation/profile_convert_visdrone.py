import cProfile
import pstats
import io
from convert_visdrone import convert_images
import os
import random

def profile_convert_visdrone():
    # Setup paths similar to main script
    src_data_dir = "data/VisDrone/VisDrone2019-DET-val"
    trg_data_dir = "data/Synthetic_VisDrone/VisDrone2019-DET-val"
    distortion_coefficient = 0.5
    crop = True

    # Create necessary directories
    if not os.path.exists(trg_data_dir):
        os.mkdir(trg_data_dir)

    if not os.path.exists(trg_data_dir):
        os.mkdir(trg_data_dir)

    src_images_dir = os.path.join(src_data_dir, "images")
    src_labels_dir = os.path.join(src_data_dir, "labels")

    trg_images_dir = os.path.join(trg_data_dir, "images")
    trg_labels_dir = os.path.join(trg_data_dir, "labels")

    if not os.path.exists(trg_images_dir):
        os.mkdir(trg_images_dir)
    if not os.path.exists(trg_labels_dir):
        os.mkdir(trg_labels_dir)

    # Get image and label paths
    _, _, img_list = next(os.walk(src_images_dir))
    if len(img_list) > 1:
        img_list = random.sample(img_list, 1)
    src_images_path = []
    src_labels_path = []
    for img_file in img_list:
        img_name = img_file.split(".")[0]
        label_file = img_name + ".txt"
        img_path = os.path.join(src_images_dir, img_file)
        label_path = os.path.join(src_labels_dir, label_file)
        src_images_path.append(img_path)
        src_labels_path.append(label_path)

    # Profile the conversion process
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Run the conversion
    convert_images(src_images_path, src_labels_path, trg_images_dir, trg_labels_dir, 
                  distortion_coefficient, crop)
    
    profiler.disable()
    
    # Print profiling results
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(30)  # Print top 30 time-consuming functions
    print(s.getvalue())

if __name__ == "__main__":
    profile_convert_visdrone() 