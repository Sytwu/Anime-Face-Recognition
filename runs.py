import os
import json
import cv2
from tqdm import tqdm
from core.detector import LFFDDetector

"""
    !!! IMPORTANT !!!
    Disable auto-tuning
    You might experience a major slow-down if you run the model on images with varied resolution / aspect ratio.
    This is because MXNet is attempting to find the best convolution algorithm for each input size, 
    so we should disable this behavior if it is not desirable.
""" 
os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
CONFIG_PATH = "configs/anime.json"

def anime_face_detect(folder):
    new_root_folder = 'face_dataset'
    new_folder = os.path.basename(folder)
    new_path = os.path.join(new_root_folder, new_folder)
    
    if not os.path.exists(new_path):
        os.makedirs(new_path)
        
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
        detector = LFFDDetector(config, use_gpu=True)
    
    WIDTH = HEIGHT = 256
    files = os.listdir(folder)
    for file in tqdm(files, desc=f"Processing {new_folder}", unit="file"):
        image = cv2.imread(os.path.join(folder, file))
        boxes = detector.detect(image)
        
        idx = 1
        for box in boxes:
            x1, y1 = box['xmin'], box['ymin']
            x2, y2 = box['xmax'], box['ymax']
            
            face_image = image[y1:y2, x1:x2]
            face_image = cv2.resize(face_image, (WIDTH, HEIGHT))
            
            save_path = f"{new_path}/{file}-{idx}.jpg"
            cv2.imwrite(save_path, face_image)
            idx += 1
    
    return 0

if __name__ == "__main__":
    
    root_folder = 'dataset'
    for folder in os.listdir(root_folder):
        path = os.path.join(root_folder, folder)
        if not os.path.isdir(path): continue
        anime_face_detect(path)
        print(f"{folder} has finished.")
    
    print("Complete!")