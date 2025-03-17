import os
import glob
import random
import pandas as pd

random_seed = 42
random.seed(random_seed)

dataset_dir = "face_dataset"

train_data = []
test_data = []

for class_name in os.listdir(dataset_dir):
    class_dir = os.path.join(dataset_dir, class_name)
    if not os.path.isdir(class_dir):
        continue

    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
        image_files.extend(glob.glob(os.path.join(class_dir, ext)))
    
    random.shuffle(image_files)
    
    split_index = int(len(image_files) * 0.8)
    train_images = image_files[:split_index]
    test_images = image_files[split_index:]
    
    train_data.extend([(img_path, class_name) for img_path in train_images])
    test_data.extend([(img_path, class_name) for img_path in test_images])

df_train = pd.DataFrame(train_data, columns=['path', 'label'])
df_test = pd.DataFrame(test_data, columns=['path', 'label'])

df_train = df_train.sample(frac=1, random_state=random_seed).reset_index(drop=True)
df_test = df_test.sample(frac=1, random_state=random_seed).reset_index(drop=True)

df_train.to_csv('train.csv', index=False)
df_test.to_csv('test.csv', index=False)

print("train.csv and test.csv has been saved.")
