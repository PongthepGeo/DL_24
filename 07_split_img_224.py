#-----------------------------------------------------------------------------------------#
import sys
sys.path.append('./Libs') 
import class_ as C
#-----------------------------------------------------------------------------------------#
import os
import shutil
import glob
from PIL import Image
import numpy as np
#-----------------------------------------------------------------------------------------#

data_dir = 'data/Fossils'
output_path = 'data/rock_split_02'
model_size = 224
sliding_window = 50
train_ratio = 0.7
val_ratio = 0.15

#-----------------------------------------------------------------------------------------#

if os.path.exists(output_path):
    shutil.rmtree(output_path)

folders = ['train', 'val', 'test']
for folder in folders:
    os.makedirs(os.path.join(output_path, folder), exist_ok=True)

#-----------------------------------------------------------------------------------------#

image_paths = []
subfolders = [f.path for f in os.scandir(data_dir) if f.is_dir()]
for subfolder in subfolders:
    for filename in os.listdir(subfolder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_paths.append(os.path.join(subfolder, filename))

#-----------------------------------------------------------------------------------------#

np.random.shuffle(image_paths)
total_images = len(image_paths)
train_end = int(total_images * train_ratio)
val_end = train_end + int(total_images * val_ratio)
train_images = image_paths[:train_end]
val_images = image_paths[train_end:val_end]
test_images = image_paths[val_end:]

#-----------------------------------------------------------------------------------------#

process = C.SplitDataset()

# Process train images
for img_path in train_images:
    img = Image.open(img_path)
    base_filename = os.path.splitext(os.path.basename(img_path))[0]
    save_path = os.path.join(output_path, 'train')
    new_filename = process.image_split_stride(img, base_filename, save_path, sliding_window)
    print(f"Saved train image: {os.path.join(save_path, new_filename)}")

# Process validation images
for img_path in val_images:
    img = Image.open(img_path)
    base_filename = os.path.splitext(os.path.basename(img_path))[0]
    save_path = os.path.join(output_path, 'val')
    new_filename = process.image_split_stride(img, base_filename, save_path, sliding_window)
    print(f"Saved validation image: {os.path.join(save_path, new_filename)}")

# Process test images
for img_path in test_images:
    img = Image.open(img_path)
    base_filename = os.path.splitext(os.path.basename(img_path))[0]
    save_path = os.path.join(output_path, 'test')
    new_filename = process.image_test(img, base_filename, save_path)
    print(f"Saved test image: {os.path.join(save_path, new_filename)}")

#-----------------------------------------------------------------------------------------#