#-----------------------------------------------------------------------------------------#
import sys
sys.path.append('./Libs') 
import utilities as U
#-----------------------------------------------------------------------------------------#
import os
import shutil
from PIL import Image
#-----------------------------------------------------------------------------------------#

MAIN_PATH = 'data/Fossils'
output_path = 'data/rock_split'
model_size = 224
sliding_window = 50

#-----------------------------------------------------------------------------------------#

if os.path.exists(output_path):
    shutil.rmtree(output_path)
    print(f"Deleted {output_path}")
os.makedirs(output_path)

#-----------------------------------------------------------------------------------------#

# Get all subfolders in the MAIN_PATH
subfolders = [f.path for f in os.scandir(MAIN_PATH) if f.is_dir()]

for subfolder in subfolders:
    for filename in os.listdir(subfolder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(subfolder, filename)
            img = Image.open(img_path)
            # Pad and center the image to the target size
            img_padded = U.pad_to_center(img, target_size=(model_size, model_size))
            # Split the image with a sliding window of xx pixels
            tiles = U.split_image(img_padded, window_size=(model_size, model_size),
                                  step_size=(sliding_window, sliding_window))
            # Create a subfolder in the output directory
            subfolder_name = os.path.basename(subfolder)
            save_path = os.path.join(output_path, subfolder_name)
            os.makedirs(save_path, exist_ok=True)
            
            # Save each tile with the specified naming convention
            base_filename = os.path.splitext(filename)[0]  # Get the base name of the file
            for i, tile in enumerate(tiles):
                tile_number = f"split_{str(i+1).zfill(3)}"
                new_filename = f"{base_filename}_{tile_number}.png"
                tile.save(os.path.join(save_path, new_filename))
                print(f"Saved: {os.path.join(save_path, new_filename)}")

#-----------------------------------------------------------------------------------------#