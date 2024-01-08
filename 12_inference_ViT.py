#-----------------------------------------------------------------------------------------#
import sys
sys.path.append('./Libs') 
import class_transformer as CT
#-----------------------------------------------------------------------------------------#
import matplotlib.pyplot as plt
from PIL import Image
#-----------------------------------------------------------------------------------------#

tabular_file = 'tabular_dataset/manifest_rocks.csv'
model_path = 'save_trained_model/ViT_rocks.ckpt'
image_unknown = 'data/rock_split/Brachiopods/Brachiopods_0001_split_009.png'

#-----------------------------------------------------------------------------------------#

vit_inference = CT.ViTInference(model_path, tabular_file)
vit_inference.load_best_model()
predicted_label = vit_inference.predict_image(image_unknown)
true_label = image_unknown.split('/')[-2]  # This splits the path and takes the second last element
image = Image.open(image_unknown)

#-----------------------------------------------------------------------------------------#

plt.imshow(image)
plt.title(f"True: {true_label}, Predicted: {predicted_label}")
plt.show()

#-----------------------------------------------------------------------------------------#