#-----------------------------------------------------------------------------------------#
import sys
sys.path.append('./Libs') 
import class_ as C
import utilities as U
#-----------------------------------------------------------------------------------------#
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
#-----------------------------------------------------------------------------------------#

image_with_meta_data = Image.open('data/gabbro.jpg')
rock_types = {
    0: 'sedimentary',  # class_1
    1: 'metamorphic'   # class_2
}
number_of_nodes = 4

#-----------------------------------------------------------------------------------------#

image2numpy_array = np.array(image_with_meta_data)
image_array2torch_tensor = torch.from_numpy(image2numpy_array).float()
torch_tensor = image_array2torch_tensor.permute(2, 0, 1)
print("Tensor shape (C, H, W):", torch_tensor.shape)
# NOTE Resize the tensor to 32x32 with antialiasing
resize_transform = transforms.Resize((32, 32), antialias=True)
resized_tensor = resize_transform(torch_tensor)
print("Resized tensor shape:", resized_tensor.shape)
# NOTE QC plot
# U.plot_torch_image(resized_tensor)

#-----------------------------------------------------------------------------------------#

perceptron = C.SimplePerceptron(input_size=number_of_nodes, rock_types=rock_types)
random_weights = torch.rand(number_of_nodes)
print("Random weights:", random_weights)
predicted_class = perceptron(random_weights)
print("Output rock type:", predicted_class)

#-----------------------------------------------------------------------------------------#