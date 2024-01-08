#-----------------------------------------------------------------------------------------#
import sys
sys.path.append('./Libs') 
import class_ as C
import utilities as U
#-----------------------------------------------------------------------------------------#
from PIL import Image
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
#-----------------------------------------------------------------------------------------#

image_file = 'data/gabbro.jpg'
model_size = 512 # should less than 512
hidden_nodes = model_size//2
rock_types = {
    0: 'sedimentary',  # class_1
    1: 'metamorphic'   # class_2
}

#-----------------------------------------------------------------------------------------#

# NOTE load 1 image and convert to pytorch tensor
image = Image.open(image_file)
image2numpy_array = np.array(image)
image_array2torch_tensor = torch.from_numpy(image2numpy_array).float()
torch_tensor = image_array2torch_tensor.permute(2, 0, 1)
# print("Tensor shape (C, H, W):", torch_tensor.shape)
resize_transform = transforms.Resize((model_size, model_size), antialias=True)
resized_tensor = resize_transform(torch_tensor)
print("Resized tensor shape:", resized_tensor.shape)
# NOTE QC image
# U.plot_torch_image(resized_tensor)

#-----------------------------------------------------------------------------------------#

model = C.SimpleMLP(resized_tensor, hidden_nodes, num_classes=len(rock_types))
output = model(resized_tensor.unsqueeze(0))  # Add a batch dimension if necessary
# print("Output:", output)
predicted_rock_type, confidence = model.interpret_output(output, rock_types)
print(f"Predicted rock type: {predicted_rock_type} with confidence: {confidence}")

#-----------------------------------------------------------------------------------------#