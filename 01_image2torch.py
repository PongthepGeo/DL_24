#-----------------------------------------------------------------------------------------#
import sys
sys.path.append('./Libs') 
import utilities as U
#-----------------------------------------------------------------------------------------#
import numpy as np
import torch
from PIL import Image
#-----------------------------------------------------------------------------------------#

image_with_meta_data = Image.open('data/gabbro.jpg')
print(image_with_meta_data.size)
image2numpy_array = np.array(image_with_meta_data)
print(image2numpy_array.shape)
image_array2torch_tensor = torch.from_numpy(image2numpy_array)
print(image_array2torch_tensor.shape)
torch_tensor = image_array2torch_tensor.permute(2, 0, 1)
print(torch_tensor.shape)

#-----------------------------------------------------------------------------------------#

U.plot_torch_image(torch_tensor)

#-----------------------------------------------------------------------------------------#