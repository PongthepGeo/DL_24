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
import torch.nn as nn
import torch.optim as optim
import os
#-----------------------------------------------------------------------------------------#

image_file = 'data/cifar_10_new/bird/0001.png'
model_size = 32 # should less than 512
hidden_nodes = model_size//2
rock_types = {
    0: 'sedimentary',  # class_1
    1: 'metamorphic'   # class_2
}
num_epochs = 50  
learning_rate = 1e-6
save_trained_model = 'save_trained_model'
save_name = 'MLP.pt'

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

#-----------------------------------------------------------------------------------------#

# NOTE Define parameters
actual_class_index = 1  # a metamorphic rock is class 1
target_tensor = torch.tensor([actual_class_index]) # convert the class index to a tensor
# NOTE Initialize the model
model = C.SimpleMLP(resized_tensor, hidden_nodes, len(rock_types))
# NOTE Loss and Optimizer
criterion = nn.CrossEntropyLoss()  # Use nn.MSELoss() for MSE
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#-----------------------------------------------------------------------------------------#

# NOTE Training loop
losses = []
for epoch in range(num_epochs):
    # NOTE Forward pass
    outputs = model(resized_tensor.unsqueeze(0))
    loss = criterion(outputs, target_tensor)
    # NOTE Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

os.makedirs(save_trained_model, exist_ok=True)
save_torch_tensor = os.path.join(save_trained_model, save_name)
torch.save(model.state_dict(), save_torch_tensor)

#-----------------------------------------------------------------------------------------#

U.plot_loss(losses)

#-----------------------------------------------------------------------------------------#

image_test = 'data/cifar_10_new/bird/0002.png'
image = Image.open(image_test)
image2numpy_array = np.array(image)
image_array2torch_tensor = torch.from_numpy(image2numpy_array).float()
torch_tensor = image_array2torch_tensor.permute(2, 0, 1)
# print("Tensor shape (C, H, W):", torch_tensor.shape)
resize_transform = transforms.Resize((model_size, model_size), antialias=True)
resized_tensor = resize_transform(torch_tensor)

loaded_model = C.SimpleMLP(resized_tensor, hidden_nodes=hidden_nodes, num_classes=len(rock_types))
loaded_model.load_state_dict(torch.load(save_torch_tensor))

# Assuming new_image_tensor is the tensor of the new image to predict
loaded_model.eval()
with torch.no_grad():
    predicted_output = loaded_model(resized_tensor.unsqueeze(0))
rock_type, probability = loaded_model.interpret_output(predicted_output, rock_types)
print("Predicted rock type:", rock_type)
print("Probability:", probability)

#-----------------------------------------------------------------------------------------#