#-----------------------------------------------------------------------------------------#
import sys
sys.path.append('./Libs') 
import utilities as U
import class_ as C
#-----------------------------------------------------------------------------------------#
import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
#-----------------------------------------------------------------------------------------#
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from torchsummary import summary
#-----------------------------------------------------------------------------------------#

device = U.set_seed(42)

#-----------------------------------------------------------------------------------------#

data_dir = 'data/rock_split_02'

rock_types = {
    0: 'Ammonites',   # class_1
    1: 'Bivalves',    # class_2
    2: 'Brachiopods', # class_3
    3: 'Gasteropods', # class_4
    4: 'Trilobites'   # class_5
}

num_workers = 32
batch_size = 1024
model_size = 224  
learning_rate = 5e-2
epochs = 50  
model_name = 'AlexNet_224'
save_trained_model = 'save_trained_model'

#-----------------------------------------------------------------------------------------#

if not os.path.exists(save_trained_model):
    os.makedirs(save_trained_model)
CHECKPOINT_PATH = os.path.join(save_trained_model, model_name + '.ckpt')

#-----------------------------------------------------------------------------------------#

class_to_idx = {value: key for key, value in rock_types.items()}
train_pattern = f"{data_dir}/train/*.png"
val_pattern = f"{data_dir}/val/*.png"
train_list = glob.glob(train_pattern)
val_list = glob.glob(val_pattern)
print('class_to_idx:', class_to_idx)
print('number of training images: ', len(train_list),
      '\nnumber of val images: ', len(val_list))

#-----------------------------------------------------------------------------------------#

mean = [0.48787489, 0.46587484, 0.42884341]
std = [0.23536755, 0.24031864, 0.24699368]
resize_size = (model_size, model_size)  
flip = transforms.RandomHorizontalFlip()
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean, std)
resize = transforms.Resize(resize_size)

transform_train = transforms.Compose([
                  resize,      # Resize the image first
                  flip,        # Then apply random horizontal flipping
                  to_tensor,   # Convert the image to a tensor
                  normalize    # Normalize the image
])
transform_val = transforms.Compose([
                resize,      # Resize the image first
                flip,        # Then apply random horizontal flipping
                to_tensor,   # Convert the image to a tensor
                normalize    # Normalize the image
])

#-----------------------------------------------------------------------------------------#

train_dataset = C.Image2Torch_02(train_list, class_to_idx, transform_train)
val_dataset = C.Image2Torch_02(val_list, class_to_idx, transform_val)
train_loader = DataLoader(train_dataset, batch_size=batch_size,
               shuffle=False, drop_last=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size,
             shuffle=False, drop_last=True, num_workers=num_workers)

#-----------------------------------------------------------------------------------------#

model = C.AlexNet_224(len(rock_types))
print(model.to(device))
summary(model, (3, model_size, model_size))
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()
model = model.to(device)
criterion = criterion.to(device)

#-----------------------------------------------------------------------------------------#

trainer = C.ModelTrainer(model, optimizer, criterion, device)
history_train_loss, history_valid_loss, best_valid_loss = trainer.run(train_loader,
    val_loader, epochs, CHECKPOINT_PATH)
U.loss_history_plot(history_train_loss, history_valid_loss, model_name)

#-----------------------------------------------------------------------------------------#
