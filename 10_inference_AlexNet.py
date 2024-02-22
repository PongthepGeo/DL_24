#-----------------------------------------------------------------------------------------#
import sys
sys.path.append('./Libs') 
import utilities as U
import class_ as C
#-----------------------------------------------------------------------------------------#
import os
import glob
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
#-----------------------------------------------------------------------------------------#
from torchsummary import summary
#-----------------------------------------------------------------------------------------#

device = U.set_seed(42)

#-----------------------------------------------------------------------------------------#

data_dir = 'data/rock_split_02'
model_name = 'AlexNet_224'
save_trained_model = 'save_trained_model'

rock_types = {
    0: 'Ammonites',   # class_1
    1: 'Bivalves',    # class_2
    2: 'Brachiopods', # class_3
    3: 'Gasteropods', # class_4
    4: 'Trilobites'   # class_5
}

sliding_window = 224
output_path = f'{data_dir}/test_split_224'  # Directly setting the intended output path

num_workers = 0
batch_size = 2
model_size = 224  

#-----------------------------------------------------------------------------------------#

if not os.path.exists(output_path):  
            os.makedirs(output_path)  

if not os.path.exists(save_trained_model):
    os.makedirs(save_trained_model)
CHECKPOINT_PATH = os.path.join(save_trained_model, model_name + '.ckpt')

#-----------------------------------------------------------------------------------------#

class_to_idx = {value: key for key, value in rock_types.items()}
test_pattern = f"{data_dir}/test/*.png"
test_list = glob.glob(test_pattern)
print('class_to_idx:', class_to_idx)
print('number of training images: ', len(test_list))

process = C.SplitDataset()

# Process test images
for img_path in test_list:
    img = Image.open(img_path)
    base_filename = os.path.splitext(os.path.basename(img_path))[0]
    new_filename = process.image_split_stride(img, base_filename, output_path, sliding_window)
    print(f"Saved test image: {os.path.join(output_path, new_filename)}")

#-----------------------------------------------------------------------------------------#

test_split_pattern = f'{data_dir}/test_split_224/*.png'
test_split_list = glob.glob(test_split_pattern)

mean = [0.48787489, 0.46587484, 0.42884341]
std = [0.23536755, 0.24031864, 0.24699368]
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean, std)
transform_test = transforms.Compose([
                to_tensor,   # Convert the image to a tensor
                normalize    # Normalize the image
])

test_dataset = C.Image2Torch_02(test_split_list, class_to_idx, transform_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size,
              shuffle=False, drop_last=True, num_workers=num_workers)

model = C.AlexNet_224(len(rock_types))
print(model.to(device))
summary(model, (3, model_size, model_size))
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()
model = model.to(device)
criterion = criterion.to(device)

model.load_state_dict(torch.load(CHECKPOINT_PATH))
test_loss, test_acc = U.evaluate(model, test_loader, criterion, device)
print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')

#-----------------------------------------------------------------------------------------#

images, labels, probs = U.get_predictions(model, test_loader, device)
# print(f"labels: {labels}, probs: {probs}")
pred_labels = torch.argmax(probs, 1)
# print(f"pred_labels: {pred_labels}")
U.plot_confusion_matrix_less_classes(labels, pred_labels, list(rock_types.values()))

#-----------------------------------------------------------------------------------------#
