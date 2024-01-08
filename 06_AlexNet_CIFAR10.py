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

DATASET_PATH = 'data/cifar_10_new'
train_size = 0.70; val_size = 0.29; test_size = 0.01
num_workers = 32
batch_size = 128
model_size = 32  
learning_rate = 1e-3
epochs = 100  
model_name = 'AlexNet_32'
save_trained_model = 'save_trained_model'
image2plot = 20

#-----------------------------------------------------------------------------------------#

if not os.path.exists(save_trained_model):
    os.makedirs(save_trained_model)

#-----------------------------------------------------------------------------------------#

class_names = sorted(os.listdir(DATASET_PATH), reverse=False)
print('class names: ', class_names)
num_class = len(class_names)
image_files = glob.glob(DATASET_PATH + '/*/*.png', recursive=True)
print('total images in: ', DATASET_PATH, ' is ', len(image_files))
idx_to_class = {i:j for i, j in enumerate(class_names)}
class_to_idx = {value:key for key, value in idx_to_class.items()}
train_idx, test_idx, val_idx = random_split(image_files, [train_size, val_size, test_size])
train_list=[image_files[i] for i in train_idx.indices]
val_list=[image_files[i] for i in test_idx.indices]
test_list=[image_files[i] for i in val_idx.indices]
print('number of training images: ', len(train_list),
	'\nnumber of val images: ', len(val_list),
	'\nnumber of test images: ', len(test_list))

#-----------------------------------------------------------------------------------------#

mean, std = U.means_std(train_list)  
flip = transforms.RandomHorizontalFlip()
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean, std)
transform_train = transforms.Compose([
    flip,    # Then apply random horizontal flipping
    to_tensor,  # Convert the image to a tensor
    normalize  # Normalize the image
])
transform_val = transforms.Compose([
    to_tensor,  # Convert the image to a tensor
    normalize  # Normalize the image
])
transform_test = transforms.Compose([
    to_tensor,  # Convert the image to a tensor
    normalize  # Normalize the image
])

#-----------------------------------------------------------------------------------------#

train_dataset = C.Image2Torch(train_list, class_to_idx, transform_train)
val_dataset = C.Image2Torch(val_list, class_to_idx, transform_val)
test_dataset = C.Image2Torch(test_list, class_to_idx, transform_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size,
               shuffle=True, drop_last=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size,
             shuffle=False, drop_last=False, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size,
              shuffle=False, drop_last=False, num_workers=num_workers)

#-----------------------------------------------------------------------------------------#

model = C.AlexNet_32(num_class)
print(model.to(device))
summary(model, (3, model_size, model_size))
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()
model = model.to(device)
criterion = criterion.to(device)

#-----------------------------------------------------------------------------------------#

CHECKPOINT_PATH = os.path.join(save_trained_model, model_name + '.ckpt')
if not os.path.exists(save_trained_model):
    os.makedirs(save_trained_model)

#-----------------------------------------------------------------------------------------#

trainer = C.ModelTrainer(model, optimizer, criterion, device)
history_train_loss, history_valid_loss, best_valid_loss = trainer.run(train_loader,
    val_loader, epochs, CHECKPOINT_PATH)
U.loss_history_plot(history_train_loss, history_valid_loss, model_name)

#-----------------------------------------------------------------------------------------#

model.load_state_dict(torch.load(CHECKPOINT_PATH))
test_loss, test_acc = U.evaluate(model, test_loader, criterion, device)
print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')

#-----------------------------------------------------------------------------------------#

images, labels, probs = U.get_predictions(model, test_loader, device)
pred_labels = torch.argmax(probs, 1)
U.plot_confusion_matrix(labels, pred_labels, class_names)

#-----------------------------------------------------------------------------------------#

corrects = torch.eq(labels, pred_labels)
incorrect_examples = []
for image, label, prob, correct in zip(images, labels, probs, corrects):
    if not correct:
        incorrect_examples.append((image, label, prob))
incorrect_examples.sort(reverse = True, key=lambda x: torch.max(x[2], dim=0).values)
if len(incorrect_examples) >= image2plot:
    U.plot_most_incorrect(incorrect_examples, class_names, image2plot)
else:
    print('reduce the number of image2plot')

#-----------------------------------------------------------------------------------------#