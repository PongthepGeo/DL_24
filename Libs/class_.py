#-----------------------------------------------------------------------------------------#
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import os
from tqdm import tqdm
import time
from PIL import Image, ImageOps
#-----------------------------------------------------------------------------------------#
import matplotlib
params = {
	'savefig.dpi': 300,  
	'figure.dpi' : 300,
	'axes.labelsize':12,  
	'axes.titlesize':12,
	'axes.titleweight': 'bold',
	'legend.fontsize': 10,
	'xtick.labelsize':10,
	'ytick.labelsize':10,
	'font.family': 'serif',
	'font.serif': 'Times New Roman'
}
matplotlib.rcParams.update(params)
#-----------------------------------------------------------------------------------------#

class SimplePerceptron(nn.Module):
	def __init__(self, input_size, rock_types):
		super(SimplePerceptron, self).__init__()
		self.rock_types = rock_types
		# Initialize weights and bias
		self.weights = nn.Parameter(torch.randn(input_size))
		self.bias = nn.Parameter(torch.randn(1))

	def forward(self, x):
		# Weighted sum of inputs + bias
		linear_output = torch.dot(self.weights, x) + self.bias
		if linear_output >= 0:
			rock = self.rock_types[0]
		else:
			rock = self.rock_types[1]
		return rock

#-----------------------------------------------------------------------------------------#

class SimpleMLP(nn.Module):
	def __init__(self, image_tensor, hidden_nodes, num_classes=2):
		super(SimpleMLP, self).__init__()
		C, H, W = image_tensor.size()  # C: Channels, H: Height, W: Width
		input_features = C * H * W
		self.flatten = nn.Flatten()
		self.fc1 = nn.Linear(input_features, hidden_nodes)
		self.fc2 = nn.Linear(hidden_nodes, hidden_nodes)
		self.output = nn.Linear(hidden_nodes, num_classes)

	def forward(self, x):
		x = self.flatten(x)
		x = F.relu(self.fc1(x))  # Apply ReLU activation function
		x = F.relu(self.fc2(x))  # Apply ReLU activation function
		x = self.output(x)
		x = torch.sigmoid(x)  # Apply Sigmoid activation function for binary classification
		return x
	
	def interpret_output(self, output, rock_types):
		probabilities = output.detach().numpy()  
		# print("Probabilities:", probabilities)
		class_index = probabilities.argmax()  # Find the index of the maximum probability
		return rock_types[class_index], probabilities[0, class_index]

#-----------------------------------------------------------------------------------------#

class Image2Torch(Dataset):
    def __init__(self, image_paths, class_to_idx, transform=None):
        self.image_paths = image_paths
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        image = Image.open(image_filepath).convert('RGB')
        label = image_filepath.split('/')[-2]
        label = self.class_to_idx[label]
        if self.transform is not None:
            image = self.transform(image)
        return image, label

#-----------------------------------------------------------------------------------------#

class Image2Torch_02(Dataset):
    def __init__(self, image_paths, class_to_idx, transform):
        self.image_paths = image_paths
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        image = Image.open(image_filepath).convert('RGB')
        image = self.transform(image)
        label_name = os.path.basename(image_filepath).split('_')[0]  # Extract class name from filename
        label = self.class_to_idx[label_name]  # Convert class name to label
        return image, label

#-----------------------------------------------------------------------------------------#

class ModelTrainer:
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.best_valid_loss = float('inf')
        self.history_train_loss = []
        self.history_valid_loss = []

    def calculate_accuracy(self, y_pred, y):
        top_pred = y_pred.argmax(1, keepdim=True)
        correct = top_pred.eq(y.view_as(top_pred)).sum()
        acc = correct.float() / y.shape[0]
        return acc

    def train(self, iterator):
        epoch_loss = 0
        epoch_acc = 0
        self.model.train()
        for (x, y) in tqdm(iterator, desc='Training', leave=False):
            x = x.to(self.device)
            y = y.to(self.device)
            self.optimizer.zero_grad()
            y_pred, _ = self.model(x)
            loss = self.criterion(y_pred, y)
            acc = self.calculate_accuracy(y_pred, y)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        return epoch_loss / len(iterator), epoch_acc / len(iterator)

    def evaluate(self, iterator):
        epoch_loss = 0
        epoch_acc = 0
        self.model.eval()
        with torch.no_grad():
            for (x, y) in tqdm(iterator, desc='Evaluating', leave=False):
                x = x.to(self.device)
                y = y.to(self.device)
                y_pred, _ = self.model(x)
                loss = self.criterion(y_pred, y)
                acc = self.calculate_accuracy(y_pred, y)
                epoch_loss += loss.item()
                epoch_acc += acc.item()
        return epoch_loss / len(iterator), epoch_acc / len(iterator)

    def epoch_time(self, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    def run(self, train_loader, val_loader, epochs, checkpoint_path):
        for epoch in range(epochs):
            start_time = time.monotonic()
            train_loss, train_acc = self.train(train_loader)
            valid_loss, valid_acc = self.evaluate(val_loader)
            self.history_train_loss.append(train_loss)
            self.history_valid_loss.append(valid_loss)
            if valid_loss < self.best_valid_loss:
                self.best_valid_loss = valid_loss
                torch.save(self.model.state_dict(), checkpoint_path)
            end_time = time.monotonic()
            epoch_mins, epoch_secs = self.epoch_time(start_time, end_time)
            print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
            print(f'\tVal. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.2f}%')
        return self.history_train_loss, self.history_valid_loss, self.best_valid_loss

#-----------------------------------------------------------------------------------------#

class AlexNet_32(nn.Module):
	def __init__(self, output_dim):
		super().__init__()
		self.features = nn.Sequential(
			nn.Conv2d(3, 64, 3, 2, 1),  # in_channels, out_channels, kernel_size, stride, padding
			nn.MaxPool2d(2),  # kernel_size
			nn.ReLU(inplace=True),
			nn.Conv2d(64, 192, 3, padding=1),
			nn.MaxPool2d(2),
			nn.ReLU(inplace=True),
			nn.Conv2d(192, 384, 3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(384, 256, 3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(256, 256, 3, padding=1),
			nn.MaxPool2d(2),
			nn.ReLU(inplace=True)
		)
		self.classifier = nn.Sequential(
			nn.Dropout(0.5),
			nn.Linear(256 * 2 * 2, 4096),
			nn.ReLU(inplace=True),
			nn.Dropout(0.5),
			nn.Linear(4096, 4096),
			nn.ReLU(inplace=True),
			nn.Linear(4096, output_dim),
		)

	def forward(self, x):
		x = self.features(x)
		h = x.view(x.shape[0], -1)
		x = self.classifier(h)
		return x, h

#-----------------------------------------------------------------------------------------#

class AlexNet_224(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2), # Adjusted for 224x224 input
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6)) # Adaptive pooling to fit the final layer
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes) 
        )

    def forward(self, x):
        # Feature extraction part
        x = self.features(x)
        conv_output = x  # Save the output of the last conv layer
        # Adaptive pooling and classifier part
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        # Return both the final output and the last conv layer's output
        return x, conv_output

#-----------------------------------------------------------------------------------------#

class SplitDataset:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size

    def pad_to_center(self, img):
        old_size = img.size
        delta_w = max(self.target_size[0] - old_size[0], 0)
        delta_h = max(self.target_size[1] - old_size[1], 0)
        padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
        return ImageOps.expand(img, padding)

    def split_image(self, img, window_size=(224, 224), step_size=(50, 50)):
        w, h = img.size
        grid = []
        for i in range(0, w - window_size[0] + 1, step_size[0]):
            for j in range(0, h - window_size[1] + 1, step_size[1]):
                box = (i, j, i + window_size[0], j + window_size[1])
                grid.append(img.crop(box))
        return grid

    def image_split_stride(self, img, base_filename, save_path, sliding_window):
        img_padded = self.pad_to_center(img)
        tiles = self.split_image(img_padded, window_size=self.target_size, step_size=(sliding_window, sliding_window))
        for i, tile in enumerate(tiles):
            tile_number = f"split_{str(i + 1).zfill(3)}"
            new_filename = f"{base_filename}_{tile_number}.png"
            tile.save(os.path.join(save_path, new_filename))
        return new_filename

    def image_test(self, img, base_filename, save_path):
        new_filename = f"{base_filename}.png"
        img.save(os.path.join(save_path, new_filename))
        return new_filename

#-----------------------------------------------------------------------------------------#