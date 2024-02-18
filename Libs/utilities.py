#-----------------------------------------------------------------------------------------#
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import ImageOps
# from torchvision import transforms
import os
from skimage import io, transform
from sklearn.preprocessing import StandardScaler
import torch
import random
from tqdm import tqdm
import torch.nn.functional as TF
from sklearn import metrics
import pickle
import pandas as pd
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

def normalize_data(arr, a=0, b=1):
	min_val = np.min(arr)
	max_val = np.max(arr)
	return a + (b - a) * (arr - min_val) / (max_val - min_val)

def plot_torch_image(torch_tensor):
	plt.figure()  
	# Permute the tensor dimensions to HxWxC
	torch_tensor = torch_tensor.permute(1, 2, 0)
	image_for_plot = torch_tensor.numpy()
	image_for_plot = normalize_data(image_for_plot, a=0, b=1)
	# image_for_plot = (image_for_plot - image_for_plot.min()) / (image_for_plot.max() - image_for_plot.min())
	plt.imshow(image_for_plot)
	plt.xlabel('Pixel-X')
	plt.ylabel('Pixel-Y')
	plt.show()

def plot_loss(losses):
	plt.figure()
	plt.plot(losses, label='Training Loss')
	plt.title('Loss Over Epochs')
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.legend()
	plt.show()

def plot_loss_train_val(epoch_losses, val_epoch_losses):
	plt.figure()
	plt.plot(epoch_losses[3:], label='Training Loss')
	plt.plot(val_epoch_losses[3:], label='Validation Loss')
	plt.title('Loss vs. Epoch')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.legend()
	plt.grid(True)
	plt.show()

def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
	print('Using PyTorch version:', torch.__version__, ' Device:', device)
	return device

def means_std(train_list):
	train_data = []
	for img_path in train_list:
		img = io.imread(img_path)
		img = img / 255.0  # limit value to be between 0 and 1
		train_data.append(img)
	train_data = np.array(train_data)
	train_data = np.transpose(train_data, (0, 3, 1, 2))  # fit PyTorch format
	train_data_flat = train_data.reshape(train_data.shape[0], -1)
	scaler = StandardScaler()
	scaler.fit(train_data_flat)
	mean = scaler.mean_.reshape(3, -1).mean(axis=1)
	std = scaler.scale_.reshape(3, -1).std(axis=1)
	print('mean: ', mean)
	print('standard deviation: ', std)
	return mean, std

def means_std_512(train_list, desired_shape=(512, 512)):
	train_data = []
	for img_path in train_list:
		img = io.imread(img_path) / 255.0  # Read and normalize the image
		
		# Resize image to ensure consistency
		img_resized = transform.resize(img, desired_shape, mode='reflect', anti_aliasing=True)
		
		# If the image is grayscale, convert it to a 3-channel image
		if img_resized.ndim == 2:
			img_resized = np.stack((img_resized,)*3, axis=-1)

		# Append the image data
		train_data.append(img_resized)

	# Convert the list of images to a NumPy array and reshape for PyTorch format
	train_data = np.array(train_data)
	train_data = np.transpose(train_data, (0, 3, 1, 2))  # Shape will be [N, 3, 512, 512]

	# Flatten the data for the scaler
	train_data_flat = train_data.reshape(train_data.shape[0], -1)

	# Calculate mean and standard deviation
	scaler = StandardScaler()
	scaler.fit(train_data_flat)
	mean = scaler.mean_.reshape(3, -1).mean(axis=1)
	std = scaler.scale_.reshape(3, -1).std(axis=1)

	print('mean: ', mean)
	print('standard deviation: ', std)
	return mean, std

def loss_history_plot(history_train, history_valid, model_name):
	axis_x = np.linspace(0, len(history_train), len(history_train))
	plt.plot(axis_x, history_train, linestyle='solid',
			 color='red', linewidth=1, marker='o', ms=5, label='train')
	plt.plot(axis_x, history_valid, linestyle='solid',
			 color='blue', linewidth=1, marker='o', ms=5, label='valid')
	plt.xlabel('epoch')
	plt.ylabel('loss')
	plt.legend(['train', 'valid'])
	plt.title(model_name + ': ' + 'Accuracy', fontweight='bold')
	# plt.savefig('data_out/' + 'resnet' + '.svg', format='svg', bbox_inches='tight', transparent=True, pad_inches=0)
	plt.show()

def evaluate(model, iterator, criterion, device):
	epoch_loss = 0; epoch_acc = 0
	model.eval()
	with torch.no_grad():
		for (x, y) in tqdm(iterator, desc='Evaluating', leave=False):
			x = x.to(device)
			y = y.to(device)
			y_pred, _ = model(x)
			loss = criterion(y_pred, y)
			acc = calculate_accuracy(y_pred, y)
			epoch_loss += loss.item()
			epoch_acc += acc.item()
	return epoch_loss / len(iterator), epoch_acc / len(iterator)

def get_predictions(model, iterator, device):
	model.eval()
	images = []; labels = []; probs = []
	with torch.no_grad():
		for (x, y) in iterator:
			x = x.to(device)
			y_pred, _ = model(x)
			y_prob = TF.softmax(y_pred, dim=-1)
			images.append(x.cpu())
			labels.append(y.cpu())
			probs.append(y_prob.cpu())
	images = torch.cat(images, dim=0)
	labels = torch.cat(labels, dim=0)
	probs = torch.cat(probs, dim=0)
	return images, labels, probs

def calculate_accuracy(y_pred, y):
	top_pred = y_pred.argmax(1, keepdim=True)
	correct = top_pred.eq(y.view_as(top_pred)).sum()
	acc = correct.float() / y.shape[0]
	return acc

def plot_confusion_matrix(labels, pred_labels, classes):
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)
	cm = metrics.confusion_matrix(labels, pred_labels)
	cm_display = metrics.ConfusionMatrixDisplay(cm, display_labels=classes)
	cm_display.plot(values_format='d', cmap='Greens', ax=ax)
	accuracy = metrics.accuracy_score(labels, pred_labels)
	accuracy_percentage = accuracy * 100
	ax.set_title(f'Confusion Matrix - Accuracy: {accuracy_percentage:.2f}%')
	plt.xticks(rotation=90)  # Rotate x-tick labels for better visibility
	# plt.savefig('data_out/CM_resnet.svg', format='svg', bbox_inches='tight', transparent=True, pad_inches=0)
	plt.show()

def normalize_image(image):
	image_min = image.min()
	image_max = image.max()
	image.clamp_(min=image_min, max=image_max)
	image.add_(-image_min).div_(image_max - image_min + 1e-5)
	return image

def plot_most_incorrect(incorrect, classes, n_images, normalize=True):
	rows = int(np.sqrt(n_images))
	cols = int(np.sqrt(n_images))
	fig = plt.figure()
	for i in range(rows*cols):
		ax = fig.add_subplot(rows, cols, i+1)
		image, true_label, probs = incorrect[i]
		image = image.permute(1, 2, 0)
		true_prob = probs[true_label]
		incorrect_prob, incorrect_label = torch.max(probs, dim=0)
		true_class = classes[true_label]
		incorrect_class = classes[incorrect_label]
		if normalize:
			image = normalize_image(image)
		ax.imshow(image.cpu().numpy())
		ax.set_title(f'true label: {true_class} ({true_prob:.3f})\n'
			 f'pred label: {incorrect_class} ({incorrect_prob:.3f})',
			 fontsize=6)
		ax.axis('off')
	fig.subplots_adjust(hspace=0.6)
	# plt.savefig('data_out/' + 'incorrect_resnet' + '.svg', format='svg', bbox_inches='tight', transparent=True, pad_inches=0)
	plt.show()

def plot_confusion_matrix_low(labels, pred_labels):
	# unique_labels = sorted(set(labels).union(set(pred_labels)))
	max_label = max(max(labels), max(pred_labels))
	display_labels = range(max_label + 1)
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)
	cm = metrics.confusion_matrix(labels, pred_labels, labels=display_labels)
	disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
	disp.plot(values_format='d', cmap='Greens', ax=ax)
	plt.xticks(rotation=90)  # Rotate x-tick labels for better visibility
	plt.show()

def save_callback(callback, filename):
	with open(filename, 'wb') as f:
		pickle.dump(callback, f)

def load_callback(filename):
	with open(filename, 'rb') as f:
		return pickle.load(f)

def pad_to_center(img, target_size=(224, 224)):
	old_size = img.size  
	delta_w = max(target_size[0] - old_size[0], 0)
	delta_h = max(target_size[1] - old_size[1], 0)
	padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
	return ImageOps.expand(img, padding)

def split_image(img, window_size=(224, 224), step_size=(50, 50)):
	w, h = img.size
	grid = []
	for i in range(0, w - window_size[0] + 1, step_size[0]):
		for j in range(0, h - window_size[1] + 1, step_size[1]):
			box = (i, j, i + window_size[0], j + window_size[1])
			grid.append(img.crop(box))
	return grid

def generate_dataset_manifest(data_dir, tabular_dataset_folder, output_filename):
	class_names = sorted(os.listdir(data_dir), reverse=False)
	sorted_label_mapping = {label: index for index, label in enumerate(class_names)}
	if not os.path.exists(tabular_dataset_folder):
		os.makedirs(tabular_dataset_folder)
	output_file = os.path.join(tabular_dataset_folder, output_filename)

	tabular_data = []

	for class_name in class_names:
		class_dir = os.path.join(data_dir, class_name)
		if os.path.isdir(class_dir):
			for image_name in os.listdir(class_dir):
				image_path = os.path.join(class_dir, image_name)
				# Append the path, class name, and corresponding label ID to the tabular dataset
				tabular_data.append([image_path, class_name, sorted_label_mapping[class_name]])
	df = pd.DataFrame(tabular_data, columns=['image_path', 'label', 'label2id'])
	df.to_csv(output_file, index=False)
	print('Saved dataset manifest to:', output_file)

def compute_mse(m, b, x, y):
    return np.mean((y - (m * x + b))**2)

def compute_gradients(m, b, x, y):
    N = len(x)
    dm = -2/N * np.sum(x * (y - (m * x + b)))
    db = -2/N * np.sum(y - (m * x + b))
    return dm, db

def basic_gradient_descent(x, y_pred, y, i, m, b, current_mse):
    plt.plot(x, y_pred, color='red')
    plt.scatter(x, y, marker='x')
    plt.title(f"Iteration {i+1}: m = {m:.4f}, b = {b:.4f} | MSE: {current_mse:.4f}")
    plt.xlabel("Data X")
    plt.ylabel("Data Y")
    plt.show()