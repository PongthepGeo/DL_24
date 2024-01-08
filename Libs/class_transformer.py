#-----------------------------------------------------------------------------------------#
import sys
sys.path.append('./Libs') 
import utilities as U
#-----------------------------------------------------------------------------------------#
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision.transforms import (Compose, RandomHorizontalFlip, ToTensor, Normalize, CenterCrop,
	Resize, RandomResizedCrop)
from transformers import (ViTImageProcessor, ViTForImageClassification, TrainingArguments,
	Trainer, get_linear_schedule_with_warmup)
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score
from torch.optim import AdamW
# import os
import pandas as pd
#-----------------------------------------------------------------------------------------#

class ImageDatasetProcessor:
	def __init__(self, data_file, train_ratio, val_size, test_size, batch_size):
		# Load dataset
		self.dataset = load_dataset('csv', data_files=data_file, split='train')
		# Split dataset
		self.train_ds, self.val_ds, self.test_ds = self.split_dataset(train_ratio, val_size, test_size)
		# Define label mappings
		self.define_label_mappings()
		# Initialize image processor
		self.processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
		# Define transformations
		self.define_transformations()
		# Set transformations
		self.set_transforms()
		# Create DataLoaders
		self.create_dataloaders(batch_size)

	def split_dataset(self, train_ratio, val_size, test_size):
		train_testvalid = self.dataset.train_test_split(test_size=(1 - train_ratio))
		test_valid = train_testvalid['test'].train_test_split(test_size=test_size / (test_size + val_size))
		return train_testvalid['train'], test_valid['train'], test_valid['test']

	def define_label_mappings(self):
		self.unique_labels = sorted(set(self.train_ds['label']))
		# print('Unique labels:', self.unique_labels)
		self.label2id = {label: id for id, label in enumerate(self.unique_labels)}
		# print('Label to ID mapping:', self.label2id)
		self.id2label = {id: label for label, id in self.label2id.items()}
		# print('ID to Label mapping:', self.id2label)

	def define_transformations(self):
		image_mean, image_std = self.processor.image_mean, self.processor.image_std
		size = self.processor.size["height"]
		normalize = Normalize(mean=image_mean, std=image_std)
		self._train_transforms = Compose([RandomResizedCrop(size), RandomHorizontalFlip(), ToTensor(),
								 normalize])
		self._val_test_transforms = Compose([Resize(size), CenterCrop(size), ToTensor(), normalize])

	def train_transforms(self, examples):
		return self._apply_transforms(examples, self._train_transforms)

	def val_transforms(self, examples):
		return self._apply_transforms(examples, self._val_test_transforms)

	def test_transforms(self, examples):
		return self._apply_transforms(examples, self._val_test_transforms)

	def _apply_transforms(self, examples, transforms):
		transformed_examples = []
		for path in examples['image_path']:
			image = Image.open(path).convert('RGB')
			transformed_image = transforms(image)
			transformed_examples.append(transformed_image)
		examples['pixel_values'] = transformed_examples
		return examples

	def set_transforms(self):
		self.train_ds.set_transform(self.train_transforms)
		self.val_ds.set_transform(self.val_transforms)
		self.test_ds.set_transform(self.val_transforms)

	def create_dataloaders(self, batch_size):
		self.train_loader = DataLoader(self.train_ds, batch_size=batch_size, shuffle=True,
							collate_fn=self.collate_fn)
		self.val_loader = DataLoader(self.val_ds, batch_size=batch_size, shuffle=False,
							collate_fn=self.collate_fn)
		self.test_loader = DataLoader(self.test_ds, batch_size=batch_size, shuffle=False,
							collate_fn=self.collate_fn)

	def collate_fn(self, examples):
		pixel_values = torch.stack([example["pixel_values"] for example in examples])
		labels = torch.tensor([example["label2id"] for example in examples])
		return {"pixel_values": pixel_values, "labels": labels}

#-----------------------------------------------------------------------------------------#

class ViTTrainer:
	def __init__(self, train_ds, val_ds, test_ds, id2label, label2id, processor, num_epochs, learning_rate,
				 train_batch_size, eval_batch_size, save_checkpoint, save_steps, eval_steps, weight_decay,
				 save_log, save_confusion_matrix, save_trained_model):
		
		self.device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
		print(f'Using device: {self.device}')

		self.train_ds = train_ds
		self.val_ds = val_ds
		self.test_ds = test_ds
		self.processor = processor
		self.save_confusion_matrix = save_confusion_matrix
		self.id2label = id2label  
		self.label2id = label2id  
		self.save_trained_model = save_trained_model

		self.model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k',
					 id2label=id2label, label2id=label2id).to(self.device)

		self.args = TrainingArguments(
			save_checkpoint,  
			save_steps=save_steps,
			eval_steps=eval_steps,
			logging_steps=10,
			disable_tqdm=True,
			evaluation_strategy="steps",
			learning_rate=learning_rate,
			per_device_train_batch_size=train_batch_size,
			per_device_eval_batch_size=eval_batch_size,
			num_train_epochs=num_epochs,
			weight_decay=weight_decay,  
			load_best_model_at_end=True,
			metric_for_best_model="accuracy",
			report_to='tensorboard',
			remove_unused_columns=False,
			logging_dir=save_log
		)

		# Custom optimizer
		optimizer = AdamW(self.model.parameters(), lr=learning_rate)

		# Learning rate scheduler
		lr_scheduler = get_linear_schedule_with_warmup(
			optimizer,
			num_warmup_steps=0,
			num_training_steps=len(self.train_ds) * num_epochs
		)

		self.trainer = Trainer(
			model=self.model,
			args=self.args,
			train_dataset=self.train_ds,
			eval_dataset=self.val_ds,
			data_collator=self.collate_fn,
			compute_metrics=self.compute_metrics,
			tokenizer=self.processor,
			optimizers=(optimizer, lr_scheduler)
		)

	def collate_fn(self, examples):
		pixel_values = torch.stack([example["pixel_values"] for example in examples])
		labels = torch.tensor([example["label2id"] for example in examples])
		return {"pixel_values": pixel_values, "labels": labels}

	def compute_metrics(self, eval_pred):
		predictions, labels = eval_pred
		predictions = np.argmax(predictions, axis=1)
		return dict(accuracy=accuracy_score(predictions, labels))

	def run_training(self):
		self.trainer.train()
		self.save_best_model()

	def evaluate_model(self):
		return self.trainer.evaluate()

	def predict(self):
		outputs = self.trainer.predict(self.test_ds)
		U.save_callback(outputs, self.save_confusion_matrix)
		print(outputs.metrics)
	
	def save_best_model(self):
		self.model.save_pretrained(self.save_trained_model)
		print(f"Best model saved to {self.save_trained_model}")

#-----------------------------------------------------------------------------------------#

class ViTInference:
	def __init__(self, model_path, csv_path, device='cpu'):
		self.device = device
		self.csv_path = csv_path  
		self.dataset = load_dataset('csv', data_files=csv_path)
		self.define_label_mappings()  
		self.model_path = model_path
		self.processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
		self.model = ViTForImageClassification.from_pretrained(model_path).to(self.device)

	def define_label_mappings(self):
		data = pd.read_csv(self.csv_path)
		self.unique_labels = sorted(set(data['label']))
		self.label2id = {label: id for id, label in enumerate(self.unique_labels)}
		self.id2label = {id: label for label, id in self.label2id.items()}

	def load_best_model(self):
		self.model = ViTForImageClassification.from_pretrained(self.model_path)
		self.model.to(self.device)

	def predict_image(self, image_path):
		image = Image.open(image_path)
		processed_image = self.processor(images=image, return_tensors="pt").to(self.device)
		with torch.no_grad():
			outputs = self.model(**processed_image)
			logits = outputs.logits
		probs = logits.softmax(-1)
		predicted_label_idx = probs.argmax(-1).item()
		predicted_label = self.id2label[predicted_label_idx]
		return predicted_label

#-----------------------------------------------------------------------------------------#
