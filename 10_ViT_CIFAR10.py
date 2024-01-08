#-----------------------------------------------------------------------------------------#
import sys
sys.path.append('./Libs') 
import utilities as U
import class_transformer as CT
#-----------------------------------------------------------------------------------------#
import os
#-----------------------------------------------------------------------------------------#

tabular_file = 'tabular_dataset/manifest.csv'
train_ratio = 0.70; val_size = 0.25; test_size = 0.05
batch_size = 128
num_epochs = 1
learning_rate = 1e-1
train_batch_size = batch_size
eval_batch_size = batch_size
save_checkpoint = 'save_checkpoint'
save_steps = 50
eval_steps = 50
weight_decay = 1e-3
save_log = './logs'
save_confusion_matrix = os.path.join('save_trained_model', 'outputs.pkl')

#-----------------------------------------------------------------------------------------#

processor = CT.ImageDatasetProcessor(tabular_file, train_ratio, val_size, test_size, batch_size)
vit_trainer = CT.ViTTrainer(processor.train_ds, processor.val_ds, processor.test_ds,
                            processor.id2label, processor.label2id, processor.processor,
							num_epochs, learning_rate, train_batch_size,
							eval_batch_size, save_checkpoint, save_steps,
							eval_steps, weight_decay, save_log,
                            save_confusion_matrix)
vit_trainer.run_training()
vit_trainer.predict()

#-----------------------------------------------------------------------------------------#

# Save the trained model and log metrics
train_results = vit_trainer.trainer.train()  
vit_trainer.trainer.save_model()  
vit_trainer.trainer.log_metrics("train", train_results.metrics)
vit_trainer.trainer.save_metrics("train", train_results.metrics)
vit_trainer.trainer.save_state()

# Evaluate the model on the validation set and log metrics
metrics = vit_trainer.trainer.evaluate(processor.val_ds)  
vit_trainer.trainer.log_metrics("eval", metrics)
vit_trainer.trainer.save_metrics("eval", metrics)

#-----------------------------------------------------------------------------------------#

outputs = U.load_callback(save_confusion_matrix)
y_true = outputs.label_ids
y_pred = outputs.predictions.argmax(1)
class_names = processor.unique_labels
U.plot_confusion_matrix(y_true, y_pred, class_names)

#-----------------------------------------------------------------------------------------#