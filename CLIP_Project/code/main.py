from load_images import load_images_with_augmentations, load_images_without_augmentations
from clip_model import load_clip_model
from train_model import train_model
from evaluate_model import evaluate_model
from config import criterion, num_epochs, learning_rate as lr, bad_seed_descriptors, good_seed_descriptors, batch_size
from plot_confusion_matrix import create_confusion_matrix
from utils import setup_output_dir
from plot_training_curve import plot_training_curve

# load CLIP model
model, preprocess, device = load_clip_model()

# setup output directory for saving plots
setup_output_dir("CLIP_Project/Plots")

# load training set with augmentations applied
training_path = 'CLIP_Project/data/train'
training_set = load_images_with_augmentations(training_path, batch_size)

# train model on our training set using seed descriptors
train_losses = train_model(model, device, training_set, criterion, num_epochs, lr, bad_seed_descriptors, good_seed_descriptors)

# plot training curve
plot_training_curve(train_losses)

# load test set without augmentations
test_path = 'CLIP_Project/data/test'
test_set = load_images_without_augmentations(test_path, batch_size)

# evaluate model on test set
avg_loss, accuracy, test_labels, test_preds = evaluate_model(model, test_set, criterion, 
                                                    device, bad_seed_descriptors, good_seed_descriptors)

# create and save confusion matrix for test set
create_confusion_matrix(test_labels, test_preds, 'Test')

# test model on Normal Room Lighting conditions
NormalRoomLight_path = 'CLIP_Project/data/NormalRoomLighting'

NormalRoomLight_set = load_images_without_augmentations(NormalRoomLight_path, batch_size)

avg_loss, accuracy, NRL_labels, NRL_preds = evaluate_model(model, NormalRoomLight_set, criterion, 
                                                    device, bad_seed_descriptors, good_seed_descriptors)

create_confusion_matrix(NRL_labels, NRL_preds, 'NormalRoomLighting')

# test model on Lightbox lighting conditions
LightBox_path = 'CLIP_Project/data/LightBox'

LightBox_set = load_images_without_augmentations(LightBox_path, batch_size)

avg_loss, accuracy, LightBox_labels, LightBox_preds = evaluate_model(model, LightBox_set, criterion, 
                                                    device, bad_seed_descriptors, good_seed_descriptors)

create_confusion_matrix(LightBox_labels, LightBox_preds, 'LightBox')