import torch.nn as nn

# define the criterion that the model will optimize
criterion = nn.CrossEntropyLoss()

# number of cycles that the model will train for
num_epochs = 10

# learning rate that the model will use during training
learning_rate = 1e-5

# number of images in each batch
batch_size = 32

# define the text descriptions for each class of seed
bad_seed_descriptors = ['a photo of a bad germinated oil palm seed']
good_seed_descriptors = ['a photo of a good germinated oil palm seed']