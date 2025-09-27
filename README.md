# CLIP Seed Quality Classifier

This project involves fine-tuning OpenAI's CLIP model on a datset of oil palm seeds to differentiate between healthy and unhealhty ones.

CLIP uses images combined with textual descriptions to learn features of each class of seed(healthy and unhealthy).

We use one dataset to train our model then we test it on 3 different datasets, each of which was captured under different lighting conditions, we do this to test the generalizability of our model and ensure it does not overfit on our training images.

The textual descriptions that CLIP will use for training are inside the 'config.py' file, changing these affects how the model learns the features

We can also change more learning parameters from inside this file like the number of epochs or the learning rate of the model.
