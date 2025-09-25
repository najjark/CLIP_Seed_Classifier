# CLIP Seed Quality Classifier

This project involves fine-tuning OpenAI's CLIP model on a datset of oil palm seeds to differentiate between healthy and unhealhty ones.

CLIP uses images combined with text to learn features of each class of seed(healthy and unhealthy).

The textual descriptions that CLIP will use for training are inside the 'config.py' file, changing these affects how the model learns the features

You can also change more of the training parameters like the number of epochs and batch size from the 'config.py' file.