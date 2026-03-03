# CLIP Seed Quality Classifier

## Project Description
This project involves fine-tuning OpenAI's CLIP model on a datset of oil palm seeds to differentiate between healthy and unhealhty ones.

## CLIP Model Overview
CLIP is a vison-langauge model (vlm) developed by OpenAI that can utilise both images and text in it's learning process.

For our project uses we Fine-tine CLIP on images of germinated oil palm seeds and short textual descriptions, allowing it to learn in distinguishing between the two seed classes (healthy and unhealthy).

## Training Process
We use one dataset of images to train our model then we test it on 3 different datasets, each of which was captured under different lighting conditions, we do this to test the generalizability of our model and ensure it does not overfit on our training images.

## Model Configuration
The textual descriptions that CLIP will use for training are inside the 'config.py' file, changing these affects how the model learns the features.

We can also change more learning parameters from inside this file like the number of epochs or the learning rate of the model.
