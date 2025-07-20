# CLIP Seed Quality Classifier

This project involves fine-tuning OpenAI's CLIP model on a datset of oil palm seeds so it can learn to differentiate between healthy and unhealhty ones.

CLIP uses images combined with text to learn features of each class.

To streamline the process we use ChatGPT to automatically generate prompts describing each class of seed (good and bad).

You will need to use your own OpenAI API key to be able to automatically generate the prompts, you can use it on **Line 1** of the code

Alternatively, you can skip the API call by hardcoding the 'bad_seed_prompts' and 'good_seed_prompts' lists directly in the code. (ensure they are defined in that order)

# Modifying Model Parameters
In the code, you can customize the model and training parameters by modifying the 'run_params' dictionary, the dictionary is located on **Line 63** in the code.
