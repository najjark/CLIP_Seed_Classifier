import clip
import torch

def load_clip_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    
    # Ensure the model is in float32
    clip_model = clip_model.float()

    return clip_model, preprocess, device