from torchvision import datasets
from torchvision.transforms import v2
import torch
from torch.utils.data import DataLoader

def load_images_with_augmentations(image_folder, batch_size):
    # transforms with image augmentations for training set
    transform = v2.Compose([
        v2.Resize(256),
        v2.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.75, 1.33)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.3),
        v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])

    training_set = datasets.ImageFolder(root=image_folder, transform=transform)
    
    # shuffle to ensure random batches
    training_loader = DataLoader(training_set, batch_size, shuffle=True)
    
    return training_loader

def load_images_without_augmentations(image_folder, batch_size):
    # transforms without augmentations for test sets
    transform = v2.Compose([        
        v2.Resize((224, 224), interpolation=v2.InterpolationMode.BICUBIC),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale = True),
        v2.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))       
    ])

    test_set = datasets.ImageFolder(root=image_folder, transform=transform)
    
    test_loader = DataLoader(test_set, batch_size, shuffle=False)
    
    return test_loader