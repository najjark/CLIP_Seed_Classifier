my_api_key = "OPENAI_API_KEY"

import clip
import torch
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader, Dataset, random_split
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import numpy as np
import os
import json
from datetime import datetime
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
from openai import OpenAI
import re
import math
from sklearn.metrics import precision_recall_curve
from torch.utils.data import TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create a run directory for storing results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = f"tests/test_{timestamp}"
os.makedirs(run_dir, exist_ok=True)

# Function to extract transformation names
def get_transform_names(transform):
    return [t.__class__.__name__ for t in transform.transforms]

# Augmentations used for training
training_transform = v2.Compose([
    v2.Resize((256, 256), interpolation=v2.InterpolationMode.BICUBIC),
    v2.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.75, 1.33)),
    v2.RandomRotation(degrees=30),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.3),
    v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    v2.RandomAffine(degrees=15,translate=(0.1, 0.1),scale=(0.9, 1.1),shear=10),
    v2.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
    v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])

testing_transform = v2.Compose([
    v2.Resize((224, 224), interpolation=v2.InterpolationMode.BICUBIC),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale = True),
    v2.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])

# Store run parameters
run_params = {
    "timestamp": timestamp,
    "model": "ViT-B/32",
    "num_descriptors": 3,
    "batch_size": 64,
    "num_epochs": 1,
    "learning_rate": 1e-5,
    "optimizer": "AdamW",
    "scheduler": "CosineAnnealingLR",
    "initial_unfrozen_layers": 0,
    "target_unfrozen_layers": 0,
    "weight_decay": 1e-5,
    "betas": [0.9, 0.99],
    "data_augmentation": get_transform_names(training_transform),   
}

# API key
client = OpenAI(
    api_key=my_api_key
)

# Function to generate prmopts for each class of seeds
def generate_descriptors(client, seed_type, num_descriptors):
    prompt = (
        f"""
        List {num_descriptors} specific features for distinguishing a {seed_type} germinated oil palm seed.
        
        Each feature should be:
        - Visual Only (e.g., color, shape or other visual features, ignore non-visual features like odor or feel)
        - Concise (1 short phrase or sentence)
        - Ensure the descriptors are very specific to the class of seed (i.e. bad or good)
        - Do not describe the seeds with the abscence of a feature
        - Do not make a comparison to the other class of seeds

        Provide the features in a numbered list format.
        Do not include any special characters in your response.
        """ 
    )
    
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are an expert in seed morphology and classification."},
            {"role": "user", "content": prompt},
        ],
        model="gpt-4o-mini",
    )
    return response.choices[0].message.content

# Process the generated text into a list format
def process_descriptors(raw_text):
    descriptors = raw_text.strip().split("\n")
    return [desc.split(". ", 1)[1].strip() for desc in descriptors if ". " in desc]

# Generate descriptors for bad and good seeds
bad_descriptions = generate_descriptors(client, "bad", run_params["num_descriptors"])
good_descriptions = generate_descriptors(client, "good", run_params["num_descriptors"])

# Process descriptors into lists
bad_seed_prompts = process_descriptors(bad_descriptions)
good_seed_prompts = process_descriptors(good_descriptions)

def sanitize_filename(name):
    # Replace invalid characters with underscores
    return re.sub(r'[\/:*?"<>|]', '_', name).strip()

def select_high_similarity_samples(model, data_source, text_features, bad_seed_prompts, good_seed_prompts, num_bad=0, num_good=0):
    model.eval()
    bad_samples = []
    good_samples = []

    if isinstance(data_source, torch.utils.data.DataLoader):
        iterator = iter(data_source)
    else:
        data_source = torch.utils.data.DataLoader(data_source, batch_size=1, shuffle=True)
        iterator = iter(data_source)
    
    # Process dataset to find correctly classified samples
    with torch.no_grad():
        for images, labels in iterator:
            images, labels = images.to(device), labels.to(device)
            
            # Get image features and similarities
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            similarities = image_features @ text_features.T
            
            # Calculate average similarities for bad and good classes
            bad_sim = similarities[:, :len(bad_seed_prompts)].mean(dim=1)
            good_sim = similarities[:, len(bad_seed_prompts):].mean(dim=1)
            
            # Determine predicted class
            predicted = (good_sim > bad_sim).long()
            
            for i in range(images.shape[0]):
                label = labels[i].item()
                img = images[i]
                pred = predicted[i].item()
                
                if label == 0 and pred == 0:
                    bad_samples.append((bad_sim[i].item(), img, label))
                elif label == 1 and pred == 1:
                    good_samples.append((good_sim[i].item(), img, label))

    bad_samples.sort(reverse=True, key=lambda x: x[0])
    good_samples.sort(reverse=True, key=lambda x: x[0])
    
    selected_samples = bad_samples[:num_bad] + good_samples[:num_good]

    if not selected_samples:
        print("Warning: No correctly classified seeds found.")
        return torch.tensor([]).to(device), torch.tensor([]).to(device)
    
    selected_images = torch.stack([x[1] for x in selected_samples])
    selected_labels = torch.tensor([x[2] for x in selected_samples])
    
    return selected_images, selected_labels

def compute_scorecam(model, image, text_features, bad_seed_prompts, good_seed_prompts, target_class=None, layer_name='visual.conv1'):
    model.eval()
    image = image.to(device)
    
    # Validate input image
    if image.shape[0] != 1 or len(image.shape) != 4 or image.shape[2:] != (224, 224):
        raise ValueError(f"Expected image shape [1, C, 224, 224], got {image.shape}")
    
    # Get target layer
    try:
        target_layer = dict(model.named_modules())[layer_name]
    except KeyError:
        raise ValueError(f"Layer {layer_name} not found in model")
    
    # Hook for capturing activations
    activations = None
    def forward_hook(module, input, output):
        nonlocal activations
        activations = output.detach()
    
    handle = target_layer.register_forward_hook(forward_hook)
    
    try:
        with torch.no_grad():
            image_features = model.encode_image(image)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Calculate similarities to all prompts
            similarities = image_features @ text_features.T
            
            # Determine target class if not specified
            bad_avg = similarities[:, :len(bad_seed_prompts)].mean(dim=1)
            good_avg = similarities[:, len(bad_seed_prompts):].mean(dim=1)
            logits = torch.stack([bad_avg, good_avg], dim=1)
            if target_class is None:
                target_class = torch.argmax(logits, dim=1).item()
        
        # Validate activations
        if activations is None or activations.numel() == 0:
            raise ValueError(f"No valid activations captured from layer {layer_name}")            
        
        # After getting activations, filter channels by variance
        act = activations[0]
        channel_variances = act.var(dim=(1,2))
        # Use most variable channels
        top_k_channels = torch.topk(channel_variances, k=20).indices
        # Only process these channels
        act = act[top_k_channels]

        mask = F.interpolate(act.unsqueeze(0), 
                            size=image.shape[-2:], 
                            mode='bilinear',
                            align_corners=False).squeeze(0)

        all_prompts = bad_seed_prompts + good_seed_prompts
        prompt_heatmaps = []
        
        for i, prompt in enumerate(all_prompts):
            scores = []
            
            for j in range(mask.shape[0]):
                # Apply activation mask to input image
                masked_image = image * mask[j].unsqueeze(0)

                with torch.no_grad():
                    masked_features = model.encode_image(masked_image)
                    masked_features = masked_features / masked_features.norm(dim=-1, keepdim=True)

                    prompt_feature = text_features[i].unsqueeze(0)
                    score = (masked_features @ prompt_feature.T).item()
                
                scores.append(score)
            
            # Convert scores to tensor and normalize
            scores = torch.tensor(scores, device=device)
            scores = F.softmax(scores, dim=0)
            
            # Weight activations by normalized scores
            cam = (act * scores.view(-1, 1, 1)).sum(dim=0)
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), 
                               size=image.shape[-2:], 
                               mode='bilinear',
                               align_corners=False).squeeze().cpu().numpy()
            
            prompt_heatmaps.append(cam)
        
        # For aggregated heatmap, weight by original prompt similarities
        prompt_weights = F.softmax(similarities[0], dim=0).cpu().numpy()
        aggregated_heatmap = np.sum([hm * weight for hm, weight in zip(prompt_heatmaps, prompt_weights)], axis=0)
        aggregated_heatmap = (aggregated_heatmap - aggregated_heatmap.min()) / (aggregated_heatmap.max() - aggregated_heatmap.min() + 1e-8)
        
        return aggregated_heatmap, prompt_heatmaps, all_prompts
    
    finally:
        handle.remove()        
             
def visualize_heatmap(original_image, heatmap, alpha=0.5, cmap='jet', title=None, save_path=None, show=True):
    plt.figure(figsize=(10, 10))
    plt.imshow(original_image)
    plt.imshow(heatmap, cmap=cmap, alpha=alpha)
    plt.axis('off')
    if title:
        plt.title(title)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    
    if show:
        plt.show()
    
    plt.close()
    
def generate_and_visualize_scorecam(model, data_source, text_features, 
                                  bad_seed_prompts, good_seed_prompts,
                                  num_bad=2, num_good=2, subset_dir='heatmaps', 
                                  run_dir=None, **kwargs):
    model.eval()
    results = []

    if run_dir is None:
        raise ValueError("run_dir must be provided to save heatmaps")
    heatmap_dir = os.path.join(run_dir, subset_dir)
    os.makedirs(heatmap_dir, exist_ok=True)
    
    # Select correctly classified samples
    selected_images, selected_labels = select_high_similarity_samples(
        model, data_source, text_features, bad_seed_prompts, good_seed_prompts, 
        num_bad=num_bad, num_good=num_good
    )
    
    if selected_images.numel() == 0:
        print("No correctly classified seeds available to process.")
        return results
    
    compute_scorecam_kwargs = {k: v for k, v in kwargs.items() if k in ['target_class', 'layer_start', 'layer_end', 'discard_ratio']}

    # Process each selected image
    for i in range(selected_images.shape[0]):
        single_image = selected_images[i:i+1]
        label = selected_labels[i].item()

        # Compute heatmap
        aggregated_heatmap, _, _ = compute_scorecam(
            model, single_image, text_features, bad_seed_prompts, good_seed_prompts,
            target_class=label, **compute_scorecam_kwargs
        )

        # Prepare original image
        original_image = single_image.squeeze().permute(1, 2, 0).detach().cpu().numpy()
        original_image = (original_image * np.array([0.26862954, 0.26130258, 0.27577711]) + 
                         np.array([0.48145466, 0.4578275, 0.40821073]))
        original_image = np.clip(original_image, 0, 1)
        
        # Save heatmap
        save_path = os.path.join(heatmap_dir, f"heatmap_img{i}_label{'GOOD' if label == 1 else 'BAD'}_aggregated.png")
        title = kwargs.get('title', f"Aggregated Score-CAM Heatmap (Label: {'GOOD' if label == 1 else 'BAD'})")
        
        visualize_heatmap(
            original_image,
            aggregated_heatmap,
            alpha=kwargs.get('alpha', 0.5),
            cmap=kwargs.get('cmap', 'jet'),
            title=title,
            save_path=save_path,
            show=kwargs.get('show', True)
        )
        
        results.append((aggregated_heatmap, original_image))
    
    return results

def analyze_prompt_influence(model, image, text_features, bad_seed_prompts, good_seed_prompts):
    model.eval()
    image = image.to(device)
    
    with torch.no_grad():
        # Get image features
        image_features = model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Compute similarities with all prompts
        similarities = image_features @ text_features.T
        
        # Calculate average similarities per class
        bad_similarities = similarities[:, :len(bad_seed_prompts)]
        good_similarities = similarities[:, len(bad_seed_prompts):]
        
        # Get individual prompt contributions
        bad_prompt_scores = {prompt: score.item() for prompt, score in zip(bad_seed_prompts, bad_similarities[0])}
        good_prompt_scores = {prompt: score.item() for prompt, score in zip(good_seed_prompts, good_similarities[0])}
        
        # Determine predicted class
        bad_avg = bad_similarities.mean().item()
        good_avg = good_similarities.mean().item()
        predicted_class = 1 if good_avg > bad_avg else 0
        
        return {
            'bad_prompts': bad_prompt_scores,
            'good_prompts': good_prompt_scores,
            'bad_avg': bad_avg,
            'good_avg': good_avg,
            'predicted_class': predicted_class
        }

def compute_prompt_effectiveness(model, data_loader, text_features, bad_seed_prompts, good_seed_prompts, contrast_factor=1.3, verbose=False):
    model.eval()
    device = next(model.parameters()).device

    prompt_stats = {
        prompt: {
            'bad': {'sum': 0.0, 'sum_sq': 0.0, 'count': 0, 'values': []},
            'good': {'sum': 0.0, 'sum_sq': 0.0, 'count': 0, 'values': []}
        } 
        for prompt in bad_seed_prompts + good_seed_prompts
    }

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            similarities = (image_features @ text_features.T).cpu().numpy()
            
            for i, prompt in enumerate(bad_seed_prompts + good_seed_prompts):
                for img_idx in range(len(labels)):
                    label = labels[img_idx].item()
                    class_key = 'bad' if label == 0 else 'good'
                    sim = similarities[img_idx, i]

                    sim = np.tanh(sim * contrast_factor)
                    
                    stats = prompt_stats[prompt][class_key]
                    stats['sum'] += sim
                    stats['sum_sq'] += sim ** 2
                    stats['count'] += 1
                    stats['values'].append(sim)

    prompt_metrics = {}
    for prompt in bad_seed_prompts + good_seed_prompts:
        bad_stats = prompt_stats[prompt]['bad']
        good_stats = prompt_stats[prompt]['good']

        bad_mean = bad_stats['sum'] / bad_stats['count'] if bad_stats['count'] else 0.0
        good_mean = good_stats['sum'] / good_stats['count'] if good_stats['count'] else 0.0
        
        bad_std = math.sqrt(
            (bad_stats['sum_sq'] - bad_stats['sum']**2 / bad_stats['count']) / 
            max(1, (bad_stats['count'] - 1))
        ) if bad_stats['count'] else 0.0
        
        good_std = math.sqrt(
            (good_stats['sum_sq'] - good_stats['sum']**2 / good_stats['count']) / 
            max(1, (good_stats['count'] - 1))
        ) if good_stats['count'] else 0.0

        target_class = 'bad' if bad_mean > good_mean else 'good'
        pos_sims = bad_stats['values'] if target_class == 'bad' else good_stats['values']
        neg_sims = good_stats['values'] if target_class == 'bad' else bad_stats['values']
        pos_label = 0 if target_class == 'bad' else 1

        all_sims = pos_sims + neg_sims
        all_labels = [1] * len(pos_sims) + [0] * len(neg_sims)

        if not pos_sims or not neg_sims:
            optimal_threshold = (bad_mean + good_mean) / 2
        else:
            try:
                precisions, recalls, thresholds = precision_recall_curve(all_labels, all_sims)
                f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
                optimal_idx = np.argmax(f1_scores)
                optimal_threshold = thresholds[optimal_idx]

                if np.isinf(optimal_threshold):
                    max_neg = max(neg_sims)
                    min_pos = min(pos_sims)
                    optimal_threshold = max_neg + (min_pos - max_neg) * 0.3
            except ValueError:
                optimal_threshold = (bad_mean + good_mean) / 2

        preds = [1 if sim > optimal_threshold else 0 for sim in all_sims]
        tn, fp, fn, tp = confusion_matrix(all_labels, preds).ravel()

        eps = 1e-10
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * (precision * recall) / (precision + recall + eps)

        def cohens_d(pos, neg):
            n_pos, n_neg = len(pos), len(neg)
            if n_pos < 2 or n_neg < 2:
                return 0.0
            mean_pos = np.mean(pos)
            mean_neg = np.mean(neg)
            pooled_std = np.sqrt((np.var(pos, ddof=1) + np.var(neg, ddof=1)) / 2)
            return (mean_pos - mean_neg) / pooled_std
        
        effect_size = cohens_d(pos_sims, neg_sims)

        prompt_metrics[prompt] = {
            'target_class': target_class,
            'threshold': float(optimal_threshold),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'avg_similarity_bad': float(bad_mean),
            'avg_similarity_good': float(good_mean),
            'std_similarity_bad': float(bad_std),
            'std_similarity_good': float(good_std),
            'cohens_d': float(effect_size),
            'true_pos': int(tp),
            'false_pos': int(fp),
            'true_neg': int(tn),
            'false_neg': int(fn),
            'support_pos': len(pos_sims),
            'support_neg': len(neg_sims)
        }
        
        if verbose:
            print(f"\nPrompt: {prompt}")
            print(f"Target class: {target_class} (Cohen's d: {effect_size:.2f})")
            print(f"Threshold: {optimal_threshold:.4f}")
            print(f"Similarity - Bad: μ={bad_mean:.4f} ± {bad_std:.4f}")
            print(f"Similarity - Good: μ={good_mean:.4f} ± {good_std:.4f}")
            print(f"Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
            print(f"Metrics: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
    
    return prompt_metrics

def visualize_prompt_effectiveness(prompt_results, dataset_name="Dataset", save_path=None):
    bad_prompts = [p for p in prompt_results if prompt_results[p]['target_class'] == 'bad']
    good_prompts = [p for p in prompt_results if prompt_results[p]['target_class'] == 'good']

    bad_prompts = sorted(bad_prompts, key=lambda p: prompt_results[p]['f1'], reverse=True)
    good_prompts = sorted(good_prompts, key=lambda p: prompt_results[p]['f1'], reverse=True)

    def plot_grouped_metrics(ax, prompts, title):
        x = np.arange(len(prompts))
        bar_width = 0.25

        precision = [prompt_results[p]['precision'] for p in prompts]
        recall = [prompt_results[p]['recall'] for p in prompts]
        f1 = [prompt_results[p]['f1'] for p in prompts]

        ax.bar(x - bar_width, precision, width=bar_width, color='dodgerblue', alpha=0.8, label='Precision')
        ax.bar(x, recall, width=bar_width, color='orange', alpha=0.8, label='Recall')
        ax.bar(x + bar_width, f1, width=bar_width, color='mediumseagreen', alpha=0.8, label='F1 Score')

        for i in range(len(prompts)):
            ax.text(x[i] - bar_width, precision[i] + 0.02, f"{precision[i]:.2f}", ha='center', va='bottom', fontsize=8)
            ax.text(x[i], recall[i] + 0.02, f"{recall[i]:.2f}", ha='center', va='bottom', fontsize=8)
            ax.text(x[i] + bar_width, f1[i] + 0.02, f"{f1[i]:.2f}", ha='center', va='bottom', fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(prompts, rotation=45, ha='right', fontsize=8)
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', linestyle=':', alpha=0.6)
        ax.set_ylabel('Score')
        ax.set_title(title, fontsize=12, fontweight='bold')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6), sharey=True)

    plot_grouped_metrics(ax1, bad_prompts, 'Bad Seed Prompts Effectiveness')
    plot_grouped_metrics(ax2, good_prompts, 'Good Seed Prompts Effectiveness')

    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False, fontsize=10)

    fig.suptitle(f'Prompt Effectiveness Comparison — {dataset_name}', fontsize=16, fontweight='bold')

    plt.tight_layout(rect=[0, 0.05, 1, 0.93])

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()

    plt.close()

def update_metrics(metrics_dict, predictions, labels, target_class):
    for pred, label in zip(predictions, labels):
        label = label.item()
        
        if pred == target_class:
            if label == target_class:
                metrics_dict['true_pos'] += 1
            else:
                metrics_dict['false_pos'] += 1
        else:
            if label == target_class:
                metrics_dict['false_neg'] += 1
            else:
                metrics_dict['true_neg'] += 1

def calculate_final_metrics(metrics_dict, avg_similarity, std_similarity):
    tp = metrics_dict['true_pos']
    fp = metrics_dict['false_pos']
    fn = metrics_dict['false_neg']
    tn = metrics_dict['true_neg']
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'avg_similarity': avg_similarity,
        'std_similarity': std_similarity,
        'true_pos': tp,
        'false_pos': fp,
        'true_neg': tn,
        'false_neg': fn
    }

def visualize_average_prompt_similarity(prompt_results_dict, bad_prompts, good_prompts, run_dir):
    datasets = list(prompt_results_dict.keys())

    def plot_dataset(dataset, prompt_results, save_path):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
        bar_width = 0.35

        n_bad = len(bad_prompts)
        index_bad = np.arange(n_bad)
        
        bad_sims_bad = [abs(prompt_results[prompt]['avg_similarity_bad']) for prompt in bad_prompts]
        good_sims_bad = [abs(prompt_results[prompt]['avg_similarity_good']) for prompt in bad_prompts]

        ax1.bar(index_bad - bar_width/2, bad_sims_bad, bar_width, label='Bad Seeds', color='red', alpha=0.8)
        ax1.bar(index_bad + bar_width/2, good_sims_bad, bar_width, label='Good Seeds', color='green', alpha=0.8)
        ax1.set_title('Bad Seed Prompts')
        ax1.set_xticks(index_bad)
        ax1.set_xticklabels(bad_prompts, rotation=45, ha='right')
        ax1.set_ylabel('Average Similarity')
        ax1.grid(True, axis='y', linestyle='--', alpha=0.6)
        ax1.legend()

        n_good = len(good_prompts)
        index_good = np.arange(n_good)
        
        bad_sims_good = [abs(prompt_results[prompt]['avg_similarity_bad']) for prompt in good_prompts]
        good_sims_good = [abs(prompt_results[prompt]['avg_similarity_good']) for prompt in good_prompts]

        ax2.bar(index_good - bar_width/2, bad_sims_good, bar_width, label='Bad Seeds', color='red', alpha=0.8)
        ax2.bar(index_good + bar_width/2, good_sims_good, bar_width, label='Good Seeds', color='green', alpha=0.8)
        ax2.set_title('Good Seed Prompts')
        ax2.set_xticks(index_good)
        ax1.set_ylabel('Average Similarity')
        ax2.set_xticklabels(good_prompts, rotation=45, ha='right')
        ax2.grid(True, axis='y', linestyle='--', alpha=0.6)
        ax2.legend()

        fig.suptitle(f'Average Prompt Similarity for {dataset} Set', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    for dataset in datasets:
        prompt_results = prompt_results_dict[dataset]
        save_path = f"{run_dir}/prompt_similarity_{dataset.lower()}.png"
        plot_dataset(dataset, prompt_results, save_path)



def show_augmented_examples(dataset, transform, save_path=None):
    import matplotlib.pyplot as plt

    # Find one BAD (label 0) and one GOOD (label 1) sample
    bad_img, good_img = None, None
    for img, label in dataset:
        if label == 0 and bad_img is None:
            bad_img = img
        elif label == 1 and good_img is None:
            good_img = img
        if bad_img and good_img:
            break

    if bad_img is None or good_img is None:
        print("Could not find both BAD and GOOD samples in the dataset.")
        return

    examples = [(bad_img, "BAD"), (good_img, "GOOD")]

    fig, axes = plt.subplots(2, 2, figsize=(6, 6))
    fig.suptitle("Original vs Augmented: BAD and GOOD Examples", fontsize=14)

    for i, (image, label_name) in enumerate(examples):
        # Original
        axes[i, 0].imshow(image)
        axes[i, 0].set_title(f"Original ({label_name})")
        axes[i, 0].axis('off')

        # Augmented
        augmented = transform(image)
        augmented_np = augmented.permute(1, 2, 0).numpy()
        augmented_np = (
            augmented_np * np.array([0.26862954, 0.26130258, 0.27577711]) +
            np.array([0.48145466, 0.4578275, 0.40821073])
        )
        augmented_np = np.clip(augmented_np, 0, 1)

        axes[i, 1].imshow(augmented_np)
        axes[i, 1].set_title(f"Augmented ({label_name})")
        axes[i, 1].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')






# Add prompts to run parameters
run_params["bad_seed_prompts"] = bad_seed_prompts
run_params["good_seed_prompts"] = good_seed_prompts

# Print descriptor lists for validation
print("Bad Seed Prompts:", bad_seed_prompts)
print("Good Seed Prompts:", good_seed_prompts)

# Load CLIP model
model, preprocess = clip.load(run_params["model"])
run_params["device"] = str(device)
model = model.to(device).float()

# Tokenize prompts for both classes
bad_text_inputs = clip.tokenize(bad_seed_prompts).to(device) 
good_text_inputs = clip.tokenize(good_seed_prompts).to(device)

# Encoding prompts for both classes
with torch.no_grad():
    bad_text_features = model.encode_text(bad_text_inputs)
    bad_text_features = bad_text_features / (bad_text_features.norm(dim=-1, keepdim=True))

    good_text_features = model.encode_text(good_text_inputs)
    good_text_features = good_text_features / (good_text_features.norm(dim=-1, keepdim=True))

# Combine text features for both classes
text_features = torch.cat([bad_text_features, good_text_features], dim=0).to(device)

# Load training dataset
full_dataset = datasets.ImageFolder(root="data/train")

# Split dataset (80% training, 20% validation)
train_dataset, val_dataset = random_split(
    full_dataset, 
    [int(0.8 * len(full_dataset)), len(full_dataset) - int(0.8 * len(full_dataset))]
)
        
class CustomTransformDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
    
    def __len__(self):
        return len(self.dataset)

# Create separate datasets with transforms
train_dataset_with_transform = CustomTransformDataset(train_dataset, training_transform)
val_dataset_with_transform = CustomTransformDataset(val_dataset, testing_transform)

# Data loaders
train_loader = DataLoader(train_dataset_with_transform, batch_size=run_params["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset_with_transform, batch_size=run_params["batch_size"], shuffle=False)

# 1. Freeze everything first
for param in model.parameters():
    param.requires_grad = False

# 2. Unfreeze the basic visual processing pipeline
model.visual.conv1.requires_grad_(True)
model.visual.ln_pre.requires_grad_(True)
model.visual.ln_post.requires_grad_(True)
model.visual.proj.requires_grad_(True)

# 3. Unfreeze initial transformer blocks
initial_unfrozen = run_params.get("initial_unfrozen_layers", 0)
current_unfrozen = initial_unfrozen

for i in range(12 - initial_unfrozen, 12):
    model.visual.transformer.resblocks[i].requires_grad_(True)

if initial_unfrozen > 0:
    print(f"Initially unfroze last {initial_unfrozen} transformer blocks")

# 4. Freeze text encoder
for param in model.transformer.parameters():
    param.requires_grad = False

# Define optimizer and loss function
optimizer = optim.AdamW(model.parameters(), lr=run_params["learning_rate"], 
                      betas=(run_params["betas"][0], run_params["betas"][1]), 
                      weight_decay=run_params["weight_decay"])

criterion = nn.CrossEntropyLoss()

# Define number of epochs and learning rate scheduler
num_epochs = run_params["num_epochs"]
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

# Variable to track best validation loss
best_val_loss = float("inf")

# Lists to track metrics for visualization
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
train_f1_scores = []
val_f1_scores = []

# Training and validation loop, with gradual unfreezing
for epoch in range(num_epochs):
    # Gradual unfreezing logic
    target_layers = run_params.get("target_unfrozen_layers", initial_unfrozen)
    if epoch > 0 and current_unfrozen < target_layers:
        unfreeze_interval = max(1, num_epochs // (target_layers - initial_unfrozen))
        
        if (epoch % unfreeze_interval == 0) or (epoch == num_epochs - 1):
            layer_to_unfreeze = 12 - current_unfrozen - 1
            model.visual.transformer.resblocks[layer_to_unfreeze].requires_grad_(True)
            current_unfrozen += 1
            print(f"Unfroze layer {layer_to_unfreeze} (now {current_unfrozen}/{target_layers} layers unfrozen)")

    model.train()
    pbar = tqdm(train_loader, total=len(train_loader))  
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    
    for images, labels in train_loader:
        images, labels = images.to(device).float(), labels.to(device)

        optimizer.zero_grad()

        # Encode image features
        image_features = model.encode_image(images)
        image_features = image_features / (image_features.norm(dim=-1, keepdim=True))
        
        # Compute similarity with text features
        similarities = image_features @ text_features.T

        # Average similarities per class
        bad_avg_similarity = similarities[:, :len(bad_seed_prompts)].mean(dim=1, keepdim=True)
        good_avg_similarity = similarities[:, len(bad_seed_prompts):].mean(dim=1, keepdim=True)

        # Create final logits for classification
        logits = torch.cat([bad_avg_similarity, good_avg_similarity], dim=1)

        # Compute loss
        loss = criterion(logits, labels)

        # Backpropagation
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        # Compute predictions
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Store for precision, recall, F1 score
        all_preds.extend(predicted.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        
        pbar.set_description(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Acc: {100 * correct / total:.2f}%")
        pbar.update(1)

    pbar.close()
    
    # Compute overall accuracy, precision, recall, and F1-score
    train_acc = 100 * correct / total
    avg_train_loss = running_loss / len(train_loader)
    train_precision = precision_score(all_labels, all_preds, average="weighted")
    train_recall = recall_score(all_labels, all_preds, average="weighted")
    train_f1 = f1_score(all_labels, all_preds, average="weighted")
    
    # Store metrics for visualization
    train_losses.append(avg_train_loss)
    train_accuracies.append(train_acc)
    train_f1_scores.append(train_f1)
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
          f"Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1-score: {train_f1:.4f}")
    
    # Update learning rate scheduler
    scheduler.step()
    
    # Validation Step
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    all_val_preds, all_val_labels = [], []
    pbar = tqdm(val_loader, total=len(val_loader))  
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Encode image features
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Compute similarity
            similarities = image_features @ text_features.T
            bad_avg_similarity = similarities[:, :len(bad_seed_prompts)].mean(dim=1, keepdim=True)
            good_avg_similarity = similarities[:, len(bad_seed_prompts):].mean(dim=1, keepdim=True)

            # Create logits for classification
            logits = torch.cat([bad_avg_similarity, good_avg_similarity], dim=1)

            # Compute loss
            loss = criterion(logits, labels)
            val_loss += loss.item()

            # Compute accuracy
            _, predicted = torch.max(logits, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

            # Store for precision, recall, F1
            all_val_preds.extend(predicted.cpu().tolist())
            all_val_labels.extend(labels.cpu().tolist())
        
            pbar.update(1)

    pbar.close()
    
    val_acc = 100 * val_correct / val_total
    avg_val_loss = val_loss / len(val_loader)
    val_precision = precision_score(all_val_labels, all_val_preds, average="weighted")
    val_recall = recall_score(all_val_labels, all_val_preds, average="weighted")
    val_f1 = f1_score(all_val_labels, all_val_preds, average="weighted")

    # Store metrics for visualization
    val_losses.append(avg_val_loss)
    val_accuracies.append(val_acc)
    val_f1_scores.append(val_f1)

    print(f"Validation Loss: {avg_val_loss:.4f}, Validation Acc: {val_acc:.2f}%, "
        f"Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1-score: {val_f1:.4f}")
    
    # Save model if it's the best one so far based on validation loss
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), f"{run_dir}/best_model.pth")
        print(f"Best model saved with validation loss: {best_val_loss:.4f}")
            
# Save training curve
plt.figure(figsize=(15, 10))

# Plot accuracy
plt.subplot(1, 1, 1)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

# Save training curve
plt.tight_layout()
plt.savefig(f"{run_dir}/training_curve.png")
plt.close()

# Update run_params with final training information
run_params["actual_epochs"] = num_epochs
run_params["best_val_loss"] = best_val_loss
run_params["final_metrics"] = {
    "train_loss": train_losses[-1],
    "val_loss": val_losses[-1],
    "train_accuracy": train_accuracies[-1],
    "val_accuracy": val_accuracies[-1],
    "train_f1": train_f1_scores[-1],
    "val_f1": val_f1_scores[-1]
}

# Load best model before evaluation
model.load_state_dict(torch.load(f"{run_dir}/best_model.pth", weights_only=True))
print("Best model loaded after training.")

model.eval()    
    
# Custom Dataset for Testing
class SeedDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = row['file_name']
        x_min, y_min, x_max, y_max = row[['x_min', 'y_min', 'x_max', 'y_max']]
        label = 0 if row['bbox_label'] == "BAD" else 1

        # Read image using PIL
        image = Image.open(img_path).convert("RGB")
        # Crop seed image using coordinates
        seed_crop = image.crop((x_min, y_min, x_max, y_max))

        if self.transform:
            seed_crop = self.transform(seed_crop)

        return seed_crop, label

# Testing logic for each dataset  
def evaluate_dataset(dataset_name, run_dir, csv_path=None, dataset=None):
    if dataset is None:
        assert csv_path is not None, "Either csv_path or dataset must be provided"
        dataset = SeedDataset(csv_file=csv_path, transform=testing_transform)

    loader = DataLoader(dataset, batch_size=run_params["batch_size"], shuffle=False)
    
    # Store predictions and true labels
    labels, predictions = [], []
    pbar = tqdm(loader, desc=f"{dataset_name} Dataset: ")
    
    with torch.no_grad():
        for images, label_batch in pbar:
            images, label_batch = images.to(device), label_batch.to(device)
            
            # Encode image features
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Compute similarity
            similarities = image_features @ text_features.T
            bad_avg_similarity = similarities[:, :len(bad_seed_prompts)].mean(dim=1)
            good_avg_similarity = similarities[:, len(bad_seed_prompts):].mean(dim=1)
            
            # Predict class
            batch_predictions = (good_avg_similarity > bad_avg_similarity).long()
            
            # Store results
            labels.extend(label_batch.cpu().tolist())
            predictions.extend(batch_predictions.cpu().tolist())
    
    # Compute accuracy
    accuracy = (np.array(labels) == np.array(predictions)).mean() * 100
    print(f"{dataset_name} Accuracy: {accuracy:.2f}%")
    
    # Compute classification report
    report = classification_report(labels, predictions, target_names=["BAD", "GOOD"], digits=4)
    print(report)
    
    # Create confusion matrix visualization
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["BAD", "GOOD"])
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title(f'{dataset_name} Confusion Matrix')
    plt.savefig(f"{run_dir}/{dataset_name.lower()}_confusion_matrix.png")
    plt.close()
    
    # Save classification report
    with open(f"{run_dir}/{dataset_name.lower()}_classification_report.txt", "w") as f:
        f.write(report)
    print(f"Classification report saved to {run_dir}/{dataset_name.lower()}_classification_report.txt")
    
    # Return metrics for run_params
    return {
        "accuracy": accuracy,
        "precision": precision_score(labels, predictions, average="weighted"),
        "recall": recall_score(labels, predictions, average="weighted"),
        "f1": f1_score(labels, predictions, average="weighted"),
        "confusion_matrix": cm.tolist()
    }, loader

# Load Test dataset
test_dataset = datasets.ImageFolder(root="data/test", transform=testing_transform)
test_loader = DataLoader(test_dataset, batch_size=run_params["batch_size"], shuffle=False)

# Show samples from original dataset before and after augmentation
show_augmented_examples(train_dataset, training_transform, save_path=f"{run_dir}/augment_preview.png")

# Evaluate prompt effectiveness on Test set
test_prompt_results = compute_prompt_effectiveness(
    model=model,
    data_loader=test_loader,
    text_features=text_features,
    bad_seed_prompts=bad_seed_prompts,
    good_seed_prompts=good_seed_prompts
)

# Visualize prompt effectiveness for Test set
visualize_prompt_effectiveness(
    test_prompt_results,
    dataset_name="Test Set",
    save_path=f"{run_dir}/Test_prompt_metrics.png"
)

# Add prompt metrics to run_params
run_params["prompt_metrics"] = test_prompt_results

# Testing model on Test set
run_params["test_metrics"], test_loader = evaluate_dataset(
    "TestSet", run_dir, dataset=test_dataset
)

# Select bad and good seed samples for heatmap generation
selected_images, selected_labels = select_high_similarity_samples(
    model=model,
    data_source=test_loader,
    text_features=text_features,
    bad_seed_prompts=bad_seed_prompts,
    good_seed_prompts=good_seed_prompts,
    num_bad=2,
    num_good=2
)

selected_dataset = TensorDataset(selected_images, selected_labels)

# Generate and save heatmaps for selected seed samples
results = generate_and_visualize_scorecam(
    model=model,
    data_source=selected_dataset,
    text_features=text_features,
    bad_seed_prompts=bad_seed_prompts,
    good_seed_prompts=good_seed_prompts,
    num_bad=2,
    num_good=2,
    subset_dir='heatmaps',
    run_dir=run_dir,
    layer_name="visual.conv1",    
    alpha=0.5,
    cmap='jet',
    show=False
)

# Evaluate NormalRoomLight dataset
NormalRoomLight_csv_path = r'data\NormalRoomLight.csv'

# Testing model on NormalRoomLight set
run_params["NormalRoomLight_metrics"], NormalRoomLight_loader = evaluate_dataset(
    "NormalRoomLight", run_dir, csv_path=NormalRoomLight_csv_path
)

# Evaluate prompt effectiveness on NormalRoomLight set
NRL_prompt_results = compute_prompt_effectiveness(
    model=model,
    data_loader=NormalRoomLight_loader,
    text_features=text_features,
    bad_seed_prompts=bad_seed_prompts,
    good_seed_prompts=good_seed_prompts
)

# Visualize prompt effectiveness for NormalRoomLight set
visualize_prompt_effectiveness(
    NRL_prompt_results,
    dataset_name="NormalRoomLight",
    save_path=f"{run_dir}/NRL_prompt_metrics.png"
)

# Evaluate LightBox dataset
Lightbox_csv_path = r'data\LightBox.csv'

# Testing model on LightBox set
run_params["LightBox_metrics"], LightBox_loader = evaluate_dataset(
    "LightBox", run_dir, csv_path=Lightbox_csv_path
)

# Evaluate prompt effectiveness on Lightbox set
Lightbox_prompt_results = compute_prompt_effectiveness(
    model=model,
    data_loader=LightBox_loader,
    text_features=text_features,
    bad_seed_prompts=bad_seed_prompts,
    good_seed_prompts=good_seed_prompts
)

# Visualize prompt effectiveness for LightBox set
visualize_prompt_effectiveness(
    Lightbox_prompt_results,
    dataset_name="LightBox",
    save_path=f"{run_dir}/Lightbox_prompt_metrics.png"
)


prompt_results_dict = {
    'Test': test_prompt_results,
    'NormalRoomLight': NRL_prompt_results,
    'LightBox': Lightbox_prompt_results
}

# Visualize prompt similarity curves for all sets
visualize_average_prompt_similarity(
    prompt_results_dict=prompt_results_dict,
    bad_prompts=bad_seed_prompts,
    good_prompts=good_seed_prompts,
    run_dir=run_dir,
)

# Save run parameters to JSON file
with open(f"{run_dir}/run_params.json", "w") as f:
    json.dump(run_params, f, indent=4)
print(f"Run parameters and metrics saved to {run_dir}/run_params.json")

# Create comparison visualization of accuracy across datasets
datasets = ["Test", "NormalRoomLight", "LightBox"]

# Extract accuracy values from run_params
accuracy_values = [
    run_params["test_metrics"]["accuracy"],
    run_params["NormalRoomLight_metrics"]["accuracy"],
    run_params["LightBox_metrics"]["accuracy"]
]

# Plot comparison of accuracy across datasets
plt.figure(figsize=(10, 6))
bars = plt.bar(datasets, accuracy_values, color=['blue', 'green', 'orange'])

plt.ylabel('Accuracy (%)')
plt.title('Accuracy Comparison Across Datasets')
plt.ylim(0, 100)
plt.grid(True, axis='y')

for bar, val in zip(bars, accuracy_values):
    plt.text(bar.get_x() + bar.get_width()/2, val + 1, f"{val:.1f}%", ha='center', va='bottom')

plt.tight_layout()
plt.savefig(f"{run_dir}/accuracy_comparison.png")
plt.close()

print(f"\nRun completed, results saved to: {run_dir}")
