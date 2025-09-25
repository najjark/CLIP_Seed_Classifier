import torch
import torch.nn.functional as F
import clip

def evaluate_model(model, test_loader, criterion, device, bad_seed_descriptors, good_seed_descriptors):
    # model in eval mode
    model.eval()

    # compute text features for each class
    bad_text_tokens = clip.tokenize(bad_seed_descriptors).to(device)
    good_text_tokens = clip.tokenize(good_seed_descriptors).to(device)
    
    with torch.no_grad():
        bad_text_features = model.encode_text(bad_text_tokens).float()
        good_text_features = model.encode_text(good_text_tokens).float()
        
        # normalize text features with small epsilon to avoid zero division
        bad_text_features = F.normalize(bad_text_features, dim=-1, eps=1e-8)
        good_text_features = F.normalize(good_text_features, dim=-1, eps=1e-8)
        
        # average and concatenate text features
        bad_mean = bad_text_features.mean(dim=0, keepdim=True)
        good_mean = good_text_features.mean(dim=0, keepdim=True)
        
        text_features = torch.cat([bad_mean, good_mean], dim=0)
        
        # ensure text features are float32 and detached
        text_features = text_features.float().detach()
    
    # initialize metrics 
    total_loss, correct, total = 0, 0, 0
    
    # store predictions and labels for plotting
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for images, labels in test_loader:
            # move images and labels to device
            images, labels = images.to(device), labels.to(device)

            # ensure images in float32
            images = images.float()

            # encode image features
            image_features = model.encode_image(images).float()  # force float32
            
            # normalize image features, with small epsilon to avoid zero division
            image_features = F.normalize(image_features, dim=-1, eps=1e-8)
            
            # compute logits based on image-text similarity
            logits = image_features @ text_features.T
            
            # compute loss
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            
            _, predicted = torch.max(logits, 1)
            
            total += labels.size(0)
            
            correct += (predicted == labels).sum().item()
            
            # append labels and predictions to lists
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy, all_labels, all_preds