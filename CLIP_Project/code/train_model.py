import tqdm
import torch.optim as optim
import torch
import torch.nn.functional as F
import clip

def train_model(model, device, training_loader, criterion, epochs, lr, bad_seed_descriptors, good_seed_descriptors):        
    # model in train mode
    model.train()
    
    # define optimimzer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # tokenize text descriptors
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
    
    # keep track of training losses for plotting    
    train_losses = []

    for epoch in range(epochs):
        pbar = tqdm.tqdm(training_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(pbar):
            # move images and labels to device
            images, labels = images.to(device), labels.to(device)

            # ensure float32
            images = images.float()
            
            # zero optimizer gradients
            optimizer.zero_grad()
            
            # encode images
            image_features = model.encode_image(images).float()  # Force float32
            
            # normalize images features with small epsilon
            image_features = F.normalize(image_features, dim=-1, eps=1e-8)
            
            # compute logits, with temperature scaling
            logits = (image_features @ text_features.T) * 100.0
            
            # Compute loss
            loss = criterion(logits, labels)
            
            # backward pass
            loss.backward()
            
            # gradient clipping to avoid exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # optimizer step
            optimizer.step()
            
            # calculating loss and accuracy
            running_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100*correct/total:.1f}%'
            })
        
        # compute average loss for this epoch
        avg_loss = running_loss / len(training_loader)
        train_losses.append(avg_loss)
    
        pbar.close()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(training_loader):.4f}, Accuracy: {100*correct/total:.2f}%")
        
    return train_losses