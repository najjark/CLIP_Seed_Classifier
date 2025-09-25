import matplotlib.pyplot as plt
import os

def plot_training_curve(train_losses):    
    # training loss curve
    plt.figure(figsize=(8,6))
    plt.plot(train_losses, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig("CLIP_Project/Plots/Training_Curve.png", dpi = 300, bbox_inches="tight")
    plt.close()