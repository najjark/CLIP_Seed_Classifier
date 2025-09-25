from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def create_confusion_matrix(true_labels, predicted_labels, set_name):
    
    # create confusion matrix using true and predicted labels
    cm = confusion_matrix(true_labels, predicted_labels)
    
    # plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Unhealthy', 'Healthy'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix for ' + set_name + ' Set')
    plt.savefig("CLIP_Project/Plots/confusion_matrix_" + set_name + ".png")
    plt.close()