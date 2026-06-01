import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_model_comparison(model_names, accuracies):
    plt.figure(figsize=(8, 5))
    bars = plt.bar(model_names, accuracies, color=['skyblue', 'lightgreen', 'salmon'], edgecolor='black')
    plt.ylim(0, 1)
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Comparison')
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{acc:.3f}', ha='center', va='bottom')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def plot_confusion_matrices(y_test, predictions, model_names):
    fig, axes = plt.subplots(1, len(predictions), figsize=(5*len(predictions), 4))
    for ax, name, y_pred in zip(axes, model_names, predictions):
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
        ax.set_title(f'{name}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
    plt.tight_layout()
    plt.show()