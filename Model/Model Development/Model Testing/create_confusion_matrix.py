# File to generate confusion matrix for the model predictions

# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

# Load data
data = pd.read_csv("../prediction_details.csv")

# Extract true and predicted labels
y_true = data['True_Action']
y_pred = data['Predicted_Action']

# Get all actions
actions = sorted(data['True_Action'].unique())

# Generate confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=actions)

# Split the labels and matrix into two halves
mid = len(actions) // 2
actions1 = actions[:mid]
actions2 = actions[mid:]

cm1 = cm[:mid, :mid]
cm2 = cm[mid:, mid:]

# Plot for first 50 classes
plt.figure(figsize=(20, 18))
sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues',
            xticklabels=actions1, yticklabels=actions1,
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix (Classes 1–50)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('confusion_matrix_1-50.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot for next 50 classes
plt.figure(figsize=(20, 18))
sns.heatmap(cm2, annot=True, fmt='d', cmap='Blues',
            xticklabels=actions2, yticklabels=actions2,
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix (Classes 51–100)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('confusion_matrix_51-100.png', dpi=300, bbox_inches='tight')
plt.show()