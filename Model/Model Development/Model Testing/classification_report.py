# This script generates a classification report from a CSV file containing true and predicted labels.
# It uses the sklearn library to compute precision, recall and F1-score for each class and visualizes the results using matplotlib.

import pandas as pd
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Load predictions CSV
df = pd.read_csv('../prediction_details.csv')
true_labels = df['True_Label']
predicted_labels = df['Predicted_Label']

# Generate classification report as dictionary
report_dict = classification_report(true_labels, predicted_labels, output_dict=True)

# Convert dictionary to DataFrame for easier plotting
report_df = pd.DataFrame(report_dict).transpose()

# Drop accuracy, macro avg, weighted avg rows for per-class metrics
report_df = report_df.drop(['accuracy', 'macro avg', 'weighted avg'], errors='ignore')

# Plot Precision, Recall and F1-score for each class
fig, ax = plt.subplots(figsize=(12, 6))
report_df[['precision', 'recall', 'f1-score']].plot(kind='bar', ax=ax)
ax.set_title('Classification Report Metrics per Class')
ax.set_xlabel('Class')
ax.set_ylabel('Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

