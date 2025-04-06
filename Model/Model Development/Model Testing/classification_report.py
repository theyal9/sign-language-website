# This script generates a classification report from a CSV file containing true and predicted class names.

import pandas as pd
from sklearn.metrics import classification_report

# Load predictions CSV
df = pd.read_csv('../prediction_details.csv')

# Extract true and predicted class names from the CSV
true_actions = df['True_Action']
predicted_actions = df['Predicted_Action']

# Generate classification report as a dictionary using the class names
report_dict = classification_report(true_actions, predicted_actions, output_dict=True)

# Convert the report dictionary into a DataFrame
report_df = pd.DataFrame(report_dict).transpose()

# Save the classification report DataFrame as a CSV file
report_df.to_csv('classification_report.csv', index=True)

# Print classification report to console
print(classification_report(true_actions, predicted_actions))

print("Classification report saved as 'classification_report.csv'")