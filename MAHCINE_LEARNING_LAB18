import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

# Load dataset
data = pd.read_csv('heart.csv')

# Preprocessing
X = data.drop('target', axis=1)
y = data['target']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression classifier
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions
y_prob = model.predict_proba(X_test)[:, 1]

# Varying threshold and calculating metrics
thresholds = np.linspace(0, 1, 100)
metrics = []
for threshold in thresholds:
    y_pred_binary = (y_prob >= threshold)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_binary).ravel()
    accuracy = accuracy_score(y_test, y_pred_binary)
    precision = precision_score(y_test, y_pred_binary)
    recall = recall_score(y_test, y_pred_binary)
    specificity = tn / (tn + fp)

    f1 = f1_score(y_test, y_pred_binary)
    metrics.append([accuracy, precision, recall, specificity, f1])

metrics = np.array(metrics)

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic')
# plt.legend(loc="lower right")
plt.show()

# Print AUC
print("AUC:", roc_auc)

# Print metrics for different thresholds
threshold_metrics_df = pd.DataFrame(metrics, columns=['Accuracy', 'Precision', 'Recall/Senstivity','Senstivity','F1-score'])
# threshold_metrics_df.index = thresholds
print(threshold_metrics_df)
