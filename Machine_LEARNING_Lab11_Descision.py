from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import  r2_score
import pandas as pd
import numpy as np


data = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")
X = data[['age', 'BP', 'BMI', 'blood_sugar', 'Gender']]
y = data['disease_score_fluct']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
regressor = DecisionTreeRegressor()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
r2 = r2_score(y_test, y_pred)
print()
print(r2)
plt.figure(figsize=(60, 50))
plot_tree(regressor, feature_names=X.columns)
plt.show()

unique_values, value_counts = np.unique(y, return_counts=True)
print(unique_values)
probabilities = value_counts / np.sum(value_counts)
entropy = -np.sum(probabilities * np.log2(probabilities))
print("Entropy for Target Variable (y):", entropy)


