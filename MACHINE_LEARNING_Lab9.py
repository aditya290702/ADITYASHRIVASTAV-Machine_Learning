from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import  r2_score
import pandas as pd


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


#Working on DecisionTreeClassifier
