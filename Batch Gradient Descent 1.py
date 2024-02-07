import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
def load_data():
    Data = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")
    X = Data[['age', 'BP', 'BMI', 'blood_sugar', 'Gender']]
    X.insert(0, 'Ones', 1)
    y = Data['disease_score']
    return X, y

def hypothesis(theta, X):
    y_pred = np.ones(X.shape[0])

    for i in range(X.shape[0]):
        y_pred[i] = sum(X.iloc[i]*theta)
    return y_pred
    accuracy_score = accuracy_score(y_pred, y)
    plt.plot(y_pred, X_test)
    plt.show()

def main():
    X, y = load_data()
    theta = np.full(X.shape[1], 1)
    alpha = 0.01
    iterations = 1000


    for i in range(X.shape[0]):
        y_pred = hypothesis(theta, X)
        gradient_D = np.dot(X.T, (y_pred - y)) / len(y)
        theta = (theta - (alpha * gradient_D))
        SSE = 1 / 2 * np.sum((y_pred - y) ** 2)


    print("Y_PRED",y_pred)
    print("COST FUNCT",SSE)
    print("Gradient",gradient_D)
    print("Theta",theta)
    print("Accuracy",accuracy_score)

if __name__ == "__main__":
    main()
