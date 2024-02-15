import sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import  numpy as np

#WITH SCIKIT
def lab1():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = print("ACCURACY",accuracy_score(y_test, y_pred) * 100,"%")
    print("Coefficients:", model.coef_)
    print("Intercept:", model.intercept_)
    plt.plot(y_pred, X_test)
    plt.show()
def load_data():
    breast_cancer = datasets.load_breast_cancer()
    X = breast_cancer.data
    y = breast_cancer.target
    return X, y
def main():
    lab1()
if __name__ == "__main__":
    main()


#Without Scikit

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def initialize_params(num_features):
    return np.zeros((num_features, 1))

def compute_cost(X, y, theta):
    m = len(y)
    h = sigmoid(X @ theta)
    cost = (-1/m) * (y.T @ np.log(h) + (1 - y).T @ np.log(1 - h))
    return cost[0, 0]

def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    for i in range(iterations):
        h = sigmoid(X @ theta)
        gradient = (1/2) * (X.T @ (h - y)**2)
        theta = theta - alpha * gradient
        cost = compute_cost(X, y, theta)
        print(f"Iteration {i + 1}: Cost = {cost}")
    return theta
def predict(X, theta):
    probabilities = sigmoid(X @ theta)
    return (probabilities >= 0.5)
def lab1():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
    X_train = np.c_[np.ones((X_train.shape[0], 1)), X_train]  # Add bias term
    X_test = np.c_[np.ones((X_test.shape[0], 1)), X_test]

    theta = initialize_params(X_train.shape[1])
    alpha = 0.01
    iterations = 1000

    theta = gradient_descent(X_train, y_train.reshape(-1, 1), theta, alpha, iterations)

    y_pred = predict(X_test, theta)

  #  accuracy = accuracy_score(y_test, y_pred) * 100
    #print("ACCURACY:", accuracy, "%")
    print("Coefficients:", theta[1:])
    print("Intercept:", theta[0])

def load_data():
    breast_cancer = datasets.load_breast_cancer()
    X = breast_cancer.data
    y = breast_cancer.target
    return X, y

def main():
    lab1()

if __name__ == "__main__":
    main()
