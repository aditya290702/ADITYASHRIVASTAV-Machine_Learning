import sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import Ridge

def lab1():
    load_data()

def load_data():
    x, y = datasets.load_breast_cancer(return_X_y=True)
    print(x, y)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)

    print(X_train)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Predictions:", y_pred)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print("R2 Score:", r2)
    print("Mean Squared Error:", mse)

    # For regression, accuracy score doesn't apply.
    # accuracy = accuracy_score(y_test, y_pred)
    # print("Accuracy:", accuracy)

    print("Coefficients:", model.coef_)
    print("Intercept:", model.intercept_)


def main():
    lab1()


if __name__ == "__main__":
    main()
