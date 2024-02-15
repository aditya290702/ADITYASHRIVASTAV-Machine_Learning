#L1

import pandas as pd
from sklearn import preprocessing

my_data = {
    "Gender": ['F', 'M', 'M', 'F', 'M', 'F', 'M', 'F', 'F', 'M'],
    "Name": ['Cindy', 'Carl', 'Johnny', 'Stacey', 'Andy', 'Sara', 'Victor', 'Martha', 'Mindy', 'Max'],
    "Marks": ['5', '4', '3', '2', '1', '6', '7', '8', '9', '10']
}

blk = pd.DataFrame(my_data)
print("Original Data Frame:\n")
print(blk)

# Convert 'Marks' column to numeric values
blk['Marks'] = pd.to_numeric(blk['Marks'], errors='coerce')

my_label_gender = preprocessing.OrdinalEncoder()
my_label_marks = preprocessing.OrdinalEncoder()

# Encode 'Gender' column
blk['Gender'] = my_label_gender.fit_transform(blk[['Gender']])

# Encode 'Marks' column
blk['Marks'] = my_label_marks.fit_transform(blk[['Marks']])

print("Data Frame after Label Encoding:\n")
print(blk)


#L2

import sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import Lasso
def load_data():
    x, y = datasets.load_breast_cancer(return_X_y=True)
    print(x, y)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)
    print(X_train)
    model = Lasso()
    model.fit(X_train, y_train)R
    y_pred = model.predict(X_test)
    print("Predictions:", y_pred)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print("R2 Score:", r2)
    print("Mean Squared Error:", mse)
    print("Coefficients:", model.coef_)
    print("Intercept:", model.intercept_)

def main():
    load_data()

if __name__ == "__main__":
    main()


#Label Encoding

import pandas as pd
from sklearn import preprocessing

my_data = {
    "Gender": ['F', 'M', 'M', 'F', 'M', 'F', 'M', 'F', 'F', 'M'],
    "Name": ['Cindy', 'Carl', 'Johnny', 'Stacey', 'Andy', 'Sara', 'Victor', 'Martha', 'Mindy', 'Max']

}
blk = pd.DataFrame(my_data)
print("Geniune Data Frame:\n")
print(blk)

my_label = preprocessing.LabelEncoder()

blk['Gender'] = my_label.fit_transform(blk['Gender'])
print(blk['Gender'].unique())
print("Data Frame after Label Encoding:\n")
print(blk)



#Ordinal Encoding

import pandas as pd
from sklearn import preprocessing

my_data = {
    "Gender": ['F', 'M', 'M', 'F', 'M', 'F', 'M', 'F', 'F', 'M'],
    "Name": ['Cindy', 'Carl', 'Johnny', 'Stacey', 'Andy', 'Sara', 'Victor', 'Martha', 'Mindy', 'Max'],
    "Marks": ['5', '4', '3', '2', '1', '6', '7', '8', '9', '10']
}
blk = pd.DataFrame(my_data)
print("Original Data Frame:\n")
print(blk)
blk['Marks'] = pd.to_numeric(blk['Marks'], errors= 'ignore' )
my_label_gender = preprocessing.OrdinalEncoder()
my_label_marks = preprocessing.OrdinalEncoder()
blk['Gender'] = my_label_gender.fit_transform(blk[['Gender']])
blk['Marks'] = my_label_marks.fit_transform(blk[['Marks']])
print("Data Frame after Label Encoding:\n")
print(blk)
