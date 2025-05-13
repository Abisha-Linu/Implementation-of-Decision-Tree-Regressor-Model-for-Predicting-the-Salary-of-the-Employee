# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and read the data frame using pandas.
2. Calculate the null values present in the dataset and apply label encoder. 
3. Determine test and training data set and apply decison tree regression in dataset.
4. Calculate Mean square error,data prediction and r2.

## Program:
```
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: ABISHA LINU L
RegisterNumber:  212224040011
```
```
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
data = pd.read_csv("/content/Salary.csv")
print(data.head())
print(data.info())
print(data.isnull().sum())

le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])
print(data.head())  

x = data[["Position", "Level"]]
y = data["Salary"]
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=2
)
dt = DecisionTreeRegressor()
dt.fit(x_train, y_train)

y_pred = dt.predict(x_test)
mse = metrics.mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

r2 = metrics.r2_score(y_test, y_pred)
print("R2 Score:", r2)

print("Predicted Salary for [5,6]:", dt.predict([[5, 6]]))

plt.figure(figsize=(20, 8))
plot_tree(dt, feature_names=x.columns, filled=True)
plt.show()
```
## Output:
![Decision Tree Regressor Model for Predicting the Salary of the Employee](sam.png)

![image](https://github.com/user-attachments/assets/ec392786-245f-47eb-ba52-a205811650e0)

![image](https://github.com/user-attachments/assets/2e3e47a5-a330-4d35-81ab-b1ab146730c4)

![image](https://github.com/user-attachments/assets/181db9a6-b53d-4942-b635-b70c1cffd249)

![image](https://github.com/user-attachments/assets/0834351c-1545-4688-bbbd-fe8af1e1df56)

![image](https://github.com/user-attachments/assets/1a09d4a6-5f38-48c1-8bf2-3246d0644a92)

![image](https://github.com/user-attachments/assets/4bda45cd-15b6-4dbd-9f21-4773474c475c)

![image](https://github.com/user-attachments/assets/5654dab9-be79-44f4-a140-6d97a7e645d6)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
