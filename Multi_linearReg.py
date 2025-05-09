import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import seaborn as sns
from sklearn.model_selection import train_test_split

#load dataset
data = pd.read_csv("D:\\50_Startups.csv")
#choosing the input features and class label

X = data.iloc[: , :-1].values
y = data.iloc[: , -1].values

#one hot encoding (converting the categorical variables into numerical values)
ct = ColumnTransformer(transformers=[('encoded' , OneHotEncoder(), [3])],remainder='passthrough')
X = np.array(ct.fit_transform(X))


# training and test data

# print(X)
# split test and training data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# train the model 

regressor = LinearRegression()
regressor.fit(X_train, y_train)


y_pred = regressor.predict(X_test) #predict the y using unseen x values 
np.set_printoptions(precision=2) #up to 2 decimal palces
ytr_reshaped = y_pred.reshape(len(y_pred), 1)
ytst_reshaped = y_test.reshape(len(y_test), 1)

concat = np.concatenate((ytr_reshaped, ytst_reshaped), axis=1)

print(concat)


# we cannot draw a plot or regression because of multi linear regression , there are many input features
