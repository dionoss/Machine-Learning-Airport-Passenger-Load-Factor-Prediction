import pandas as pd
import csv
import sklearn
import pickle
import dc
import numpy as np
from sklearn.model_selection import train_test_split # for splitting the data
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import make_column_transformer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt 
%matplotlib inline

df = pd.read_csv('./Datadump.csv',
                 encoding='latin-1')
X, y= dc.cleaning(df)

ohe = OneHotEncoder(sparse=False)

column_trans = make_column_transformer(
    (OneHotEncoder(), ['Vliegtuigtype','Luchthaven', 'Maatschappij']),
    remainder='passthrough')

X = column_trans.fit_transform(X).toarray()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

filename = 'knn_model.sav'
model = pickle.load(open(filename, 'rb'))

y_test_predicted = np.loadtxt('knn_y_test_predicted.csv', delimiter=",")

y_train_predicted = np.loadtxt('knn_y_train_predicted.csv', delimiter=",")
fig, ax = plt.subplots()
ax.scatter(y_test_predicted, y_test, edgecolors=(0, 0, 1))
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=3)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
plt.show()

# model evaluation for testing set

mae = sklearn.metrics.mean_absolute_error(y_test, y_test_predicted)
rmse = sklearn.metrics.mean_squared_error(y_test, y_test_predicted, squared=False)
r2 = sklearn.metrics.r2_score(y_test, y_test_predicted)

print("The model performance for testing set")
print("--------------------------------------")
print('MAE is {}'.format(mae))
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))

from sklearn.metrics import mean_squared_error, r2_score
train_mse = mean_squared_error(y_train, y_train_predicted, squared=False)
train_r2 = r2_score(y_train, y_train_predicted)
test_mse = mean_squared_error(y_test, y_test_predicted, squared=False)
test_r2 = r2_score(y_test, y_test_predicted)
results = pd.DataFrame(['KNN',train_mse, train_r2, test_mse, test_r2]).transpose()
results.columns = ['Method','Training RMSE','Training R2','Test RMSE','Test R2']
results
