import pandas as pd
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
import time
start = time.time()

#Load dataset
df = pd.read_csv('./Datadump.csv',
                 encoding='latin-1')
X, y= dc.cleaning(df)
end = time.time()
preprocessing = end - start, "seconds"
preprocessing

#One Hot encode dataset
start = time.time()
ohe = OneHotEncoder(sparse=False)
column_trans = make_column_transformer(
    (OneHotEncoder(), ['Vliegtuigtype','Luchthaven', 'Maatschappij']),
    remainder='passthrough')

#Build model
knn_model = KNeighborsRegressor(n_neighbors=5)
X = column_trans.fit_transform(X).toarray()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
knn_model.fit(X_train, y_train)
y_test_predicted = knn_model.predict(X_test)
y_train_predicted = knn_model.predict(X_train)
rmse = mean_squared_error(y_train, y_train_predicted, squared=False)
rmse

filename = 'knn_model.sav'
pickle.dump(knn_model, open(filename, 'wb'))
np.savetxt('knn_y_test_predicted.csv', y_test_predicted, delimiter=',')
np.savetxt('knn_y_train_predicted.csv', y_train_predicted, delimiter=',')
end = time.time()
modeltime = end - start, "seconds"
modeltime
