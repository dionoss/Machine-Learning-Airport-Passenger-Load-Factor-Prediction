import pandas as pd
import numpy as np
import sklearn
import pickle
import dc
from sklearn import metrics
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor, Pool, metrics, cv
import time

#dataset loading and cleaning:
df = pd.read_csv('./Datadump.csv', encoding='latin-1')
X, y= dc.cleaning(df)
start = time.time()
categorical = ['Vliegtuigtype', 'Luchthaven', ' Maatschappij']
def column_index(df, query_cols):
    cols = df.columns.values
    sidx = np.argsort(cols)
    return sidx[np.searchsorted(cols, query_cols, sorter=sidx)]
categorical_features_indices = column_index(X, categorical)
categorical_features_indices = np.where(X.dtypes != float)[0]
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=42)

#building the model:
model = CatBoostRegressor(iterations=438,
                             learning_rate=0.104502,
                             task_type = 'GPU',
                             depth=16,
                             eval_metric='RMSE',
                             random_seed = 42,
                             bagging_temperature = 0.2,
                             od_type='Iter',
                             metric_period = 75,
                             od_wait=100)
model.fit(X_train, y_train,
                 eval_set=(X_valid, y_valid),
                 cat_features=categorical_features_indices,
                 use_best_model=True,
          plot=True
         )

#The following code will write the model to .sav
filename = 'catboost_model.sav'
pickle.dump(model, open(filename, 'wb'))
