import pandas as pd
import sklearn
import pickle
import dc
from sklearn import metrics
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor, Pool, metrics, cv
import matplotlib.pyplot as plt 
import scikitplot as skplt
%matplotlib inline

df = pd.read_csv('./Datadump.csv',
                 encoding='latin-1')

X, y= dc.cleaning(df)
X

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

filename = 'catboost_model.sav'
model = pickle.load(open(filename, 'rb'))

y_test_predicted = model.predict(X_test)
y_train_predicted = model.predict(X_train)
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
results = pd.DataFrame(['CatBoost',train_mse, train_r2, test_mse, test_r2]).transpose()
results.columns = ['Method','Training RMSE','Training R2','Test RMSE','Test R2']
results

feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(20).plot(kind='barh')
feat_importances.nlargest(20)
X = X.rename(columns={'Max_zitplaatsen': 'max_seats', 'Luchthaven': 'airport', 'Maatschappij': 'airline', 'Vliegtuigtype': 'aircraft_type'})

feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(20).plot(kind='barh')
