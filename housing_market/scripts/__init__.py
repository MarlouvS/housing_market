# Set up code checking
import os
if not os.path.exists("../home-data-for-ml-course/train.csv"):
    os.symlink("../home-data-for-ml-course/train.csv", "../home-data-for-ml-course/train.csv")
    os.symlink("../home-data-for-ml-course/test.csv", "../home-data-for-ml-course/test.csv")
from learntools.core import binder
binder.bind(globals())
# from learntools.ml_intermediate.ex1 import *
print("Setup Complete")

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from methods import *


# Read the data
train_df, test_df = read_train_test_data()
numeric_columns = get_numeric_columns(train_df)
categorical_columns = get_categorical_columns(train_df)

def  preprocess_data(df):
    le = LabelEncoder()

    # Obtain target and predictors
    if 'SalePrice' in df:
        df.drop('SalePrice', axis=1)

    failed_features = []
    for aFeature in categorical_columns:
        try:
            df[aFeature] = le.fit_transform(df[aFeature])
        except:
            failed_features.append(aFeature)
    df.drop(columns=failed_features, inplace=True)
    return df

X_submission = preprocess_data(test_df)
y_full = train_df.SalePrice
X_full = preprocess_data(train_df)


# Transform categorical_columns
imputer = SimpleImputer(strategy='median')
X_full = X_full.fillna(X_full.median())
X_submission = X_submission.fillna(X_submission.median())
X_submission.drop('Electrical', axis=1, inplace=True)

X_full = X_full[list(X_submission.columns.values)]
print(X_full.shape)
print(X_submission.shape)
print(X_submission.columns.values)

# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X_full, y_full, train_size=0.8, test_size=0.2,
                                                      random_state=0)

# Define the models
params = {'n_estimators':[90,100,110], 'random_state':[0], 'criterion':['mae'],'min_samples_split':[2], 'max_depth':[20]}
model_6 = GridSearchCV(RandomForestRegressor(), params, n_jobs=-1, cv=4)
model_7 = XGBRegressor()
gbr_model = GradientBoostingRegressor()

models = [ model_6, model_7, gbr_model]
model_scores = {}

# Function for comparing different models
def score_model(model, X_t=X_train, X_v=X_valid, y_t=y_train, y_v=y_valid):
    model.fit(X_t, y_t)
    preds = model.predict(X_v)
    return mean_absolute_error(y_v, preds)

for i in range(0, len(models)):
    mae = score_model(models[i])
    model_scores[i]=mae
    print("Model %d MAE: %d" % (i+1, mae))

min_score=min(model_scores.values())
my_model=get_key(model_scores,min_score)
print("Best model is: %s with score %d" % ( my_model,min_score))

print(model_6.get_params())
print(model_6.best_params_)
print(model_6.best_score_)

# Fit the model to the training data
models[my_model].fit(X_full, y_full)

# Generate test predictions
preds_test = models[my_model].predict(X_submission)

# Save predictions in format used for competition scoring
output = pd.DataFrame({'Id': X_submission.index,
                       'SalePrice': preds_test})
output.to_csv('submission.csv', index=False)

