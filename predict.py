import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor,AdaBoostClassifier
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
data = pd.read_csv('train.csv')
t_data = pd.read_csv('test.csv')

# Separate features and target variable
X = data.drop(columns=['id', 'True_value'])  # Features
y = data['True_value']  # Target variable

# Handling missing values
# Let's impute missing values in numerical columns with mean, and in categorical columns with the most frequent value
num_cols = X.select_dtypes(include=['float64']).columns
cat_cols = X.select_dtypes(include=['object', 'bool']).columns

num_imputer = SimpleImputer(strategy='mean')
cat_imputer = SimpleImputer(strategy='most_frequent')

X[num_cols] = num_imputer.fit_transform(X[num_cols])
X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])

# Handling missing values for test data
t_data[num_cols] = num_imputer.transform(t_data[num_cols])  # Impute missing values in test numerical columns
t_data[cat_cols] = cat_imputer.transform(t_data[cat_cols])  # Impute missing values in test categorical columns

# Encoding categorical variables
# Since there are categorical variables, let's apply one-hot encoding to them
# First, we create a ColumnTransformer to apply transformations to specific columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', num_cols),  # numerical columns will remain unchanged
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)  # one-hot encode categorical columns
    ])

# Fit and transform the data
X_preprocessed = preprocessor.fit_transform(X)

# Models Selection and Training
results = {"id": [], "True_value": []}

models = {
    "RandomForest": RandomForestRegressor(random_state=11),
    
}

for name, model in models.items():
    model.fit(X_preprocessed, y)
    y_pred = model.predict(X_preprocessed)
    rmse = mean_squared_error(y, y_pred)
    print(f"Model: {name}, Root Mean Squared Error: {rmse}")
    
    # Predict on the test data
    t_data_preprocessed = preprocessor.transform(t_data.drop(columns=['id']))
    t_data_pred = model.predict(t_data_preprocessed)
    
    # Append the results
    results["id"].extend(t_data['id'])
    results["True_value"].extend(t_data_pred)

# Save results to a new file
result_df = pd.DataFrame(results)
# result_df.to_csv("predicted_results_2.csv", index=False)
