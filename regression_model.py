import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from xgboost import XGBRegressor
from pre_proc import calculate_grade, parse_activity_scores, sort_activities_by_keys, normalize_scores
import optuna


# Read the data from a CSV file named "processed_data.csv"
df = pd.read_csv("processed_data.csv")

# Apply the parsing function to the "activities_scores" column
df["activities_scores"] = df["activities_scores"].apply(parse_activity_scores)

# Sort activity scores based on the defined activities
df["activities_scores"] = df["activities_scores"].apply(sort_activities_by_keys)

# Apply the normalization function to each row
df['activities_scores'] = df.apply(normalize_scores, axis=1)
# Function to calculate a total score based on activity scores


# Add a new "grade" column with the calculated score for each row
df["grade"] = df["activities_scores"].apply(calculate_grade)

# Select only relevant columns for further analysis
df = df[["id", "industry", "followers", "recommendations_count", "grade"]]

#remove the rows with industry = 4 as it is not a valid industry
df = df[df['industry'] != 4]

# save the result in a new csv file  df
df.to_csv('result_df.csv', index=False)



# Prepare data for model training
# Features (independent variables)
X = df[["industry", "followers", "recommendations_count"]]
# Target variable (dependent variable)
y = df["grade"]

def objective(trial):
    # Extract trial parameters
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.1)
    n_estimators = trial.suggest_int("n_estimators", 100, 500)
    max_depth = trial.suggest_int("max_depth", 10, 30)
    subsample = trial.suggest_float("subsample", 0.1, 1.0)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)
    gamma = trial.suggest_float("gamma", 0.0, 1.0)
    reg_alpha = trial.suggest_float("reg_alpha", 0.0, 1.0)
    reg_lambda = trial.suggest_float("reg_lambda", 0.0, 1.0)

    # Define KFold cross-validation strategy
    cv = KFold(n_splits=5, shuffle=True)

    # Track mean squared errors and r2 scores across folds
    mse_scores = []
    r2_scores = []
    for train_index, test_index in cv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Create and train XGBoost model with trial parameters
        model = XGBRegressor(objective='reg:squarederror', learning_rate=learning_rate,
                                 n_estimators=n_estimators, max_depth=max_depth,
                                 subsample=subsample, colsample_bytree=colsample_bytree,
                                 gamma=gamma, reg_alpha=reg_alpha, reg_lambda=reg_lambda)
        model.fit(X_train, y_train)

        # Make predictions and calculate metrics
        y_predicted = model.predict(X_test)
        mse = mean_squared_error(y_test, y_predicted)
        r2 = r2_score(y_test, y_predicted)
        mse_scores.append(mse)
        r2_scores.append(r2)

    # Return mean MSE (you can also return mean R2 or a combination)
    return np.mean(mse_scores)

# Create Optuna study
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=1000)  # Adjust n_trials as needed

# Print best trial results (optional)
best_trial = study.best_trial
print("Best Trial Parameters:", best_trial.params)
print("Best Mean Squared Error:", best_trial.value)

# Save the best parameters and train the final model
best_params = best_trial.params
model = XGBRegressor(objective='reg:squarederror', **best_params)
model.fit(X, y)

# Save the model to a file
import pickle
with open("regression_model.pkl", "wb") as file:
    pickle.dump(model, file)
