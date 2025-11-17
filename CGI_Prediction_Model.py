import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from joblib import dump as joblib_dump

def promethee(models_results):
    """
    Selects the best model based on multiple criteria using a PROMETHEE-like method.
    Criteria: RMSE and MAE to minimize, R² to maximize.
    """
    # Define preferences for each metric
    preferences = {
        'rmse': 'min',
        'mae': 'min',
        'r2': 'max'
    }

    # Define weights for each criterion
    weights = {
        'rmse': 0.33,
        'mae': 0.33,
        'r2': 0.34
    }

    # Normalize scores for each criterion
    for criterion, direction in preferences.items():
        values = [result[criterion] for result in models_results.values()]
        max_value, min_value = max(values), min(values)
        for model in models_results:
            if direction == 'min':
                models_results[model][criterion] = (max_value - models_results[model][criterion]) / (max_value - min_value)
            elif direction == 'max':
                models_results[model][criterion] = (models_results[model][criterion] - min_value) / (max_value - min_value)

    # Calculate weighted scores for each model
    scores = {}
    for model, metrics in models_results.items():
        scores[model] = sum(weights[criterion] * value for criterion, value in metrics.items())

    print("Scores:", scores)

    # Select the model with the highest score
    best_model_name = max(scores, key=scores.get)
    return best_model_name

def build_and_train_model(data_path):
    """
    Loads dataset, trains multiple regression models, evaluates them,
    and selects the best model using the PROMETHEE method.
    """
    # Load dataset
    df = pd.read_csv(data_path, sep=";")
    print(df)

    # Prepare features and target
    X = df[['day', 'hour']]
    y = df['nb_machine']

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(),
        'Lasso Regression': Lasso(),
        'Random Forest': RandomForestRegressor(),
        'Support Vector Regressor': SVR(),
        'Gradient Boosting Regressor': GradientBoostingRegressor()
    }

    # Dictionary to store model performance metrics
    models_results = {}

    # Train and evaluate each model
    for name, model in models.items():
        print(f"\nTraining model: {name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Compute metrics
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        # Print metrics
        print(f"RMSE: {rmse}")
        print(f"R²: {r2}")
        print(f"MAE: {mae}")

        # Store results for PROMETHEE
        models_results[name] = {'rmse': rmse, 'r2': r2, 'mae': mae}

    print("All model results:", models_results)

    # Select the best model using PROMETHEE
    best_model_name = promethee(models_results)
    best_model = models[best_model_name]
    print(f"\nThe best model selected by PROMETHEE is: {best_model_name}")

    # Save the best model
    joblib_dump(best_model, 'best_model.joblib')

    return best_model

def predict_vms(day_of_week, hour, model):
    """
    Predicts the number of VMs for a given day and hour using the trained model.
    """
    future_data = pd.DataFrame({'day': [day_of_week], 'hour': [hour]})
    prediction = model.predict(future_data)
    return prediction

if __name__ == "__main__":
    # Check command-line arguments
    if len(sys.argv) != 2:
        print("Usage: python CGI_Prediction_Model.py <data_file.csv>")
        sys.exit(1)

    # Retrieve arguments
    data_path = sys.argv[1]         # First argument: CSV file path

    # Build and train the model
    best_model = build_and_train_model(data_path)

    # Generate predictions for one week (7 days x 24 hours)
    data = []
    for day in range(7):
        for hour in range(24):
            pred = predict_vms(day, hour, best_model)
            data.append([day, hour, pred[0]])

    # Save predictions to CSV
    df_out = pd.DataFrame(data, columns=["day", "hour", "nb_machines"])
    df_out.to_csv('out_prediction_one_week.csv', index=False)
