import sys
import time
import warnings
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------
HAS_ARIMA = False
HAS_PROPHET = False
HAS_TF = False
TF_IMPORT_ERROR = None

try:
    from statsmodels.tsa.arima.model import ARIMA
    HAS_ARIMA = True
except Exception:
    ARIMA = None

try:
    from prophet import Prophet
    HAS_PROPHET = True
except Exception:
    Prophet = None

try:
    import tensorflow as tf
    from tensorflow.keras import Sequential, Model, Input
    from tensorflow.keras.layers import (
        Dense, Dropout, LSTM, MultiHeadAttention,
        LayerNormalization, GlobalAveragePooling1D
    )
    from tensorflow.keras.callbacks import EarlyStopping
    HAS_TF = True
except Exception as e:
    HAS_TF = False
    TF_IMPORT_ERROR = str(e)

TARGET_COL = "nb_machine"
TEST_SIZE = 0.2

BASE_LAGS = [1, 2, 3, 6, 12, 24]
ROLLING_WINDOW = 24
SEQ_WINDOW = 12


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def read_input_csv(data_path):
    df = pd.read_csv(data_path, sep=None, engine="python")
    df = df.loc[:, ~df.columns.astype(str).str.contains(r"^Unnamed")]
    return df


def format_seconds(seconds):
    return round(float(seconds), 4)


def compute_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else float("nan")
    return {"rmse": rmse, "mae": mae, "r2": r2}


def promethee(models_results):
    start_time = time.perf_counter()

    preferences = {"rmse": "min", "mae": "min", "r2": "max"}
    weights = {"rmse": 0.33, "mae": 0.33, "r2": 0.34}
    scores = {name: 0.0 for name in models_results.keys()}

    print("\n=== PROMETHEE configuration ===")
    print("Weights used:", weights)

    for criterion, direction in preferences.items():
        values = np.array([models_results[name][criterion] for name in models_results], dtype=float)
        min_v = values.min()
        max_v = values.max()

        if np.isclose(max_v, min_v):
            normalized = np.ones_like(values)
        else:
            if direction == "min":
                normalized = (max_v - values) / (max_v - min_v)
            else:
                normalized = (values - min_v) / (max_v - min_v)

        print(f"\nCriterion: {criterion}")
        print("Normalized values:", dict(zip(models_results.keys(), normalized)))

        for idx, name in enumerate(models_results.keys()):
            scores[name] += weights[criterion] * normalized[idx]

    best_model = max(scores, key=scores.get)
    promethee_time = time.perf_counter() - start_time

    print("\nPROMETHEE scores:", scores)
    print(f"PROMETHEE execution time (s): {format_seconds(promethee_time)}")

    return best_model, promethee_time, scores


def save_summary_csv(models_results, execution_times, promethee_scores=None, filename_prefix="model_summary"):
    rows = []
    for model_name in models_results:
        row = {
            "model": model_name,
            "rmse": models_results[model_name]["rmse"],
            "mae": models_results[model_name]["mae"],
            "r2": models_results[model_name]["r2"],
            "execution_time_s": execution_times.get(model_name, np.nan),
        }
        if promethee_scores is not None:
            row["promethee_score"] = promethee_scores.get(model_name, np.nan)
        rows.append(row)

    df_summary = pd.DataFrame(rows)
    df_summary.to_csv(f"{filename_prefix}.csv", index=False)
    print(f"\nSaved: {filename_prefix}.csv")


# ---------------------------------------------------------------------
# Script 1 style: classical regression models
# ---------------------------------------------------------------------
def evaluate_classical_models_script1_style(df):
    df = df.copy()

    if "heure" not in df.columns and "hour" in df.columns:
        df = df.rename(columns={"hour": "heure"})

    required = ["day", "heure", TARGET_COL]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}. Available columns: {list(df.columns)}")

    X = df[["day", "heure"]]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(),
        "Lasso Regression": Lasso(),
        "Random Forest": RandomForestRegressor(),
        "Support Vector Regressor": SVR(),
        "Gradient Boosting Regressor": GradientBoostingRegressor(),
    }

    models_results = {}
    trained_models = {}
    execution_times = {}

    for name, model in models.items():
        print(f"\nModèle en cours d'entraînement : {name}")
        start_time = time.perf_counter()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        elapsed = time.perf_counter() - start_time

        rmse = root_mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        print(f"RMSE: {rmse}")
        print(f"R²: {r2}")
        print(f"MAE: {mae}")
        print(f"Execution time (s): {format_seconds(elapsed)}")

        models_results[name] = {"rmse": rmse, "r2": r2, "mae": mae}
        trained_models[name] = model
        execution_times[name] = elapsed

    return models_results, trained_models, execution_times


def predict_vms(day_of_week, hour, model):
    future_data = pd.DataFrame({"day": [day_of_week], "heure": [hour]})
    return model.predict(future_data)


def build_prediction_grid_classical_day(model):
    data = []
    for days in range(1):
        for hours in range(24):
            pred = predict_vms(days, hours, model)
            data.append([days, hours, float(pred[0])])
    return pd.DataFrame(data, columns=["day", "heure", "nb_machines"])


def build_prediction_grid_classical_week(model):
    data = []
    for days in range(7):
        for hours in range(24):
            pred = predict_vms(days, hours, model)
            data.append([days, hours, float(pred[0])])
    return pd.DataFrame(data, columns=["day", "heure", "nb_machines"])


def build_prediction_grid_classical_2months(model):
    data = []
    for cycle in range(8):
        for days in range(7):
            for hours in range(24):
                pred = predict_vms(days, hours, model)
                data.append([days, hours, float(pred[0])])
    return pd.DataFrame(data, columns=["day", "heure", "nb_machines"])


# ---------------------------------------------------------------------
# Script 2 style: time series models
# ---------------------------------------------------------------------
def normalize_and_order(df):
    df = df.copy()

    if "hour" not in df.columns and "heure" in df.columns:
        df = df.rename(columns={"heure": "hour"})

    if "week" not in df.columns:
        df["week"] = np.arange(len(df)) // 168

    if "value" not in df.columns:
        df["value"] = np.arange(1, len(df) + 1)

    required = ["week", "day", "hour", TARGET_COL]
    for col in required:
        if col not in df.columns:
            raise ValueError(
                f"Missing required column: {col}. Available columns: {list(df.columns)}"
            )

    for col in ["value", "week", "day", "hour", TARGET_COL]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["week", "day", "hour", TARGET_COL])
    df = df.sort_values(["week", "day", "hour", "value"]).reset_index(drop=True)

    df["timestamp"] = pd.date_range(
        start="2024-01-01 00:00:00",
        periods=len(df),
        freq="H"
    )
    df = df.set_index("timestamp")
    return df


def get_adaptive_lags(n_train, base_lags=BASE_LAGS):
    lags = [lag for lag in base_lags if lag < n_train]
    return lags if lags else [1]


def get_feature_columns(lags):
    cols = [
        "week", "day", "hour", "day_hour", "is_weekend",
        "hour_sin", "hour_cos", "day_sin", "day_cos"
    ]
    cols += [f"lag_{lag}" for lag in lags]
    cols += [f"rolling_mean_{ROLLING_WINDOW}", f"rolling_std_{ROLLING_WINDOW}"]
    return cols


def add_features(df, lags=None, rolling_window=ROLLING_WINDOW):
    out = df.copy()

    if lags is None:
        lags = BASE_LAGS

    out["day_hour"] = out["day"] * 24 + out["hour"]
    out["is_weekend"] = (out["day"] >= 5).astype(int)

    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24.0)
    out["day_sin"] = np.sin(2 * np.pi * out["day"] / 7.0)
    out["day_cos"] = np.cos(2 * np.pi * out["day"] / 7.0)

    for lag in lags:
        out[f"lag_{lag}"] = out[TARGET_COL].shift(lag)

    out[f"rolling_mean_{rolling_window}"] = out[TARGET_COL].shift(1).rolling(rolling_window).mean()
    out[f"rolling_std_{rolling_window}"] = out[TARGET_COL].shift(1).rolling(rolling_window).std()

    return out.dropna()


def make_feature_row(history_values, next_timestamp, lags):
    history_values = np.asarray(history_values, dtype=float)

    row = {
        "week": int((next_timestamp - pd.Timestamp("2024-01-01")).days // 7),
        "day": int(next_timestamp.dayofweek),
        "hour": int(next_timestamp.hour),
        "day_hour": int(next_timestamp.dayofweek * 24 + next_timestamp.hour),
        "is_weekend": int(next_timestamp.dayofweek >= 5),
        "hour_sin": float(np.sin(2 * np.pi * next_timestamp.hour / 24.0)),
        "hour_cos": float(np.cos(2 * np.pi * next_timestamp.hour / 24.0)),
        "day_sin": float(np.sin(2 * np.pi * next_timestamp.dayofweek / 7.0)),
        "day_cos": float(np.cos(2 * np.pi * next_timestamp.dayofweek / 7.0)),
    }

    for lag in lags:
        if len(history_values) < lag:
            raise ValueError(f"Not enough history for lag_{lag}")
        row[f"lag_{lag}"] = float(history_values[-lag])

    window = history_values[-ROLLING_WINDOW:] if len(history_values) >= ROLLING_WINDOW else history_values
    row[f"rolling_mean_{ROLLING_WINDOW}"] = float(np.mean(window))
    row[f"rolling_std_{ROLLING_WINDOW}"] = float(np.std(window, ddof=0))

    return pd.DataFrame([row])


def fit_sklearn_model(model, train_supervised, feature_cols):
    X_train = train_supervised[feature_cols]
    y_train = train_supervised[TARGET_COL]
    model.fit(X_train, y_train)
    return {
        "kind": "sklearn",
        "model": model,
        "feature_cols": feature_cols,
    }


def recursive_forecast_sklearn(model, history_values, start_timestamp, horizon, freq, lags, feature_cols):
    history = list(map(float, history_values))
    future_index = pd.date_range(
        start=start_timestamp + pd.tseries.frequencies.to_offset(freq),
        periods=horizon,
        freq=freq,
    )

    preds = []
    for ts in future_index:
        X_next = make_feature_row(history, ts, lags)
        X_next = X_next.reindex(columns=feature_cols, fill_value=0)
        yhat = float(model.predict(X_next)[0])
        preds.append(yhat)
        history.append(yhat)

    return future_index, np.array(preds)


def build_lstm_model(window_size):
    model = Sequential([
        Input(shape=(window_size, 1)),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dense(16, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def build_transformer_model(window_size):
    inputs = Input(shape=(window_size, 1))

    x = Dense(32)(inputs)
    attn = MultiHeadAttention(num_heads=2, key_dim=16)(x, x)
    x = LayerNormalization()(x + attn)

    x2 = Dense(32, activation="relu")(x)
    x2 = Dense(32)(x2)
    x = LayerNormalization()(x + x2)

    x = GlobalAveragePooling1D()(x)
    x = Dense(32, activation="relu")(x)
    outputs = Dense(1)(x)

    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse")
    return model


def make_sequences(values, window_size):
    X, y = [], []
    for i in range(window_size, len(values)):
        X.append(values[i - window_size:i])
        y.append(values[i])
    X = np.array(X)
    y = np.array(y)
    return X[..., np.newaxis], y


def recursive_forecast_sequence_model(model, scaler, history_values, horizon, window_size):
    history_values = np.asarray(history_values, dtype=float).reshape(-1, 1)
    scaled_history = scaler.transform(history_values).ravel().tolist()

    if len(scaled_history) < window_size:
        raise ValueError(f"Not enough history for sequence window_size={window_size}")

    window = np.array(scaled_history[-window_size:], dtype=float)
    preds_scaled = []

    for _ in range(horizon):
        x = window.reshape(1, window_size, 1)
        yhat = float(model.predict(x, verbose=0)[0, 0])
        preds_scaled.append(yhat)
        window = np.append(window[1:], yhat)

    preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).ravel()
    return preds


def fit_arima(train_series):
    if not HAS_ARIMA:
        raise ImportError("statsmodels is not installed, ARIMA unavailable.")

    best_aic = np.inf
    best_fit = None
    best_order = None

    for p in [0, 1, 2]:
        for d in [0, 1]:
            for q in [0, 1, 2]:
                try:
                    fit = ARIMA(train_series, order=(p, d, q)).fit()
                    if fit.aic < best_aic:
                        best_aic = fit.aic
                        best_fit = fit
                        best_order = (p, d, q)
                except Exception:
                    continue

    if best_fit is None:
        raise RuntimeError("ARIMA grid search failed.")

    print(f"Best ARIMA order: {best_order} | AIC={best_aic}")
    return {
        "kind": "arima",
        "model": best_fit,
        "order": best_order,
    }


def fit_prophet(train_df):
    if not HAS_PROPHET:
        raise ImportError("prophet is not installed, Prophet unavailable.")

    prophet_df = train_df.reset_index()[["timestamp", TARGET_COL]].rename(
        columns={"timestamp": "ds", TARGET_COL: "y"}
    )
    m = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False)
    m.fit(prophet_df)
    return {
        "kind": "prophet",
        "model": m,
    }


def fit_sequence_model(train_series, model_type="lstm", window_size=SEQ_WINDOW, epochs=50):
    if not HAS_TF:
        raise ImportError(f"TensorFlow unavailable: {TF_IMPORT_ERROR}")

    train_values = np.asarray(train_series, dtype=float).reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled_train = scaler.fit_transform(train_values).ravel()

    X_train, y_train = make_sequences(scaled_train, window_size)
    if len(X_train) == 0:
        raise ValueError("Not enough data to train sequence model.")

    if model_type == "lstm":
        model = build_lstm_model(window_size)
    elif model_type == "transformer":
        model = build_transformer_model(window_size)
    else:
        raise ValueError("model_type must be 'lstm' or 'transformer'.")

    es = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)

    fit_kwargs = dict(
        x=X_train,
        y=y_train,
        epochs=epochs,
        batch_size=32,
        callbacks=[es],
        verbose=0,
        shuffle=False,
    )

    if len(X_train) >= 20:
        fit_kwargs["validation_split"] = 0.2

    model.fit(**fit_kwargs)

    return {
        "kind": model_type,
        "model": model,
        "scaler": scaler,
        "window_size": window_size,
    }


def evaluate_classical_time_series(train_df, test_df, model, lags):
    start_time = time.perf_counter()

    train_supervised = add_features(train_df, lags=lags)
    feature_cols = [c for c in get_feature_columns(lags) if c in train_supervised.columns]

    if len(train_supervised) < 5:
        raise ValueError("Not enough rows after feature engineering.")

    bundle = fit_sklearn_model(model, train_supervised, feature_cols)
    fitted_model = bundle["model"]

    history_values = train_df[TARGET_COL].tolist()
    freq = pd.infer_freq(train_df.index) or "H"

    _, preds = recursive_forecast_sklearn(
        fitted_model,
        history_values=history_values,
        start_timestamp=train_df.index[-1],
        horizon=len(test_df),
        freq=freq,
        lags=lags,
        feature_cols=feature_cols,
    )

    elapsed = time.perf_counter() - start_time
    metrics = compute_metrics(test_df[TARGET_COL].values, preds)
    return bundle, metrics, elapsed


def evaluate_arima(train_series, test_series):
    start_time = time.perf_counter()

    bundle = fit_arima(train_series)
    model = bundle["model"]
    preds = model.forecast(steps=len(test_series))
    preds = np.asarray(preds, dtype=float)

    elapsed = time.perf_counter() - start_time
    metrics = compute_metrics(test_series.values, preds)
    return bundle, metrics, elapsed


def evaluate_prophet(train_df, test_df):
    start_time = time.perf_counter()

    bundle = fit_prophet(train_df)
    model = bundle["model"]

    future_df = pd.DataFrame({"ds": test_df.index})
    forecast = model.predict(future_df)
    preds = forecast["yhat"].values.astype(float)

    elapsed = time.perf_counter() - start_time
    metrics = compute_metrics(test_df[TARGET_COL].values, preds)
    return bundle, metrics, elapsed


def evaluate_sequence_model(train_series, test_series, model_type="lstm", window_size=SEQ_WINDOW):
    start_time = time.perf_counter()

    bundle = fit_sequence_model(train_series, model_type=model_type, window_size=window_size)

    preds = recursive_forecast_sequence_model(
        bundle["model"],
        bundle["scaler"],
        history_values=train_series.values,
        horizon=len(test_series),
        window_size=window_size,
    )

    elapsed = time.perf_counter() - start_time
    metrics = compute_metrics(test_series.values, preds)
    return bundle, metrics, elapsed


# ---------------------------------------------------------------------
# Refit on full data for final selected model
# ---------------------------------------------------------------------
def refit_full_model(model_name, raw_df, ts_df, lags):
    classical_names = {
        "Linear Regression", "Ridge Regression", "Lasso Regression",
        "Random Forest", "Support Vector Regressor", "Gradient Boosting Regressor"
    }

    if model_name in classical_names:
        df = raw_df.copy()
        if "heure" not in df.columns and "hour" in df.columns:
            df = df.rename(columns={"hour": "heure"})

        X = df[["day", "heure"]]
        y = df[TARGET_COL]

        model_map = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(),
            "Lasso Regression": Lasso(),
            "Random Forest": RandomForestRegressor(),
            "Support Vector Regressor": SVR(),
            "Gradient Boosting Regressor": GradientBoostingRegressor(),
        }
        model = model_map[model_name]
        model.fit(X, y)
        return {"kind": "sklearn_legacy", "model": model}

    if model_name == "ARIMA":
        return fit_arima(ts_df[TARGET_COL])

    if model_name == "Prophet":
        return fit_prophet(ts_df)

    if model_name == "LSTM":
        return fit_sequence_model(ts_df[TARGET_COL], model_type="lstm", window_size=SEQ_WINDOW)

    if model_name == "Transformer":
        return fit_sequence_model(ts_df[TARGET_COL], model_type="transformer", window_size=SEQ_WINDOW)

    raise ValueError(f"Unknown model name: {model_name}")


def forecast_next_horizon(bundle, history_df, horizon=24, lags=None):
    freq = pd.infer_freq(history_df.index) or "H"
    last_ts = history_df.index[-1]

    if bundle["kind"] == "sklearn_legacy":
        future_rows = []
        for i in range(horizon):
            ts = last_ts + pd.tseries.frequencies.to_offset(freq) * (i + 1)
            future_rows.append([ts.dayofweek, ts.hour])

        future_df = pd.DataFrame(future_rows, columns=["day", "heure"])
        preds = bundle["model"].predict(future_df)
        future_index = pd.date_range(
            start=last_ts + pd.tseries.frequencies.to_offset(freq),
            periods=horizon,
            freq=freq,
        )
        return future_index, np.asarray(preds, dtype=float)

    if bundle["kind"] == "sklearn":
        future_index, preds = recursive_forecast_sklearn(
            bundle["model"],
            history_values=history_df[TARGET_COL].tolist(),
            start_timestamp=last_ts,
            horizon=horizon,
            freq=freq,
            lags=lags or BASE_LAGS,
            feature_cols=bundle["feature_cols"],
        )
        return future_index, preds

    if bundle["kind"] == "arima":
        preds = np.asarray(bundle["model"].forecast(steps=horizon), dtype=float)
        future_index = pd.date_range(
            start=last_ts + pd.tseries.frequencies.to_offset(freq),
            periods=horizon,
            freq=freq,
        )
        return future_index, preds

    if bundle["kind"] == "prophet":
        future_index = pd.date_range(
            start=last_ts + pd.tseries.frequencies.to_offset(freq),
            periods=horizon,
            freq=freq,
        )
        future_df = pd.DataFrame({"ds": future_index})
        forecast = bundle["model"].predict(future_df)
        preds = forecast["yhat"].values.astype(float)
        return future_index, preds

    if bundle["kind"] in {"lstm", "transformer"}:
        preds = recursive_forecast_sequence_model(
            bundle["model"],
            bundle["scaler"],
            history_values=history_df[TARGET_COL].values,
            horizon=horizon,
            window_size=bundle["window_size"],
        )
        future_index = pd.date_range(
            start=last_ts + pd.tseries.frequencies.to_offset(freq),
            periods=horizon,
            freq=freq,
        )
        return future_index, preds

    raise ValueError(f"Unknown bundle kind: {bundle['kind']}")


def save_best_model(bundle, base_path="best_model"):
    if bundle["kind"] in {"lstm", "transformer"}:
        bundle["model"].save(f"{base_path}.keras")
        meta = {k: v for k, v in bundle.items() if k != "model"}
        with open(f"{base_path}_meta.pkl", "wb") as f:
            pickle.dump(meta, f)
    else:
        with open(f"{base_path}.pkl", "wb") as f:
            pickle.dump(bundle, f)


def build_prediction_grid(preds):
    rows = []
    for i, yhat in enumerate(preds):
        day = (i // 24) % 7
        heure = i % 24
        rows.append([day, heure, float(yhat)])
    return pd.DataFrame(rows, columns=["day", "heure", "nb_machines"])


def save_multiple_outputs(best_bundle, history_df, lags):
    outputs = {
        "one_day": 24,
        "one_week": 7 * 24,
        "two_months": 8 * 7 * 24,
    }

    prediction_times = {}

    for name, horizon in outputs.items():
        start_time = time.perf_counter()

        _, preds = forecast_next_horizon(
            best_bundle,
            history_df,
            horizon=horizon,
            lags=lags,
        )

        elapsed = time.perf_counter() - start_time
        prediction_times[name] = elapsed

        out_df = build_prediction_grid(preds)
        filename = f"out_prediction_{name}.csv"
        out_df.to_csv(filename, index=False)

        print(f"Saved: {filename}")
        print(f"Prediction time for {name} (s): {format_seconds(elapsed)}")

    return prediction_times


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def build_and_train_model(data_path):
    total_start_time = time.perf_counter()

    raw_df = read_input_csv(data_path)
    print(raw_df.head())

    # 1) Classical regression models -> script1 logic
    classical_results, classical_models, classical_times = evaluate_classical_models_script1_style(raw_df)

    # 2) Time series models -> script2 logic
    ts_df = normalize_and_order(raw_df)
    split_idx = int(len(ts_df) * (1 - TEST_SIZE))
    train_df = ts_df.iloc[:split_idx].copy()
    test_df = ts_df.iloc[split_idx:].copy()

    if len(train_df) < 10 or len(test_df) < 1:
        raise ValueError("Dataset too small for train/test split.")

    adaptive_lags = get_adaptive_lags(len(train_df))
    adaptive_seq_window = min(SEQ_WINDOW, max(3, len(train_df) // 4))

    print(f"Adaptive lags used: {adaptive_lags}")
    print(f"Adaptive sequence window used: {adaptive_seq_window}")

    if HAS_TF:
        print("TensorFlow detected.")
    else:
        print(f"TensorFlow unavailable: {TF_IMPORT_ERROR}")

    ts_results = {}
    ts_models = {}
    ts_times = {}

    # ARIMA
    if HAS_ARIMA and len(train_df) >= 20:
        print("\nTraining model: ARIMA")
        try:
            bundle, metrics, elapsed = evaluate_arima(train_df[TARGET_COL], test_df[TARGET_COL])
            print(metrics)
            print(f"Execution time (s): {format_seconds(elapsed)}")
            ts_results["ARIMA"] = metrics
            ts_models["ARIMA"] = bundle
            ts_times["ARIMA"] = elapsed
        except Exception as e:
            print(f"Skipped ARIMA: {e}")
    else:
        print("\nARIMA skipped: not enough data or statsmodels not installed.")

    # Prophet
    if HAS_PROPHET:
        print("\nTraining model: Prophet")
        try:
            bundle, metrics, elapsed = evaluate_prophet(train_df, test_df)
            print(metrics)
            print(f"Execution time (s): {format_seconds(elapsed)}")
            ts_results["Prophet"] = metrics
            ts_models["Prophet"] = bundle
            ts_times["Prophet"] = elapsed
        except Exception as e:
            print(f"Skipped Prophet: {e}")
    else:
        print("\nProphet skipped: prophet not installed.")

    # LSTM
    if HAS_TF:
        print("\nTraining model: LSTM")
        try:
            bundle, metrics, elapsed = evaluate_sequence_model(
                train_df[TARGET_COL],
                test_df[TARGET_COL],
                model_type="lstm",
                window_size=adaptive_seq_window,
            )
            print(metrics)
            print(f"Execution time (s): {format_seconds(elapsed)}")
            ts_results["LSTM"] = metrics
            ts_models["LSTM"] = bundle
            ts_times["LSTM"] = elapsed
        except Exception as e:
            print(f"Skipped LSTM: {e}")
    else:
        print(f"\nLSTM skipped: TensorFlow not available ({TF_IMPORT_ERROR}).")

    # Transformer
    if HAS_TF:
        print("\nTraining model: Transformer")
        try:
            bundle, metrics, elapsed = evaluate_sequence_model(
                train_df[TARGET_COL],
                test_df[TARGET_COL],
                model_type="transformer",
                window_size=adaptive_seq_window,
            )
            print(metrics)
            print(f"Execution time (s): {format_seconds(elapsed)}")
            ts_results["Transformer"] = metrics
            ts_models["Transformer"] = bundle
            ts_times["Transformer"] = elapsed
        except Exception as e:
            print(f"Skipped Transformer: {e}")
    else:
        print(f"\nTransformer skipped: TensorFlow not available ({TF_IMPORT_ERROR}).")

    # Merge all results for PROMETHEE
    models_results = {}
    models_results.update(classical_results)
    models_results.update(ts_results)

    all_execution_times = {}
    all_execution_times.update(classical_times)
    all_execution_times.update(ts_times)

    if not models_results:
        raise RuntimeError("No model could be trained successfully.")

    print("\nAll model results:", models_results)

    print("\nExecution times (seconds):")
    for model_name, exec_time in all_execution_times.items():
        print(f"{model_name}: {format_seconds(exec_time)}")

    best_model_name, promethee_time, promethee_scores = promethee(models_results)
    print(f"\nBest model selected by PROMETHEE: {best_model_name}")

    models_total_time = sum(all_execution_times.values())
    total_models_plus_promethee_time = models_total_time + promethee_time
    wall_clock_time = time.perf_counter() - total_start_time

    print(f"\nTotal time for all model executions (s): {format_seconds(models_total_time)}")
    print(f"PROMETHEE time (s): {format_seconds(promethee_time)}")
    print(f"Total time (models + PROMETHEE) (s): {format_seconds(total_models_plus_promethee_time)}")
    print(f"Measured wall-clock time (s): {format_seconds(wall_clock_time)}")

    save_summary_csv(models_results, all_execution_times, promethee_scores, filename_prefix="model_summary")

    best_bundle = refit_full_model(best_model_name, raw_df, ts_df, lags=adaptive_lags)
    save_best_model(best_bundle, base_path="best_model")

    # Save outputs for the selected model and measure prediction time
    if best_bundle["kind"] == "sklearn_legacy":
        prediction_times = {}

        start_time = time.perf_counter()
        out_df = build_prediction_grid_classical_day(best_bundle["model"])
        prediction_times["one_day"] = time.perf_counter() - start_time
        out_df.to_csv("out_prediction_one_day.csv", index=False)
        print("Saved: out_prediction_one_day.csv")
        print(f"Prediction time for one_day (s): {format_seconds(prediction_times['one_day'])}")

        start_time = time.perf_counter()
        out_df = build_prediction_grid_classical_week(best_bundle["model"])
        prediction_times["one_week"] = time.perf_counter() - start_time
        out_df.to_csv("out_prediction_one_week.csv", index=False)
        print("Saved: out_prediction_one_week.csv")
        print(f"Prediction time for one_week (s): {format_seconds(prediction_times['one_week'])}")

        start_time = time.perf_counter()
        out_df = build_prediction_grid_classical_2months(best_bundle["model"])
        prediction_times["two_months"] = time.perf_counter() - start_time
        out_df.to_csv("out_prediction_two_months.csv", index=False)
        print("Saved: out_prediction_two_months.csv")
        print(f"Prediction time for two_months (s): {format_seconds(prediction_times['two_months'])}")
    else:
        prediction_times = save_multiple_outputs(best_bundle, ts_df, adaptive_lags)

    print("\nPrediction times for the best model:")
    for horizon_name, pred_time in prediction_times.items():
        print(f"{horizon_name}: {format_seconds(pred_time)} s")

    measured_prediction_time = sum(prediction_times.values())
    end_to_end_time = total_models_plus_promethee_time + measured_prediction_time

    print(f"\nTotal prediction time (s): {format_seconds(measured_prediction_time)}")
    print(f"End-to-end total time (models + PROMETHEE + predictions) (s): {format_seconds(end_to_end_time)}")

    return best_bundle


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python CGI_Prediction_Model.py <data_file.csv>")
        sys.exit(1)

    data_path = sys.argv[1]
    best_model = build_and_train_model(data_path)
    print("\nDone.")
