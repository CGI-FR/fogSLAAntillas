# CGI Prediction Model – Dynamic Virtual Machine Forecasting

This project implements a system for **predicting the number of virtual machines (VMs)** to deploy based on the **day of the week** and the **hour of the day**.

It trains multiple regression models, evaluates their performance, and automatically selects the **best model** using a PROMETHEE-like multi‑criteria decision method.

---

## ✨ Key Features

- Loads dataset containing: `day ; hour ; nb_machine`
- Trains multiple regression models:
  - Linear Regression
  - Ridge Regression
  - Lasso Regression
  - Random Forest
  - Support Vector Regressor (SVR)
  - Gradient Boosting Regressor
  -  The AutoRegressive Integrated Moving Average (ARIMA)
  -  Prophet
  -  Long Short-Term Memory (LSTM) networks
  -  Transformer-based models
- Evaluates models using RMSE, MAE, R²
- Selects the best model through a PROMETHEE-like scoring method
- Saves the best model as `best_model.pkl`
- Generates predictions for 24 hours → out_prediction_one_day.csv
- Generates predictions for 7×24 hours → `out_prediction_one_week.csv`
- Generates predictions for 8×7×24 hours → out_prediction_two_months.csv

---

## 📦 Installation

```bash
git clone https://github.com/CGI-FR/fogSLAAntillas.git
cd fogSLAAntillas
pip install -r requirements.txt
```

Requires:  
`numpy==2.4.4`,`pandas==3.0.2`,`prophet==1.3.0`,`scikit_learn==1.8.0`,`statsmodels==0.14.6`,`tensorflow==2.21.0`

---

## 📊 Input CSV Format

| day | hour | nb_machine |
|-----|------|------------|
| 0   | 13   | 5          |
| 0   | 14   | 7          |
| ... | ...  | ...        |

Separator: `;`

---

## 🚀 Usage

```bash
python CGI_Prediction_Model.py data.csv
```

Outputs:
- `best_model.pkl`
- `model_summary.csv`
- `out_prediction_one_day.csv`
- `out_prediction_one_week.csv`
- `out_prediction_two_months.csv`

---

## 🧠 How It Works

1. Loads dataset from CSV  
2. Splits training/testing sets  
3. Trains all regression models  
4. Computes RMSE, MAE, R²  
5. Selects best model with PROMETHEE-like method  
6. Saves best model  
7. Generates one-week hourly predictions  

---

## 🧪 Manual Prediction Example

```python
import pickle
import CGI_Prediction_Model as cgi

with open("best_model.pkl", "rb") as f:
    bundle = pickle.load(f)

model = bundle["model"]
prediction = cgi.predict_vms(0, 2, model)
print("Predicted VMs:", prediction[0])
```
---

This code loads a previously trained machine learning model and uses it to predict the number of virtual machines (VMs) needed for a specific time.

-  2 corresponds to the 3rd day of the week (counting starts from 0).
-  15 corresponds to 15:00 (3 PM).

The function predict_vms() sends these values to the model, which returns the estimated number of VMs required at that time.

## 📈 Example output: out_prediction_one_week.csv
| day | hour | nb_machine |
|-----|------|------------|
| 0   | 0   | 4          |
| 0   | 1   | 4          |
| ... | ... | ...        |




---

## 📄 License

BSD-3 License
