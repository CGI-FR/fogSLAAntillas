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
- Evaluates models using RMSE, MAE, R²
- Selects the best model through a PROMETHEE-like scoring method
- Saves the best model as `best_model.joblib`
- Generates predictions for 7×24 hours → `out_prediction_one_week.csv`

---

## 📦 Installation

```bash
git clone https://github.com/CGI-FR/fogSLAAntillas.git
cd fogSLAAntillas
pip install -r requirements.txt
```

Requires:  
`pandas`, `scikit-learn`, `joblib`

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
- `best_model.joblib`  
- `out_prediction_one_week.csv`

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
from joblib import load
import CGI_Prediction_Model as cgi

model = load("best_model.joblib")
prediction = cgi.predict_vms(2, 15, model)
print("Predicted VMs:", prediction[0])
```
---

## 📈 Example output: out_prediction_one_week.csv
| day | hour | nb_machine |
|-----|------|------------|
| 0   | 0   | 4          |
| 0   | 1   | 4          |
| ... | ... | ...        |




---

## 📄 License

BSD-3 License
