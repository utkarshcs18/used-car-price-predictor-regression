# Used Car Price Predictor

**A regression project: clean real-world used-car listings, engineer features from messy specification columns, compare 7 models, tune the best one, and save it for reuse with a command-line prediction script.**

<p align="center">
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Regression-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-LightGBM-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
</p>

> **Best model:** Gradient Boosting Regressor · **R² = `93.38`** · **MAE = ₹`14.40`** on a held-out test set

---

## 📌 Problem Statement

Pricing a used car is hard. Buyers and sellers often disagree because price depends on many interacting factors: age, engine specifications, fuel type, brand, transmission and ownership history.

**Goal:** build a model that predicts the selling price of a used car from its listing details, and identify which features matter most for price.

**Who could use this:** online car marketplaces (instant price estimates), dealerships (fair buying and selling prices), and individual sellers (checking whether a listing is over- or under-priced).

---

## 📊 Dataset

- **Source:** [CarDekho Used Car Data on Kaggle](`https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho`) (file: `CarData.csv`)
- **Size:** `6926` listings after cleaning (1,202 duplicate rows removed)
- **Target:** `selling_price` (INR)
- **Raw features:** `name`, `year`, `km_driven`, `fuel`, `seller_type`, `transmission`, `owner`, `mileage`, `engine`, `max_power`, `torque`, `seats`

---

## 🔧 Approach

| Step | What I did |
|------|------------|
| **1. Inspection** | Checked data types, missing values and duplicate records |
| **2. Cleaning** | Removed duplicates, handled missing values, and extracted numeric values from text columns stored with units (e.g. "1248 CC", "74 bhp") |
| **3. Feature engineering** | Parsed the messy `torque` column into a numeric `torque_nm` feature and extracted `brand` from the car name |
| **4. Encoding** | Converted categorical features (fuel, seller type, owner, brand) into numeric indicators |
| **5. Feature selection** | Kept 13 features based on their relationship with price (listed below) |
| **6. Modeling** | Trained and compared 7 regression models on the same train/test split |
| **7. Tuning** | Hyperparameter tuning on the top-performing models |
| **8. Packaging** | Saved the final model with Joblib and wrote `predict.py` to make predictions from user input |

**Selected features (13):** `max_power`, `torque_nm`, `engine`, `year`, `transmission`, `fuel_Diesel`, `fuel_Petrol`, `seller_type_Individual`, `owner_Test Drive Car`, `brand_BMW`, `brand_Mercedes-Benz`, `brand_Audi`, `brand_Volvo`

---

## 🏆 Results

All models were evaluated on the same held-out test set.

| Model | R² | MAE (₹) | RMSE (₹) |
|-------|----|---------|----------|
| Linear Regression | 62.57 | 29.16 | 52.73 |
| Lasso | -0.2 | 52.97 | 90.20 |
| Ridge | 66.22 | 29.30 | 52.37 |
| Random Forest | 92.77 | 14.02 | 24.23 |
| XGBoost | 91.03 | 16.57 | 26.99 |

| **Gradient Boosting (tuned)** | **93.38** | **14.40** | **23.18** |

**Takeaway:** "Boosting models clearly outperformed others linear models, which suggests the relationship between specifications and price is non-linear. Tuning improved R² from 91.03 to 93.38."

### What drives the price? (Feature Importance)

Based on this dataset, the `max_power` shows the higher Importance.

---

## 🧠 What I Learned

- **Real data is messy:** several columns stored numbers as text with units, and `torque` mixed different formats, so parsing and cleaning took more work than modeling.
- **Duplicates can inflate scores:** removing duplicate rows before splitting keeps the same car from appearing in both train and test sets.
- **Metrics must match the problem:** R² shows overall fit, but MAE is in rupees and is the number a real user cares about.
- **Model choice should be tested, not assumed:** I compared simple and complex models on the same split rather than picking one in advance.
- **Training and inference must match:** the model only works if input features are encoded exactly as they were during training, which is why packaging preprocessing with the model is my next step.

---

## 📁 Project Structure

```
used-car-price-predictor-regression/
├── data/
│   └── CarData.csv
├── notebooks/
│   └── car_price_prediction.ipynb   # Cleaning → features → modeling → evaluation
├── models/
│   └── best_car_price_model.pkl     # Saved trained model
├── predict.py                       # Predict a price from the command line
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ▶️ How to Run

**1. Clone and install**
```bash
git clone https://github.com/utkarshcs18/used-car-price-predictor-regression.git
cd used-car-price-predictor-regression
pip install -r requirements.txt
```

**2. Predict a price**
```bash
python predict.py
```
Enter the car's specifications when prompted to get an estimated selling price.

**3. Explore the analysis**
```bash
jupyter notebook notebooks/car_price_prediction.ipynb
```

**4. Load the saved model in Python**
```python
import joblib

model = joblib.load("models/best_car_price_model.pkl")
```

Input features must be prepared and encoded the same way as in training (see the 13 selected features above).

> **Note:** the saved model was trained with scikit-learn `v1.9.1`. Use the versions in `requirements.txt` to load it without warnings.

---

## ⚠️ Limitations

- The model uses a selected subset of the available features, and only four luxury brands (BMW, Mercedes-Benz, Audi, Volvo) are encoded explicitly.
- Condition, accident history, location and service records are not captured.
- Prices reflect the period when the data was collected and would need retraining as the market changes.
- `predict.py` builds the input features by hand, so inputs must match the training encoding exactly.

## 🛠️ Tech Stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `XGBoost` · `Matplotlib` · `Seaborn` · `Jupyter` · `joblib`

---
