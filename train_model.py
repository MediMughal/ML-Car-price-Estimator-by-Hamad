import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("pakwheels_used_cars.csv")
print("✅ Dataset Loaded")

# =========================
# 2. CLEANING
# =========================
df['year'] = df['year'].replace(1905, df['year'].median())

df['engine_cc'] = pd.to_numeric(df['engine_cc'], errors='coerce')
df['mileage'] = pd.to_numeric(df['mileage'], errors='coerce')

df = df.dropna(subset=['price', 'engine_cc', 'mileage', 'year'])

# =========================
# 3. OUTLIERS
# =========================
df = df[df['price'] < df['price'].quantile(0.99)]
df = df[df['price'] > df['price'].quantile(0.01)]

# =========================
# 4. FEATURE ENGINEERING
# =========================
current_year = 2026

df['car_age'] = current_year - df['year']
df['mileage_per_year'] = df['mileage'] / (df['car_age'] + 1)
df['engine_per_age'] = df['engine_cc'] / (df['car_age'] + 1)

# =========================
# 5. ENCODING
# =========================
le_make = LabelEncoder()
df['make_enc'] = le_make.fit_transform(df['make'].astype(str))

# =========================
# 6. FEATURES & TARGET
# =========================
feature_columns = [
    'make_enc',
    'engine_cc',
    'mileage',
    'year',
    'mileage_per_year',
    'engine_per_age'
]

X = df[feature_columns]
y = np.log1p(df['price'])

# =========================
# 7. TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 8. MODELS (NO SVM HERE)
# =========================
models = {

    "Linear Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),

    "Decision Tree": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", DecisionTreeRegressor(random_state=42))
    ]),

    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=42
        ))
    ]),

    "KNN": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", KNeighborsRegressor(n_neighbors=5))
    ]),

    "XGBoost": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        ))
    ]),

    "CatBoost": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", CatBoostRegressor(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            verbose=0,
            random_state=42
        ))
    ])
}

# =========================
# 9. TRAIN & COMPARE
# =========================
best_model = None
best_mae = float("inf")
best_name = ""

results = []

print("\n📊 Model Comparison:\n")

for name, model in models.items():
    print(f"🚀 Training {name}...")

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    results.append((name, r2, mae))

    print(f"{name}")
    print(f"R2: {r2:.4f}")
    print(f"MAE: {mae:.4f}\n")

    if mae < best_mae:
        best_mae = mae
        best_model = model
        best_name = name
    joblib.dump(best_name, "best_model_name.pkl")
# =========================
# 10. RESULTS
# =========================
results_df = pd.DataFrame(results, columns=["Model", "R2", "MAE"])
results_df = results_df.sort_values(by="MAE")

print("\n🏆 FINAL RANKING:")
print(results_df)

print("\n🔥 BEST MODEL:", best_name)
print("🔥 BEST MAE:", best_mae)

# =========================
# 11. SAVE MODEL
# =========================
joblib.dump(best_model, "car_price_model.pkl")
joblib.dump(le_make, "make_encoder.pkl")
joblib.dump(feature_columns, "feature_columns.pkl")

print("\n✅ Model saved successfully!")

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# === Classification banane ke liye ===
def price_category(price):
    if price < 1000000:
        return 0
    elif price < 3000000:
        return 1
    else:
        return 2

y_class = df['price'].apply(price_category)

X_cls = df[['engine_cc', 'mileage', 'year']]

clf = RandomForestClassifier()
clf.fit(X_cls, y_class)

y_pred = clf.predict(X_cls)

accuracy = accuracy_score(y_class, y_pred)
precision = precision_score(y_class, y_pred, average='weighted')
recall = recall_score(y_class, y_pred, average='weighted')
f1 = f1_score(y_class, y_pred, average='weighted')
cm = confusion_matrix(y_class, y_pred)

# SAVE FILE
joblib.dump({
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "cm": cm
}, "evaluation.pkl")

# for Run this train_model.py  (C:\Users\mediq\miniconda3\python.exe train_model.py)
