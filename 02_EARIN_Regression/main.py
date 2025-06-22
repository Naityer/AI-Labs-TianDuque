import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

# 1. Data preparation
# Load the diabetes dataset and convert to a pandas DataFrame
raw = load_diabetes()
X = pd.DataFrame(raw.data, columns=raw.feature_names)
y = np.log1p(raw.target)  # Apply log1p transformation to the target to reduce skewness

# Select and create relevant features based on correlation and interactions
X = X[['age', 'bmi', 'bp', 's4', 's5']].copy()
X['bmi_s5'] = X['bmi'] * X['s5']            # Nonlinear interaction between BMI and S5
X['bp_s4'] = X['bp'] / (X['s4'] + 1e-6)     # Division interaction between BP and S4
X['log_s4'] = np.log1p(X['s4'] - X['s4'].min())  # Log transformation of S4 to reduce skewness

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Model definitions
# Model 1: Polynomial Ridge Regression (degree=2, regularization alpha=1.0)
model1 = make_pipeline(PolynomialFeatures(degree=2, include_bias=False), Ridge(alpha=1.0))

# Model 2: Support Vector Regression with RBF kernel
model2 = SVR(kernel='rbf', C=1.0, epsilon=0.2)

# 4. Model evaluation using 4-fold cross-validation (R² as metric)
cv = KFold(n_splits=4, shuffle=True, random_state=42)
scores1 = cross_val_score(model1, X_train, y_train, cv=cv, scoring="r2")
scores2 = cross_val_score(model2, X_train, y_train, cv=cv, scoring="r2")

print("Cross-Validation Results:")
print(f"Polynomial Ridge CV R2: {scores1.mean():.3f}")
print(f"SVR (RBF) CV R2: {scores2.mean():.3f}\n")

# Final model training on full training data
model1.fit(X_train, y_train)
model2.fit(X_train, y_train)

# Apply inverse log1p to get original target scale
pred1 = np.expm1(model1.predict(X_test))
pred2 = np.expm1(model2.predict(X_test))
y_test_exp = np.expm1(y_test)

# Final performance metrics on test set
print("Final Evaluation on Test Set:")
print(f"Polynomial Ridge - MSE: {mean_squared_error(y_test_exp, pred1):.2f} | R2: {r2_score(y_test_exp, pred1):.3f}")
print(f"SVR (RBF) - MSE: {mean_squared_error(y_test_exp, pred2):.2f} | R2: {r2_score(y_test_exp, pred2):.3f}")
