import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
import sys
from contextlib import contextmanager
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import StackingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline

# Suppress warnings and configure environment
warnings.filterwarnings('ignore')
os.environ['LOKY_MAX_CPU_COUNT'] = '4'
np.random.seed(42)
sns.set_palette("husl")
plt.style.use('seaborn-v0_8-darkgrid')
pd.set_option('display.max_columns', None)

output_dir = "diabetes_model_outputs"
os.makedirs(output_dir, exist_ok=True)

# Context manager to suppress output for clean execution
@contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

# Custom transformer for advanced feature engineering
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.keep = ['age', 'bmi', 'bp', 's4', 's5']

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X, columns=['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6'])
        X_df = X_df[self.keep].copy()
        X_df['bmi_s5'] = X_df['bmi'] * X_df['s5']  # Interaction term
        X_df['bp_s4'] = X_df['bp'] / (X_df['s4'] + 1e-6)  # Avoid division by zero
        X_df['log_s4'] = np.log1p(X_df['s4'] - X_df['s4'].min())  # Normalize skew
        return X_df

    def get_feature_names(self):
        return ['age', 'bmi', 'bp', 's4', 's5', 'bmi_s5', 'bp_s4', 'log_s4']

# Main predictor class to run all models
class DiabetesPredictor:
    def __init__(self):
        self.results = pd.DataFrame(columns=['Model', 'Test MSE', 'Test R2', 'Test MAE', 'CV R2', 'Time'])
        self.best_model = None
        self._load_data()
        self._engineer_features()
        self._plot_target_distribution()

    def _load_data(self):
        data = load_diabetes()
        self.X = data.data
        self.y = np.log1p(data.target)  # Apply log1p to normalize target distribution

    def _engineer_features(self):
        self.fe = FeatureEngineer()
        X_transformed = self.fe.fit_transform(self.X)
        X_train, X_test, y_train, y_test = train_test_split(X_transformed, self.y, test_size=0.2, random_state=42)
        self.scaler = StandardScaler()
        self.X_train = pd.DataFrame(self.scaler.fit_transform(X_train), columns=self.fe.get_feature_names())
        self.X_test = pd.DataFrame(self.scaler.transform(X_test), columns=self.fe.get_feature_names())
        self.y_train = y_train
        self.y_test = y_test

    def _plot_target_distribution(self):
        plt.figure(figsize=(8, 4))
        sns.histplot(np.expm1(self.y), kde=True, bins=30, color='orange')
        plt.title("Distribución del target original (sin log)")
        plt.xlabel("Valor del target (progresión de diabetes)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "target_distribution.png"))
        plt.close()

    def _evaluate_model(self, model, name):
        start = time.time()
        model.fit(self.X_train, self.y_train)

        # Perform 4-fold cross-validation
        cv = KFold(n_splits=4, shuffle=True, random_state=42)
        cv_r2_scores = cross_val_score(model, self.X_train, self.y_train, cv=cv, scoring='r2', n_jobs=-1)
        mean_cv_r2 = np.mean(cv_r2_scores)

        # Evaluate on test set (after inverse log1p)
        test_pred = np.expm1(model.predict(self.X_test))
        y_true = np.expm1(self.y_test)
        mse = mean_squared_error(y_true, test_pred)
        r2 = r2_score(y_true, test_pred)
        mae = mean_absolute_error(y_true, test_pred)

        # Record results
        self.results = pd.concat([self.results, pd.DataFrame([{
            'Model': name,
            'Test MSE': mse,
            'Test R2': r2,
            'Test MAE': mae,
            'CV R2': mean_cv_r2,
            'Time': time.time() - start
        }])], ignore_index=True)

        # Select best model based on test MSE
        if self.best_model is None or mse < self.best_model[1]:
            self.best_model = (model, mse, name, r2)

        print(f"{name.ljust(25)} MSE: {mse:.2f} | R2: {r2:.3f} | MAE: {mae:.2f} | CV R2: {mean_cv_r2:.3f} | Time: {time.time() - start:.1f}s")

    def train_models(self):
        # Evaluate Ridge (base and optimized)
        base_ridge = make_pipeline(PolynomialFeatures(2, include_bias=False), Ridge(alpha=1.0))
        grid_ridge = GridSearchCV(
            estimator=make_pipeline(PolynomialFeatures(2, include_bias=False), Ridge()),
            param_grid={'ridge__alpha': [0.01, 0.1, 1.0, 10.0, 100.0]},
            scoring='r2', cv=4, n_jobs=-1
        )
        self._evaluate_model(base_ridge, "Polynomial Ridge (alpha=1.0)")
        self._evaluate_model(grid_ridge, "Polynomial Ridge (GridSearch)")

        # Evaluate SVR (base and optimized)
        base_svr = make_pipeline(PolynomialFeatures(2), SVR(C=1.0, epsilon=0.2))
        grid_svr = GridSearchCV(
            estimator=make_pipeline(PolynomialFeatures(2), SVR()),
            param_grid={
                'svr__C': [0.1, 1.0, 10.0],
                'svr__epsilon': [0.01, 0.1, 0.2]
            },
            scoring='r2', cv=4, n_jobs=-1
        )
        self._evaluate_model(base_svr, "SVR (C=1.0, eps=0.2)")
        self._evaluate_model(grid_svr, "SVR (GridSearch)")

        # Evaluate additional models
        self._evaluate_model(Lasso(alpha=0.1, max_iter=10000), "Lasso")
        self._evaluate_model(XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, subsample=0.9, colsample_bytree=0.8, n_jobs=4, random_state=42), "XGBoost")
        self._evaluate_model(LGBMRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, subsample=0.9, colsample_bytree=0.8, n_jobs=4, random_state=42, verbose=-1), "LightGBM")
        self._evaluate_model(MLPRegressor(hidden_layer_sizes=(64, 64), activation='relu', max_iter=1000, random_state=42), "Neural Network")
        self._evaluate_model(StackingRegressor(estimators=[
            ('xgb', XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, n_jobs=4, random_state=42)),
            ('lgb', LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, n_jobs=4, random_state=42, verbose=-1)),
        ], final_estimator=Ridge(alpha=1.0), n_jobs=4), "Stacking Ensemble")

    def analyze_results(self):
        print("\n=== FINAL RESULTS ===")
        final_results = self.results.sort_values(by='Test MSE')
        print(final_results.to_string(index=False))
        self.results.to_csv(os.path.join(output_dir, 'model_results.csv'), index=False)
        print(f"\n=== BEST MODEL: {self.best_model[2]} ===")
        print(f"MSE: {self.best_model[1]:.2f}")
        print(f"R²: {self.best_model[3]:.3f}")

        # Plot performance comparison
        plt.figure(figsize=(14, 6))

        plt.subplot(1, 2, 1)
        sns.barplot(data=final_results, y='Model', x='Test R2', palette='Blues_d')
        plt.title('Test R² por Modelo')
        plt.xlabel('R² (Test)')

        plt.subplot(1, 2, 2)
        sns.barplot(data=final_results, y='Model', x='CV R2', palette='Greens_d')
        plt.title('CV R² por Modelo')
        plt.xlabel('R² (Cross-Validation)')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=300)
        plt.close()

if __name__ == "__main__":
    print("=== DIABETES PREDICTION SYSTEM - v3.5 (Best-on-Test Strategy) ===")
    predictor = DiabetesPredictor()
    predictor.train_models()
    predictor.analyze_results()
