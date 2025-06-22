# EARIN_Regression - Diabetes Progression Prediction

This project was developed as part of the EARIN (Evolutionary and Artificial Intelligence) course. It focuses on solving a regression problem using the classic `diabetes` dataset from `sklearn.datasets`.

The goal is to **predict the disease progression** using advanced machine learning models and proper data preprocessing techniques.

---

## 👨‍💻 Author

## Professor
- Filip Szatkowski

## Date
20/04  Summer 2025

## Variant 2
Predict the disease progression

## Students
- Tian Duque Rey
- Eduardo Sánchez Belchí

---

## 📊 Dataset

The dataset contains:
- 442 samples
- 10 numeric features
- Target: quantitative measure of disease progression one year after baseline

---

## 🧪 Project Structure

### `main.py`
Contains the final implementation that:
- Selects and engineers the most relevant features
- Applies log-transform to stabilize target distribution
- Tests two regression models: Polynomial Ridge & SVR (RBF)
- Performs 4-fold cross-validation and evaluates performance on test set

### `compare_models.py`
Extended experimentation with:
- More models: Lasso, XGBoost, LightGBM, MLP, Stacking
- CV metrics, test metrics, and visual comparisons
- Feature engineering class and pipelines

### `requirements.txt`
List of all Python dependencies used in the project.

### `Diabetes_Model_Report.pdf`
Final report explaining:
- Feature analysis and decisions
- Model design and evaluation
- Cross-validation results
- Final conclusions

---

## 📈 Best Results

| Model             | Test R² | CV R²  | MAE  |
|------------------|---------|--------|------|
| Polynomial Ridge | 0.461   | 0.325  | 42.4 |
| SVR (RBF)        | 0.380   | 0.354  | 44.8 |

> Ridge Regression with polynomial expansion provided the best trade-off between training and generalization error.

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
python main.py
```
