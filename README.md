# Real Estate Price Prediction Using Machine Learning and 10-Fold Cross-Validation

**Course:** Machine Learning Fundamentals (AIC354), COMSATS University Islamabad, Lahore Campus — Assignment 3

**Live application URL:** PENDING — add the real Streamlit link here after deployment succeeds.

## Problem statement
Predict the numerical *house price of unit area* from six property features (supervised regression).

## Dataset
Kaggle — Real Estate Price Prediction (quantbruce): https://www.kaggle.com/datasets/quantbruce/real-estate-price-prediction  
File: `Real estate.csv` (commonly 414 rows; the notebook prints the actual count).  
**Target:** `Y house price of unit area`  
**Features:** `X1 transaction date`, `X2 house age`, `X3 distance to the nearest MRT station`, `X4 number of convenience stores`, `X5 latitude`, `X6 longitude`. The `No` column is only an ID and is not used.

## Five algorithms (different from the Car Price example)
1. Support Vector Regression (SVR) 2. AdaBoost Regressor 3. Extra Trees Regressor 4. MLP Regressor 5. Gaussian Process Regressor

## Evaluation method
`KFold(n_splits=10, shuffle=True, random_state=42)`, identical folds for all models. Scaling lives inside scikit-learn pipelines so it is fitted only on training folds. Primary metric: **RMSE** (mean and standard deviation over 10 folds); MAE and R² are supplementary.

## Results summary
Measured with 10-fold cross-validation on 414 rows (from `model_results.csv`; lower RMSE is better):

| Rank | Model | Mean RMSE | Std RMSE | Mean MAE | Mean R² |
|------|-------|-----------|----------|----------|---------|
| 1 | SVR | 7.3334 | 2.1055 | 4.8245 | 0.6984 |
| 2 | Gaussian Process | 7.3977 | 2.0485 | 4.9983 | 0.6917 |
| 3 | Extra Trees | 7.7897 | 1.8066 | 4.9190 | 0.6563 |
| 4 | AdaBoost | 8.1735 | 1.8709 | 5.7965 | 0.6115 |
| 5 | MLP Regressor | 9.9276 | 2.6120 | 6.2906 | 0.4275 |

**Best model: SVR** (mean RMSE 7.3334). The gap between SVR and Gaussian Process (0.064) is much smaller than the fold-to-fold standard deviation (about 2), so these two perform very similarly. The chart is in `rmse_comparison.png`.

## Files
| File | Purpose |
|------|---------|
| `Real_Estate_Price_Prediction.ipynb` | Colab notebook: exploration, 5 models, 10-fold CV, charts, model export |
| `Real estate.csv` | The dataset downloaded from Kaggle — **add this file yourself** (see note below) |
| `app.py` | Streamlit web app that loads the saved model and predicts |
| `requirements.txt` | Packages the web app needs |
| `README.md` | This file |
| `best_model.joblib` | Fitted best pipeline — created by the notebook |
| `model_results.csv` | Real comparison results — created by the notebook |
| `rmse_comparison.png` | RMSE bar chart — created by the notebook |
| `best_model_info.json` | Extra file created by the notebook: model name, feature order, input ranges used by the app |
| `requirements_from_colab.txt` | Exact package versions used in Colab (created by the notebook) |

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud (free; check their site for current requirements)
1. Create a GitHub account and a new **public** repository, e.g. `real-estate-price-prediction`.
2. Upload: `app.py`, `requirements.txt`, `best_model.joblib`, `best_model_info.json`, `README.md` (and optionally the notebook, CSV, results).
3. Replace `requirements.txt` contents with the versions in `requirements_from_colab.txt` so the saved model loads on the same scikit-learn version.
4. Go to https://share.streamlit.io, sign in with GitHub, click **Create app / New app**.
5. Choose the repository, branch `main`, main file path `app.py`.
6. (Advanced settings) choose a Python version close to Colab's if offered, then click **Deploy**.
7. Wait for the build, then test with a few inputs.
8. Copy the real URL into the first cell of the notebook and into this README.

## Limitations
Small dataset from one region (Taiwan), six features, initial parameters only. Predictions are estimates, not market prices.
