# Short explanation for my instructor

1. **Why this project:** It is similar to car price prediction: predict a price from properties of an item, using a real dataset.
2. **Supervised regression:** The data has known prices (labels) and the output is a continuous number.
3. **Inputs / output:** Inputs are transaction date, house age, distance to nearest MRT station, number of convenience stores, latitude, longitude. Output is house price of unit area.
4. **10-Fold Cross-Validation:** The data is split into 10 parts; the model trains on 9 and is tested on 1, repeated 10 times so every row is tested once. Results are averaged.
5. **Why these five algorithms:** SVR, AdaBoost, Extra Trees, MLP Regressor and Gaussian Process are all regression methods with different ideas (margin-based, boosting, randomised trees, neural network, probabilistic kernel) and none is among the five used in the Car Price example.
6. **RMSE:** The typical size of the prediction error in the same units as the price, with big mistakes penalised more. Lower is better.
7. **Best model:** **SVR** with mean 10-fold CV RMSE = **7.3334** (std 2.1055). Gaussian Process was very close (7.3977), then Extra Trees (7.7897), AdaBoost (8.1735) and MLP Regressor (9.9276).
8. **Deployment:** The best pipeline is saved with joblib, loaded by a Streamlit app (`app.py`) hosted on Streamlit Community Cloud; the user enters six values and clicks Predict. Live link: (add after deployment)
9. **Limitations:** Small dataset from one area, only six features, little hyperparameter tuning, prices may change over time, so predictions are estimates only.
