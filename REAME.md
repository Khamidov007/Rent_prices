Price Prediction Project

This project implements a complete machine learning pipeline to predict rental prices for housing listings in the USA. The workflow spans from initial data exploration and rigorous cleaning to advanced feature engineering and the development of a stacking ensemble model.
The primary goal of this project is to build a robust regression model capable of accurately predicting listing prices while handling the high cardinality and noise often found in real estate datasets. The project is organized into six sequential Jupyter notebooks, each handling a critical phase of the machine learning lifecycle.

### 1. Exploratory Data Analysis (`01_EDA.ipynb`)
The project begins with a deep dive into the raw dataset (`housing.csv`), which contains 384,977 entries and 22 columns.
    * Significant missing data was identified in `laundry_options` and `parking_options`.
    * Extreme outliers were detected: listings with 1,100 bedrooms, 75 bathrooms, and prices in the billions.
    * Target variable (`price`) and `sqfeet` displayed heavy right-skewness.
 Outlier removal and non-linear scaling (log1p) are mandatory for stable model performance.

### 2. Feature Engineering & Preprocessing (`02_feature_engineering.ipynb`)
This notebook transforms raw data into a model-ready format through a reproducible pipeline.
 Removed irrelevant high-cardinality text columns (`id`, `url`, `image_url`, `description`) and filtered outliers to keep values within realistic 99.9th percentile ranges.
Applied `log1p` transformation to `price` and `sqfeet` to normalize distributions.
* **Encoding Strategy:**
    * **One-Hot Encoding:** Applied to features like `type`, `state`, `laundry_options`, and `parking_options`.
    * **Target Encoding:** Used for the high-cardinality `region` column to capture geographic price trends.
* **Pipeline:** Built a unified `ColumnTransformer` and saved it as an artifact (`preprocessor.pkl`).

### 3. Model Training & Stacking (`03_model_training.ipynb` & `03_model_training2.ipynb`)
The training phase involved rigorous hyperparameter optimization and the creation of an ensemble.
* **Base Models:** Trained a suite of regressors including:
    * Decision Tree
    * Random Forest
    * Bagging Regressor (Tuned & Default)
    * XGBoost
    * CatBoost
* **Stacking Meta-Model:** A final ensemble model was created using a **Linear Regression meta-learner**. This model takes the predictions of all base models as input to make a final, more stable prediction.

### 4. Model Validation (`04_model_validation.ipynb`)
Models were evaluated on a dedicated validation set to create a performance leaderboard.
* **Metrics:** RMSE, MAE, MAPE, and R².
* **Leaderboard Results:**
    * **Bagging (Default)** and the **Stacking Meta-Model** emerged as the top performers.
    * **Decision Tree** was identified as the weakest model, showing signs of overfitting.

### 5. Final Testing (`05_model_test.ipynb`)
The project concludes with an evaluation of the best model (Stacking) on unseen test data.
Metrics were calculated by reversing the logarithmic transformation to provide real-world error margins.

---

##  Final Results (Original Scale)

The **Stacking Meta-Model** achieved the following results on the test set:

| Metric | Value |
| :--- | :--- |
| **RMSE** (Root Mean Squared Error) | **208.75** |
| **MAE** (Mean Absolute Error) | **78.90** |
| **MAPE** (Mean Absolute Percentage Error) | **14.21%** |
| **R² Score** | **~0.9** |

On average, the model's predictions are within approximately **$78.90** of the actual listing price, with a typical percentage error of **14%**. This demonstrates high reliability for real-estate price estimations.

---

## Tech Stack
* **Python** (Pandas, NumPy, Scikit-Learn)
* **Visualization:** Matplotlib, Seaborn
* **Algorithms:** XGBoost, CatBoost, Bagging, Random Forest
* **Advanced Techniques:** Target Encoding, Log Transformation, Stacking Ensembles