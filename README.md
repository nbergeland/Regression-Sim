# Regression-Sim
Regression (Demand Forecasting with scikit-learn + Plotly Visualization)
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import plotly.express as px

# 1. DATA PREPARATION
X = np.random.rand(1000, 5)
y = 50 + 10*X[:, 0] + 5*X[:, 2] + np.random.randn(1000)*2  # Synthetic demand

df = pd.DataFrame(X, columns=['price','marketing_spend','seasonality_index','competitor_price','holiday_flag'])
df['demand'] = y

# 2. TRAIN/TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    df.drop('demand', axis=1),
    df['demand'],
    test_size=0.2,
    random_state=42
)

# 3. MODEL TRAINING
model = LinearRegression()
model.fit(X_train, y_train)

# 4. PREDICTION
y_pred = model.predict(X_test)

# 5. VISUALIZATION: Actual vs. Predicted Demand
results_df = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred})
results_df.reset_index(drop=True, inplace=True)

fig = px.line(results_df, title='Regression: Actual vs. Predicted Demand')
fig.update_layout(xaxis_title='Test Sample Index', yaxis_title='Demand')
fig.show()

![Screenshot of project] (https://github.com/nbergeland/Regression-Sim/blob/main/REGRESSION.png)

print("Regression Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
