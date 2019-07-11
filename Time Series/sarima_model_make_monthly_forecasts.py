from pandas import read_csv
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Specify the training data 
series = read_csv('monthly-total-upgrades.csv',
                header = 0,
                index_col = 0)
data = series.values

# Define the model
model = SARIMAX(data,
                order = (0, 1, 1),
                seasonal_order = (0, 0, 1, 0),
                trend = 'n')

# Fit the model
model_fit = model.fit()

# One step forecast
y_hat = model_fit.forecast()
print(y_hat)
