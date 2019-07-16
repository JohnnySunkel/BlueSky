# Prophet forecasts of daily total upgrades
import pandas as pd
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics

# Load the dataset
df = pd.read_csv('daily_total_upgrades_fbp.csv')
df.head()

# Define and fit the model with holidays
m = Prophet()
m.add_country_holidays(country_name = 'US')
m.fit(df)

# Cross validation
df_cv = cross_validation(m, 
                         initial = '730 days',
                         period = '180 days',
                         horizon = '90 days')
df_cv.head()

# Performance metrics
df_p = performance_metrics(df_cv)
df_p.head()

# Create a dataframe to hold predictions
future = m.make_future_dataframe(periods = 31)
future.tail()

# Make predictions
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(31)

# Plot forecasts
fig1 = m.plot(forecast)

# Plot forecast components
fig2 = m.plot_components(forecast)
