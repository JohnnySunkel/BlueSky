# Prophet forecasting example
import pandas as pd
from fbprophet import Prophet

# Load the dataset
df = pd.read_csv('example_wp_log_peyton_manning.csv')
df.head()

# Define and fit the model
m = Prophet()
m.fit(df)

# Create a dataframe to hold predictions
future = m.make_future_dataframe(periods = 365)
future.tail()

# Make predictions
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

# Plot forecasts
fig1 = m.plot(forecast)

# Plot forecast components
fig2 = m.plot_components(forecast)
