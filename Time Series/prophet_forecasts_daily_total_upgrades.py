# Prophet forecasts of daily total upgrades
import pandas as pd
from fbprophet import Prophet

# Load the dataset
df = pd.read_csv('daily_total_upgrades_fbp.csv')
df.head()

# Define and fit the model
m = Prophet()
m.fit(df)

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
