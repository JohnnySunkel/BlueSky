# Prophet forecasts of daily total aal data
import pandas as pd
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics

# Load the dataset
df = pd.read_csv('daily_total_aal_data_no_apr.csv')
df.head()
df.tail()

# Remove Black Friday outliers
df.loc[(df['ds'] == '2017-11-24') & (df['ds'] == '2018-11-23'), 'y'] = None

# Define holidays
easter = pd.DataFrame({
        'holiday': 'easter',
        'ds': pd.to_datetime(['2017-04-16', 
                              '2018-04-01',
                              '2019-04-21']),
        'lower_window': 0,
        'upper_window': 0,
})

samsung_launch = pd.DataFrame({
        'holiday': 'samsung_launch',
        'ds': pd.to_datetime(['2017-04-21',
                              '2018-03-16',
                              '2019-03-08']),
        'lower_window': 0,
        'upper_window': 0,
})

apple_launch = pd.DataFrame({
        'holiday': 'apple_launch',
        'ds': pd.to_datetime(['2017-09-22',
                              '2017-11-03',
                              '2018-09-21',
                              '2018-10-26']),
        'lower_window': 0,
        'upper_window': 1,
})

holidays = pd.concat((easter,
                      samsung_launch,
                      apple_launch))

# Define and fit the model
m = Prophet(holidays = holidays)
m.add_country_holidays(country_name = 'US')
m.train_holiday_names
m.fit(df)

# Cross validation
df_cv = cross_validation(m, 
                         initial = '730 days',
                         period = '180 days',
                         horizon = '60 days')
df_cv.head()

# Performance metrics
df_p = performance_metrics(df_cv)
df_p.head()

# Create a dataframe to hold predictions
future = m.make_future_dataframe(periods = 30)
future.tail()

# Make predictions
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30)

# Plot forecasts
m.plot(forecast).savefig('daily_total_aal_forecast.png')

# Plot forecast components
m.plot_components(forecast).savefig('daily_total_aal_forecast_components.png')
