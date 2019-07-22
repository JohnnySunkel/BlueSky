# Prophet forecasts of daily total upgrades
import pandas as pd
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics

# Load the dataset
df = pd.read_csv('daily_total_upgrades_fbp.csv')
df.head()
df.tail()

# Define holidays
easter = pd.DataFrame({
        'holiday': 'easter',
        'ds': pd.to_datetime(['2017-04-16', 
                              '2018-04-01',
                              '2019-04-21']),
        'lower_window': 0,
        'upper_window': 0,
})

samsung_preorder = pd.DataFrame({
        'holiday': 'samsung_preorder',
        'ds': pd.to_datetime(['2017-03-30',
                              '2018-03-02',
                              '2019-02-21']),
        'lower_window': 0,
        'upper_window': 0,
})

samsung_launch = pd.DataFrame({
        'holiday': 'samsung_launch',
        'ds': pd.to_datetime(['2017-04-21',
                              '2018-03-16',
                              '2019-03-08']),
        'lower_window': 0,
        'upper_window': 1,
})

apple_preorder = pd.DataFrame({
        'holiday': 'apple_preorder',
        'ds': pd.to_datetime(['2017-09-15',
                              '2017-10-27',
                              '2018-09-14',
                              '2018-10-19']),
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

black_friday = pd.DataFrame({
        'holiday': 'black_friday',
        'ds': pd.to_datetime(['2017-11-24',
                              '2018-11-23']),
        'lower_window': 0,
        'upper_window': 1,
})

cyber_monday = pd.DataFrame({
        'holiday': 'cyber_monday',
        'ds': pd.to_datetime(['2017-11-27',
                              '2018-11-26']),
        'lower_window': 0,
        'upper_window': 0,
})

bogo = pd.DataFrame({
        'holiday': 'bogo',
        'ds': pd.to_datetime(['2018-05-03',
                              '2018-05-07',
                              '2018-06-03',
                              '2018-06-04',
                              '2018-06-07',
                              '2018-06-08',
                              '2018-06-09',
                              '2018-06-12',
                              '2018-06-13',
                              '2018-06-18',
                              '2018-07-06',
                              '2018-07-07',
                              '2018-07-13',
                              '2018-07-14',
                              '2019-05-01',
                              '2019-05-02',
                              '2019-05-03',
                              '2019-05-04',
                              '2019-05-06',
                              '2019-05-07',
                              '2019-05-08',
                              '2019-05-09',
                              '2019-05-10',
                              '2019-05-12',
                              '2019-05-13',
                              '2019-05-14',
                              '2019-05-15',
                              '2019-05-18',
                              '2019-05-21',
                              '2019-05-25',
                              '2019-05-31',
                              '2019-06-01',
                              '2019-06-03',
                              '2019-06-06',
                              '2019-06-08',
                              '2019-06-13',
                              '2019-06-17',
                              '2019-06-18',
                              '2019-06-19',
                              '2019-06-20',
                              '2019-06-22',
                              '2019-06-25',
                              '2019-06-26',
                              '2019-07-01']),
        'lower_window': 0,
        'upper_window': 0,
})

holidays = pd.concat((easter,
                      samsung_preorder,
                      samsung_launch,
                      apple_preorder,
                      apple_launch,
                      black_friday,
                      cyber_monday,
                      bogo))

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
future = m.make_future_dataframe(periods = 62)
future.tail()

# Make predictions
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(31)

# Plot forecasts
m.plot(forecast).savefig('daily_upgrades_forecast.png')

# Plot forecast components
m.plot_components(forecast).savefig('daily_upgrades_forecast_components.png')
